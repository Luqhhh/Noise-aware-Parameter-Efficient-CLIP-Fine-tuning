"""SCOPE-K2 parent/evidence cache contracts and deterministic persistence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.scope_protocol import (
    EVIDENCE_VIEW_ORDER,
    EVIDENCE_VIEW_WEIGHTS,
    PARENT_BRANCH_ORDER,
    ScopeProtocol,
    four_neighbor_edges,
)


@dataclass(frozen=True)
class ParentBatchScores:
    constituent_scores: torch.Tensor
    constituent_top1: torch.Tensor
    fused_log_scores: torch.Tensor


@dataclass(frozen=True)
class ScopeCacheManifest:
    path: Path
    sha256: str
    semantic_sha256: str


VALIDATION_DIAGNOSTIC_FIELDS = (
    "label", "clean_probability", "pseudo_label", "correction_alpha",
)
LINEAGE_FIELDS = {
    "checkpoint_sha256", "split_sha256", "class_to_idx_sha256",
    "idx_to_class_sha256", "trust_bundle_sha256", "exact_group_sha256",
    "protocol_sha256", "code_sha256", "dirty_diff_sha256", "lockfile_sha256",
}


def scope_parent_batch_scores(
    global_original: torch.Tensor,
    local_original: Sequence[torch.Tensor],
    global_flipped: torch.Tensor,
    local_flipped: Sequence[torch.Tensor],
    *,
    global_temperature: float = 1.5,
    local_temperature: float = 1.5,
    local_scale_weights: Sequence[float] = (0.2, 0.3, 0.4, 0.1),
    local_weight: float = 0.4,
    flip_weight: float = 0.5,
) -> ParentBatchScores:
    """Apply the frozen four-branch, eight-local-view FULLFT_DUAL fusion."""
    if len(local_original) != 4 or len(local_flipped) != 4:
        raise ValueError("SCOPE parent requires exactly four local scales per orientation")
    views = [global_original, global_flipped, *local_original, *local_flipped]
    if global_original.ndim != 2 or any(value.shape != global_original.shape for value in views[1:]):
        raise ValueError("SCOPE parent logits must share one [N,C] shape")
    if not all(torch.isfinite(value).all() for value in views):
        raise ValueError("SCOPE parent logits must be finite")
    weights = tuple(float(value) for value in local_scale_weights)
    if len(weights) != 4 or any(value < 0.0 for value in weights) or abs(sum(weights) - 1.0) > 1.0e-12:
        raise ValueError("SCOPE local scale weights must be four probabilities")
    if min(float(global_temperature), float(local_temperature)) <= 0.0:
        raise ValueError("SCOPE branch temperatures must be positive")
    if not 0.0 <= float(local_weight) <= 1.0 or not 0.0 <= float(flip_weight) <= 1.0:
        raise ValueError("SCOPE fusion weights must be in [0,1]")
    global_o = F.softmax(global_original.float() / float(global_temperature), dim=1)
    global_f = F.softmax(global_flipped.float() / float(global_temperature), dim=1)
    local_o = sum(
        weight * F.softmax(logits.float() / float(local_temperature), dim=1)
        for weight, logits in zip(weights, local_original)
    )
    local_f = sum(
        weight * F.softmax(logits.float() / float(local_temperature), dim=1)
        for weight, logits in zip(weights, local_flipped)
    )
    constituents = torch.stack((global_o, local_o, global_f, local_f), dim=1)
    global_pair = (1.0 - float(flip_weight)) * global_o + float(flip_weight) * global_f
    local_pair = (1.0 - float(flip_weight)) * local_o + float(flip_weight) * local_f
    fused = (1.0 - float(local_weight)) * global_pair + float(local_weight) * local_pair
    return ParentBatchScores(
        constituent_scores=constituents,
        constituent_top1=torch.argmax(constituents, dim=2).to(torch.int64),
        fused_log_scores=fused.clamp_min(torch.finfo(fused.dtype).tiny).log(),
    )


def stable_top2(log_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(log_scores)
    if values.ndim != 2 or values.shape[1] < 2 or not torch.isfinite(values).all():
        raise ValueError("parent log scores must be finite [N,C] with C>=2")
    order = torch.argsort(values, dim=1, descending=True, stable=True)[:, :2]
    return order.to(torch.int64), torch.gather(values, 1, order).to(torch.float64)


def pack_crop_boxes(
    original: Sequence[Sequence[tuple[int, int, int, int]]],
    flipped: Sequence[Sequence[tuple[int, int, int, int]]],
) -> torch.Tensor:
    if len(original) != 4 or len(flipped) != 4:
        raise ValueError("SCOPE crop boxes require four scales per orientation")
    counts = [len(value) for value in (*original, *flipped)]
    if not counts or len(set(counts)) != 1:
        raise ValueError("SCOPE crop boxes are batch-misaligned")
    orientations = [
        torch.stack([torch.tensor(value, dtype=torch.int64) for value in scales], dim=1)
        for scales in (original, flipped)
    ]
    packed = torch.stack(orientations, dim=1)
    if packed.shape != (counts[0], 2, 4, 4):
        raise ValueError("SCOPE crop boxes have an invalid shape")
    return packed


def formal_row_binding_hash(ids: torch.Tensor, paths: Sequence[str]) -> str:
    rows = torch.as_tensor(ids, dtype=torch.int64).flatten().cpu()
    if rows.numel() != len(paths):
        raise ValueError("formal row IDs and paths are misaligned")
    hasher = hashlib.sha256()
    for row, path in zip(rows.tolist(), paths):
        hasher.update(f"{int(row)}\t{str(path)}\n".encode("utf-8"))
    return hasher.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().contiguous().cpu()
    hasher = hashlib.sha256()
    hasher.update(str(tensor.dtype).encode("ascii"))
    hasher.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    hasher.update(tensor.numpy().tobytes(order="C"))
    return hasher.hexdigest()


def semantic_sha256(value: Any) -> str:
    hasher = hashlib.sha256()
    _update_semantic_hash(hasher, value)
    return hasher.hexdigest()


def replicate_semantic_sha256(value: Any) -> str:
    """Hash experiment semantics while ignoring a reproducible parent's file instance.

    Evidence caches retain ``parent_cache_sha256`` for strict provenance.  Two
    byte-distinct parent serializations can nevertheless have identical semantic
    hashes, so replicate comparison excludes only that file-instance digest and
    continues to bind ``parent_semantic_sha256`` and every evidence value.
    """
    hasher = hashlib.sha256()
    _update_semantic_hash(hasher, value, ignored_keys=frozenset({"parent_cache_sha256"}))
    return hasher.hexdigest()


def validate_parent_cache(payload: Mapping[str, Any], protocol: ScopeProtocol, split: str) -> int:
    _reject_raw_patch_storage(payload)
    if payload.get("schema") != protocol.fixed["schemas"]["parent"]:
        raise ValueError("parent cache schema mismatch")
    _validate_split(payload, split)
    paths, rows, count = _validate_rows(payload)
    candidates = _tensor(payload, "candidate_indices", torch.int64, (count, 2))
    _validate_candidates(candidates)
    scores = _tensor(payload, "candidate_parent_log_scores", torch.float64, (count, 2), finite=True)
    if torch.any(scores[:, 0] < scores[:, 1]):
        raise ValueError("candidate parent log scores must be descending")
    margin = _tensor(payload, "parent_margin", torch.float64, (count,), finite=True)
    if not torch.equal(margin, scores[:, 1] - scores[:, 0]):
        raise ValueError("parent_margin must be runner-up minus top1")
    prediction = _tensor(payload, "parent_predictions", torch.int64, (count,))
    if not torch.equal(prediction, candidates[:, 0]):
        raise ValueError("parent_predictions must equal candidate top1")
    constituents = _tensor(payload, "constituent_scores", torch.float32, (count, 4, 500), finite=True)
    top1 = _tensor(payload, "constituent_top1", torch.int64, (count, 4))
    if not torch.equal(top1, constituents.argmax(dim=2)):
        raise ValueError("constituent_top1 disagrees with constituent_scores")
    if tuple(payload.get("constituent_order", ())) != PARENT_BRANCH_ORDER:
        raise ValueError("constituent order mismatch")
    if payload.get("constituent_scores_sha256") != tensor_sha256(constituents):
        raise ValueError("constituent score hash mismatch")
    boxes = _tensor(payload, "crop_boxes", torch.int64, (count, 2, 4, 4))
    _validate_crop_boxes(boxes)
    _tensor(payload, "corrupt", torch.bool, (count,))
    _tensor(payload, "prior_bias", torch.float64, (500,), finite=True)
    if not isinstance(payload.get("prior_iterations"), int) or int(payload["prior_iterations"]) < 0:
        raise ValueError("prior_iterations must be a non-negative integer")
    _digest(payload.get("prior_report_sha256"), "prior_report_sha256")
    if payload.get("aligned_log_scores_shape") != [count, 500]:
        raise ValueError("aligned log score shape mismatch")
    if payload.get("aligned_log_scores_dtype") != "float32":
        raise ValueError("aligned log score dtype mismatch")
    _digest(payload.get("aligned_log_scores_sha256"), "aligned_log_scores_sha256")
    _validate_lineage(payload.get("lineage"))
    _validate_diagnostics(payload, split, count)
    if formal_row_binding_hash(rows, paths) != payload["formal_row_binding_sha256"]:
        raise ValueError("formal row binding hash mismatch")
    return count


def validate_evidence_cache(
    payload: Mapping[str, Any], parent: Mapping[str, Any], protocol: ScopeProtocol, split: str
) -> int:
    _reject_raw_patch_storage(payload)
    if payload.get("schema") != protocol.fixed["schemas"]["evidence"]:
        raise ValueError("evidence cache schema mismatch")
    _validate_split(payload, split)
    paths, rows, count = _validate_rows(payload)
    for field in ("formal_row_id", "candidate_indices", "crop_boxes", "corrupt"):
        current = torch.as_tensor(payload[field])
        expected = torch.as_tensor(parent[field])
        if current.dtype != expected.dtype or current.shape != expected.shape or not torch.equal(current, expected):
            raise ValueError(f"evidence {field} disagrees with parent")
    if paths != list(parent["paths"]):
        raise ValueError("evidence paths disagree with parent")
    if payload.get("formal_row_binding_sha256") != parent.get("formal_row_binding_sha256"):
        raise ValueError("evidence formal row binding disagrees with parent")
    if payload.get("parent_semantic_sha256") != semantic_sha256(parent):
        raise ValueError("parent semantic hash mismatch")
    _digest(payload.get("parent_cache_sha256"), "parent_cache_sha256")
    if tuple(payload.get("view_order", ())) != EVIDENCE_VIEW_ORDER:
        raise ValueError("evidence view order mismatch")
    if tuple(payload.get("view_weights", ())) != EVIDENCE_VIEW_WEIGHTS:
        raise ValueError("evidence view weights mismatch")
    if payload.get("grid_shape") != [7, 7] or payload.get("adjacency") != "four_neighbor_row_major_v1":
        raise ValueError("evidence grid/adjacency mismatch")
    edges = _tensor(payload, "edges", torch.int64, (84, 2))
    expected_edges = torch.tensor(four_neighbor_edges(), dtype=torch.int64)
    if not torch.equal(edges, expected_edges):
        raise ValueError("evidence edges are not canonical")
    if payload.get("edges_sha256") != tensor_sha256(edges):
        raise ValueError("edge hash mismatch")
    _digest(payload.get("classifier_weight_sha256"), "classifier_weight_sha256")
    _tensor(payload, "weight_norm", torch.float64, (count,), finite=True)
    _tensor(payload, "weight_norm_valid", torch.bool, (count,))
    for audit_name in ("classifier_space_audit", "antisymmetry_audit"):
        if not isinstance(payload.get(audit_name), Mapping):
            raise ValueError(f"{audit_name} is missing")
    for family in ("scope", "pace", "no_topology"):
        _validate_family(payload.get(family), count)
    _validate_lineage(payload.get("lineage"))
    _validate_diagnostics(payload, split, count)
    if formal_row_binding_hash(rows, paths) != payload["formal_row_binding_sha256"]:
        raise ValueError("formal row binding hash mismatch")
    return count


def atomic_save_scope_cache(payload: Mapping[str, Any], destination: str | Path) -> ScopeCacheManifest:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    semantic = semantic_sha256(payload)
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists():
        if not manifest_path.is_file():
            raise FileExistsError("existing SCOPE cache has no manifest")
        existing = load_scope_cache(path)
        if semantic_sha256(existing) != semantic:
            raise FileExistsError("existing SCOPE cache has different semantics")
        return ScopeCacheManifest(path, sha256_file(path), semantic)
    stored = dict(payload)
    stored["_semantic_sha256"] = semantic
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(stored, temporary)
        file_sha = sha256_file(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json_dump({"sha256": file_sha, "semantic_sha256": semantic}, manifest_path)
    return ScopeCacheManifest(path, file_sha, semantic)


def load_scope_cache(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    manifest_path = source.with_suffix(".manifest.json")
    if not source.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("SCOPE cache or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(source) != manifest.get("sha256"):
        raise ValueError("SCOPE cache file SHA-256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("SCOPE cache root must be a mapping")
    expected = payload.pop("_semantic_sha256", None)
    observed = semantic_sha256(payload)
    if expected != observed or manifest.get("semantic_sha256") != observed:
        raise ValueError("SCOPE cache semantic SHA-256 mismatch")
    payload["_cache_sha256"] = manifest["sha256"]
    payload["_semantic_sha256"] = observed
    return payload


def _validate_family(value: Any, count: int) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("evidence family must be a mapping")
    per_view = _tensor(value, "view_evidence", torch.float64, (count, 6), finite=True)
    aggregate = _tensor(value, "aggregate", torch.float64, (count,), finite=True)
    weights = torch.tensor(EVIDENCE_VIEW_WEIGHTS, dtype=torch.float64)
    if not torch.allclose(aggregate, (per_view * weights).sum(dim=1), atol=1.0e-12, rtol=0.0):
        raise ValueError("family aggregate disagrees with per-view evidence")
    positive = _tensor(value, "positive_count", torch.int64, (count,))
    if not torch.equal(positive, (per_view > 0.0).sum(dim=1)):
        raise ValueError("family positive_count mismatch")
    _tensor(value, "orientation", torch.float64, (count, 2), finite=True)
    _tensor(value, "leave_one_scale", torch.float64, (count, 3), finite=True)
    _tensor(value, "eligibility", torch.bool, (count,))


def _validate_rows(payload: Mapping[str, Any]) -> tuple[list[str], torch.Tensor, int]:
    paths = [str(path) for path in payload.get("paths", [])]
    if not paths or len(paths) != len(set(paths)) or paths != sorted(paths):
        raise ValueError("paths must be non-empty, unique, and canonical-sorted")
    rows = _tensor(payload, "formal_row_id", torch.int64, (len(paths),))
    if not torch.equal(rows, torch.arange(len(paths), dtype=torch.int64)):
        raise ValueError("formal_row_id must be contiguous from zero")
    if not isinstance(payload.get("formal_row_binding_sha256"), str):
        raise ValueError("formal row binding hash is missing")
    return paths, rows, len(paths)


def _validate_split(payload: Mapping[str, Any], split: str) -> None:
    if split not in {"validation", "test"} or payload.get("split") != split:
        raise ValueError("cache split mismatch")


def _validate_candidates(candidates: torch.Tensor) -> None:
    if candidates.numel() and (int(candidates.min()) < 0 or int(candidates.max()) >= 500):
        raise ValueError("candidate class index is out of range")
    if torch.any(candidates[:, 0] == candidates[:, 1]):
        raise ValueError("candidate classes must be distinct")


def _validate_crop_boxes(boxes: torch.Tensor) -> None:
    x0, y0, x1, y1 = boxes.unbind(dim=-1)
    if torch.any(x0 < 0) or torch.any(y0 < 0) or torch.any(x1 > 224) or torch.any(y1 > 224):
        raise ValueError("crop_boxes are out of range")
    if torch.any(x0 >= x1) or torch.any(y0 >= y1):
        raise ValueError("crop_boxes are empty")


def _validate_lineage(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != LINEAGE_FIELDS:
        raise ValueError("lineage fields mismatch")
    for key, digest in value.items():
        _digest(digest, f"lineage.{key}")


def _validate_diagnostics(payload: Mapping[str, Any], split: str, count: int) -> None:
    present = {field for field in VALIDATION_DIAGNOSTIC_FIELDS if field in payload}
    if split == "test":
        if present:
            raise ValueError(f"test cache contains validation-only fields: {sorted(present)}")
        return
    if present != set(VALIDATION_DIAGNOSTIC_FIELDS):
        raise ValueError("validation cache diagnostics are incomplete")
    _tensor(payload, "label", torch.int64, (count,))
    _tensor(payload, "pseudo_label", torch.int64, (count,))
    clean = torch.as_tensor(payload["clean_probability"])
    alpha = torch.as_tensor(payload["correction_alpha"])
    for name, value in (("clean_probability", clean), ("correction_alpha", alpha)):
        if value.shape != (count,) or value.dtype not in {torch.float32, torch.float64} or not torch.isfinite(value).all():
            raise ValueError(f"{name} is malformed")


def _tensor(
    payload: Mapping[str, Any], name: str, dtype: torch.dtype,
    shape: tuple[int, ...], finite: bool = False,
) -> torch.Tensor:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor) or value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be {dtype} with shape {shape}")
    if finite and not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value.cpu()


def _digest(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be 64 lowercase hex characters")
    return text


def _reject_raw_patch_storage(value: Any, path: str = "cache") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if "patch" in lowered and any(token in lowered for token in ("raw", "feature", "tensor")):
                raise ValueError(f"raw patch storage is forbidden: {path}.{key}")
            _reject_raw_patch_storage(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_raw_patch_storage(item, f"{path}[{index}]")


def _update_semantic_hash(
    hasher: Any, value: Any, *, ignored_keys: frozenset[str] = frozenset(),
) -> None:
    if isinstance(value, torch.Tensor):
        hasher.update(b"tensor:")
        hasher.update(tensor_sha256(value).encode("ascii"))
    elif isinstance(value, Mapping):
        hasher.update(b"mapping{")
        for key in sorted(value, key=str):
            if str(key).startswith("_") or str(key) in ignored_keys:
                continue
            hasher.update(str(key).encode("utf-8") + b"=")
            _update_semantic_hash(hasher, value[key], ignored_keys=ignored_keys)
        hasher.update(b"}")
    elif isinstance(value, (list, tuple)):
        hasher.update(b"sequence[")
        for item in value:
            _update_semantic_hash(hasher, item, ignored_keys=ignored_keys)
        hasher.update(b"]")
    elif isinstance(value, float):
        hasher.update(("float:" + value.hex()).encode("ascii"))
    elif isinstance(value, Path):
        hasher.update(("path:" + str(value)).encode("utf-8"))
    else:
        hasher.update((type(value).__name__ + ":" + repr(value)).encode("utf-8"))
