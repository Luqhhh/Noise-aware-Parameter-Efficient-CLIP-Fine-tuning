"""Build the compact SCOPE-K2 Pass-B evidence cache."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.cli.cache_scope_parent import ValidationImages
from aegis_clip.data import TestImageDataset, TrustBundle
from aegis_clip.features import canonical_sample_path
from aegis_clip.local_feature_adapter import load_local_feature_adapter
from aegis_clip.local_inference import native_visual_forward_with_patch_features
from aegis_clip.part_token_adapter import (
    anchored_classifier_residual_logits,
    load_part_token_adapter,
    pool_cls_aligned_patch_features,
)
from aegis_clip.runtime import atomic_json_dump, seed_worker, set_seed, sha256_file
from aegis_clip.scope_cache import (
    VALIDATION_DIAGNOSTIC_FIELDS,
    atomic_save_scope_cache,
    load_scope_cache,
    semantic_sha256,
    tensor_sha256,
    validate_evidence_cache,
    validate_parent_cache,
)
from aegis_clip.scope_evidence import (
    aggregate_family_evidence,
    family_eligibility,
    matched_pace_evidence,
    no_topology_view_evidence,
    pairwise_residual_grid,
    scope_view_evidence,
    validate_classifier_space_batch,
    validate_scope_parent_model,
)
from aegis_clip.scope_protocol import (
    EVIDENCE_VIEW_ORDER,
    EVIDENCE_VIEW_WEIGHTS,
    four_neighbor_edges,
    load_scope_protocol,
    verify_scope_assets,
)


def replay_cached_crops(images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or tuple(images.shape[-2:]) != (224, 224):
        raise ValueError("SCOPE crop replay requires [N,C,224,224] images")
    if boxes.dtype != torch.int64 or boxes.shape != (images.shape[0], 4):
        raise ValueError("SCOPE cached crop boxes must be int64 [N,4]")
    crops: list[torch.Tensor] = []
    shapes: set[tuple[int, ...]] = set()
    for image, coordinates in zip(images, boxes.tolist()):
        x0, y0, x1, y1 = (int(value) for value in coordinates)
        if not (0 <= x0 < x1 <= 224 and 0 <= y0 < y1 <= 224):
            raise ValueError("SCOPE cached crop coordinates are out of range")
        crop = image[:, y0:y1, x0:x1]
        shapes.add(tuple(crop.shape))
        crops.append(crop)
    if not crops or len(shapes) != 1:
        raise ValueError("SCOPE crop replay requires one fixed size per view")
    return F.interpolate(
        torch.stack(crops), size=(224, 224), mode="bilinear", align_corners=False
    )


def _family_payload(
    per_view: torch.Tensor,
    *,
    constituent_top1: torch.Tensor,
    parent_corrupt: torch.Tensor,
    evidence_corrupt: torch.Tensor,
    weight_norm_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    summary = aggregate_family_evidence(per_view)
    eligibility = family_eligibility(
        constituent_top1=constituent_top1,
        parent_corrupt=parent_corrupt,
        evidence_corrupt=evidence_corrupt,
        weight_norm_valid=weight_norm_valid,
        summary=summary,
    )
    return {
        "view_evidence": summary.per_view,
        "aggregate": summary.total,
        "positive_count": summary.positive_view_count,
        "orientation": torch.stack((summary.original, summary.flipped), dim=1),
        "leave_one_scale": summary.leave_one_scale,
        "eligibility": eligibility.eligible,
    }


def _independent_residual(
    cls: torch.Tensor,
    patches: torch.Tensor,
    weight: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    cls64 = cls.detach().to(device="cpu", dtype=torch.float64)
    patches64 = patches.detach().to(device="cpu", dtype=torch.float64)
    weight64 = weight.detach().to(device="cpu", dtype=torch.float64)
    pairs = candidates.detach().to(device="cpu", dtype=torch.int64)
    direction = weight64[pairs[:, 1]] - weight64[pairs[:, 0]]
    norm = torch.linalg.vector_norm(direction, dim=1)
    direction = direction / norm[:, None]
    return torch.einsum("npd,nd->np", patches64 - cls64[:, None, :], direction)


@torch.inference_mode()
def build_scope_evidence_cache(
    checkpoint: str | Path,
    parent_cache: str | Path,
    config: str | Path,
    output: str | Path,
    split: str,
    device: str | torch.device,
    batch_size: int,
    num_workers: int,
) -> Path:
    protocol = load_scope_protocol(config)
    verify_scope_assets(protocol)
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if int(batch_size) != int(protocol.fixed["execution"]["batch_size"]):
        raise ValueError("formal SCOPE evidence cache requires batch_size=128")
    if int(num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    parent_path = Path(parent_cache)
    parent = load_scope_cache(parent_path)
    row_count = validate_parent_cache(parent, protocol, split)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if checkpoint_path != protocol.assets.checkpoint or sha256_file(checkpoint_path) != protocol.assets.checkpoint_sha256:
        raise ValueError("SCOPE checkpoint path or SHA-256 mismatch")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("SCOPE CUDA execution requested but CUDA is unavailable")

    model, preprocess, checkpoint_payload = build_from_checkpoint(checkpoint_path, target_device)
    model.eval()
    validate_scope_parent_model(model, checkpoint_payload)
    local_adapter = load_local_feature_adapter(checkpoint_payload, target_device)
    part_adapter = load_part_token_adapter(checkpoint_payload, target_device)
    pool = checkpoint_payload["part_token_adapter"]["spec"]["part_pool_spec"]
    if split == "validation":
        dataset: Dataset = ValidationImages(
            protocol.assets.val_csv, protocol.assets.train_root, preprocess,
            TrustBundle(protocol.assets.trust_bundle),
        )
        paths = list(dataset.paths)
    else:
        dataset = TestImageDataset(protocol.assets.test_root, preprocess)
        paths = [canonical_sample_path(path.name) for path in dataset.paths]
    if row_count > len(dataset):
        raise ValueError("parent cache has more rows than the frozen dataset")
    if row_count != len(dataset):
        dataset = Subset(dataset, range(row_count))
        paths = paths[:row_count]
    if paths != parent["paths"]:
        raise ValueError("SCOPE dataset paths disagree with parent cache")
    set_seed(int(protocol.fixed["crossfit"]["seed"]), deterministic=True)
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
        timeout=120 if int(num_workers) else 0, pin_memory=target_device.type == "cuda",
        persistent_workers=int(num_workers) > 0, worker_init_fn=seed_worker,
    )

    family_batches: dict[str, list[torch.Tensor]] = {
        "scope": [], "pace": [], "no_topology": [],
    }
    norm_batches: list[torch.Tensor] = []
    valid_batches: list[torch.Tensor] = []
    corrupt_batches: list[torch.Tensor] = []
    max_base_error = 0.0
    max_dual_error = 0.0
    max_independent_error = 0.0
    canonical_bitwise = True
    classifier_weight = model.classifier.weight.detach().float()
    evidence_spec = protocol.fixed["evidence"]
    offset = 0
    for batch in tqdm(loader, desc=f"SCOPE evidence {split}"):
        images = batch["images"].to(target_device, non_blocking=True)
        count = int(images.shape[0])
        end = offset + count
        candidates = parent["candidate_indices"][offset:end].to(target_device)
        boxes = parent["crop_boxes"][offset:end]
        current: dict[str, list[torch.Tensor]] = {
            "scope": [], "pace": [], "no_topology": [],
        }
        batch_norm: torch.Tensor | None = None
        batch_valid: torch.Tensor | None = None
        for orientation in range(2):
            oriented = images if orientation == 0 else torch.flip(images, dims=(3,))
            for scale_index in (1, 2, 3):
                local_images = replay_cached_crops(
                    oriented, boxes[:, orientation, scale_index].to(target_device)
                )
                base_logits, base_cls, patch_features = native_visual_forward_with_patch_features(
                    model, local_images
                )
                part_features = pool_cls_aligned_patch_features(
                    base_cls, patch_features, top_patches=int(pool["top_patches"]),
                    temperature=float(pool["temperature"]),
                )
                dual_cls = local_adapter(base_cls) + part_adapter(base_cls, part_features) - base_cls
                dual_logits = anchored_classifier_residual_logits(
                    base_logits, base_cls, dual_cls, classifier_weight
                )
                audit = validate_classifier_space_batch(
                    model, base_logits, base_cls, dual_logits=dual_logits, dual_cls=dual_cls,
                    atol=float(evidence_spec["linear_gate_atol"]),
                    rtol=float(evidence_spec["linear_gate_rtol"]),
                )
                max_base_error = max(max_base_error, audit.base_max_abs_error)
                max_dual_error = max(max_dual_error, float(audit.dual_max_abs_error or 0.0))
                residual = pairwise_residual_grid(
                    base_cls, patch_features, classifier_weight, candidates,
                    epsilon=float(evidence_spec["weight_norm_epsilon"]),
                )
                reverse = pairwise_residual_grid(
                    base_cls, patch_features, classifier_weight, candidates.flip(dims=(1,)),
                    epsilon=float(evidence_spec["weight_norm_epsilon"]),
                )
                canonical_bitwise = canonical_bitwise and torch.equal(reverse.residual, -residual.residual)
                independent = _independent_residual(
                    base_cls, patch_features, classifier_weight, candidates
                )
                max_independent_error = max(
                    max_independent_error,
                    float((independent - residual.residual).abs().max().item()),
                )
                if batch_norm is None:
                    batch_norm, batch_valid = residual.weight_norm, residual.valid
                elif not torch.equal(batch_norm, residual.weight_norm) or not torch.equal(batch_valid, residual.valid):
                    raise ValueError("pair weight norm changed across evidence views")
                current["scope"].append(scope_view_evidence(residual.residual))
                current["pace"].append(
                    matched_pace_evidence(residual.residual, tail_size=int(evidence_spec["tail_size"]))
                )
                current["no_topology"].append(no_topology_view_evidence(residual.residual))
        if batch_norm is None or batch_valid is None:
            raise RuntimeError("SCOPE evidence produced no views")
        for family in family_batches:
            family_batches[family].append(torch.stack(current[family], dim=1))
        norm_batches.append(batch_norm)
        valid_batches.append(batch_valid)
        corrupt_batches.append(torch.as_tensor(batch["corrupt"], dtype=torch.bool).cpu())
        offset = end
    if offset != row_count:
        raise ValueError("SCOPE evidence loader did not cover every parent row")
    observed_corrupt = torch.cat(corrupt_batches)
    if not torch.equal(observed_corrupt, parent["corrupt"]):
        raise ValueError("SCOPE parent/evidence corrupt states disagree")
    if not canonical_bitwise or max_independent_error > float(evidence_spec["antisymmetry_atol"]):
        raise ValueError("SCOPE antisymmetry audit failed")
    weight_norm = torch.cat(norm_batches).to(torch.float64)
    weight_valid = torch.cat(valid_batches).bool()
    family_payloads = {
        family: _family_payload(
            torch.cat(batches).to(torch.float64),
            constituent_top1=parent["constituent_top1"],
            parent_corrupt=parent["corrupt"], evidence_corrupt=observed_corrupt,
            weight_norm_valid=weight_valid,
        )
        for family, batches in family_batches.items()
    }
    edges = torch.tensor(four_neighbor_edges(), dtype=torch.int64)
    payload: dict[str, Any] = {
        "schema": protocol.fixed["schemas"]["evidence"], "split": split,
        "paths": list(parent["paths"]), "formal_row_id": parent["formal_row_id"].clone(),
        "formal_row_binding_sha256": parent["formal_row_binding_sha256"],
        "candidate_indices": parent["candidate_indices"].clone(),
        "crop_boxes": parent["crop_boxes"].clone(), "corrupt": observed_corrupt,
        "parent_cache_sha256": parent["_cache_sha256"],
        "parent_semantic_sha256": semantic_sha256(parent),
        "view_order": list(EVIDENCE_VIEW_ORDER), "view_weights": list(EVIDENCE_VIEW_WEIGHTS),
        "grid_shape": [7, 7], "adjacency": "four_neighbor_row_major_v1",
        "edges": edges, "edges_sha256": tensor_sha256(edges),
        "classifier_weight_sha256": tensor_sha256(classifier_weight.cpu()),
        "weight_norm": weight_norm, "weight_norm_valid": weight_valid,
        "classifier_space_audit": {
            "base_max_abs_error": max_base_error, "dual_max_abs_error": max_dual_error,
        },
        "antisymmetry_audit": {
            "canonical_bitwise": canonical_bitwise,
            "independent_max_abs_error": max_independent_error,
        },
        **family_payloads,
        "lineage": dict(parent["lineage"]),
        "_run_metadata": {
            "argv": list(sys.argv), "cwd": os.getcwd(), "python": sys.version,
            "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
        },
    }
    if split == "validation":
        payload.update({field: parent[field].clone() for field in VALIDATION_DIAGNOSTIC_FIELDS})
    validate_evidence_cache(payload, parent, protocol, split)
    output_path = Path(output)
    manifest = atomic_save_scope_cache(payload, output_path)
    atomic_json_dump(
        {"cache_sha256": manifest.sha256, "semantic_sha256": manifest.semantic_sha256,
         "parent_cache_sha256": parent["_cache_sha256"], "rows": row_count, "split": split},
        output_path.with_suffix(".run.json"),
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    build_scope_evidence_cache(
        args.checkpoint, args.parent_cache, args.config, args.output, args.split,
        args.device, args.batch_size, args.num_workers,
    )


if __name__ == "__main__":
    main()
