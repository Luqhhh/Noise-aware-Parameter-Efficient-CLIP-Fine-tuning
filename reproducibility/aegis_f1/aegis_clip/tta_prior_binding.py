"""Fail-closed provenance binding for flip-TTA + validation-fitted prior calibration.

The L05 recipe fits one additive class-bias vector on the validation branch cache
and applies it, frozen, to the test logits.  Competition rules forbid fitting the
prior on test predictions, so the only thing that makes the pipeline auditable is
that every downstream artifact names the exact upstream artifact it was derived
from.  This module makes that chain explicit and fail-closed:

* a validation branch cache is bound to the checkpoint that produced it (SHA-256
  of the checkpoint file plus the trainer-written ``*.binding.json`` lineage),
* the checkpoint lineage is bound to the class mapping, the frozen split
  (``dataset_manifest_sha256`` + ``dataset_fingerprint``), the validation CSV and
  the content-group partition,
* the input resolution, preprocessing description and numeric precision come from
  that checkpoint lineage instead of being re-declared by the caller,
* the TTA fusion rule, temperature and prior strength are declared once, and
* a fitted bias is recorded with its own tensor SHA-256 and the SHA-256 of the
  exact cache bytes it was fitted on.

A new-seed checkpoint therefore cannot silently consume the previous model's
prior bias: the recorded ``checkpoint_sha256`` and ``cache_sha256`` will not match.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from aegis_clip.prior_alignment import apply_prior_bias, fit_prior_bias
from aegis_clip.rematch_protocol import validate_checkpoint
from aegis_clip.runtime import sha256_file, sha256_lines
from aegis_clip.tta import TTA_FUSION_MODES, fuse_paired_logits


PROVENANCE_SCHEMA_VERSION = 1

#: The single frozen L05 inference/calibration recipe.  Later work verifies this
#: point instead of re-scanning temperature or prior strength.
FIXED_RECIPE: dict[str, Any] = {
    "tta": "horizontal_flip",
    "fusion": "mean_probabilities",
    "temperature": 1.4,
    "prior_strength": 0.6,
}

#: The data split was created once with this seed and must never be re-drawn when
#: only the *training* seed changes.
SPLIT_SEED = 42


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"TTA/prior provenance: {message}")


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Deterministic SHA-256 of a tensor's canonical little-endian float32 bytes."""
    array = np.ascontiguousarray(
        torch.as_tensor(tensor).detach().cpu().numpy(), dtype="<f4"
    )
    digest = hashlib.sha256()
    digest.update(repr((tuple(array.shape), str(array.dtype))).encode("utf-8"))
    digest.update(b"\n")
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_cache_paths(paths: Sequence[str]) -> list[str]:
    """Normalise cache/split paths to the train-root-relative POSIX form."""
    return [
        str(path).replace("\\", "/").removeprefix("train/").lstrip("./")
        for path in paths
    ]


def checkpoint_identity(
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    *,
    require_config_binding: bool = True,
) -> dict[str, Any]:
    """Validate and describe the checkpoint that a cache/bias must be bound to."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    _require(checkpoint.is_file(), f"checkpoint missing: {checkpoint}")
    digest = sha256_file(checkpoint)
    meta = validate_checkpoint(checkpoint, dict(config), parent=False)
    _require(
        meta.get("checkpoint_sha256") == digest,
        "checkpoint binding hash does not match the checkpoint file",
    )
    _require(isinstance(meta.get("binding"), dict), "checkpoint binding block missing")
    binding = dict(meta["binding"])
    training_config_sha256 = meta.get("training_config_sha256")
    if require_config_binding:
        config_path = config.get("_config_path")
        _require(bool(config_path), "config lacks _config_path; cannot bind training config")
        _require(
            training_config_sha256 == sha256_file(config_path),
            "checkpoint was not trained with the supplied config",
        )
    identity = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "training_config_sha256": training_config_sha256,
        "experiment_id": meta.get("experiment_id"),
        "epoch": meta.get("epoch"),
        "class_mapping_sha256": binding.get("class_mapping_sha256"),
        "dataset_manifest_sha256": binding.get("dataset_manifest_sha256"),
        "dataset_fingerprint": binding.get("dataset_fingerprint"),
        "train_csv_sha256": binding.get("train_csv_sha256"),
        "official_checkpoint_sha256": binding.get("official_checkpoint_sha256"),
        "input_resolution": binding.get("input_resolution"),
        "reference_resolution": binding.get("reference_resolution"),
        "preprocessing": binding.get("preprocessing"),
        "encoder_precision": binding.get("encoder_precision"),
        "feature_precision": binding.get("feature_precision"),
        "autocast": binding.get("autocast"),
        "protocol": binding.get("protocol"),
        "data_version": binding.get("data_version"),
        "stage": binding.get("stage"),
    }
    for key in (
        "class_mapping_sha256",
        "dataset_manifest_sha256",
        "dataset_fingerprint",
        "input_resolution",
        "preprocessing",
        "encoder_precision",
        "feature_precision",
    ):
        _require(identity.get(key) not in (None, ""), f"checkpoint binding lacks {key}")
    return identity


def validation_cache_identity(
    cache_path: str | Path,
    config: Mapping[str, Any],
    checkpoint: str | Path | Mapping[str, Any],
    *,
    check_payload: bool = True,
) -> dict[str, Any]:
    """Validate a validation branch cache against its checkpoint and split.

    Every field is recomputed from on-disk assets, so a cache written before the
    provenance block existed is still verifiable; the returned identity is the
    binding that later prior-bias records must reproduce.
    """
    cache = Path(cache_path).expanduser().resolve()
    _require(cache.is_file(), f"validation branch cache missing: {cache}")
    identity = (
        dict(checkpoint)
        if isinstance(checkpoint, Mapping)
        else checkpoint_identity(checkpoint, config)
    )
    payload = torch.load(cache, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict), "validation branch cache is not a mapping")
    missing = {"original_logits", "flip_logits", "paths"} - set(payload)
    _require(not missing, f"validation branch cache missing keys: {sorted(missing)}")

    recorded_checkpoint = payload.get("checkpoint")
    _require(recorded_checkpoint is not None, "cache does not record its checkpoint")
    _require(
        Path(str(recorded_checkpoint)).resolve() == Path(identity["checkpoint"]),
        "cache was produced from a different checkpoint path",
    )
    _require(
        payload.get("checkpoint_sha256") == identity["checkpoint_sha256"],
        "cache checkpoint SHA-256 does not match the supplied checkpoint",
    )

    config = dict(config)
    mapping_path = Path(config["data"]["class_mapping"]).resolve()
    _require(mapping_path.is_file(), f"class mapping missing: {mapping_path}")
    class_mapping_sha256 = sha256_file(mapping_path)
    _require(
        class_mapping_sha256 == identity["class_mapping_sha256"],
        "class mapping changed since checkpoint training",
    )
    mapping = _read_json(mapping_path)
    num_classes = len(mapping)
    original_logits = payload["original_logits"]
    _require(
        int(original_logits.shape[1]) == num_classes,
        "cache class count does not match the class mapping",
    )

    dataset_manifest_path = Path(config["data"]["dataset_manifest"]).resolve()
    dataset_manifest = _read_json(dataset_manifest_path)
    _require(
        sha256_file(dataset_manifest_path) == identity["dataset_manifest_sha256"],
        "dataset manifest changed since checkpoint training",
    )
    _require(
        dataset_manifest.get("train_fingerprint") == identity["dataset_fingerprint"],
        "dataset fingerprint changed since checkpoint training",
    )
    _require(
        int(dataset_manifest.get("seed", -1)) == SPLIT_SEED,
        "validation split was not drawn with the frozen split seed",
    )

    val_csv = Path(config["data"]["val_csv"]).resolve()
    _require(
        Path(str(payload.get("validation_csv"))).resolve() == val_csv,
        "cache was produced from a different validation CSV",
    )
    validation_csv_sha256 = sha256_file(val_csv)
    _require(
        payload.get("validation_csv_sha256") == validation_csv_sha256,
        "validation CSV changed since the cache was written",
    )
    _require(
        dataset_manifest.get("files", {}).get("val_dev.csv") == validation_csv_sha256,
        "validation CSV does not match the frozen dataset manifest",
    )

    num_classes = int(num_classes)
    if check_payload:
        with val_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected_paths = canonical_cache_paths(
            [row["image_path"] for row in rows]
        )
        cache_paths = canonical_cache_paths(payload["paths"])
        _require(
            cache_paths == expected_paths,
            "cache sample order does not match the validation split",
        )
        _require(
            len(cache_paths) == len(set(cache_paths)),
            "cache contains duplicate validation samples",
        )
        labels = payload.get("labels")
        _require(labels is not None, "cache does not record validation labels")
        expected_labels = torch.tensor(
            [int(row["label"]) for row in rows], dtype=torch.long
        )
        _require(
            torch.equal(torch.as_tensor(labels).long().cpu(), expected_labels),
            "cache validation labels do not match the split",
        )
        _require(
            int(original_logits.shape[0]) == len(rows),
            "cache sample count does not match the validation split",
        )
        groups_path = dataset_manifest_path.parent / "content_groups.csv"
        _require(groups_path.is_file(), f"content groups missing: {groups_path}")
        content_groups_sha256 = sha256_file(groups_path)
        with groups_path.open(newline="", encoding="utf-8") as handle:
            group_of = {
                canonical_cache_paths([row["image_path"]])[0]: row["content_group"]
                for row in csv.DictReader(handle)
            }
        unknown = [path for path in cache_paths if path not in group_of]
        _require(not unknown, f"cache samples missing from content groups: {unknown[:3]}")
        group_set = sorted({group_of[path] for path in cache_paths})
        group_set_sha256 = sha256_lines(group_set)
        sample_order_sha256 = sha256_lines(cache_paths)
    else:
        content_groups_sha256 = None
        group_set_sha256 = None
        sample_order_sha256 = None

    fusion = str(payload.get("tta_fusion", "") or "")
    _require(fusion in TTA_FUSION_MODES, f"cache records unknown TTA fusion: {fusion!r}")

    result = {
        "cache": str(cache),
        "cache_sha256": sha256_file(cache),
        "cache_format_version": payload.get("format_version"),
        "num_samples": int(original_logits.shape[0]),
        "num_classes": num_classes,
        "class_mapping_sha256": class_mapping_sha256,
        "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "dataset_fingerprint": identity["dataset_fingerprint"],
        "split_seed": SPLIT_SEED,
        "validation_csv_sha256": validation_csv_sha256,
        "sample_order_sha256": sample_order_sha256,
        "content_groups_sha256": content_groups_sha256,
        "content_group_set_sha256": group_set_sha256,
        "cache_tta_fusion": fusion,
        "cache_tta_temperature": payload.get("tta_temperature"),
        "input_resolution": identity["input_resolution"],
        "preprocessing": identity["preprocessing"],
        "encoder_precision": identity["encoder_precision"],
        "feature_precision": identity["feature_precision"],
        "autocast": identity["autocast"],
        "checkpoint": identity["checkpoint"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "training_config_sha256": identity.get("training_config_sha256"),
        "protocol": identity.get("protocol"),
    }
    return result


def cache_provenance_block(
    paths: Sequence[str],
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    tta: str,
    fusion: str,
    temperature: float,
    training_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the provenance block a validation branch cache must carry.

    The checkpoint's trainer-written ``*.binding.json`` is re-derived from the
    supplied config, so a cache can never declare a lineage the checkpoint itself
    does not claim.
    """
    from aegis_clip.rematch_protocol import checkpoint_binding

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    _require(checkpoint.is_file(), f"checkpoint missing: {checkpoint}")
    meta = _read_json(checkpoint.with_suffix(".binding.json"))
    digest = sha256_file(checkpoint)
    _require(meta.get("checkpoint_sha256") == digest, "checkpoint hash mismatch")
    _require(
        meta.get("binding") == checkpoint_binding(dict(config)),
        "checkpoint lineage does not match the supplied config",
    )
    config = dict(config)
    dataset_manifest_path = Path(config["data"]["dataset_manifest"]).resolve()
    dataset_manifest = _read_json(dataset_manifest_path)
    groups_path = dataset_manifest_path.parent / "content_groups.csv"
    _require(groups_path.is_file(), f"content groups missing: {groups_path}")
    with groups_path.open(newline="", encoding="utf-8") as handle:
        group_of = {
            canonical_cache_paths([row["image_path"]])[0]: row["content_group"]
            for row in csv.DictReader(handle)
        }
    cache_paths = canonical_cache_paths(paths)
    missing = sorted({path for path in cache_paths if path not in group_of})
    _require(not missing, f"cached samples missing from content groups: {missing[:3]}")
    binding = dict(meta["binding"])
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "checkpoint_sha256": digest,
        "training_config_sha256": (
            sha256_file(training_config_path) if training_config_path else None
        ),
        "checkpoint_binding_sha256": sha256_file(checkpoint.with_suffix(".binding.json")),
        "class_mapping_sha256": binding.get("class_mapping_sha256"),
        "dataset_manifest_sha256": binding.get("dataset_manifest_sha256"),
        "dataset_fingerprint": binding.get("dataset_fingerprint"),
        "split_seed": int(dataset_manifest.get("seed", -1)),
        "validation_csv_sha256": sha256_file(config["data"]["val_csv"]),
        "sample_order_sha256": sha256_lines(cache_paths),
        "content_groups_sha256": sha256_file(groups_path),
        "content_group_set_sha256": sha256_lines(
            sorted({group_of[path] for path in cache_paths})
        ),
        "num_samples": len(cache_paths),
        "num_classes": int(config["model"]["num_classes"]),
        "num_branch_views": 2,
        "input_resolution": binding.get("input_resolution"),
        "preprocessing": binding.get("preprocessing"),
        "encoder_precision": binding.get("encoder_precision"),
        "feature_precision": binding.get("feature_precision"),
        "autocast": binding.get("autocast"),
        "tta": str(tta),
        "tta_fusion": str(fusion),
        "tta_temperature": float(temperature),
        "test_data_used": False,
    }


def fused_validation_logits(
    payload: Mapping[str, Any],
    *,
    fusion: str,
    temperature: float,
) -> torch.Tensor:
    """Rebuild the frozen-recipe fused validation logits from a branch cache."""
    _require(fusion in TTA_FUSION_MODES, f"unknown TTA fusion mode: {fusion}")
    original = torch.as_tensor(payload["original_logits"]).float()
    flipped = torch.as_tensor(payload["flip_logits"]).float()
    _require(
        original.shape == flipped.shape and original.ndim == 2,
        "cache branch logits must share an [N,C] shape",
    )
    return fuse_paired_logits(
        original, flipped, mode=str(fusion), temperature=float(temperature)
    )


def resolved_recipe(
    *,
    tta: str = FIXED_RECIPE["tta"],
    fusion: str = FIXED_RECIPE["fusion"],
    temperature: float = FIXED_RECIPE["temperature"],
    prior_strength: float = FIXED_RECIPE["prior_strength"],
    enforce_fixed: bool = False,
) -> dict[str, Any]:
    """Return the declared recipe, optionally failing closed on any deviation."""
    recipe = {
        "tta": str(tta),
        "fusion": str(fusion),
        "temperature": float(temperature),
        "prior_strength": float(prior_strength),
    }
    _require(recipe["tta"] == FIXED_RECIPE["tta"], "only horizontal-flip TTA is frozen")
    _require(recipe["fusion"] in TTA_FUSION_MODES, "unknown TTA fusion mode")
    _require(recipe["temperature"] > 0.0, "temperature must be positive")
    _require(0.0 <= recipe["prior_strength"] <= 1.0, "prior strength must be in [0, 1]")
    if enforce_fixed:
        for key, expected in FIXED_RECIPE.items():
            _require(
                recipe[key] == expected,
                f"recipe {key}={recipe[key]!r} deviates from the frozen {expected!r}",
            )
    return recipe


def fit_bound_prior(
    payload: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    fit_mask: torch.Tensor | None = None,
    max_iterations: int = 50,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    """Fit one prior bias and emit a record bound to the exact cache and checkpoint.

    ``fit_mask`` selects the rows the bias may read.  A fold-level diagnostic
    passes the complement of the evaluated fold here; the production path passes
    ``None`` (the whole validation split, never the test set).
    """
    fused = fused_validation_logits(
        payload, fusion=str(recipe["fusion"]), temperature=float(recipe["temperature"])
    )
    if fit_mask is None:
        fit_rows = fused
        fit_scope = {
            "mode": "full_validation",
            "fit_sample_count": int(fused.shape[0]),
            "fit_index_sha256": None,
        }
    else:
        mask = torch.as_tensor(fit_mask).bool().flatten()
        _require(mask.numel() == fused.shape[0], "fit mask length mismatch")
        _require(bool(mask.any()), "fit mask selects no samples")
        fit_rows = fused[mask]
        indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        fit_scope = {
            "mode": "conditional_fold_exclusion",
            "fit_sample_count": int(mask.sum()),
            "fit_index_sha256": sha256_lines(str(index) for index in indices),
        }
    bias, report = fit_prior_bias(fit_rows, max_iterations=int(max_iterations))
    record = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "bias_sha256": tensor_sha256(bias),
        "bias_dim": int(bias.numel()),
        "prior_fit_iterations": int(report["iterations"]),
        "fit_report": report,
        "fit_scope": fit_scope,
        "test_data_used": False,
        "recipe": {
            "tta": str(recipe["tta"]),
            "fusion": str(recipe["fusion"]),
            "temperature": float(recipe["temperature"]),
            "prior_strength": float(recipe["prior_strength"]),
        },
        "cache": str(cache_identity["cache"]),
        "cache_sha256": cache_identity["cache_sha256"],
        "checkpoint": cache_identity["checkpoint"],
        "checkpoint_sha256": cache_identity["checkpoint_sha256"],
        "class_mapping_sha256": cache_identity["class_mapping_sha256"],
        "dataset_manifest_sha256": cache_identity["dataset_manifest_sha256"],
        "dataset_fingerprint": cache_identity["dataset_fingerprint"],
        "split_seed": cache_identity["split_seed"],
        "validation_csv_sha256": cache_identity["validation_csv_sha256"],
        "sample_order_sha256": cache_identity["sample_order_sha256"],
        "content_group_set_sha256": cache_identity["content_group_set_sha256"],
        "num_samples": cache_identity["num_samples"],
        "num_classes": cache_identity["num_classes"],
        "input_resolution": cache_identity["input_resolution"],
        "preprocessing": cache_identity["preprocessing"],
        "encoder_precision": cache_identity["encoder_precision"],
        "feature_precision": cache_identity["feature_precision"],
        "autocast": cache_identity["autocast"],
    }
    return bias, report, record


_BOUND_FIELDS = (
    "cache_sha256",
    "checkpoint_sha256",
    "class_mapping_sha256",
    "dataset_manifest_sha256",
    "dataset_fingerprint",
    "split_seed",
    "validation_csv_sha256",
    "sample_order_sha256",
    "content_group_set_sha256",
    "num_samples",
    "num_classes",
    "input_resolution",
    "preprocessing",
    "encoder_precision",
    "feature_precision",
    "autocast",
)


def verify_prior_record(
    record: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any] | None = None,
    bias: torch.Tensor | None = None,
) -> None:
    """Fail closed unless a recorded prior bias belongs to this cache/checkpoint."""
    _require(
        record.get("schema_version") == PROVENANCE_SCHEMA_VERSION,
        "prior record has an unsupported schema version",
    )
    _require(
        record.get("test_data_used") is False,
        "prior record does not assert test_data_used=false",
    )
    for key in _BOUND_FIELDS:
        _require(
            record.get(key) == cache_identity.get(key),
            f"prior record {key} does not match the current cache/checkpoint",
        )
    recorded_recipe = record.get("recipe", {})
    if recipe is not None:
        for key, expected in (
            ("tta", recipe["tta"]),
            ("fusion", recipe["fusion"]),
            ("temperature", float(recipe["temperature"])),
            ("prior_strength", float(recipe["prior_strength"])),
        ):
            _require(
                recorded_recipe.get(key) == expected,
                f"prior record recipe {key} does not match the requested recipe",
            )
    if bias is not None:
        _require(
            record.get("bias_sha256") == tensor_sha256(bias),
            "prior bias tensor does not match its recorded SHA-256",
        )
        _require(
            int(torch.as_tensor(bias).numel()) == int(cache_identity["num_classes"]),
            "prior bias length does not match the class count",
        )


def apply_fixed_recipe(
    payload: Mapping[str, Any],
    bias: torch.Tensor,
    *,
    recipe: Mapping[str, Any],
) -> torch.Tensor:
    """Rebuild fused logits and apply the frozen bias at the frozen strength."""
    fused = fused_validation_logits(
        payload, fusion=str(recipe["fusion"]), temperature=float(recipe["temperature"])
    )
    return apply_prior_bias(
        fused, bias, strength=float(recipe["prior_strength"])
    )
