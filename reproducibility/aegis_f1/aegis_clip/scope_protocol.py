"""Frozen SCOPE-K2 protocol and fail-closed asset preflight."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aegis_clip.runtime import sha256_file


class ScopePreflightError(ValueError):
    """Raised when a frozen SCOPE input or protocol value is invalid."""


PARENT_LOCAL_VIEW_ORDER = (
    "original_112", "original_128", "original_144", "original_160",
    "flipped_112", "flipped_128", "flipped_144", "flipped_160",
)
EVIDENCE_VIEW_ORDER = (
    "original_128", "original_144", "original_160",
    "flipped_128", "flipped_144", "flipped_160",
)
EVIDENCE_VIEW_WEIGHTS = (0.1875, 0.25, 0.0625, 0.1875, 0.25, 0.0625)
PARENT_BRANCH_ORDER = (
    "original_global", "original_local", "flipped_global", "flipped_local",
)


def four_neighbor_edges() -> tuple[tuple[int, int], ...]:
    """Return the canonical horizontal-then-vertical 7x7 edge list."""
    horizontal = tuple(
        (row * 7 + column, row * 7 + column + 1)
        for row in range(7)
        for column in range(6)
    )
    vertical = tuple(
        (row * 7 + column, (row + 1) * 7 + column)
        for row in range(6)
        for column in range(7)
    )
    return horizontal + vertical


@dataclass(frozen=True)
class ScopeAssetPaths:
    train_csv: Path
    train_csv_sha256: str
    val_csv: Path
    val_csv_sha256: str
    class_to_idx: Path
    class_to_idx_sha256: str
    idx_to_class: Path
    idx_to_class_sha256: str
    split_manifest: Path
    split_manifest_sha256: str
    train_root: Path
    test_root: Path
    checkpoint: Path
    checkpoint_sha256: str
    trust_bundle: Path
    trust_bundle_sha256: str
    group_artifact: Path
    group_artifact_sha256: str
    fallback_csv: Path
    fallback_csv_sha256: str
    fallback_zip: Path
    fallback_zip_sha256: str
    fallback_manifest: Path
    fallback_manifest_sha256: str


@dataclass(frozen=True)
class ScopeProtocol:
    protocol_id: str
    config_path: Path
    spec_path: Path
    spec_sha256: str
    assets: ScopeAssetPaths
    fixed: dict[str, Any]


@dataclass(frozen=True)
class ScopePreflightAudit:
    protocol_sha256: str
    spec_verified: bool
    split_assets_verified: bool
    model_assets_verified: bool
    group_verified: bool
    fallback_verified: bool


_TOP_LEVEL = {
    "protocol_id", "spec", "assets", "parent_assets", "group_artifact",
    "parent", "evidence", "schemas", "crossfit", "beta_solver",
    "threshold", "bootstrap", "promotion", "execution", "outputs",
}
_EXPECTED_FIXED: dict[str, Any] = {
    "parent": {
        "peft_mode": "full_finetune",
        "classifier_mode": "linear",
        "feature_adapter": "Identity",
        "crop_sizes": [112, 128, 144, 160],
        "local_scale_weights": [0.2, 0.3, 0.4, 0.1],
        "flip_weight": 0.5,
        "local_weight": 0.4,
        "global_temperature": 1.5,
        "local_temperature": 1.5,
        "local_top_k": 5,
        "part_top_patches": 8,
        "part_temperature": 0.07,
        "prior": {
            "target": "uniform", "strength": 0.9, "max_iterations": 50,
            "tolerance": 1.0e-6, "damping": 0.5,
        },
    },
    "evidence": {
        "view_order": list(EVIDENCE_VIEW_ORDER),
        "view_weights": list(EVIDENCE_VIEW_WEIGHTS),
        "grid_shape": [7, 7],
        "adjacency": "four_neighbor_row_major_v1",
        "tail_size": 7,
        "weight_norm_epsilon": 1.0e-12,
        "minimum_positive_views": 4,
        "require_orientation_positive": True,
        "require_leave_one_scale_positive": True,
        "linear_gate_atol": 1.0e-5,
        "linear_gate_rtol": 1.0e-5,
        "antisymmetry_atol": 1.0e-6,
        "antisymmetry_rtol": 0.0,
    },
    "schemas": {
        "parent": "scope_parent_cache_v1",
        "evidence": "scope_evidence_cache_v1",
        "validation_only_fields": [
            "label", "clean_probability", "pseudo_label", "correction_alpha",
        ],
        "test_forbid_validation_fields": True,
    },
    "crossfit": {
        "seed": 42, "outer_folds": 5, "inner_folds": 3,
        "inner_seed_offset": 1000, "sort_key": "canonical_path",
        "conditional_parent": True,
    },
    "beta_solver": {
        "dtype": "float64", "initial_upper": 1.0,
        "maximum_upper": 1048576.0, "maximum_iterations": 100,
        "interval_tolerance": 1.0e-12, "l2_numerator": 1.0,
        "allow_zero": True, "intercept": False,
    },
    "threshold": {
        "modes": ["all_switch", "finite", "no_switch"],
        "strict_comparator": ">",
        "minimum_accuracy_changing_precision": 0.6,
        "minimum_wilson_lower": 0.5,
        "wilson_z": 1.959963984540054,
        "rank_mapping": "nearest_integer_fewer_switches_then_higher_cut",
    },
    "bootstrap": {
        "draws": 10000, "seed": 42, "rng": "PCG64",
        "quantile_method": "linear", "confidence": 0.95,
        "strata": ["outer_fold_id", "group_majority_label"],
    },
    "promotion": {
        "minimum_raw_delta_pp": 0.2,
        "minimum_clean_core_delta_pp": 0.2,
        "minimum_net_correct": 21,
        "minimum_nonnegative_outer_folds": 4,
        "bootstrap_lower_strictly_positive": True,
        "strictly_better_than": ["pace", "no_topology"],
    },
    "execution": {
        "device": "cuda", "amp": False, "batch_size": 128,
        "expected_validation_samples": 10316, "expected_test_samples": 24967,
        "numpy_version": "2.5.1", "sklearn_version": "1.9.0",
    },
}


def load_scope_protocol(path: str | Path) -> ScopeProtocol:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"SCOPE protocol does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScopePreflightError("SCOPE protocol root must be a mapping")
    _require_exact_keys(payload, _TOP_LEVEL, "protocol")
    if payload["protocol_id"] != "scope_k2_fullft_dual_pa090_v1":
        raise ScopePreflightError("unsupported SCOPE protocol_id")
    for section, expected in _EXPECTED_FIXED.items():
        if payload.get(section) != expected:
            raise ScopePreflightError(f"frozen section mismatch: {section}")

    base = config_path.parent
    resolve = lambda value: (
        Path(value).expanduser().resolve()
        if Path(value).expanduser().is_absolute()
        else (base / Path(value)).resolve()
    )
    spec = payload["spec"]
    assets = payload["assets"]
    parent = payload["parent_assets"]
    group = payload["group_artifact"]
    for name, value in ("spec", spec), ("assets", assets), ("parent_assets", parent), ("group_artifact", group):
        if not isinstance(value, dict):
            raise ScopePreflightError(f"{name} must be a mapping")

    digests = {
        "spec.sha256": spec.get("sha256"),
        **{f"assets.{key}": value for key, value in assets.items() if key.endswith("_sha256")},
        **{f"parent_assets.{key}": value for key, value in parent.items() if key.endswith("_sha256")},
        "group_artifact.sha256": group.get("sha256"),
    }
    for name, digest in digests.items():
        _checked_digest(digest, name)

    paths = ScopeAssetPaths(
        train_csv=resolve(assets["train_csv"]),
        train_csv_sha256=assets["train_csv_sha256"],
        val_csv=resolve(assets["val_csv"]),
        val_csv_sha256=assets["val_csv_sha256"],
        class_to_idx=resolve(assets["class_to_idx"]),
        class_to_idx_sha256=assets["class_to_idx_sha256"],
        idx_to_class=resolve(assets["idx_to_class"]),
        idx_to_class_sha256=assets["idx_to_class_sha256"],
        split_manifest=resolve(assets["split_manifest"]),
        split_manifest_sha256=assets["split_manifest_sha256"],
        train_root=resolve(assets["train_root"]),
        test_root=resolve(assets["test_root"]),
        checkpoint=resolve(parent["checkpoint"]),
        checkpoint_sha256=parent["checkpoint_sha256"],
        trust_bundle=resolve(parent["trust_bundle"]),
        trust_bundle_sha256=parent["trust_bundle_sha256"],
        group_artifact=resolve(group["path"]),
        group_artifact_sha256=group["sha256"],
        fallback_csv=resolve(parent["fallback_csv"]),
        fallback_csv_sha256=parent["fallback_csv_sha256"],
        fallback_zip=resolve(parent["fallback_zip"]),
        fallback_zip_sha256=parent["fallback_zip_sha256"],
        fallback_manifest=resolve(parent["fallback_manifest"]),
        fallback_manifest_sha256=parent["fallback_manifest_sha256"],
    )
    fixed = {key: payload[key] for key in _EXPECTED_FIXED} | {
        "outputs": payload["outputs"],
        "group_expected": {
            key: group[key] for key in (
                "expected_total_rows", "expected_unique_groups",
                "expected_duplicate_extras", "expected_cross_split_overlap",
            )
        },
    }
    return ScopeProtocol(
        protocol_id=payload["protocol_id"], config_path=config_path,
        spec_path=resolve(spec["path"]), spec_sha256=spec["sha256"],
        assets=paths, fixed=fixed,
    )


def verify_scope_assets(protocol: ScopeProtocol) -> ScopePreflightAudit:
    checks = (
        ("spec", protocol.spec_path, protocol.spec_sha256),
        ("train_csv", protocol.assets.train_csv, protocol.assets.train_csv_sha256),
        ("val_csv", protocol.assets.val_csv, protocol.assets.val_csv_sha256),
        ("class_to_idx", protocol.assets.class_to_idx, protocol.assets.class_to_idx_sha256),
        ("idx_to_class", protocol.assets.idx_to_class, protocol.assets.idx_to_class_sha256),
        ("split_manifest", protocol.assets.split_manifest, protocol.assets.split_manifest_sha256),
        ("checkpoint", protocol.assets.checkpoint, protocol.assets.checkpoint_sha256),
        ("trust_bundle", protocol.assets.trust_bundle, protocol.assets.trust_bundle_sha256),
        ("group_artifact", protocol.assets.group_artifact, protocol.assets.group_artifact_sha256),
        ("fallback_csv", protocol.assets.fallback_csv, protocol.assets.fallback_csv_sha256),
        ("fallback_zip", protocol.assets.fallback_zip, protocol.assets.fallback_zip_sha256),
        ("fallback_manifest", protocol.assets.fallback_manifest, protocol.assets.fallback_manifest_sha256),
    )
    for name, path, expected in checks:
        if not path.is_file():
            raise ScopePreflightError(f"{name} is missing: {path}")
        if sha256_file(path) != expected:
            raise ScopePreflightError(f"{name} SHA-256 mismatch")
    if not protocol.assets.train_root.is_dir() or not protocol.assets.test_root.is_dir():
        raise ScopePreflightError("train/test image root is missing")
    return ScopePreflightAudit(
        protocol_sha256=sha256_file(protocol.config_path), spec_verified=True,
        split_assets_verified=True, model_assets_verified=True,
        group_verified=True, fallback_verified=True,
    )


def _require_exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ScopePreflightError(
            f"{name} keys mismatch: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _checked_digest(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ScopePreflightError(f"{name} must be 64 lowercase hex characters")
    return text
