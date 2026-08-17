"""PACE-K2 protocol loading and exact-byte duplicate artifact construction."""

from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from aegis_clip.data import resolve_image_path
from aegis_clip.features import canonical_sample_path
from aegis_clip.runtime import atomic_json_dump, sha256_file


class PacePreflightError(ValueError):
    """Raised when a frozen PACE protocol input cannot be verified."""


@dataclass(frozen=True)
class PaceAssetPaths:
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


@dataclass(frozen=True)
class PaceGroupArtifact:
    state: str
    sha256: str | None
    output_path: Path
    report_path: Path
    expected_total_rows: int
    expected_unique_groups: int
    expected_duplicate_extras: int


@dataclass(frozen=True)
class PaceProtocol:
    protocol_id: str
    config_path: Path
    assets: PaceAssetPaths
    group_artifact: PaceGroupArtifact
    fixed: dict[str, Any]


@dataclass(frozen=True)
class GroupArtifactAudit:
    map_sha256: str
    total_rows: int
    unique_groups: int
    duplicate_extras: int
    cross_split_overlap: int


@dataclass(frozen=True)
class PreflightAudit:
    protocol_sha256: str
    split_assets_verified: bool
    model_assets_verified: bool


_ASSET_PATH_KEYS = {
    "train_csv",
    "val_csv",
    "class_to_idx",
    "idx_to_class",
    "split_manifest",
    "train_root",
}
EXPECTED_PROTOCOL_SECTIONS = {
    "protocol_id",
    "spec",
    "assets",
    "parent_assets",
    "group_artifact",
    "parent",
    "evidence",
    "crossfit",
    "beta_solver",
    "threshold",
    "promotion",
    "execution",
    "outputs",
}
_ASSET_KEYS = _ASSET_PATH_KEYS | {
    "train_csv_sha256",
    "val_csv_sha256",
    "class_to_idx_sha256",
    "idx_to_class_sha256",
    "split_manifest_sha256",
}
_GROUP_KEYS = {
    "state",
    "sha256",
    "output_path",
    "report_path",
    "expected_total_rows",
    "expected_unique_groups",
    "expected_duplicate_extras",
}
_TOP_LEVEL_KEYS = EXPECTED_PROTOCOL_SECTIONS
_REQUIRED_SPLIT_COLUMNS = {"image_path", "label"}
_TRACKED_ARTIFACT_PARTS = ("protocol_artifacts", "pace_k2_r2_parttoken")
_SPEC_KEYS = {"path", "sha256", "commit"}
_PARENT_ASSET_PATH_KEYS = {"checkpoint", "trust_bundle", "submission_dir"}
_PARENT_ASSET_DIGESTS = {
    "checkpoint_sha256": "26916fd3ec96311dcab7a637f416ad3455cf7c78087844d408a38958f168962a",
    "trust_bundle_sha256": "ff8688a818219d3737715cd7dc9d11d014e7a5e973fa68f966c0ce370a88a246",
    "prediction_csv_sha256": "c6ed3e6a7f63c49a9b821f0e09222a153d926702de6f4c42505781aa7ae89fdd",
    "submission_zip_sha256": "6333375eea0f0b7575b833de16daf89c897df521c9eaa3f64a71e546c5ec4dc6",
    "submission_manifest_sha256": (
        "0ddff9c4e03b0bfefe3c8671d388679a4a33dbbcc547b9f88d1b440fdca1c06e"
    ),
}
_EXPECTED_FIXED_SECTIONS: dict[str, dict[str, Any]] = {
    "parent": {
        "peft_mode": "visual_lora_mlp_lora",
        "classifier_mode": "linear",
        "feature_adapter": "Identity",
        "batch_size": 128,
        "amp": False,
        "crop_sizes": [128, 144, 160],
        "local_scale_weights": [0.45, 0.50, 0.05],
        "flip_weight": 0.50,
        "local_weight": 0.40,
        "global_temperature": 1.5,
        "local_temperature": 1.5,
        "local_top_k": 5,
        "part_top_patches": 8,
        "part_temperature": 0.07,
        "prior": {
            "target": "uniform",
            "strength": 0.85,
            "max_iterations": 50,
            "tolerance": 1.0e-6,
            "damping": 0.5,
        },
    },
    "evidence": {
        "view_order": [
            "original_128",
            "original_144",
            "original_160",
            "flipped_128",
            "flipped_144",
            "flipped_160",
        ],
        "tail_size": 7,
        "weight_norm_epsilon": 1.0e-12,
        "minimum_positive_views": 4,
        "require_orientation_positive": True,
        "require_leave_one_scale_positive": True,
        "linear_gate_atol": 1.0e-5,
        "linear_gate_rtol": 1.0e-5,
        "antisymmetry_atol": 1.0e-6,
        "antisymmetry_rtol": 1.0e-6,
    },
    "crossfit": {
        "seed": 42,
        "outer_folds": 5,
        "inner_folds": 3,
        "sklearn_version": "1.9.0",
        "numpy_version": "2.5.1",
        "sort_key": "canonical_path",
    },
    "beta_solver": {
        "dtype": "float64",
        "initial_upper": 1.0,
        "maximum_upper": 1048576.0,
        "maximum_iterations": 100,
        "interval_tolerance": 1.0e-12,
        "allow_zero": True,
        "intercept": False,
        "margin_coefficient": 1.0,
    },
    "threshold": {
        "modes": ["all_switch", "finite", "no_switch"],
        "strict_comparator": ">",
        "minimum_accuracy_changing_precision": 0.6,
        "minimum_wilson_lower": 0.5,
        "wilson_z": 1.959963984540054,
        "no_switch_reasons": [
            "no_eligible",
            "no_qualified_candidate",
            "inner_fit_failed",
            "full_refit_failed",
            "finite_mapping_failed",
            "final_oof_no_switch",
        ],
    },
    "promotion": {
        "minimum_raw_micro_delta_pp": 0.2,
        "minimum_clean_core_micro_delta_pp": 0.2,
        "minimum_net_correct_fraction": 0.002,
        "minimum_nonnegative_outer_folds": 4,
        "minimum_worst_fold_delta_pp": -0.1,
        "minimum_macro_delta_pp": -0.05,
        "bootstrap_draws": 10000,
        "bootstrap_seed": 42,
        "bootstrap_rng": "PCG64",
        "bootstrap_quantile_method": "linear",
    },
    "execution": {
        "device": "cuda",
        "amp": False,
        "batch_size": 128,
        "expected_validation_samples": 10316,
        "expected_test_samples": 24967,
    },
}


def load_pace_protocol(
    path: str | Path,
    *,
    allow_unfrozen_group: bool = False,
) -> PaceProtocol:
    """Load a strict PACE protocol and resolve paths relative to its YAML file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"PACE protocol does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise PacePreflightError("PACE protocol root must be a mapping")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "protocol")
    assets_payload = payload["assets"]
    group_payload = payload["group_artifact"]
    if not isinstance(assets_payload, dict) or not isinstance(group_payload, dict):
        raise PacePreflightError("assets and group_artifact must be mappings")
    _require_exact_keys(assets_payload, _ASSET_KEYS, "assets")
    _require_exact_keys(group_payload, _GROUP_KEYS, "group_artifact")
    _validate_fixed_protocol(payload)
    for key in _ASSET_KEYS - _ASSET_PATH_KEYS:
        _checked_digest(assets_payload[key], f"assets.{key}")

    base = config_path.parent

    def resolved(value: Any) -> Path:
        candidate = Path(str(value)).expanduser()
        return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    assets = PaceAssetPaths(
        **{
            key: resolved(value) if key in _ASSET_PATH_KEYS else str(value)
            for key, value in assets_payload.items()
        }
    )
    group = PaceGroupArtifact(
        state=str(group_payload["state"]),
        sha256=(
            None
            if group_payload["sha256"] is None
            else _checked_digest(group_payload["sha256"], "group_artifact.sha256")
        ),
        output_path=resolved(group_payload["output_path"]),
        report_path=resolved(group_payload["report_path"]),
        expected_total_rows=int(group_payload["expected_total_rows"]),
        expected_unique_groups=int(group_payload["expected_unique_groups"]),
        expected_duplicate_extras=int(group_payload["expected_duplicate_extras"]),
    )
    if str(payload["protocol_id"]) != "pace_k2_r2_parttoken_v1":
        raise PacePreflightError("unsupported PACE protocol_id")
    if group.state not in {"bootstrap_unfrozen", "frozen"}:
        raise PacePreflightError("group_artifact.state must be bootstrap_unfrozen or frozen")
    if group.state == "bootstrap_unfrozen":
        if group.sha256 is not None:
            raise PacePreflightError("unfrozen group artifact cannot declare a SHA-256")
        if not allow_unfrozen_group:
            raise PacePreflightError("PACE group artifact is not frozen")
    elif group.sha256 is None:
        raise PacePreflightError("frozen group artifact requires a SHA-256")
    if min(
        group.expected_total_rows,
        group.expected_unique_groups,
        group.expected_duplicate_extras,
    ) < 0:
        raise PacePreflightError("group artifact expected counts must be non-negative")
    if (
        group.expected_total_rows - group.expected_unique_groups
        != group.expected_duplicate_extras
    ):
        raise PacePreflightError("group artifact expected counts are inconsistent")
    _require_tracked_artifact_path(group.output_path)
    _require_tracked_artifact_path(group.report_path)
    if group.state == "frozen":
        if not group.output_path.is_file():
            raise PacePreflightError("frozen group artifact is missing")
        if sha256_file(group.output_path) != group.sha256:
            raise PacePreflightError("group artifact SHA-256 mismatch")

    return PaceProtocol(
        protocol_id=str(payload["protocol_id"]),
        config_path=config_path,
        assets=assets,
        group_artifact=group,
        fixed={
            key: payload[key]
            for key in EXPECTED_PROTOCOL_SECTIONS
            if key not in {"protocol_id", "assets", "group_artifact"}
        },
    )


def build_exact_group_artifact(
    protocol: PaceProtocol,
    *,
    hash_workers: int,
) -> GroupArtifactAudit:
    """Hash frozen official image bytes without creating or changing a split."""
    if int(hash_workers) <= 0:
        raise ValueError("hash_workers must be positive")
    verify_protocol_assets(protocol, require_model_assets=False)
    train = _load_split(protocol.assets.train_csv, "train")
    validation = _load_split(protocol.assets.val_csv, "validation")
    rows = [
        ("train", str(value))
        for value in train["image_path"].tolist()
    ] + [
        ("validation", str(value))
        for value in validation["image_path"].tolist()
    ]
    if len(rows) != protocol.group_artifact.expected_total_rows:
        raise PacePreflightError(
            f"formal split row count mismatch: {len(rows)} != "
            f"{protocol.group_artifact.expected_total_rows}"
        )

    canonical_rows: list[tuple[str, str, Path]] = []
    seen: set[str] = set()
    for split, raw_path in rows:
        canonical = canonical_sample_path(raw_path)
        if canonical in seen:
            raise PacePreflightError(f"duplicate canonical path across formal CSVs: {canonical}")
        seen.add(canonical)
        absolute = resolve_image_path(protocol.assets.train_root, raw_path)
        if not absolute.is_file():
            raise PacePreflightError(f"official image is missing: {canonical}")
        canonical_rows.append((split, canonical, absolute))
    canonical_rows.sort(key=lambda item: item[1])

    with ThreadPoolExecutor(max_workers=int(hash_workers)) as executor:
        digests = list(executor.map(sha256_file, [row[2] for row in canonical_rows]))
    mapping = {
        canonical: _checked_digest(digest, f"image digest for {canonical}")
        for (_, canonical, _), digest in zip(canonical_rows, digests)
    }
    train_groups = {
        mapping[canonical] for split, canonical, _ in canonical_rows if split == "train"
    }
    validation_groups = {
        mapping[canonical]
        for split, canonical, _ in canonical_rows
        if split == "validation"
    }
    overlap = train_groups & validation_groups
    if overlap:
        raise PacePreflightError(
            f"exact duplicate group crosses train and validation: {len(overlap)}"
        )
    unique_groups = len(set(mapping.values()))
    duplicate_extras = len(mapping) - unique_groups
    if unique_groups != protocol.group_artifact.expected_unique_groups:
        raise PacePreflightError(
            f"unique group count mismatch: {unique_groups} != "
            f"{protocol.group_artifact.expected_unique_groups}"
        )
    if duplicate_extras != protocol.group_artifact.expected_duplicate_extras:
        raise PacePreflightError(
            f"duplicate extras mismatch: {duplicate_extras} != "
            f"{protocol.group_artifact.expected_duplicate_extras}"
        )

    atomic_json_dump(mapping, protocol.group_artifact.output_path)
    map_sha256 = sha256_file(protocol.group_artifact.output_path)
    if (
        protocol.group_artifact.state == "frozen"
        and map_sha256 != protocol.group_artifact.sha256
    ):
        raise PacePreflightError("rebuilt group artifact SHA-256 mismatch")
    audit = GroupArtifactAudit(
        map_sha256=map_sha256,
        total_rows=len(mapping),
        unique_groups=unique_groups,
        duplicate_extras=duplicate_extras,
        cross_split_overlap=0,
    )
    atomic_json_dump(
        {
            "format_version": 1,
            "protocol_id": protocol.protocol_id,
            "train_csv_sha256": protocol.assets.train_csv_sha256,
            "val_csv_sha256": protocol.assets.val_csv_sha256,
            "map_sha256": audit.map_sha256,
            "total_rows": audit.total_rows,
            "unique_groups": audit.unique_groups,
            "duplicate_extras": audit.duplicate_extras,
            "cross_split_overlap": audit.cross_split_overlap,
            "prepare_stage_called": False,
        },
        protocol.group_artifact.report_path,
    )
    return audit


def verify_protocol_assets(
    protocol: PaceProtocol,
    *,
    require_model_assets: bool,
) -> PreflightAudit:
    """Verify frozen split identities and, when requested, parent model files."""
    _verify_assets(protocol.assets)
    if require_model_assets:
        parent_assets = protocol.fixed["parent_assets"]
        for name, digest_key in (
            ("checkpoint", "checkpoint_sha256"),
            ("trust_bundle", "trust_bundle_sha256"),
        ):
            path = _resolve_protocol_path(protocol.config_path, parent_assets[name])
            if not path.is_file():
                raise PacePreflightError(f"{name} is missing: {path}")
            if sha256_file(path) != parent_assets[digest_key]:
                raise PacePreflightError(f"{name} SHA-256 mismatch")
    return PreflightAudit(
        protocol_sha256=sha256_file(protocol.config_path),
        split_assets_verified=True,
        model_assets_verified=bool(require_model_assets),
    )


def freeze_group_sha_in_config(path: str | Path, digest: str) -> None:
    """Atomically freeze the one-time exact-group digest in its protocol YAML."""
    checked = _checked_digest(digest, "group artifact digest")
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    group = payload["group_artifact"]
    current_state = str(group["state"])
    current_digest = group.get("sha256")
    if current_state == "frozen":
        if current_digest != checked:
            raise PacePreflightError("refusing to replace a frozen group SHA-256")
        return
    if current_state != "bootstrap_unfrozen" or current_digest is not None:
        raise PacePreflightError("group artifact is not in bootstrap state")
    group["state"] = "frozen"
    group["sha256"] = checked
    config_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=config_path.name + ".",
        suffix=".tmp",
        dir=config_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        os.replace(temporary, config_path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_assets(assets: PaceAssetPaths) -> None:
    for name in (
        "train_csv",
        "val_csv",
        "class_to_idx",
        "idx_to_class",
        "split_manifest",
    ):
        path = getattr(assets, name)
        if not path.is_file():
            raise PacePreflightError(f"{name} is missing: {path}")
        expected = _checked_digest(getattr(assets, f"{name}_sha256"), f"{name} SHA-256")
        actual = sha256_file(path)
        if actual != expected:
            raise PacePreflightError(f"{name} SHA-256 mismatch")
    if not assets.train_root.is_dir():
        raise PacePreflightError(f"train_root is missing: {assets.train_root}")


def _resolve_protocol_path(config_path: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _load_split(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = _REQUIRED_SPLIT_COLUMNS - set(frame)
    if missing:
        raise PacePreflightError(f"{name} CSV missing columns: {sorted(missing)}")
    if frame["image_path"].duplicated().any():
        raise PacePreflightError(f"{name} CSV contains duplicate image paths")
    return frame.reset_index(drop=True)


def _require_exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise PacePreflightError(
            f"{name} keys mismatch: missing={sorted(missing)} unknown={sorted(unknown)}"
        )


def _validate_fixed_protocol(payload: dict[str, Any]) -> None:
    spec = payload["spec"]
    if not isinstance(spec, dict):
        raise PacePreflightError("spec constants mismatch")
    _require_exact_keys(spec, _SPEC_KEYS, "spec")
    if (
        not isinstance(spec["path"], str)
        or not spec["path"]
        or spec["sha256"]
        != "668eca56452b3e2dc55ae1e8a9fceea626765f518b65688a8454d697e8f64fc9"
        or str(spec["commit"]) != "84f4705"
    ):
        raise PacePreflightError("spec constants mismatch")

    parent_assets = payload["parent_assets"]
    if not isinstance(parent_assets, dict):
        raise PacePreflightError("parent_assets constants mismatch")
    expected_parent_asset_keys = _PARENT_ASSET_PATH_KEYS | set(_PARENT_ASSET_DIGESTS)
    if set(parent_assets) != expected_parent_asset_keys:
        raise PacePreflightError("parent_assets constants mismatch")
    if any(
        not isinstance(parent_assets[key], str) or not parent_assets[key]
        for key in _PARENT_ASSET_PATH_KEYS
    ):
        raise PacePreflightError("parent_assets constants mismatch")
    if any(
        parent_assets[key] != digest
        for key, digest in _PARENT_ASSET_DIGESTS.items()
    ):
        raise PacePreflightError("parent_assets constants mismatch")

    for section, expected in _EXPECTED_FIXED_SECTIONS.items():
        if not _typed_constants_equal(payload[section], expected):
            raise PacePreflightError(f"{section} constants mismatch")

    outputs = payload["outputs"]
    if (
        not isinstance(outputs, dict)
        or set(outputs) != {"root"}
        or not isinstance(outputs["root"], str)
        or not outputs["root"]
    ):
        raise PacePreflightError("outputs constants mismatch")


def _typed_constants_equal(actual: Any, expected: Any) -> bool:
    try:
        options = {"allow_nan": False, "separators": (",", ":"), "sort_keys": True}
        return json.dumps(actual, **options) == json.dumps(expected, **options)
    except (TypeError, ValueError):
        return False


def _checked_digest(value: Any, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or digest.lower() != digest:
        raise PacePreflightError(f"{name} must be 64 lowercase hex characters")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise PacePreflightError(
            f"{name} must be 64 lowercase hex characters"
        ) from exc
    return digest


def _require_tracked_artifact_path(path: Path) -> None:
    parts = path.parts
    marker_length = len(_TRACKED_ARTIFACT_PARTS)
    if not any(
        tuple(parts[index : index + marker_length]) == _TRACKED_ARTIFACT_PARTS
        for index in range(len(parts) - marker_length + 1)
    ):
        raise PacePreflightError(
            "group artifacts must be under protocol_artifacts/pace_k2_r2_parttoken"
        )
