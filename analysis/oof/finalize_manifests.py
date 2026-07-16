"""Finalize OOF weight manifests into the training-side canonical schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.manifest_loader import ManifestLoader


OPTIONAL_SIGNAL_COLUMNS = [
    "oof_top1",
    "p_original_label",
    "p_top1",
    "prototype_margin",
    "knn_agreement",
    "flip_consistency",
    "duplicate_conflict_flag",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_canonical_manifest(
    quality: pd.DataFrame,
    weight_column: str,
    source: str,
) -> pd.DataFrame:
    """Convert sample quality rows to ManifestLoader's fail-closed schema."""
    required = {
        "sample_id",
        "image_path",
        "original_label",
        "quality",
        weight_column,
    }
    missing = required - set(quality.columns)
    if missing:
        raise ValueError(f"Missing quality columns: {sorted(missing)}")
    if quality["sample_id"].duplicated().any():
        raise ValueError("sample_quality contains duplicate sample_id values")
    if quality["image_path"].duplicated().any():
        raise ValueError("sample_quality contains duplicate image_path values")

    weights = pd.to_numeric(quality[weight_column], errors="coerce")
    scores = pd.to_numeric(quality["quality"], errors="coerce")
    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError(f"{weight_column} contains non-finite values")
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("quality contains non-finite values")
    if not weights.between(0.3, 1.0).all():
        raise ValueError(f"{weight_column} must stay within [0.3, 1.0]")
    if not scores.between(0.0, 1.0).all():
        raise ValueError("quality must stay within [0.0, 1.0]")

    result = pd.DataFrame(
        {
            "sample_id": quality["sample_id"].astype(str),
            "image_path": quality["image_path"].astype(str),
            "original_label": quality["original_label"].astype(int),
            "training_label": quality["original_label"].astype(int),
            "sample_weight": weights.astype(float),
            "quality_score": scores.astype(float),
            "source": source,
        }
    )
    for column in OPTIONAL_SIGNAL_COLUMNS:
        if column in quality.columns:
            result[column] = quality[column]
    return result


def _load_dataset_labels(strict_train_csv: Path) -> dict[str, int]:
    frame = pd.read_csv(strict_train_csv, dtype={"image_path": str, "label": int})
    required = {"image_path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Strict train CSV is missing columns: {sorted(missing)}")
    if frame["image_path"].duplicated().any():
        raise ValueError("Strict train CSV contains duplicate image_path values")
    return dict(zip(frame["image_path"].astype(str), frame["label"].astype(int)))


def _audit_manifest(
    manifest_path: Path,
    dataset_labels: dict[str, int],
    apply_low_weight_gate: bool,
) -> dict:
    loader = ManifestLoader(str(manifest_path))
    frame = loader.load()
    audit = loader.audit(frame, dataset_labels)

    finite_weights = bool(
        np.isfinite(frame["sample_weight"].to_numpy(dtype=float)).all()
    )
    finite_quality = bool(
        np.isfinite(frame["quality_score"].to_numpy(dtype=float)).all()
    )
    if not finite_weights:
        audit["errors"].append("sample_weight contains non-finite values")
    if not finite_quality:
        audit["errors"].append("quality_score contains non-finite values")

    low_fraction = (
        frame.assign(is_low=frame["sample_weight"] < 0.5)
        .groupby("original_label")["is_low"]
        .mean()
    )
    warning_classes = sorted(
        int(label) for label, fraction in low_fraction.items() if fraction > 0.30
    )
    audit.update(
        {
            "all_weights_finite": finite_weights,
            "all_quality_scores_finite": finite_quality,
            "classes_with_over_30pct_weight_below_0_5": warning_classes,
            "low_weight_gate_applied": apply_low_weight_gate,
            "manifest_path": str(manifest_path),
        }
    )
    audit["training_allowed"] = bool(
        not audit["errors"] and (not apply_low_weight_gate or not warning_classes)
    )
    return audit


def finalize_manifests(
    sample_quality_path: Path,
    strict_train_csv: Path,
    output_dir: Path,
) -> dict:
    """Write canonical soft/discrete manifests and a fail-closed weight audit."""
    quality = pd.read_csv(
        sample_quality_path,
        dtype={"sample_id": str, "image_path": str, "original_label": int},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    soft_path = output_dir / "oof_soft_weight_manifest.csv"
    discrete_path = output_dir / "oof_discrete_weight_manifest.csv"

    build_canonical_manifest(quality, "soft_weight", "oof_soft").to_csv(
        soft_path, index=False
    )
    build_canonical_manifest(quality, "discrete_weight", "oof_discrete").to_csv(
        discrete_path, index=False
    )

    dataset_labels = _load_dataset_labels(strict_train_csv)
    soft_audit = _audit_manifest(soft_path, dataset_labels, True)
    discrete_audit = _audit_manifest(discrete_path, dataset_labels, False)
    discrete_audit["requires_soft_gate"] = True
    discrete_audit["training_allowed"] = bool(
        soft_audit["training_allowed"] and discrete_audit["training_allowed"]
    )

    stop_reasons = []
    if soft_audit["errors"]:
        stop_reasons.append("soft_manifest_validation_failed")
    if soft_audit["classes_with_over_30pct_weight_below_0_5"]:
        stop_reasons.append("soft_low_weight_class_fraction_exceeds_30pct")
    audit = {
        "schema_version": 1,
        "sample_quality_path": str(sample_quality_path),
        "sample_quality_sha256": _sha256(sample_quality_path),
        "strict_train_csv": str(strict_train_csv),
        "strict_train_sha256": _sha256(strict_train_csv),
        "sample_count": len(quality),
        "soft": soft_audit,
        "discrete": discrete_audit,
        "overall_training_allowed": bool(soft_audit["training_allowed"]),
        "decision": (
            "proceed_soft_then_discrete"
            if soft_audit["training_allowed"]
            else "stop_before_weight_training"
        ),
        "stop_reasons": stop_reasons,
        "protocol_note": (
            "The >30% low-weight gate applies to the continuous soft manifest. "
            "The classwise-tertile discrete control is evaluated only after the "
            "soft gate because it assigns roughly one third of each class to 0.3."
        ),
    }
    (output_dir / "weight_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-quality",
        default="outputs/phase3/oof/sample_quality.csv",
    )
    parser.add_argument(
        "--strict-train-csv",
        default="outputs/d3_strict/seed42/train.csv",
    )
    parser.add_argument("--output-dir", default="outputs/phase3/oof")
    args = parser.parse_args()

    audit = finalize_manifests(
        Path(args.sample_quality),
        Path(args.strict_train_csv),
        Path(args.output_dir),
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
