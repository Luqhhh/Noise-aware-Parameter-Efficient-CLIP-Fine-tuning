"""Audit a purification manifest against fail-closed rules.

Usage:
    python3 scripts/audit_purification_manifest.py \\
        --manifest outputs/phase4/purification/nr_cl_knn_drop/purification_manifest.csv \\
        --strict-train outputs/data/d3_strict/seed42/train.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit_manifest(
    manifest: pd.DataFrame,
    strict_train_df: pd.DataFrame | None = None,
    max_class_reject_rate: float = 0.10,
    max_global_reject_rate: float = 0.10,
    max_global_relabel_rate: float = 0.03,
) -> tuple[list, list]:
    """Audit a purification manifest and return (errors, warnings)."""
    errors = []
    warnings = []

    required = [
        "sample_id", "image_path", "original_label", "training_label",
        "sample_weight", "quality_score", "training_role",
    ]
    for col in required:
        if col not in manifest.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return errors, warnings

    # 1. Coverage: all strict-train images in manifest
    if strict_train_df is not None:
        strict_paths = set(strict_train_df["image_path"])
        manifest_paths = set(manifest["image_path"])
        if strict_paths != manifest_paths:
            missing = strict_paths - manifest_paths
            extra = manifest_paths - strict_paths
            if missing:
                errors.append(
                    f"{len(missing)} strict-train images missing from manifest. "
                    f"First 5: {sorted(missing)[:5]}"
                )
            if extra:
                errors.append(
                    f"{len(extra)} manifest images not in strict-train. "
                    f"First 5: {sorted(extra)[:5]}"
                )

    # 2. No duplicate sample_id or image_path
    if manifest["sample_id"].duplicated().any():
        dup = manifest["sample_id"].duplicated().sum()
        errors.append(f"{dup} duplicate sample_ids")
    if manifest["image_path"].duplicated().any():
        dup = manifest["image_path"].duplicated().sum()
        errors.append(f"{dup} duplicate image_paths")

    # 3. Sample weight in [0, 1]
    if ((manifest["sample_weight"] < 0) | (manifest["sample_weight"] > 1)).any():
        errors.append("Some sample_weight values outside [0, 1]")

    # 4. training_role only allowed values
    valid_roles = {"clean", "rejected", "pseudo"}
    invalid = set(manifest["training_role"].unique()) - valid_roles
    if invalid:
        errors.append(f"Found invalid training_role values: {invalid}")

    # 5. clean: label unchanged, weight=1
    clean = manifest[manifest["training_role"] == "clean"]
    if (clean["training_label"] != clean["original_label"]).any():
        errors.append("clean samples have training_label != original_label")
    if (clean["sample_weight"] != 1.0).any():
        errors.append("clean samples have weight != 1.0")

    # 6. rejected: label unchanged, weight=0
    rejected = manifest[manifest["training_role"] == "rejected"]
    if len(rejected) > 0:
        if (rejected["training_label"] != rejected["original_label"]).any():
            errors.append("rejected samples have training_label != original_label")
        if (rejected["sample_weight"] != 0.0).any():
            errors.append("rejected samples have weight != 0.0")

    # 7. pseudo: label changed, weight=1
    pseudo = manifest[manifest["training_role"] == "pseudo"]
    if len(pseudo) > 0:
        if (pseudo["training_label"] == pseudo["original_label"]).any():
            errors.append("pseudo samples have training_label == original_label")
        if (pseudo["sample_weight"] != 1.0).any():
            errors.append("pseudo samples have weight != 1.0")

    # 8. Global caps
    n = len(manifest)
    n_rejected = len(rejected)
    n_pseudo = len(pseudo)
    if n_rejected / n > max_global_reject_rate:
        errors.append(
            f"Global reject rate {n_rejected/n:.4f} > {max_global_reject_rate}"
        )
    if n_pseudo / n > max_global_relabel_rate:
        errors.append(
            f"Global relabel rate {n_pseudo/n:.4f} > {max_global_relabel_rate}"
        )

    # 9. Original label match against strict-train
    if strict_train_df is not None:
        strict_label_map = dict(zip(strict_train_df["image_path"], strict_train_df["label"]))
        mismatches = 0
        for _, row in manifest.iterrows():
            sl = strict_label_map.get(row["image_path"])
            if sl is not None and int(sl) != int(row["original_label"]):
                mismatches += 1
        if mismatches > 0:
            errors.append(f"{mismatches} original_label mismatches vs strict-train")

    # 10. Per-class reject rate
    num_classes = manifest["original_label"].nunique()
    for c in range(num_classes):
        cls = manifest[manifest["original_label"] == c]
        if len(cls) == 0:
            continue
        r = (cls["training_role"] == "rejected").sum()
        if r / len(cls) > max_class_reject_rate:
            errors.append(
                f"Class {c} reject rate {r/len(cls):.4f} > {max_class_reject_rate}"
            )
        if (cls["training_role"] == "clean").sum() == 0:
            errors.append(f"Class {c} has zero clean samples")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="outputs/phase4/purification/nr_cl_classwise_drop/purification_manifest.csv",
    )
    parser.add_argument("--strict-train", default="outputs/data/d3_strict/seed42/train.csv")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    strict_train = (
        pd.read_csv(args.strict_train) if Path(args.strict_train).exists() else None
    )

    errors, warnings = audit_manifest(manifest, strict_train)

    result = {
        "manifest": str(args.manifest),
        "errors": errors,
        "warnings": warnings,
        "passed": len(errors) == 0,
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2))

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        raise SystemExit(1)
    else:
        for w in warnings:
            print(f"WARNING: {w}")
        print("AUDIT_PASSED")


if __name__ == "__main__":
    main()
