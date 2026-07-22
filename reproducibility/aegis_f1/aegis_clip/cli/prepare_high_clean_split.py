"""Build a fixed high-clean training split from cross-fitted trust scores."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from aegis_clip.data import TrustBundle
from aegis_clip.features import canonical_sample_path
from aegis_clip.runtime import atomic_json_dump, sha256_file


def prepare_high_clean_split(
    source_csv: str | Path,
    validation_csv: str | Path,
    trust_path: str | Path,
    output_csv: str | Path,
    *,
    threshold: float,
    expected_selected: int,
    expected_classes: int,
) -> tuple[Path, Path]:
    source_csv = Path(source_csv).resolve()
    validation_csv = Path(validation_csv).resolve()
    trust_path = Path(trust_path).resolve()
    destination = Path(output_csv).resolve()
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    source = pd.read_csv(source_csv)
    validation = pd.read_csv(validation_csv)
    required = {"image_path", "label"}
    if required - set(source) or required - set(validation):
        raise ValueError("Train/validation split is missing required columns")
    if source["image_path"].duplicated().any():
        raise ValueError("Source training split contains duplicate paths")
    trust = TrustBundle(trust_path)
    trust.verify_coverage(source["image_path"].astype(str).tolist())
    clean = [
        float(
            trust.clean_probability[
                trust.path_to_index[canonical_sample_path(path)]
            ]
        )
        for path in source["image_path"].astype(str)
    ]
    selected = source.loc[
        pd.Series(clean, index=source.index) >= float(threshold)
    ].copy()
    train_keys = {
        canonical_sample_path(path)
        for path in selected["image_path"].astype(str)
    }
    validation_keys = {
        canonical_sample_path(path)
        for path in validation["image_path"].astype(str)
    }
    overlap = train_keys & validation_keys
    if overlap:
        raise ValueError(f"High-clean split overlaps validation: {len(overlap)}")
    if len(selected) != int(expected_selected):
        raise ValueError(
            f"Selected {len(selected)} samples, expected {expected_selected}"
        )
    counts = selected["label"].astype(int).value_counts().sort_index()
    if len(counts) != int(expected_classes):
        raise ValueError(
            f"Selected split covers {len(counts)} classes, expected {expected_classes}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    selected.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    report = {
        "status": "passed",
        "threshold": float(threshold),
        "source_samples": len(source),
        "selected_samples": len(selected),
        "classes": len(counts),
        "minimum_class_samples": int(counts.min()),
        "maximum_class_samples": int(counts.max()),
        "validation_overlap": 0,
        "source_csv_sha256": sha256_file(source_csv),
        "validation_csv_sha256": sha256_file(validation_csv),
        "trust_bundle_sha256": sha256_file(trust_path),
        "output_csv_sha256": sha256_file(destination),
    }
    report_path = destination.with_suffix(".audit.json")
    atomic_json_dump(report, report_path)
    return destination, report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--trust-bundle", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--expected-selected", type=int, required=True)
    parser.add_argument("--expected-classes", type=int, default=500)
    args = parser.parse_args()
    output, report = prepare_high_clean_split(
        args.source_csv,
        args.validation_csv,
        args.trust_bundle,
        args.output_csv,
        threshold=args.threshold,
        expected_selected=args.expected_selected,
        expected_classes=args.expected_classes,
    )
    print(output)
    print(report)


if __name__ == "__main__":
    main()
