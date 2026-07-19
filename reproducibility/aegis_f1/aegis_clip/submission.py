"""Fail-closed prediction CSV, ZIP, and lineage manifest generation."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from aegis_clip.runtime import atomic_json_dump, sha256_file


def create_submission(
    predictions: Sequence[tuple[str, str]],
    expected_names: Sequence[str],
    output_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    inference_mode: str,
    tta_risk_acknowledged: bool,
    valid_labels: set[str] | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate temporary artifacts first, publish CSV/ZIP only after success."""
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Submission directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="aegis-submission-", dir=destination.parent))
    try:
        csv_path = temporary / "pred_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for name, label in predictions:
                writer.writerow([name, label])
        validate_predictions(predictions, expected_names, valid_labels=valid_labels)
        zip_path = temporary / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(csv_path, arcname="pred_results.csv")
        _validate_zip(zip_path, csv_path)

        manifest = {
            "format_version": 1,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "inference_mode": inference_mode,
            "tta_risk_acknowledged": bool(tta_risk_acknowledged),
            "prediction_count": len(predictions),
            "prediction_csv_sha256": sha256_file(csv_path),
            "submission_zip_sha256": sha256_file(zip_path),
        }
        if extra_manifest:
            protected = set(manifest)
            collisions = protected.intersection(extra_manifest)
            if collisions:
                raise ValueError(
                    f"Extra manifest cannot replace protected fields: {sorted(collisions)}"
                )
            manifest.update(extra_manifest)
        atomic_json_dump(manifest, temporary / "manifest.json")

        for source in temporary.iterdir():
            target = destination / source.name
            if target.exists():
                if not overwrite:
                    raise FileExistsError(f"Artifact already exists: {target}")
                target.unlink()
            os.replace(source, target)
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def validate_predictions(
    predictions: Sequence[tuple[str, str]],
    expected_names: Sequence[str],
    *,
    valid_labels: set[str] | None = None,
) -> None:
    expected = list(expected_names)
    if len(expected) != len(set(expected)):
        raise ValueError("Test set contains duplicate basenames")
    if len(predictions) != len(expected):
        raise ValueError(
            f"Prediction count {len(predictions)} does not match test count {len(expected)}"
        )
    names = [name for name, _ in predictions]
    if len(names) != len(set(names)):
        raise ValueError("Predictions contain duplicate image names")
    if set(names) != set(expected):
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise ValueError(
            f"Prediction coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )
    for name, label in predictions:
        if (
            name != name.strip()
            or not name
            or Path(name).name != name
            or "\\" in name
        ):
            raise ValueError(f"Invalid image name: {name!r}")
        if label != label.strip() or len(label) != 4 or not label.isdigit():
            raise ValueError(f"Invalid class label for {name}: {label!r}")
        if valid_labels is not None and label not in valid_labels:
            raise ValueError(f"Out-of-range class label for {name}: {label!r}")


def _validate_zip(zip_path: Path, source_csv: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        if names != ["pred_results.csv"]:
            raise ValueError(f"ZIP must contain only pred_results.csv, got {names}")
        extracted = archive.read("pred_results.csv")
    if extracted != source_csv.read_bytes():
        raise ValueError("ZIP-internal CSV differs from validated source CSV")
