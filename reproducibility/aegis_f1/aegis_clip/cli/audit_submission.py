"""Read-only, fail-closed audit of an existing competition submission."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

from aegis_clip.config import load_config
from aegis_clip.data import IMAGE_EXTENSIONS, load_class_mapping
from aegis_clip.runtime import sha256_file
from aegis_clip.submission import validate_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument(
        "--allow-tta",
        action="store_true",
        help="Allow an explicitly acknowledged same-model TTA artifact.",
    )
    args = parser.parse_args()
    result = audit_submission(
        config_path=args.config,
        submission_dir=args.submission_dir,
        allow_tta=args.allow_tta,
    )
    print(json.dumps(result, indent=2))


def audit_submission(
    *,
    config_path: str | Path,
    submission_dir: str | Path,
    allow_tta: bool = False,
) -> dict:
    config = load_config(config_path)
    root = Path(submission_dir).resolve()
    csv_path = root / "pred_results.csv"
    zip_path = root / "submission.zip"
    manifest_path = root / "manifest.json"
    for path in (csv_path, zip_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Submission artifact is missing: {path}")

    test_root = Path(config["data"]["test_root"])
    test_paths = sorted(
        path
        for path in test_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    expected_names = [path.name for path in test_paths]
    if len(expected_names) != int(config["data"]["expected_test_samples"]):
        raise ValueError("Official test image count changed since inference")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    malformed = [row for row in rows if len(row) != 2]
    if malformed:
        raise ValueError(f"CSV contains {len(malformed)} malformed rows")
    predictions = [(row[0], row[1]) for row in rows]
    _, idx_to_class = load_class_mapping(config["data"]["class_mapping"])
    valid_labels = {str(value).zfill(4) for value in idx_to_class.values()}
    validate_predictions(
        predictions,
        expected_names,
        valid_labels=valid_labels,
    )

    with zipfile.ZipFile(zip_path, "r") as archive:
        if archive.namelist() != ["pred_results.csv"]:
            raise ValueError("ZIP must contain only root-level pred_results.csv")
        if archive.read("pred_results.csv") != csv_path.read_bytes():
            raise ValueError("ZIP-internal CSV differs from external CSV")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = Path(manifest.get("checkpoint", ""))
    if not checkpoint.is_file():
        raise FileNotFoundError("Manifest checkpoint is missing")
    expected_hashes = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "prediction_csv_sha256": sha256_file(csv_path),
        "submission_zip_sha256": sha256_file(zip_path),
    }
    for key, expected in expected_hashes.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Manifest hash mismatch: {key}")
    if int(manifest.get("prediction_count", -1)) != len(predictions):
        raise ValueError("Manifest prediction_count mismatch")
    if int(manifest.get("corrupt_images", -1)) != 0:
        raise ValueError("Submission manifest reports corrupt test images")
    if manifest.get("inference_mode") != "none" and not allow_tta:
        raise ValueError("Official submission audit defaults to bare single-model inference")

    return {
        "status": "passed",
        "prediction_count": len(predictions),
        "classes": len(valid_labels),
        "inference_mode": manifest.get("inference_mode"),
        **expected_hashes,
    }


if __name__ == "__main__":
    main()
