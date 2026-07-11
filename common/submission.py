"""
Submission file generator.

Converts the raw prediction CSV (pred_raw.csv) into the final submission format:
    - pred_results.csv: each line formatted as "image_name.jpg, 0001"
    - submission.zip: containing only pred_results.csv

Usage:
    python -m src.submission --raw outputs/submissions/pred_raw.csv \
        --out_dir outputs/submissions
"""

import argparse
import csv
import logging
import zipfile
from pathlib import Path

import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)

# Supported image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final submission files from raw predictions."
    )
    parser.add_argument(
        "--raw",
        type=str,
        required=True,
        help="Path to pred_raw.csv (output from src/infer.py).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/submissions",
        help="Directory to save submission files.",
    )
    return parser.parse_args()


def validate_submission_coverage(test_dir: str, results_csv_path: str) -> None:
    """Validate submission coverage with explicit exceptions.

    Checks:
      1. Test basename uniqueness
      2. Prediction count == test image count
      3. No duplicate image names in predictions
      4. Set equality (no missing, no extra)
      5. Class name format (4-digit, no whitespace)

    Args:
        test_dir: Path to test image directory.
        results_csv_path: Path to pred_results.csv.

    Raises:
        ValueError: On any validation failure.
        FileNotFoundError: If paths don't exist.
    """
    test_dir = Path(test_dir)
    results_csv_path = Path(results_csv_path)

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not results_csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv_path}")

    # Collect test image basenames
    test_image_paths = sorted(
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    test_names = [p.name for p in test_image_paths]

    # Check basename uniqueness
    if len(test_names) != len(set(test_names)):
        raise ValueError("Test set contains duplicate basenames")

    expected_names = set(test_names)

    # Read predictions
    with open(results_csv_path, "r") as f:
        reader = csv.reader(f)
        submission_rows = list(reader)

    predicted_names = [row[0].strip() for row in submission_rows if row]

    # Check count
    if len(predicted_names) != len(expected_names):
        raise ValueError(
            f"Prediction count mismatch: "
            f"got {len(predicted_names)}, expected {len(expected_names)}"
        )

    # Check duplicates
    if len(predicted_names) != len(set(predicted_names)):
        raise ValueError("Submission contains duplicate image names")

    # Check set equality
    predicted_set = set(predicted_names)
    if predicted_set != expected_names:
        missing = sorted(expected_names - predicted_set)
        extra = sorted(predicted_set - expected_names)
        raise ValueError(
            f"Submission coverage mismatch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    # Check class name format
    for row in submission_rows:
        if not row:
            continue
        image_name = row[0].strip()
        class_name = row[1].strip() if len(row) > 1 else ""

        if class_name != class_name.strip():
            raise ValueError(f"Class name contains whitespace: {class_name!r}")
        if len(class_name) != 4 or not class_name.isdigit():
            raise ValueError(f"Invalid class name for {image_name}: {class_name!r}")

    logger.info(f"Submission validation passed: {len(predicted_names)} predictions")


def generate_submission(raw_csv_path: str, out_dir: str) -> tuple:
    """Generate pred_results.csv and submission.zip from raw predictions.

    Args:
        raw_csv_path: Path to pred_raw.csv with columns [image_name, pred_idx, pred_label].
        out_dir: Output directory.

    Returns:
        Tuple of (results_csv_path, zip_path).

    Raises:
        FileNotFoundError: If raw_csv_path does not exist.
        ValueError: If raw CSV is missing required columns.
    """
    raw_csv_path = Path(raw_csv_path)
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw prediction file not found: {raw_csv_path}")

    out_dir = ensure_dir(out_dir)

    # Read raw predictions
    df = pd.read_csv(raw_csv_path)
    logger.info(f"Loaded {len(df)} predictions from {raw_csv_path}")

    # Validate columns
    required_cols = {"image_name", "pred_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Generate pred_results.csv using csv.writer (no header)
    # Format: image_name.jpg, 0001
    results_path = out_dir / "pred_results.csv"

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        for _, row in df.iterrows():
            img_name = row["image_name"]
            pred_label = str(row["pred_label"]).zfill(4)
            # Prepend space to match "image_name.jpg, 0001" format
            writer.writerow([img_name, f" {pred_label}"])

    logger.info(f"pred_results.csv written to: {results_path}")
    logger.info(f"  Lines: {len(df)}")
    logger.info(f"  Format: image_name.jpg, 0001")

    # Generate submission.zip (containing only pred_results.csv, no directory structure)
    zip_path = out_dir / "submission.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # arcname ensures no directory hierarchy in the zip
        zf.write(results_path, arcname="pred_results.csv")

    logger.info(f"submission.zip written to: {zip_path}")
    logger.info(f"  Contents: pred_results.csv only")

    # Verify zip contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        logger.info(f"  Zip contains: {names}")

    return str(results_path), str(zip_path)


def main():
    args = parse_args()

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        results_path, zip_path = generate_submission(args.raw, args.out_dir)
        logger.info("Submission generation complete!")
        logger.info(f"  Results CSV: {results_path}")
        logger.info(f"  Submission ZIP: {zip_path}")
    except Exception as e:
        logger.error(f"Submission generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
