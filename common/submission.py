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
import logging
import zipfile
from pathlib import Path

import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)


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

    # Generate pred_results.csv
    # Format: image_name.jpg, 0001
    results_path = out_dir / "pred_results.csv"

    with open(results_path, "w") as f:
        for _, row in df.iterrows():
            img_name = row["image_name"]
            pred_label = str(row["pred_label"]).zfill(4)  # Ensure 4-digit format
            f.write(f"{img_name}, {pred_label}\n")

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
