import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#!/usr/bin/env python3
"""
Submission file checker.

Validates the submission files (pred_results.csv and submission.zip) against
the test set to ensure correctness before submission.

Checks:
    1. File name is exactly "pred_results.csv"
    2. Number of lines equals number of test images
    3. Each line has exactly 2 fields (image_name, label)
    4. All image names exist in the test set
    5. No duplicate image names
    6. No missing image names (all test images are covered)
    7. All labels are 4-digit strings
    8. All labels are in range [0000, {num_classes-1:04d}]
    9. ZIP file contains only pred_results.csv (no directory hierarchy)

Usage:
    python scripts/check_submission.py \
        --test_dir /path/to/test \
        --num-classes 500 \
        --csv outputs/submissions/pred_results.csv \
        --zip outputs/submissions/submission.zip

``--num-classes`` (or ``--class-mapping`` pointing at a class_to_idx.json)
must be provided explicitly so the label-range check stays stage-agnostic.
"""

import argparse
import json
import logging
import re
import sys
import zipfile
from pathlib import Path
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate submission files before submitting."
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Path to the test set directory.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Expected number of classes for the label-range check.",
    )
    parser.add_argument(
        "--class-mapping",
        type=str,
        default=None,
        help="Path to class_to_idx.json; num_classes is derived from it.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to pred_results.csv.",
    )
    parser.add_argument(
        "--zip",
        type=str,
        default=None,
        help="Path to submission.zip (optional).",
    )
    return parser.parse_args()


def get_test_image_names(test_dir: Path) -> Set[str]:
    """Collect all image filenames from the test directory."""
    return {
        path.name
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def check_csv(
    csv_path: Path,
    test_names: Set[str],
    num_classes: int | None = None,
) -> Tuple[bool, List[str]]:
    """Validate the pred_results.csv file.

    Returns:
        Tuple of (all_ok, error_messages).
    """
    errors = []

    # Check 1: File name
    if csv_path.name != "pred_results.csv":
        errors.append(f"❌ File name is '{csv_path.name}', expected 'pred_results.csv'")
    else:
        errors.append("✅ File name: pred_results.csv")

    # Read CSV
    with open(csv_path, "r") as f:
        lines = f.readlines()

    # Strip empty lines
    lines = [line.strip() for line in lines if line.strip()]

    # Check 2: Line count
    if len(lines) != len(test_names):
        errors.append(
            f"❌ Line count: {len(lines)} rows, but test set has {len(test_names)} images"
        )
    else:
        errors.append(f"✅ Line count: {len(lines)} (matches test set)")

    # Parse lines
    image_names_in_csv = []
    labels_in_csv = []

    for i, line in enumerate(lines, start=1):
        # Check 3: Two fields
        parts = line.split(",")
        if len(parts) != 2:
            errors.append(f"❌ Line {i}: expected 2 fields, got {len(parts)}: '{line}'")
            continue

        img_name = parts[0].strip()
        label = parts[1].strip()
        image_names_in_csv.append(img_name)
        labels_in_csv.append(label)

    # Check 4: Image names exist in test set
    csv_names_set = set(image_names_in_csv)
    missing_from_test = csv_names_set - test_names
    if missing_from_test:
        errors.append(
            f"❌ {len(missing_from_test)} image(s) in CSV not found in test set: "
            f"{list(missing_from_test)[:5]}..."
        )
    else:
        errors.append("✅ All image names found in test set")

    # Check 5: Duplicates
    if len(csv_names_set) != len(image_names_in_csv):
        duplicates = len(image_names_in_csv) - len(csv_names_set)
        errors.append(f"❌ {duplicates} duplicate image name(s) found in CSV")
    else:
        errors.append("✅ No duplicate image names")

    # Check 6: Missing test images
    missing_from_csv = test_names - csv_names_set
    if missing_from_csv:
        errors.append(
            f"❌ {len(missing_from_csv)} test image(s) missing from CSV: "
            f"{list(missing_from_csv)[:5]}..."
        )
    else:
        errors.append("✅ All test images covered")

    # Check 7: Label format (4 digits)
    non_4digit = [l for l in labels_in_csv if not re.match(r"^\d{4}$", l)]
    if non_4digit:
        errors.append(
            f"❌ {len(non_4digit)} label(s) are not 4-digit strings: "
            f"{non_4digit[:5]}..."
        )
    else:
        errors.append("✅ All labels are 4-digit strings")

    # Check 8: Label range
    if num_classes is None:
        errors.append("❌ num_classes is required for the label-range check")
        out_of_range = []
    else:
        out_of_range = [
            l
            for l in labels_in_csv
            if re.match(r"^\d{4}$", l)
            and (int(l) < 0 or int(l) >= num_classes)
        ]
    if out_of_range:
        errors.append(
            f"❌ {len(out_of_range)} label(s) out of range "
            f"[0000, {num_classes - 1:04d}]: "
            f"{out_of_range[:5]}..."
        )
    elif num_classes is not None:
        errors.append(f"✅ All labels in range [0000, {num_classes - 1:04d}]")

    # Summary
    all_ok = all(not e.startswith("❌") for e in errors)  # Include file name check

    return all_ok, errors


def check_zip(zip_path: Path) -> Tuple[bool, List[str]]:
    """Validate the submission.zip file.

    Returns:
        Tuple of (all_ok, error_messages).
    """
    errors = []

    if not zip_path.exists():
        errors.append(f"❌ ZIP file not found: {zip_path}")
        return False, errors

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # Check 9: Only pred_results.csv, no directory hierarchy
        if names == ["pred_results.csv"]:
            errors.append("✅ ZIP contains only pred_results.csv (no directory)")
        elif "pred_results.csv" in names and len(names) == 1:
            errors.append(f"⚠️  ZIP contains pred_results.csv but with path: {names}")
        elif "pred_results.csv" in names:
            errors.append(f"❌ ZIP contains extra files: {names}")
        else:
            errors.append(
                f"❌ ZIP does not contain pred_results.csv. Contents: {names}"
            )

    return all(not e.startswith("❌") for e in errors), errors


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    test_dir = Path(args.test_dir)
    csv_path = Path(args.csv)
    zip_path = Path(args.zip) if args.zip else None
    num_classes = args.num_classes
    if args.class_mapping:
        mapping = json.loads(Path(args.class_mapping).read_text(encoding="utf-8"))
        num_classes = len(mapping)
    if num_classes is None:
        logger.error(
            "Label-range check needs --num-classes or --class-mapping; "
            "refusing to assume the preliminary stage's 500 classes"
        )
        sys.exit(2)
    if num_classes <= 0:
        logger.error(f"Invalid num_classes: {num_classes}")
        sys.exit(2)

    # Validate paths
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        sys.exit(1)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    # Get test image names
    test_names = get_test_image_names(test_dir)
    logger.info(f"Test set contains {len(test_names)} images")

    all_ok = True

    # Check CSV
    logger.info("\n" + "=" * 60)
    logger.info("Checking pred_results.csv...")
    logger.info("=" * 60)
    csv_ok, csv_errors = check_csv(csv_path, test_names, num_classes=num_classes)
    for error in csv_errors:
        logger.info(f"  {error}")
    if not csv_ok:
        all_ok = False

    # Check ZIP
    if zip_path:
        logger.info("\n" + "=" * 60)
        logger.info("Checking submission.zip...")
        logger.info("=" * 60)
        zip_ok, zip_errors = check_zip(zip_path)
        for error in zip_errors:
            logger.info(f"  {error}")
        if not zip_ok:
            all_ok = False

    # Final verdict
    logger.info("\n" + "=" * 60)
    if all_ok:
        logger.info("🎉 All checks passed! Submission is ready.")
    else:
        logger.error("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
