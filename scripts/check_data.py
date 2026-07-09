import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#!/usr/bin/env python3
"""
Data checking and statistics script.

Scans the training and test directories, collects statistics, checks for
corrupted images, and saves results to a JSON file.

Usage:
    python scripts/check_data.py --config configs/baseline.yaml
    python scripts/check_data.py --train_dir /path/to/train --test_dir /path/to/test
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

from common.utils import ensure_dir, load_config

# Add parent directory to path


ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check dataset integrity and compute statistics."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="Path to training data directory (overrides config).",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Path to test data directory (overrides config).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory to save data_stats.json.",
    )
    return parser.parse_args()


def check_image(image_path: Path) -> bool:
    """Try to open and verify an image file.

    Args:
        image_path: Path to the image.

    Returns:
        True if the image can be opened and verified, False otherwise.
    """
    try:
        img = Image.open(image_path)
        img.verify()
        # Re-open after verify() because verify() may leave the file in a bad state
        img = Image.open(image_path)
        img.convert("RGB")
        return True
    except Exception:
        return False


def analyze_train_dir(train_dir: Path) -> Dict:
    """Analyze the training directory structure and count images.

    Returns:
        Dictionary with training data statistics.
    """
    class_dirs = sorted(
        [d for d in train_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
    )

    num_classes = len(class_dirs)
    class_counts = {}
    total_images = 0
    corrupted_images = 0
    corrupted_list = []

    logger.info(f"Counting images in {num_classes} class directories...")

    # Phase 1: Fast count (no image opening)
    for class_dir in tqdm(class_dirs, desc="Counting train images"):
        class_name = class_dir.name
        count = 0

        for ext in IMAGE_EXTENSIONS:
            count += len(list(class_dir.glob(f"*{ext}")))
            count += len(list(class_dir.glob(f"*{ext.upper()}")))

        total_images += count
        class_counts[class_name] = count

    # Phase 2: Sample-based corruption check (10 images per class)
    logger.info("Sample corruption check (10 images per class)...")
    samples_checked = 0
    for class_dir in tqdm(class_dirs, desc="Checking corruption"):
        checked = 0
        for ext in IMAGE_EXTENSIONS:
            if checked >= 10:
                break
            for img_path in class_dir.glob(f"*{ext}"):
                if checked >= 10:
                    break
                if not check_image(img_path):
                    corrupted_images += 1
                    corrupted_list.append(str(img_path))
                checked += 1
                samples_checked += 1
            for img_path in class_dir.glob(f"*{ext.upper()}"):
                if checked >= 10:
                    break
                if not check_image(img_path):
                    corrupted_images += 1
                    corrupted_list.append(str(img_path))
                checked += 1
                samples_checked += 1

    counts_array = np.array(list(class_counts.values()))

    stats = {
        "num_classes": num_classes,
        "total_images": total_images,
        "corrupted_images": corrupted_images,
        "min_samples_per_class": int(counts_array.min()),
        "max_samples_per_class": int(counts_array.max()),
        "mean_samples_per_class": float(counts_array.mean()),
        "std_samples_per_class": float(counts_array.std()),
        "median_samples_per_class": float(np.median(counts_array)),
        "classes_with_zero_images": [
            name for name, c in class_counts.items() if c == 0
        ],
        "class_counts": class_counts,
    }

    if corrupted_images > 0:
        stats["corrupted_sample_list"] = corrupted_list[:100]  # First 100

    return stats


def analyze_test_dir(test_dir: Path) -> Dict:
    """Analyze the test directory.

    Returns:
        Dictionary with test data statistics.
    """
    # Count images quickly
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(test_dir.glob(f"*{ext}"))
        images.extend(test_dir.glob(f"*{ext.upper()}"))

    images = sorted(images)
    total = len(images)

    # Sample corruption check: 1000 random images
    sample_size = min(1000, total)
    corrupted = 0
    corrupted_list = []

    import random

    random.seed(42)
    sample_images = random.sample(images, sample_size)

    logger.info(f"Sample corruption check on {sample_size}/{total} test images...")
    for img_path in tqdm(sample_images, desc="Checking test images"):
        if not check_image(img_path):
            corrupted += 1
            corrupted_list.append(str(img_path))

    # Extrapolate
    if corrupted > 0:
        corr_rate = corrupted / sample_size
        estimated = int(corr_rate * total)
        logger.info(
            f"Sample corruption rate: {corr_rate:.4f} " f"(estimated {estimated} total)"
        )

    # Check filename extensions
    extensions_found = {}
    for img in images:
        ext = img.suffix.lower()
        extensions_found[ext] = extensions_found.get(ext, 0) + 1

    stats = {
        "total_images": total,
        "corrupted_images": corrupted,
        "extensions": extensions_found,
    }

    if corrupted > 0:
        stats["corrupted_sample_list"] = corrupted_list[:100]

    return stats


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine paths
    train_dir = args.train_dir
    test_dir = args.test_dir

    if args.config:
        config = load_config(args.config)
        data_cfg = config["data"]
        if train_dir is None:
            train_dir = data_cfg.get("train_dir")
        if test_dir is None:
            test_dir = data_cfg.get("test_dir")
        if args.log_dir is None:
            log_dir = config.get("output", {}).get("log_dir", "outputs/logs")
        else:
            log_dir = args.log_dir
    else:
        log_dir = args.log_dir

    if train_dir is None and test_dir is None:
        raise ValueError(
            "At least one of --train_dir or --test_dir must be provided "
            "(or use --config)."
        )

    log_dir = ensure_dir(log_dir)

    all_stats = {}

    # Analyze training data
    if train_dir:
        train_dir = Path(train_dir)
        if not train_dir.exists():
            logger.error(f"Training directory does not exist: {train_dir}")
        else:
            logger.info(f"Analyzing training data: {train_dir}")
            train_stats = analyze_train_dir(train_dir)
            all_stats["train"] = train_stats

            logger.info("=" * 60)
            logger.info("Training Data Statistics:")
            logger.info(f"  Number of classes:          {train_stats['num_classes']}")
            logger.info(f"  Total images:               {train_stats['total_images']}")
            logger.info(
                f"  Corrupted images:           {train_stats['corrupted_images']}"
            )
            logger.info(
                f"  Min samples per class:      {train_stats['min_samples_per_class']}"
            )
            logger.info(
                f"  Max samples per class:      {train_stats['max_samples_per_class']}"
            )
            logger.info(
                f"  Mean samples per class:     {train_stats['mean_samples_per_class']:.1f}"
            )
            logger.info(
                f"  Std samples per class:      {train_stats['std_samples_per_class']:.1f}"
            )
            logger.info(
                f"  Median samples per class:   {train_stats['median_samples_per_class']:.1f}"
            )

            zero_classes = train_stats.get("classes_with_zero_images", [])
            if zero_classes:
                logger.warning(
                    f"  WARNING: {len(zero_classes)} classes have ZERO images!"
                )
                logger.warning(f"  Empty classes: {zero_classes[:10]}")

    # Analyze test data
    if test_dir:
        test_dir = Path(test_dir)
        if not test_dir.exists():
            logger.error(f"Test directory does not exist: {test_dir}")
        else:
            logger.info(f"\nAnalyzing test data: {test_dir}")
            test_stats = analyze_test_dir(test_dir)
            all_stats["test"] = test_stats

            logger.info("=" * 60)
            logger.info("Test Data Statistics:")
            logger.info(f"  Total images:         {test_stats['total_images']}")
            logger.info(f"  Corrupted images:     {test_stats['corrupted_images']}")
            logger.info(f"  Extensions:           {test_stats['extensions']}")

    # Save statistics
    stats_path = log_dir / "data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    logger.info(f"\nStatistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
