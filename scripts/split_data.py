import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#!/usr/bin/env python3
"""
Train/Validation Split Script.

Scans the training data directory for class subdirectories, then performs a
stratified (within-class) random split into train and validation sets.

Outputs:
    outputs/splits/train.csv           - [image_path, label, class_name]
    outputs/splits/val.csv             - [image_path, label, class_name]
    outputs/splits/class_to_idx.json   - {class_name: index}
    outputs/splits/idx_to_class.json   - {index: class_name}

Usage:
    python scripts/split_data.py --config configs/baseline.yaml
    python scripts/split_data.py --train_dir /path/to/train --val_ratio 0.1 --seed 42
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from common.utils import ensure_dir, load_config, set_seed

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split training data into train/val sets (within-class random split)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. If provided, reads data.train_dir, data.val_ratio, "
        "data.seed, and data.split_dir from the config.",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="Path to training data directory (overrides config).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=None,
        help="Fraction of data to use for validation (overrides config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the split (overrides config data.split_seed).",
    )
    parser.add_argument(
        "--split-seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds for multi-split generation. "
        "Overrides --seed. Each seed produces splits in "
        "split_dir/seed_{N}/.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Output directory for split files (overrides config).",
    )
    return parser.parse_args()


def find_class_directories(train_dir: Path) -> List[Path]:
    """Find all class subdirectories in the training directory.

    Args:
        train_dir: Path to the training data root.

    Returns:
        Sorted list of class directory paths.
    """
    class_dirs = sorted(
        [d for d in train_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
    )

    if not class_dirs:
        raise RuntimeError(f"No class directories found in {train_dir}")

    return class_dirs


def find_images_in_dir(directory: Path) -> List[Path]:
    """Find all image files in a directory."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(images)


def split_class_images(
    image_paths: List[Path],
    val_ratio: float,
) -> Tuple[List[Path], List[Path]]:
    """Split a class's images into train and val sets.

    Guarantees:
    - At least 1 image in val (if val_ratio > 0 and total >= 2)
    - At least 1 image in train

    Args:
        image_paths: List of image paths for one class.
        val_ratio: Fraction to use for validation.

    Returns:
        Tuple of (train_paths, val_paths).
    """
    n_total = len(image_paths)

    if n_total == 0:
        return [], []

    if n_total == 1:
        # Only one image: put in train, val gets it too for completeness
        logger.warning(
            f"Class with only 1 image. Placing in train set; val set will be empty for this class."
        )
        return image_paths, []

    n_val = max(1, int(n_total * val_ratio))
    n_val = min(n_val, n_total - 1)  # Ensure at least 1 for train

    # Shuffle deterministically (seeded globally)
    shuffled = image_paths.copy()
    random.shuffle(shuffled)

    val_paths = shuffled[:n_val]
    train_paths = shuffled[n_val:]

    return train_paths, val_paths


def build_class_mapping(
    class_dirs: List[Path],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build class name to index mapping (sorted by class name).

    Args:
        class_dirs: Sorted list of class directory paths.

    Returns:
        Tuple of (class_to_idx, idx_to_class).
    """
    class_to_idx = {}
    idx_to_class = {}

    for i, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        class_to_idx[class_name] = i
        idx_to_class[i] = class_name

    return class_to_idx, idx_to_class


def _generate_single_split(
    train_dir: Path, split_dir: Path, val_ratio: float, seed: int
) -> None:
    """Generate a single train/val split for the given seed.

    Args:
        train_dir: Path to training data root.
        split_dir: Output directory for split files.
        val_ratio: Fraction of data for validation.
        seed: Random seed for this split.
    """
    set_seed(seed)

    split_dir = ensure_dir(split_dir)

    logger.info(f"Generating split with seed={seed} -> {split_dir}")

    # Find class directories
    class_dirs = find_class_directories(train_dir)

    # Build class mapping (sorted by name)
    class_to_idx, idx_to_class = build_class_mapping(class_dirs)

    # Split each class
    train_entries = []
    val_entries = []
    class_stats = {}

    for class_dir in class_dirs:
        class_name = class_dir.name
        label = class_to_idx[class_name]
        images = find_images_in_dir(class_dir)

        train_imgs, val_imgs = split_class_images(images, val_ratio)

        for img_path in train_imgs:
            train_entries.append(
                {
                    "image_path": str(img_path.resolve()),
                    "label": label,
                    "class_name": class_name,
                }
            )

        for img_path in val_imgs:
            val_entries.append(
                {
                    "image_path": str(img_path.resolve()),
                    "label": label,
                    "class_name": class_name,
                }
            )

        class_stats[class_name] = {
            "total": len(images),
            "train": len(train_imgs),
            "val": len(val_imgs),
        }

    # Save CSV files
    train_df = pd.DataFrame(train_entries)
    val_df = pd.DataFrame(val_entries)

    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # Save class mappings
    class_to_idx_path = split_dir / "class_to_idx.json"
    idx_to_class_path = split_dir / "idx_to_class.json"

    with open(class_to_idx_path, "w") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

    with open(idx_to_class_path, "w") as f:
        json.dump(
            {str(k): v for k, v in idx_to_class.items()},
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Summary
    logger.info(f"  Seed {seed}: {len(train_entries)} train, {len(val_entries)} val, "
                f"{len(class_dirs)} classes")

    # Check for classes with very few samples
    small_classes = [
        (name, stats) for name, stats in class_stats.items() if stats["total"] < 5
    ]
    if small_classes:
        logger.warning(
            f"Classes with fewer than 5 samples ({len(small_classes)} classes):"
        )
        for name, stats in small_classes[:10]:
            logger.warning(
                f"  {name}: total={stats['total']}, "
                f"train={stats['train']}, val={stats['val']}"
            )


def main():
    args = parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine parameters: CLI args take precedence over config
    train_dir = args.train_dir
    val_ratio = args.val_ratio
    seed = args.seed
    split_seeds_str = args.split_seeds
    split_dir = args.split_dir

    if args.config:
        config = load_config(args.config)
        data_cfg = config["data"]
        if train_dir is None:
            train_dir = data_cfg["train_dir"]
        if val_ratio is None:
            val_ratio = data_cfg.get("val_ratio", 0.1)
        if seed is None and split_seeds_str is None:
            seed = data_cfg.get("split_seed", data_cfg.get("seed", 42))
        if split_dir is None:
            split_dir = data_cfg.get("split_dir", "outputs/splits")
    else:
        if train_dir is None:
            raise ValueError("Either --config or --train_dir must be provided.")
        val_ratio = val_ratio if val_ratio is not None else 0.1
        seed = seed if seed is not None else 42
        split_dir = split_dir if split_dir is not None else "outputs/splits"

    train_dir = Path(train_dir)
    base_split_dir = Path(split_dir)

    # Determine seeds
    if split_seeds_str is not None:
        seeds = [int(s.strip()) for s in split_seeds_str.split(",") if s.strip()]
        if not seeds:
            raise ValueError("--split-seeds must contain at least one seed")
        logger.info(f"Multi-split mode: generating splits for seeds {seeds}")
        logger.info(f"Training directory: {train_dir}")
        logger.info(f"Validation ratio:   {val_ratio}")
        logger.info(f"Base output dir:    {base_split_dir}")

        for s in seeds:
            seed_split_dir = base_split_dir / f"seed_{s}"
            _generate_single_split(train_dir, seed_split_dir, val_ratio, s)

        logger.info("=" * 60)
        logger.info(f"Multi-split complete! Generated {len(seeds)} splits.")
    else:
        # Single seed mode
        logger.info(f"Training directory: {train_dir}")
        logger.info(f"Validation ratio:   {val_ratio}")
        logger.info(f"Random seed:        {seed}")
        logger.info(f"Output directory:   {base_split_dir}")
        _generate_single_split(train_dir, base_split_dir, val_ratio, seed)

        logger.info("=" * 60)
        logger.info("Split complete!")


if __name__ == "__main__":
    main()
