"""
Build the canonical master train/val split.

This script generates the ONE authoritative split per seed that ALL
experiments MUST reference. No experiment is allowed to generate its
own split after this exists.

Usage:
    python3 scripts/build_master_split.py --train-dir train --output-root outputs/master_splits --seed 42 --val-ratio 0.1
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common.class_mapping import generate_canonical_mapping
from common.utils import setup_logging

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Build the canonical master train/val split."
    )
    parser.add_argument("--train-dir", default="train")
    parser.add_argument("--output-root", default="outputs/master_splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_root) / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir), name="build_master_split")

    # ------------------------------------------------------------------
    # Collect all images
    # ------------------------------------------------------------------
    records = []
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "image_path": str(img_path.relative_to(train_dir)),
                    "class_name": class_name,
                })

    if not records:
        raise ValueError(f"No images found under {train_dir}")

    df = pd.DataFrame(records)
    logger.info("Found %d images in %d classes.", len(df), df["class_name"].nunique())

    # ------------------------------------------------------------------
    # Generate class mapping (lexicographic sort of class directories)
    # ------------------------------------------------------------------
    class_to_idx, idx_to_class = generate_canonical_mapping(
        train_dir=train_dir, expected_num_classes=500,
    )

    # Persist class mapping alongside the split
    json.dump(class_to_idx, (output_dir / "class_to_idx.json").open("w"),
              indent=2, sort_keys=True)
    json.dump(idx_to_class, (output_dir / "idx_to_class.json").open("w"),
              indent=2, sort_keys=True)

    df["label"] = df["class_name"].map(class_to_idx)
    missing = df["label"].isna().sum()
    if missing:
        raise ValueError(f"{missing} images not in class_to_idx mapping.")

    # ------------------------------------------------------------------
    # Stratified split
    # ------------------------------------------------------------------
    train_df, val_df = train_test_split(
        df, test_size=args.val_ratio, random_state=args.seed,
        stratify=df["label"],
    )

    train_df = train_df.sort_values("image_path").reset_index(drop=True)
    val_df = val_df.sort_values("image_path").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Write CSV files (format: image_path,class_name,label)
    # ------------------------------------------------------------------
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    train_df[["image_path", "class_name", "label"]].to_csv(train_csv, index=False)
    val_df[["image_path", "class_name", "label"]].to_csv(val_csv, index=False)

    # ------------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------------
    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    overlap = train_paths & val_paths
    if overlap:
        raise AssertionError(f"Train/val overlap: {len(overlap)} images")
    union = train_paths | val_paths
    if len(union) != len(df):
        raise AssertionError(f"Union ({len(union)}) != total ({len(df)})")

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    manifest = {
        "split_seed": args.seed,
        "source_root": str(train_dir.resolve()),
        "source_file_count": int(len(df)),
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "num_classes": int(df["class_name"].nunique()),
        "train_csv_sha256": sha256_file(train_csv),
        "val_csv_sha256": sha256_file(val_csv),
        "created_by_git_commit": None,
        "duplicate_grouping_enabled": False,
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info("Master split: %d train / %d val / %d classes",
                len(train_df), len(val_df), df["class_name"].nunique())


if __name__ == "__main__":
    main()
