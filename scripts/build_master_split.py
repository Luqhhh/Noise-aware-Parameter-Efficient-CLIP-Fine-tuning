"""
Build the canonical master train/val split.

This script generates the ONE authoritative split per seed that ALL
experiments MUST reference. No experiment is allowed to generate its
own split after this exists.

Strict-v2: SHA-256 group-aware splitting.
Images with the same SHA-256 hash (exact duplicates, including
cross-class copies) are assigned to the SAME split, preventing
content-level leakage between train and val.

Usage:
    python3 scripts/build_master_split.py --train-dir train --output-root outputs/master_splits --seed 42 --val-ratio 0.1
"""

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common.class_mapping import generate_canonical_mapping
from common.utils import setup_logging

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_groups_stratified(
    groups: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list, list]:
    """Split SHA-256 groups into train/val, stratified by majority class.

    Each group is a dict with keys: sha256, image_paths, class_names, labels, size.
    The entire group goes to either train or val — never split.

    Stratification uses the group's majority label to preserve per-class
    proportions as closely as possible.
    """
    # Assign each group a single "representative" label (majority vote)
    group_records = []
    for g in groups:
        majority_label = Counter(g["labels"]).most_common(1)[0][0]
        group_records.append({
            "sha256": g["sha256"],
            "majority_label": majority_label,
            "size": g["size"],
        })

    df_groups = pd.DataFrame(group_records)

    # Split groups (not images)
    train_groups, val_groups = train_test_split(
        df_groups,
        test_size=val_ratio,
        random_state=seed,
        stratify=df_groups["majority_label"],
    )

    train_sha = set(train_groups["sha256"])
    val_sha = set(val_groups["sha256"])

    return train_sha, val_sha


def main():
    parser = argparse.ArgumentParser(
        description="Build the canonical master train/val split."
    )
    parser.add_argument("--train-dir", default="train")
    parser.add_argument("--output-root", default="outputs/master_splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--no-sha256-dedup",
        action="store_true",
        help="Disable SHA-256 group-aware splitting (strict-v1 mode).",
    )
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_root) / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir), name="build_master_split")

    use_sha256_dedup = not args.no_sha256_dedup

    # ------------------------------------------------------------------
    # Collect all images + SHA-256
    # ------------------------------------------------------------------
    records = []
    sha256_map = {}  # absolute_path → sha256

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                rel_path = str(Path(args.train_dir) / class_dir.name / img_path.name)
                abs_path = train_dir / class_dir.name / img_path.name
                records.append({
                    "image_path": rel_path,
                    "class_name": class_name,
                    "_abs_path": abs_path,  # temporary, dropped before CSV
                })

    if not records:
        raise ValueError(f"No images found under {train_dir}")

    df = pd.DataFrame(records)
    logger.info("Found %d images in %d classes.", len(df), df["class_name"].nunique())

    # ------------------------------------------------------------------
    # SHA-256 group-aware splitting (strict-v2)
    # ------------------------------------------------------------------
    sha256_groups = {}
    duplicate_image_count = 0

    if use_sha256_dedup:
        logger.info("Computing SHA-256 hashes for all %d images...", len(df))
        for i, row in df.iterrows():
            h = sha256_file(row["_abs_path"])
            df.at[i, "sha256"] = h
            if h not in sha256_groups:
                sha256_groups[h] = []
            sha256_groups[h].append(i)

        # Count duplicates
        for h, indices in sha256_groups.items():
            if len(indices) > 1:
                duplicate_image_count += len(indices) - 1  # extra copies

        n_groups = len(sha256_groups)
        logger.info(
            "SHA-256: %d unique hashes, %d duplicate images (%d groups with >1 copy).",
            n_groups,
            duplicate_image_count,
            sum(1 for idxs in sha256_groups.values() if len(idxs) > 1),
        )

        # Build group objects
        groups = []
        for h, indices in sha256_groups.items():
            group_labels = [df.at[i, "label"] for i in indices]
            groups.append({
                "sha256": h,
                "indices": indices,
                "labels": group_labels,
                "size": len(indices),
            })

        # Split at group level
        train_sha, val_sha = _split_groups_stratified(groups, args.val_ratio, args.seed)

        # Assign images to splits based on group membership
        train_mask = df["sha256"].isin(train_sha)
        val_mask = df["sha256"].isin(val_sha)

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()

        # Verify no SHA group is split
        train_sha_set = set(train_df["sha256"])
        val_sha_set = set(val_df["sha256"])
        split_groups = train_sha_set & val_sha_set
        if split_groups:
            raise AssertionError(
                f"SHA-256 group split detected! {len(split_groups)} hashes "
                f"appear in both train and val."
            )

        # Drop temporary columns
        train_df = train_df.drop(columns=["sha256", "_abs_path"])
        val_df = val_df.drop(columns=["sha256", "_abs_path"])
        df = df.drop(columns=["sha256"])

        logger.info(
            "SHA-256 dedup: %d groups → %d train / %d val.",
            n_groups, len(train_df), len(val_df),
        )
    else:
        logger.info("SHA-256 dedup DISABLED (strict-v1 mode).")

        # Drop temporary column
        df = df.drop(columns=["_abs_path"])

        # Standard image-level stratified split
        train_df, val_df = train_test_split(
            df, test_size=args.val_ratio, random_state=args.seed,
            stratify=df["label"],
        )

    # Drop _abs_path from df
    if "_abs_path" in df.columns:
        df = df.drop(columns=["_abs_path"])

    train_df = train_df.sort_values("image_path").reset_index(drop=True)
    val_df = val_df.sort_values("image_path").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Generate class mapping (lexicographic sort of class directories)
    # ------------------------------------------------------------------
    class_to_idx, idx_to_class = generate_canonical_mapping(
        train_dir=train_dir, expected_num_classes=500,
    )

    # Recompute labels (in case any filtered images affected mapping)
    # Actually labels were computed before split, they're fine.

    # Persist class mapping alongside the split
    json.dump(class_to_idx, (output_dir / "class_to_idx.json").open("w"),
              indent=2, sort_keys=True)
    json.dump(idx_to_class, (output_dir / "idx_to_class.json").open("w"),
              indent=2, sort_keys=True)

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
        raise AssertionError(f"Train/val path overlap: {len(overlap)} images")
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
        "duplicate_grouping_enabled": use_sha256_dedup,
        "sha256_unique_groups": len(sha256_groups) if use_sha256_dedup else None,
        "sha256_duplicate_images": duplicate_image_count if use_sha256_dedup else None,
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Master split (strict-%s): %d train / %d val / %d classes",
        "v2" if use_sha256_dedup else "v1",
        len(train_df), len(val_df), df["class_name"].nunique(),
    )


if __name__ == "__main__":
    main()
