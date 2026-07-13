#!/usr/bin/env python3
"""
Apply D3 train-only dedup cleaning to a multi-seed master split.

Reads the removal list from seed 42 (content-based, seed-independent),
filters the master train.csv for a target seed, and writes a cleaned
train.csv pointing to train_dedup/ paths.

Usage:
    python3 scripts/apply_d3_cleaning.py --seed 3407
    python3 scripts/apply_d3_cleaning.py --seed 2026
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Apply D3 dedup cleaning to a multi-seed master split."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--removal-list",
                        default="outputs/ref/seed42/removal_list_train_only.txt")
    parser.add_argument("--master-split-root",
                        default="outputs/master_splits")
    parser.add_argument("--output-root",
                        default="outputs/ref")
    args = parser.parse_args()

    seed = args.seed
    removal_path = Path(args.removal_list)
    master_dir = Path(args.master_split_root) / f"seed{seed}"
    output_dir = Path(args.output_root) / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load removal list (relative paths like "0001/xxx.jpg")
    with open(removal_path) as f:
        removed = set(line.strip() for line in f if line.strip())
    print(f"Removal list: {len(removed)} images to exclude")

    # Load master train CSV
    train_csv = master_dir / "train.csv"
    df = pd.read_csv(train_csv)
    print(f"Master train: {len(df)} images")

    # Build a key for each row: relative path without train/ prefix
    # master CSV has paths like "train/0000/xxx.jpg"
    def make_key(image_path: str) -> str:
        p = image_path
        if p.startswith("train/"):
            p = p[len("train/"):]
        elif p.startswith("train_dedup/"):
            p = p[len("train_dedup/"):]
        return p

    df["_key"] = df["image_path"].apply(make_key)

    # Filter
    clean_df = df[~df["_key"].isin(removed)].copy()
    removed_count = len(df) - len(clean_df)
    print(f"Removed: {removed_count} images")
    print(f"Clean train: {len(clean_df)} images")

    # Change paths from train/ to train_dedup/
    clean_df["image_path"] = clean_df["image_path"].str.replace(
        "^train/", "train_dedup/", regex=True
    )

    # Drop temporary key
    clean_df = clean_df.drop(columns=["_key"])

    # Write cleaned train.csv
    clean_csv = output_dir / "train.csv"
    clean_df.to_csv(clean_csv, index=False)
    print(f"Wrote: {clean_csv}")

    # Copy val.csv from master (unchanged)
    val_csv = master_dir / "val.csv"
    val_df = pd.read_csv(val_csv)
    val_out = output_dir / "val.csv"
    val_df.to_csv(val_out, index=False)
    print(f"Wrote: {val_out} ({len(val_df)} images)")

    # Copy class mapping
    import shutil
    for fname in ["class_to_idx.json", "idx_to_class.json"]:
        shutil.copy(master_dir / fname, output_dir / fname)

    # Write cleaning report
    import json
    report = {
        "master_train_count": len(df),
        "clean_train_count": len(clean_df),
        "removed_count": removed_count,
        "removed_ratio": round(removed_count / len(df), 4),
        "val_untouched": True,
        "val_count": len(val_df),
        "removal_list_source": str(removal_path),
        "note": f"train-only cleaning applied to seed {seed}; "
                f"paths point to train_dedup/; "
                f"removal list from seed 42 (content-based, seed-independent)",
    }
    report_path = output_dir / "cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
