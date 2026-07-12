#!/usr/bin/env python3
"""
SHA-256 group-aware split dedup analysis.

Checks whether SHA-256 duplicate groups (from duplicate_scan.json)
straddle the train/val boundary in the master split.

Reports:
  - Cross-boundary SHA-256 groups (content leakage)
  - Images affected
  - Per-class statistics
  - Ratio of unique SHA-256 hashes to total images

Usage:
    python3 scripts/check_split_dedup.py
    python3 scripts/check_split_dedup.py --seed 3407
    python3 scripts/check_split_dedup.py --all-seeds
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def load_duplicate_scan(scan_path: str) -> list:
    """Load duplicate groups from duplicate_scan.json."""
    with open(scan_path) as f:
        data = json.load(f)
    return data["duplicates"]  # list of {sha256, classes, paths, ...}


def load_split_paths(csv_path: str) -> set:
    """Load image_path values from a split CSV."""
    df = pd.read_csv(csv_path)
    return set(df["image_path"])


def analyze_split(seed: int,
                  split_dir: str,
                  duplicate_scan_path: str):
    """Analyze one seed's split for SHA-256 cross-boundary leakage."""
    split_dir = Path(split_dir) / f"seed{seed}"
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"

    if not train_csv.exists():
        print(f"Split not found: {train_csv}")
        return None

    # Load split paths
    train_paths = load_split_paths(str(train_csv))
    val_paths = load_split_paths(str(val_csv))
    all_paths = train_paths | val_paths

    print(f"\n{'='*60}")
    print(f"Seed {seed} Split Dedup Analysis")
    print(f"{'='*60}")
    print(f"Train: {len(train_paths):,} images")
    print(f"Val:   {len(val_paths):,} images")
    print(f"Total: {len(all_paths):,} images")

    # Load duplicate groups
    dup_groups = load_duplicate_scan(duplicate_scan_path)
    print(f"Duplicate groups (from scan): {len(dup_groups):,}")

    # Check each group for cross-boundary leakage
    cross_boundary_groups = []
    cross_boundary_images = 0
    total_dup_images_in_split = 0

    for group in dup_groups:
        paths = set(group["paths"])
        in_split = paths & all_paths
        if not in_split:
            continue  # group not in this split

        in_train = paths & train_paths
        in_val = paths & val_paths
        total_dup_images_in_split += len(in_split)

        if in_train and in_val:
            # Leakage! This SHA-256 group is split across train/val
            cross_boundary_groups.append({
                "sha256": group["sha256"][:16],
                "classes": group["classes"],
                "in_train": sorted(in_train),
                "in_val": sorted(in_val),
                "num_copies": group["num_copies"],
            })
            cross_boundary_images += len(in_split)

    # Report
    print(f"\n--- Leakage Report ---")
    print(f"Duplicate images in split: {total_dup_images_in_split:,}")
    print(f"Cross-boundary SHA-256 groups: {len(cross_boundary_groups)}")
    print(f"Cross-boundary images affected: {cross_boundary_images}")

    if cross_boundary_groups:
        pct = cross_boundary_images / len(val_paths) * 100
        print(f"\n⚠ CONTENT LEAKAGE DETECTED: {cross_boundary_images} images "
              f"({pct:.2f}% of val) share SHA-256 with training data!")
        print(f"\nWorst offenders (first 5):")
        for g in cross_boundary_groups[:5]:
            print(f"  SHA-256 {g['sha256']}...: {len(g['in_train'])} train, "
                  f"{len(g['in_val'])} val across classes {g['classes']}")
    else:
        print("\n✅ No SHA-256 cross-boundary leakage detected.")

    # Per-class statistics
    print(f"\n--- Per-Class Statistics ---")
    df_train = pd.read_csv(train_csv)
    class_counts = df_train["class_name"].value_counts()
    print(f"Train per-class: min={class_counts.min()}, "
          f"median={class_counts.median():.0f}, max={class_counts.max()}")

    df_val = pd.read_csv(val_csv)
    val_class_counts = df_val["class_name"].value_counts()
    print(f"Val per-class:   min={val_class_counts.min()}, "
          f"median={val_class_counts.median():.0f}, max={val_class_counts.max()}")

    # SHA-256 uniqueness
    print(f"\n--- SHA-256 Uniqueness ---")
    total_images = len(all_paths)
    # Unique SHA-256 = total images - (duplicate copies)
    # Each group has num_copies; if all copies are in split, redundant = num_copies - 1
    unique_sha = total_images - (total_dup_images_in_split - len(
        [g for g in dup_groups if set(g["paths"]) & all_paths]
    ))
    print(f"Total images in split:    {total_images:,}")
    print(f"Images in dup groups:     {total_dup_images_in_split:,}")
    print(f"Duplicate groups present: {len([g for g in dup_groups if set(g['paths']) & all_paths])}")

    result = {
        "seed": seed,
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "cross_boundary_groups": len(cross_boundary_groups),
        "cross_boundary_images": cross_boundary_images,
        "leakage_pct_of_val": round(cross_boundary_images / len(val_paths) * 100, 4),
        "has_leakage": len(cross_boundary_groups) > 0,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="SHA-256 group-aware split dedup analysis."
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed to analyze (default: 42)")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Analyze all available seeds")
    parser.add_argument("--split-root", default="outputs/master_splits")
    parser.add_argument("--duplicate-scan",
                        default="outputs/duplicate_scan.json")
    args = parser.parse_args()

    results = []

    if args.all_seeds:
        # Find all seed dirs
        split_root = Path(args.split_root)
        seeds = []
        for d in sorted(split_root.iterdir()):
            if d.is_dir() and d.name.startswith("seed"):
                try:
                    seeds.append(int(d.name.replace("seed", "")))
                except ValueError:
                    pass
        print(f"Found seeds: {seeds}")
    else:
        seeds = [args.seed]

    for seed in seeds:
        result = analyze_split(seed, args.split_root, args.duplicate_scan)
        if result:
            results.append(result)

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Multi-Seed Summary")
        print(f"{'='*60}")
        print(f"{'Seed':<8} {'Train':>8} {'Val':>8} {'X-Boundary':>12} {'Images':>8} {'%Val':>8}")
        print("-" * 60)
        for r in results:
            print(f"{r['seed']:<8} {r['train_count']:>8,} {r['val_count']:>8,} "
                  f"{r['cross_boundary_groups']:>12} {r['cross_boundary_images']:>8} "
                  f"{r['leakage_pct_of_val']:>7.2f}%")

        any_leakage = any(r["has_leakage"] for r in results)
        if any_leakage:
            print("\n⚠ CONTENT LEAKAGE DETECTED in one or more seeds!")
            print("Recommendation: rebuild splits with SHA-256 dedup enabled "
                  "(build_master_split.py without --no-sha256-dedup)")
        else:
            print("\n✅ All seeds clean — no SHA-256 cross-boundary leakage.")


if __name__ == "__main__":
    main()
