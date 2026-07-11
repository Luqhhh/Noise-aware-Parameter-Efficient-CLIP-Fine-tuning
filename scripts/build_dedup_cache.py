#!/usr/bin/env python3
"""One-command pipeline: build dedup feature cache from original cache + removal list.

Prerequisites:
    - Original feature cache at cache/preliminary/clip_vit_b32_openai/
    - outputs/dedup/removal_list.txt (in git)
    - Original training data at train/

Usage:
    python scripts/build_dedup_cache.py

Output:
    - cache/preliminary/clip_vit_b32_openai_dedup/  (filtered cache with correct fingerprints)
    - Train/test split generated via split_data.py on the resulting dedup dataset
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.cache import compute_full_fingerprint, compute_quick_fingerprint


def main():
    src_cache = ROOT / "cache/preliminary/clip_vit_b32_openai"
    dst_cache = ROOT / "cache/preliminary/clip_vit_b32_openai_dedup"
    removal_path = ROOT / "outputs/dedup/removal_list.txt"
    train_dir = ROOT / "train"

    # --- 1. Validate prerequisites ---
    if not (src_cache / "manifest.json").exists():
        print("ERROR: Original feature cache not found. Build it first:")
        print("  python scripts/cache_features.py --config configs/e0_hyper_search.yaml")
        sys.exit(1)

    if not removal_path.exists():
        print("ERROR: removal_list.txt not found. Pull the latest repo first.")
        sys.exit(1)

    if not train_dir.exists():
        print("ERROR: train/ directory not found. Place training data at train/")
        sys.exit(1)

    # --- 2. Load removal list ---
    with open(removal_path) as f:
        removed = set(line.strip() for line in f if line.strip())
    print(f"Removal list: {len(removed)} files to exclude")

    # --- 3. Load original cache ---
    image_paths = json.load(open(src_cache / "image_paths.json"))
    features = torch.load(src_cache / "features.pt", map_location="cpu", weights_only=True)
    labels = json.load(open(src_cache / "labels.json"))

    print(f"Original cache: {len(image_paths)} images")

    # --- 4. Filter ---
    keep_idx = []
    kept_paths = []
    for i, p in enumerate(image_paths):
        if p not in removed:
            keep_idx.append(i)
            kept_paths.append(p)

    filtered_features = features[keep_idx]
    filtered_labels = [labels[i] for i in keep_idx]

    print(f"Filtered: {len(kept_paths)} images kept, {len(image_paths) - len(kept_paths)} removed")

    # --- 5. Save filtered cache ---
    dst_cache.mkdir(parents=True, exist_ok=True)
    torch.save(filtered_features, dst_cache / "features.pt")
    json.dump(kept_paths, open(dst_cache / "image_paths.json", "w"), ensure_ascii=False)
    json.dump(filtered_labels, open(dst_cache / "labels.json", "w"), ensure_ascii=False)

    # Copy class mapping
    for f in ["class_to_idx.json", "idx_to_class.json"]:
        shutil.copy(src_cache / f, dst_cache / f)

    # --- 6. Build manifest with correct fingerprints ---
    manifest = json.load(open(src_cache / "manifest.json"))
    manifest["dataset_root"] = str(train_dir.resolve())
    manifest["dataset_size"] = len(kept_paths)
    manifest["dedup_source"] = "filtered_from_original"
    manifest["dedup_removed_count"] = len(removed)
    manifest["dataset_quick_fingerprint"] = compute_quick_fingerprint(train_dir)
    manifest["dataset_full_fingerprint"] = compute_full_fingerprint(train_dir)
    json.dump(manifest, open(dst_cache / "manifest.json", "w"), indent=2, ensure_ascii=False)

    print(f"Manifest updated with fingerprints from {train_dir}")

    # --- 7. Generate train/val split ---
    config_path = ROOT / "configs/d3_dedup.yaml"
    if not config_path.exists():
        print("WARNING: configs/d3_dedup.yaml not found, skipping split generation.")
        print(f"Dedup cache ready at: {dst_cache}")
        return

    print("\nGenerating train/val split...")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/split_data.py"), "--config", str(config_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print("WARNING: split_data.py failed. You may need to adjust config.")
        print(f"Dedup cache ready at: {dst_cache}")
        sys.exit(1)

    # --- 8. Summary ---
    print("=" * 60)
    print("Dedup cache ready!")
    print(f"  Cache: {dst_cache}")
    print(f"  Split: {ROOT / 'outputs/d3/splits'}")
    print(f"  Images: {len(kept_paths)}")
    print(f"\nTo train D3:")
    print(f"  python -m experiments.baseline.train --config configs/d3_dedup.yaml")


if __name__ == "__main__":
    main()
