#!/usr/bin/env python3
"""Build a tiny synthetic stage dataset for pipeline dry-runs.

Generates a long-tailed ``num_classes``-class training tree plus a flat test
directory of unique image names.  Images are deterministic noise JPEGs, so the
result exercises the full split/cache/train/infer/submission path (including
the 1500-class label range) without touching official data.

Usage:
    python3 scripts/build_synthetic_stage.py \
        --output-dir /tmp/dryrun_repechage \
        --num-classes 1500 --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def longtail_counts(
    num_classes: int,
    *,
    minimum: int,
    maximum: int,
    alpha: float,
) -> list[int]:
    """Deterministic Zipf-like counts: head classes get ``maximum`` images."""
    counts = []
    for index in range(num_classes):
        value = float(maximum) * (float(index + 1) ** (-alpha))
        counts.append(max(int(minimum), int(round(value))))
    return counts


def _noise_image(seed: int, size: int) -> Image.Image:
    generator = np.random.default_rng(seed)
    pixels = generator.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classes", type=int, default=1500)
    parser.add_argument("--train-min-per-class", type=int, default=2)
    parser.add_argument("--train-max-per-class", type=int, default=30)
    parser.add_argument("--train-alpha", type=float, default=1.1)
    parser.add_argument("--test-per-class", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_classes < 3:
        raise ValueError("num_classes must be at least 3")
    if args.train_min_per_class < 1 or args.train_max_per_class < args.train_min_per_class:
        raise ValueError("invalid train per-class bounds")
    if args.test_per_class < 1:
        raise ValueError("test-per-class must be positive")

    root = Path(args.output_dir)
    train_root = root / "train"
    test_root = root / "test"
    counts = longtail_counts(
        args.num_classes,
        minimum=args.train_min_per_class,
        maximum=args.train_max_per_class,
        alpha=args.train_alpha,
    )
    seed_counter = args.seed
    for class_index, count in enumerate(counts):
        directory = train_root / f"{class_index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        for sample in range(count):
            seed_counter += 1
            image = _noise_image(seed_counter, args.image_size)
            image.save(directory / f"{class_index:04d}_{sample:05d}.jpg", quality=90)

    test_root.mkdir(parents=True, exist_ok=True)
    for class_index in range(args.num_classes):
        for sample in range(args.test_per_class):
            seed_counter += 1
            image = _noise_image(seed_counter, args.image_size)
            image.save(
                test_root / f"test_{class_index:04d}_{sample:05d}.jpg", quality=90
            )

    manifest = {
        "format_version": 1,
        "num_classes": args.num_classes,
        "train_samples": sum(counts),
        "test_samples": args.num_classes * args.test_per_class,
        "class_counts": counts,
        "seed": args.seed,
        "image_size": args.image_size,
        "train_root": str(train_root.resolve()),
        "test_root": str(test_root.resolve()),
    }
    (root / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
