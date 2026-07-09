#!/usr/bin/env python3
"""Generate a tiny dataset for smoke testing.

Creates 5 classes with 4 images each (train) + 3 test images.
Images are solid-color PNGs of varying sizes.
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="data/tiny/train")
    parser.add_argument("--test_dir", default="data/tiny/test")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--images_per_class", type=int, default=4)
    parser.add_argument("--num_test", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    # Clean and recreate
    if train_dir.exists():
        import shutil

        shutil.rmtree(train_dir)
    if test_dir.exists():
        import shutil

        shutil.rmtree(test_dir)

    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    # Create class dirs with padded names (like real competition format)
    for c in range(args.num_classes):
        class_name = f"{c:04d}"
        class_dir = train_dir / class_name
        class_dir.mkdir()

        for i in range(args.images_per_class):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            size = random.choice([224, 256, 300, 400])
            img = Image.new("RGB", (size, size), color)
            img_path = class_dir / f"train_{class_name}_{i:02d}.jpg"
            img.save(img_path)

    # Create test images
    for i in range(args.num_test):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        size = random.choice([224, 256, 300, 400])
        img = Image.new("RGB", (size, size), color)
        img_path = test_dir / f"test_{i:05d}.jpg"
        img.save(img_path)

    print(f"Created tiny dataset:")
    print(
        f"  Train: {args.num_classes} classes x {args.images_per_class} images = {args.num_classes * args.images_per_class} images"
    )
    print(f"  Test:  {args.num_test} images")
    print(f"  Train dir: {train_dir.resolve()}")
    print(f"  Test dir:  {test_dir.resolve()}")


if __name__ == "__main__":
    main()
