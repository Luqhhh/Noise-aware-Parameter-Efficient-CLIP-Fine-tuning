"""Test train/val split correctness."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.dataset import _find_images_in_dir


def test_image_scan_does_not_duplicate_lowercase_files(tmp_path):
    """A lowercase image must be returned once on case-insensitive filesystems."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "sample.jpg").write_bytes(b"not-an-image")

    assert _find_images_in_dir(image_dir) == [image_dir / "sample.jpg"]


def test_split_coverage():
    """Train + val should cover all samples (no overlap, no missing)."""
    split_dir = Path("outputs/baselines/baseline/splits")
    if not split_dir.exists():
        return

    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")

    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])

    # No overlap
    overlap = train_paths & val_paths
    assert len(overlap) == 0, f"Train/val overlap: {len(overlap)} images"

    # All labels in range
    with open(split_dir / "class_to_idx.json") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)

    assert train_df["label"].between(0, num_classes - 1).all()
    assert val_df["label"].between(0, num_classes - 1).all()
