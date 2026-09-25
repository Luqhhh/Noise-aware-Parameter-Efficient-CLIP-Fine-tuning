"""Wiring tests for the NEW01 geometry-aware training dataset."""

import csv
import hashlib
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)

from aegis_clip.data import (
    GeometryAwareImageDataset,
    collate_geometry_batch,
    split_geometry_augmentation,
)


class _FakeFeatureStore:
    def __init__(self, dim=4):
        self.dim = dim

    def get(self, path):
        return torch.zeros(self.dim)

    def verify_coverage(self, paths):
        return None


def _fixture(tmp_path: Path, size=(40, 30)):
    image_dir = tmp_path / "train" / "0000"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "a.jpg"
    Image.new("RGB", size, (120, 200, 60)).save(image_path)
    csv_path = tmp_path / "train_dev.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label"])
        writer.writerow(["train/0000/a.jpg", 0])
    return csv_path, tmp_path / "train", image_path


def _transform(size=32):
    return Compose(
        [
            RandomResizedCrop(
                size, scale=(0.7, 1.0), ratio=(0.85, 1.15),
                interpolation=InterpolationMode.BICUBIC,
            ),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def test_split_geometry_augmentation_extracts_params():
    size, scale, ratio, interpolation, probability, tail = split_geometry_augmentation(
        _transform(32)
    )
    assert size == (32, 32)
    assert scale == (0.7, 1.0)
    assert ratio == (0.85, 1.15)
    assert interpolation == InterpolationMode.BICUBIC
    assert probability == 0.5
    assert len(tail.transforms) == 2


def test_geometry_dataset_records_true_rrc_box_and_original(tmp_path):
    csv_path, root, source = _fixture(tmp_path, size=(40, 30))
    dataset = GeometryAwareImageDataset(
        csv_path, root, _transform(32), _FakeFeatureStore()
    )
    torch.manual_seed(7)
    item = dataset[0]
    geometry = item["geometry"]
    height0, width0, top, left, crop_h, crop_w, flip, size = geometry.tolist()
    assert (height0, width0) == (30.0, 40.0)
    assert size == 32.0
    assert 0 <= top <= height0 - crop_h
    assert 0 <= left <= width0 - crop_w
    assert crop_h > 0 and crop_w > 0
    assert flip in (0.0, 1.0)
    assert item["images"].shape == (3, 32, 32)
    # Original is the untouched native-resolution RGB image.
    assert item["original"].shape == (3, 30, 40)
    assert item["original"].dtype == torch.uint8
    assert item["original"].shape[1:] == (height0, width0)


def test_geometry_dataset_is_deterministic_given_the_rng(tmp_path):
    csv_path, root, _ = _fixture(tmp_path)
    dataset = GeometryAwareImageDataset(
        csv_path, root, _transform(32), _FakeFeatureStore()
    )
    torch.manual_seed(123)
    first = dataset[0]
    torch.manual_seed(123)
    second = dataset[0]
    assert torch.equal(first["geometry"], second["geometry"])
    assert torch.equal(first["images"], second["images"])


def test_collate_geometry_batch_pads_and_reports_sizes(tmp_path):
    csv_path = tmp_path / "train_dev.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label"])
        writer.writerow(["train/0000/a.jpg", 0])
        writer.writerow(["train/0000/b.jpg", 1])
    directory = tmp_path / "train" / "0000"
    directory.mkdir(parents=True)
    Image.new("RGB", (40, 30), (10, 20, 30)).save(directory / "a.jpg")
    Image.new("RGB", (20, 50), (30, 40, 50)).save(directory / "b.jpg")
    dataset = GeometryAwareImageDataset(
        csv_path, tmp_path / "train", _transform(32), _FakeFeatureStore()
    )
    batch = collate_geometry_batch([dataset[0], dataset[1]])
    assert batch["originals"].shape == (2, 3, 50, 40)
    assert batch["original_sizes"].tolist() == [[30, 40], [50, 20]]
    assert batch["geometry"].shape == (2, 8)
    assert batch["images"].shape == (2, 3, 32, 32)
    # Padding is zero and never overwrites the real extent.
    assert float(batch["originals"][0, :, 30:, :].abs().sum()) == 0.0
