"""Datasets for cached-head training, online PEFT, and fail-closed inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from aegis_clip.features import FrozenFeatureStore, canonical_sample_path


REQUIRED_SPLIT_COLUMNS = {"image_path", "label"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TrustBundle:
    """Sparse OOF trust and correction metadata indexed by canonical sample path."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        required = {
            "paths",
            "clean_probability",
            "pseudo_label",
            "pseudo_confidence",
            "correction_alpha",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Trust bundle missing keys: {sorted(missing)}")
        self.metadata = dict(payload.get("metadata", {}))
        self.paths = [canonical_sample_path(path) for path in payload["paths"]]
        self.path_to_index = {path: index for index, path in enumerate(self.paths)}
        if len(self.path_to_index) != len(self.paths):
            raise ValueError("Trust bundle contains duplicate canonical paths")
        size = len(self.paths)
        self.clean_probability = _checked_vector(
            payload["clean_probability"], size, torch.float32, "clean_probability"
        ).clamp(0.0, 1.0)
        self.pseudo_label = _checked_vector(
            payload["pseudo_label"], size, torch.long, "pseudo_label"
        )
        self.pseudo_confidence = _checked_vector(
            payload["pseudo_confidence"], size, torch.float32, "pseudo_confidence"
        ).clamp(0.0, 1.0)
        self.correction_alpha = _checked_vector(
            payload["correction_alpha"], size, torch.float32, "correction_alpha"
        ).clamp(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.paths)

    def values_for(self, path: str | Path, noisy_label: int) -> dict[str, torch.Tensor]:
        key = canonical_sample_path(path)
        if key not in self.path_to_index:
            raise KeyError(f"Sample is missing from trust bundle: {key}")
        index = self.path_to_index[key]
        pseudo = int(self.pseudo_label[index])
        if pseudo < 0:
            pseudo = int(noisy_label)
        return {
            "clean_probability": self.clean_probability[index],
            "pseudo_label": torch.tensor(pseudo, dtype=torch.long),
            "pseudo_confidence": self.pseudo_confidence[index],
            "correction_alpha": self.correction_alpha[index],
        }

    def verify_coverage(self, paths: list[str]) -> None:
        missing = [
            canonical_sample_path(path)
            for path in paths
            if canonical_sample_path(path) not in self.path_to_index
        ]
        if missing:
            raise ValueError(
                f"Trust bundle misses {len(missing)} split samples; first={missing[0]}"
            )


class CachedFeatureDataset(Dataset):
    """Fast training dataset backed by deterministic frozen CLIP features."""

    def __init__(
        self,
        split_csv: str | Path,
        feature_store: FrozenFeatureStore,
        trust_bundle: TrustBundle | None = None,
    ) -> None:
        self.frame = _load_split(split_csv)
        self.paths = self.frame["image_path"].astype(str).tolist()
        self.labels = self.frame["label"].astype(int).tolist()
        self.feature_store = feature_store
        self.trust_bundle = trust_bundle
        feature_store.verify_coverage(self.paths)
        if trust_bundle is not None:
            trust_bundle.verify_coverage(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        label = self.labels[index]
        item: dict[str, torch.Tensor | str] = {
            "features": self.feature_store.get(path),
            "reference_features": self.feature_store.get(path),
            "index": torch.tensor(index, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "path": canonical_sample_path(path),
        }
        item.update(_trust_values(self.trust_bundle, path, label))
        return item


class OnlineImageDataset(Dataset):
    """Online image dataset for PEFT, paired with cached parent features."""

    def __init__(
        self,
        split_csv: str | Path,
        image_root: str | Path,
        transform: Callable,
        feature_store: FrozenFeatureStore,
        trust_bundle: TrustBundle | None = None,
    ) -> None:
        self.frame = _load_split(split_csv)
        self.paths = self.frame["image_path"].astype(str).tolist()
        self.labels = self.frame["label"].astype(int).tolist()
        self.image_root = Path(image_root)
        self.transform = transform
        self.feature_store = feature_store
        self.trust_bundle = trust_bundle
        feature_store.verify_coverage(self.paths)
        if trust_bundle is not None:
            trust_bundle.verify_coverage(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        relative_path = self.paths[index]
        label = self.labels[index]
        absolute_path = resolve_image_path(self.image_root, relative_path)
        try:
            with Image.open(absolute_path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode training image: {absolute_path}") from exc
        item: dict[str, torch.Tensor | str] = {
            "images": tensor,
            "reference_features": self.feature_store.get(relative_path),
            "index": torch.tensor(index, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "path": canonical_sample_path(relative_path),
        }
        item.update(_trust_values(self.trust_bundle, relative_path, label))
        return item


class TestImageDataset(Dataset):
    """Deterministically enumerate every test image and preserve corrupt entries."""

    def __init__(self, image_root: str | Path, transform: Callable) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        self.paths = sorted(
            path
            for path in self.image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No test images found under {self.image_root}")
        names = [path.name for path in self.paths]
        if len(names) != len(set(names)):
            raise ValueError("Test images must have unique basenames for submission")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | bool]:
        path = self.paths[index]
        corrupt = False
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception:
            corrupt = True
            tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
        return {"images": tensor, "name": path.name, "corrupt": corrupt}


def resolve_image_path(root: Path, value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == root.name:
        path = Path(*parts[1:])
    # Map train_dedup → train so that files referenced under the dedup
    # prefix resolve to the physical train/ directory (critical when
    # child splits mirror a parent trained with d3 strict dedup).
    if parts and parts[0] == "train_dedup":
        path = Path("train") / Path(*parts[1:])
    return root / path


def load_class_mapping(path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        class_to_idx = {str(key): int(value) for key, value in json.load(handle).items()}
    idx_to_class = {value: key for key, value in class_to_idx.items()}
    if len(class_to_idx) != len(idx_to_class):
        raise ValueError("Class mapping is not one-to-one")
    if set(idx_to_class) != set(range(len(class_to_idx))):
        raise ValueError("Class mapping indices must be contiguous from zero")
    malformed = [
        name for name in class_to_idx if len(name) != 4 or not name.isdigit()
    ]
    if malformed:
        raise ValueError(
            f"Competition class names must be four digits; first={malformed[0]!r}"
        )
    return class_to_idx, idx_to_class


def _load_split(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_SPLIT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Split CSV missing columns: {sorted(missing)}")
    if frame["image_path"].duplicated().any():
        raise ValueError(f"Split CSV contains duplicate paths: {path}")
    return frame.reset_index(drop=True)


def _trust_values(
    trust_bundle: TrustBundle | None, path: str, label: int
) -> dict[str, torch.Tensor]:
    if trust_bundle is None:
        return {
            "clean_probability": torch.tensor(1.0),
            "pseudo_label": torch.tensor(label, dtype=torch.long),
            "pseudo_confidence": torch.tensor(0.0),
            "correction_alpha": torch.tensor(0.0),
        }
    return trust_bundle.values_for(path, label)


def _checked_vector(
    value: torch.Tensor, size: int, dtype: torch.dtype, name: str
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype).flatten()
    if tensor.numel() != size:
        raise ValueError(f"{name} has {tensor.numel()} values, expected {size}")
    return tensor
