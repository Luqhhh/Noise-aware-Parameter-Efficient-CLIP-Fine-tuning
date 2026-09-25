"""Datasets for cached-head training, online PEFT, and fail-closed inference."""

from __future__ import annotations

import json
import hashlib
import io
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

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
        self.pseudo_soft = None
        if "pseudo_soft" in payload:
            soft = torch.as_tensor(payload["pseudo_soft"])
            if soft.ndim != 2 or soft.shape[0] != size:
                raise ValueError("pseudo_soft must have shape [N, num_classes]")
            if not torch.isfinite(soft).all() or (soft < 0).any():
                raise ValueError("pseudo_soft must be finite and non-negative")
            self.pseudo_soft = soft.to(dtype=torch.float32)

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
        result = {
            "clean_probability": self.clean_probability[index],
            "pseudo_label": torch.tensor(pseudo, dtype=torch.long),
            "pseudo_confidence": self.pseudo_confidence[index],
            "correction_alpha": self.correction_alpha[index],
        }
        if self.pseudo_soft is not None:
            result["pseudo_soft"] = self.pseudo_soft[index]
        return result

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
            source = absolute_path
            if "file_sha256" in self.frame.columns:
                ImageFile.LOAD_TRUNCATED_IMAGES = False
                raw = absolute_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != self.frame.iloc[index]["file_sha256"]:
                    raise ValueError("Training image changed after stage audit")
                source = io.BytesIO(raw)
            with Image.open(source) as image:
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


class GeometryAwareImageDataset(Dataset):
    """Online image dataset that also returns the original RGB and its geometry.

    NEW01 needs the *original* image so the local view can be re-extracted at the
    native resolution instead of being upscaled from the already-downscaled model
    input.  The global view is built by exactly the same recipe as
    :class:`OnlineImageDataset` (RandomResizedCrop -> RandomHorizontalFlip ->
    CLIP tensor conversion and normalization), but the sampled crop ``(i, j, h, w)``
    and the flip flag are recorded so the local attention box can be mapped back
    to original-image coordinates.

    ``geometry`` is a float32 vector ``(H0, W0, i, j, h, w, flip, S)``:
    the original height/width, the RRC top/left/crop-height/crop-width in original
    pixels, the flip flag, and the model input size ``S``.
    """

    def __init__(
        self,
        split_csv: str | Path,
        image_root: str | Path,
        preprocess: Callable,
        feature_store: FrozenFeatureStore,
        trust_bundle: TrustBundle | None = None,
    ) -> None:
        self.frame = _load_split(split_csv)
        self.paths = self.frame["image_path"].astype(str).tolist()
        self.labels = self.frame["label"].astype(int).tolist()
        self.image_root = Path(image_root)
        self.transform = preprocess
        self.feature_store = feature_store
        self.trust_bundle = trust_bundle
        (
            self.rrc_size,
            self.rrc_scale,
            self.rrc_ratio,
            self.rrc_interpolation,
            self.flip_probability,
            self.tail_transform,
        ) = split_geometry_augmentation(preprocess)
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
            source = absolute_path
            if "file_sha256" in self.frame.columns:
                ImageFile.LOAD_TRUNCATED_IMAGES = False
                raw = absolute_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != self.frame.iloc[index]["file_sha256"]:
                    raise ValueError("Training image changed after stage audit")
                source = io.BytesIO(raw)
            with Image.open(source) as image:
                image = image.convert("RGB")
                width, height = image.size
                top, left, crop_height, crop_width = _sample_rrc_params(
                    height,
                    width,
                    self.rrc_scale,
                    self.rrc_ratio,
                )
                crop = _resized_crop(
                    image,
                    top,
                    left,
                    crop_height,
                    crop_width,
                    self.rrc_size,
                    self.rrc_interpolation,
                )
                flip = bool(torch.rand(()).item() < self.flip_probability)
                if flip:
                    crop = _hflip(crop)
                tensor = self.tail_transform(crop)
                original = _pil_to_tensor(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode training image: {absolute_path}") from exc
        item: dict[str, torch.Tensor | str] = {
            "images": tensor,
            "original": original,
            "geometry": torch.tensor(
                [
                    float(height),
                    float(width),
                    float(top),
                    float(left),
                    float(crop_height),
                    float(crop_width),
                    1.0 if flip else 0.0,
                    float(self.rrc_size[0]),
                ],
                dtype=torch.float32,
            ),
            "reference_features": self.feature_store.get(relative_path),
            "index": torch.tensor(index, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "path": canonical_sample_path(relative_path),
        }
        item.update(_trust_values(self.trust_bundle, relative_path, label))
        return item


def split_geometry_augmentation(preprocess: Callable) -> tuple:
    """Split a geometry-aware training transform into (params, tail)."""
    from torchvision.transforms import Compose, RandomHorizontalFlip, RandomResizedCrop

    transforms = list(getattr(preprocess, "transforms", []))
    rrc = next((t for t in transforms if isinstance(t, RandomResizedCrop)), None)
    if rrc is None:
        raise ValueError(
            "Geometry-aware training requires a RandomResizedCrop in the transform"
        )
    flip = next((t for t in transforms if isinstance(t, RandomHorizontalFlip)), None)
    tail = [t for t in transforms if t is not rrc and t is not flip]
    if not tail:
        raise ValueError("Geometry-aware training requires tensor/normalize transforms")
    size = rrc.size if isinstance(rrc.size, (tuple, list)) else (rrc.size, rrc.size)
    return (
        (int(size[0]), int(size[1])),
        (float(rrc.scale[0]), float(rrc.scale[1])),
        (float(rrc.ratio[0]), float(rrc.ratio[1])),
        rrc.interpolation,
        float(flip.p) if flip is not None else 0.0,
        Compose(tail),
    )


def _sample_rrc_params(
    height: int,
    width: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
) -> tuple[int, int, int, int]:
    """Replicate torchvision RandomResizedCrop.get_params with the torch RNG."""
    area = height * width
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * float(torch.empty(1).uniform_(scale[0], scale[1]).item())
        aspect_ratio = math.exp(
            float(torch.empty(1).uniform_(log_ratio[0], log_ratio[1]).item())
        )
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = int(torch.randint(0, height - crop_height + 1, size=(1,)).item())
            left = int(torch.randint(0, width - crop_width + 1, size=(1,)).item())
            return top, left, crop_height, crop_width
    in_ratio = float(width) / float(height)
    if in_ratio < ratio[0]:
        crop_width = width
        crop_height = int(round(crop_width / ratio[0]))
    elif in_ratio > ratio[1]:
        crop_height = height
        crop_width = int(round(crop_height * ratio[1]))
    else:
        crop_width = width
        crop_height = height
    return (height - crop_height) // 2, (width - crop_width) // 2, crop_height, crop_width


def _resized_crop(image, top, left, height, width, size, interpolation):
    from torchvision.transforms import functional as TF

    return TF.resized_crop(image, top, left, height, width, list(size), interpolation)


def _hflip(image):
    from torchvision.transforms import functional as TF

    return TF.hflip(image)


def _pil_to_tensor(image) -> torch.Tensor:
    from torchvision.transforms import functional as TF

    return TF.pil_to_tensor(image)


def collate_geometry_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a geometry batch, zero-padding variable-size originals.

    Only the current batch's originals are kept; the true sizes travel alongside
    so the ROI crop is bounded by the real image extent, not the padding.
    """
    originals = [item["original"] for item in batch]
    max_height = max(int(o.shape[1]) for o in originals)
    max_width = max(int(o.shape[2]) for o in originals)
    padded = torch.zeros(
        len(batch), 3, max_height, max_width, dtype=originals[0].dtype
    )
    sizes = torch.zeros(len(batch), 2, dtype=torch.long)
    for row, original in enumerate(originals):
        height, width = int(original.shape[1]), int(original.shape[2])
        padded[row, :, :height, :width] = original
        sizes[row, 0] = height
        sizes[row, 1] = width
    collated = default_collate(
        [{key: value for key, value in item.items() if key != "original"} for item in batch]
    )
    collated["originals"] = padded
    collated["original_sizes"] = sizes
    return collated


class TestImageDataset(Dataset):
    """Deterministically enumerate every test image and preserve corrupt entries."""

    def __init__(self, image_root: str | Path, transform: Callable, source_hashes=None) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        self.source_hashes = source_hashes
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
            source = path
            if self.source_hashes is not None:
                ImageFile.LOAD_TRUNCATED_IMAGES = False
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != self.source_hashes[path.name]:
                    raise ValueError("Test image changed after stage audit")
                source = io.BytesIO(raw)
            with Image.open(source) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception:
            if self.source_hashes is not None:
                raise RuntimeError(f"Strict test decode/integrity failure: {path}")
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
