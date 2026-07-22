"""Frozen CLIP feature cache access with path and lineage validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from aegis_clip.runtime import sha256_lines


def canonical_sample_path(path: str | Path) -> str:
    """Canonicalise a training sample independently of storage directory."""
    value = str(path).replace("\\", "/").strip()
    while value.startswith("./"):
        value = value[2:]
    for marker in ("/train_dedup/", "/train/"):
        if marker in value:
            return value.split(marker, 1)[1].lstrip("/")

    for prefix in ("train_dedup/", "train/"):
        if value.startswith(prefix):
            return value[len(prefix):].lstrip("/")
    return value.lstrip("/")


class FrozenFeatureStore:
    """In-memory index over a deterministic frozen CLIP feature cache."""

    def __init__(
        self,
        tensor_path: str | Path,
        paths_path: str | Path,
        manifest_path: str | Path | None = None,
        expected_dim: int = 512,
        expected_backbone: str = "ViT-B/32",
        expected_pretrained: str = "openai",
    ) -> None:
        self.tensor_path = Path(tensor_path)
        self.paths_path = Path(paths_path)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.features = torch.load(
            self.tensor_path, map_location="cpu", weights_only=True
        ).float()
        with self.paths_path.open("r", encoding="utf-8") as handle:
            raw_paths = json.load(handle)
        if self.features.ndim != 2 or self.features.shape[1] != expected_dim:
            raise ValueError(
                f"Feature cache shape must be [N,{expected_dim}], got "
                f"{tuple(self.features.shape)}"
            )
        if len(raw_paths) != self.features.shape[0]:
            raise ValueError("Feature tensor and path index have different lengths")

        self.paths = [canonical_sample_path(path) for path in raw_paths]
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("Feature cache contains duplicate canonical paths")
        self.path_to_index = {path: index for index, path in enumerate(self.paths)}
        self.features = F.normalize(self.features, dim=1)

        self.manifest: dict = {}
        if self.manifest_path:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                self.manifest = json.load(handle)
            if self.manifest.get("backbone") != expected_backbone:
                raise ValueError(
                    f"Expected cache backbone {expected_backbone}, got "
                    f"{self.manifest.get('backbone')}"
                )
            if self.manifest.get("pretrained") != expected_pretrained:
                raise ValueError(
                    f"Expected cache pretrained source {expected_pretrained}, got "
                    f"{self.manifest.get('pretrained')}"
                )
            if not self.manifest.get("normalized", False):
                raise ValueError("Feature cache manifest must declare normalized=true")
            if int(self.manifest.get("dataset_size", -1)) != len(self.paths):
                raise ValueError("Feature cache manifest dataset_size mismatch")
            if self.manifest.get("external_data") is not False:
                raise ValueError("Feature cache must declare external_data=false")
            if self.manifest.get("test_data_used") is not False:
                raise ValueError("Feature cache must declare test_data_used=false")
            if self.manifest.get("path_index_sha256") != sha256_lines(raw_paths):
                raise ValueError("Feature cache path index hash mismatch")

    def __len__(self) -> int:
        return len(self.paths)

    def index_of(self, path: str | Path) -> int:
        key = canonical_sample_path(path)
        try:
            return self.path_to_index[key]
        except KeyError as exc:
            raise KeyError(f"Sample is missing from feature cache: {key}") from exc

    def get(self, path: str | Path) -> torch.Tensor:
        return self.features[self.index_of(path)]

    def get_many(self, paths: Iterable[str | Path]) -> torch.Tensor:
        indices = [self.index_of(path) for path in paths]
        return self.features[indices]

    def verify_coverage(self, paths: Iterable[str | Path]) -> None:
        missing = [
            canonical_sample_path(path)
            for path in paths
            if canonical_sample_path(path) not in self.path_to_index
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Feature cache misses {len(missing)} requested samples: {preview}"
            )
