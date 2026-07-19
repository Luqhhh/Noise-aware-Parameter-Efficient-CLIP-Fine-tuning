"""Fixed OOF soft targets and continuous confidence weights."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


def stable_image_key(value: str) -> str:
    """Return a machine-independent class/file key for an image path."""
    parts = str(value).replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Image path must contain class and filename: {value!r}")
    return "/".join(parts[-2:])


class OOFSoftTargetProvider:
    """Look up fixed OOF logits and continuous weights by image path.

    The OOF tensor stores rows by ``sample_id`` while the training loader
    yields paths.  ``sample_quality.csv`` provides the stable path-to-id join
    and the confidence-derived weight used for the hard-label term.
    """

    def __init__(
        self,
        logits_path: str,
        quality_path: str,
        min_weight: float = 0.6,
        max_weight: float = 1.0,
        missing_policy: str = "error",
    ) -> None:
        if not 0.0 <= min_weight <= max_weight <= 1.0:
            raise ValueError(
                f"Invalid OOF weight range: {min_weight}, {max_weight}"
            )
        if missing_policy not in ("error", "warn"):
            raise ValueError(f"Unknown missing_policy: {missing_policy}")

        payload = torch.load(logits_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "sample_ids" not in payload or "logits" not in payload:
            raise ValueError("OOF logits payload must contain sample_ids and logits")
        logits = payload["logits"]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("OOF logits must be a 2-D tensor")
        sample_ids = [str(x) for x in payload["sample_ids"]]
        if len(sample_ids) != logits.shape[0] or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("OOF sample_ids must be unique and align with logits")

        quality = pd.read_csv(quality_path, dtype={"sample_id": str, "image_path": str})
        required = {"sample_id", "image_path", "soft_weight"}
        missing = required - set(quality.columns)
        if missing:
            raise ValueError(f"OOF quality file missing columns: {sorted(missing)}")
        if quality["image_path"].duplicated().any():
            raise ValueError("OOF quality file contains duplicate image_path values")
        id_to_index = {sample_id: i for i, sample_id in enumerate(sample_ids)}

        self._indices: dict[str, int] = {}
        self._weights: dict[str, float] = {}
        for row in quality.itertuples(index=False):
            key = stable_image_key(row.image_path)
            if key in self._indices:
                raise ValueError(f"Duplicate stable OOF image key: {key}")
            if str(row.sample_id) not in id_to_index:
                raise ValueError(f"Quality row missing from OOF logits: {row.sample_id}")
            weight = float(row.soft_weight)
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"Invalid soft_weight for {key}: {weight}")
            self._indices[key] = id_to_index[str(row.sample_id)]
            self._weights[key] = max(min_weight, min(max_weight, weight))

        self._logits = logits.float().contiguous()
        self._missing = missing_policy
        self._min_weight = min_weight
        self._max_weight = max_weight
        self._warned_missing = False

    def _lookup(self, paths: list[str]) -> tuple[list[int], list[float]]:
        indices: list[int] = []
        weights: list[float] = []
        missing: list[str] = []
        for path in paths:
            key = stable_image_key(path)
            if key not in self._indices:
                missing.append(path)
                indices.append(-1)
                weights.append(1.0)
            else:
                indices.append(self._indices[key])
                weights.append(self._weights[key])
        if missing and self._missing == "error":
            raise KeyError(f"missing OOF soft target for {missing[0]}")
        if missing and not self._warned_missing:
            self._warned_missing = True
        return indices, weights

    def get_weights(self, sample_paths, labels, epoch=0, per_sample_loss=None):
        _, weights = self._lookup(list(sample_paths))
        return torch.tensor(weights, device=labels.device, dtype=torch.float32)

    def get_training_labels(self, sample_paths, original_labels):
        return original_labels

    def get_batch(self, sample_paths, device: torch.device, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        indices, weights = self._lookup(list(sample_paths))
        if any(index < 0 for index in indices):
            logits = torch.zeros(len(indices), self._logits.shape[1])
        else:
            logits = self._logits[torch.tensor(indices, dtype=torch.long)]
        targets = torch.softmax(logits.to(device) / temperature, dim=1)
        return targets, torch.tensor(weights, device=device, dtype=torch.float32)
