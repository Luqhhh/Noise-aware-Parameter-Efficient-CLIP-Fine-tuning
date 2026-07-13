"""Unified sample weight provider.

Replaces the ad-hoc JSON loading in ``train.py`` with a strategy-based
provider that supports stateful (EMA loss) and stateless (manifest, prototype)
weighting schemes.

Usage in config:

    sample_weighting:
      type: static_manifest        # or ema_loss, prototype, hybrid, ...
      manifest_path: outputs/phase2/prototype_weights/sample_weights.json
      momentum: 0.9                # ema_loss only
      warmup_epochs: 5             # ema_loss only
      ranking: classwise           # ema_loss only
      min_weight: 0.4
      max_weight: 1.0
      normalize_by_weight_sum: true
      missing_weight_policy: error
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────


class BaseWeightProvider(ABC):
    """Abstract base for sample weight providers."""

    @abstractmethod
    def get_weights(
        self,
        sample_paths: list,
        labels: torch.Tensor,
        epoch: int,
        per_sample_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return weight tensor of shape (batch_size,)."""

    def state_dict(self) -> dict:
        """State for checkpoint serialisation (stateful providers only)."""
        return {}

    def load_state_dict(self, d: dict):
        """Restore state from checkpoint."""

    def on_epoch_start(self, epoch: int):
        """Hook called at the start of each epoch."""


# ──────────────────────────────────────────────────────────────────────
# None (identity weights)
# ──────────────────────────────────────────────────────────────────────


class NoneWeightProvider(BaseWeightProvider):
    """Returns 1.0 for every sample — equivalent to unweighted training."""

    def get_weights(self, sample_paths, labels, epoch, per_sample_loss=None):
        return torch.ones(len(sample_paths), device=labels.device)


# ──────────────────────────────────────────────────────────────────────
# Static manifest (backward-compatible with old sample_weighting.enabled)
# ──────────────────────────────────────────────────────────────────────


class StaticManifestProvider(BaseWeightProvider):
    """Loads per-sample weights from a JSON manifest file.

    The manifest is a dict mapping image_path → {"weight": float}.
    This preserves exact backward compatibility with the existing
    ``sample_weighting.enabled + weights_path`` config in train.py.
    """

    def __init__(
        self,
        manifest_path: str,
        normalize_by_weight_sum: bool = True,
        missing_policy: str = "error",
    ):
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Weight manifest not found: {path}")
        with open(path) as f:
            raw = json.load(f)

        # Build lookup: path → weight
        self._weights: Dict[str, float] = {}
        for img_path, entry in raw.items():
            w = float(entry["weight"]) if isinstance(entry, dict) else float(entry)
            self._weights[str(img_path)] = w

        self._normalize = normalize_by_weight_sum
        self._missing = missing_policy
        logger.info(
            "StaticManifestProvider: %d weights loaded from %s",
            len(self._weights), manifest_path,
        )

    def get_weights(self, sample_paths, labels, epoch, per_sample_loss=None):
        w_vals = []
        for p in sample_paths:
            entry = self._weights.get(p)
            if entry is None:
                if self._missing == "error":
                    raise KeyError(f"Sample weight missing for: {p}")
                w_vals.append(1.0)
            else:
                w_vals.append(entry)
        return torch.tensor(w_vals, device=labels.device, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────
# EMA loss weighting (stateful)
# ──────────────────────────────────────────────────────────────────────


class EMALossProvider(BaseWeightProvider):
    """Tracks per-sample EMA of training loss and assigns lower weights
    to samples with consistently high loss (likely noisy labels).

    Ranking is classwise — the highest-loss samples within each class
    get the lowest weights.
    """

    def __init__(
        self,
        num_samples: int,
        momentum: float = 0.9,
        warmup_epochs: int = 5,
        ranking: str = "classwise",
        min_weight: float = 0.4,
        max_weight: float = 1.0,
        high_loss_fraction_epoch_6_15: float = 0.10,
        high_loss_fraction_epoch_16_plus: float = 0.20,
    ):
        if not 0.0 < momentum <= 1.0:
            raise ValueError(f"momentum must be in (0, 1], got {momentum}")
        if ranking not in ("classwise",):
            raise ValueError(f"ranking must be 'classwise', got {ranking}")

        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.ranking = ranking
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.frac_6_15 = high_loss_fraction_epoch_6_15
        self.frac_16_plus = high_loss_fraction_epoch_16_plus

        # State
        self.register_buffer("ema_loss", torch.full((num_samples,), float("nan")))
        self._sample_to_idx: Dict[str, int] = {}
        self._label_of_idx: Optional[torch.Tensor] = None  # (num_samples,)

    def register_buffer(self, name, tensor):
        """Simulate nn.Module.register_buffer for standalone use."""
        setattr(self, name, tensor)

    def init_sample_index(self, sample_paths: list, labels: torch.Tensor):
        """Build path→index mapping (call once before training)."""
        if self._sample_to_idx:
            return  # already initialized
        for i, p in enumerate(sample_paths):
            if p in self._sample_to_idx:
                raise ValueError(f"Duplicate sample path: {p}")
            self._sample_to_idx[p] = i
        self._label_of_idx = labels.clone()
        logger.info(
            "EMALossProvider: initialised with %d samples", len(self._sample_to_idx)
        )

    def get_weights(self, sample_paths, labels, epoch, per_sample_loss=None):
        n = len(sample_paths)

        # Warmup: all weights = 1.0
        if epoch <= self.warmup_epochs:
            return torch.ones(n)

        # Determine device from per_sample_loss
        device = per_sample_loss.device if per_sample_loss is not None else torch.device("cpu")

        # Update EMA
        if per_sample_loss is not None:
            indices = [self._sample_to_idx[p] for p in sample_paths]
            idx_t = torch.tensor(indices, device=device)
            new_ema = torch.where(
                self.ema_loss[idx_t].to(device).isnan(),
                per_sample_loss.detach(),
                self.momentum * self.ema_loss[idx_t].to(device)
                + (1.0 - self.momentum) * per_sample_loss.detach(),
            )
            self.ema_loss[idx_t] = new_ema.cpu()

        # Determine fraction of high-loss samples to downweight
        if epoch <= 15:
            frac = self.frac_6_15
        else:
            frac = self.frac_16_plus

        # Classwise ranking
        weights = torch.ones(n, device=device)
        unique_labels = labels.unique()
        for c in unique_labels:
            c_mask = labels == c
            c_indices = c_mask.nonzero(as_tuple=True)[0]
            if len(c_indices) <= 1:
                continue
            c_paths = [sample_paths[i.item()] for i in c_indices]
            c_ema_indices = [self._sample_to_idx[p] for p in c_paths]
            c_ema = self.ema_loss[torch.tensor(c_ema_indices)].clone()
            # NaN → treat as 0 (no history yet, default to max weight)
            c_ema = torch.where(c_ema.isnan(), torch.zeros_like(c_ema), c_ema)
            n_high = max(1, int(len(c_indices) * frac))
            _, high_idx = torch.topk(c_ema, n_high)
            weights[c_indices[high_idx]] = self.min_weight

        return weights

    def state_dict(self) -> dict:
        return {
            "ema_loss": self.ema_loss.clone(),
            "sample_to_idx_keys": list(self._sample_to_idx.keys()),
            "sample_to_idx_values": list(self._sample_to_idx.values()),
            "label_of_idx": self._label_of_idx.clone()
            if self._label_of_idx is not None
            else None,
        }

    def load_state_dict(self, d: dict):
        self.ema_loss = d["ema_loss"]
        self._sample_to_idx = dict(zip(d["sample_to_idx_keys"], d["sample_to_idx_values"]))
        self._label_of_idx = d.get("label_of_idx")


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────


def build_weight_provider(config: dict, num_train_samples: int = 0) -> BaseWeightProvider:
    """Build a sample weight provider from config.

    Args:
        config: Full project config dict.  Reads ``sample_weighting``.
        num_train_samples: Total number of training samples (needed for
            stateful providers like ema_loss).

    Returns:
        A ``BaseWeightProvider`` instance.
    """
    sw = config.get("sample_weighting", {})
    sw_type = sw.get("type", "none")

    if sw_type == "none" or not sw:
        return NoneWeightProvider()

    if sw_type == "static_manifest":
        return StaticManifestProvider(
            manifest_path=sw["manifest_path"],
            normalize_by_weight_sum=sw.get("normalize_by_weight_sum", True),
            missing_policy=sw.get("missing_weight_policy", "error"),
        )

    if sw_type == "ema_loss":
        if num_train_samples <= 0:
            raise ValueError(
                "num_train_samples is required for ema_loss provider"
            )
        return EMALossProvider(
            num_samples=num_train_samples,
            momentum=sw.get("momentum", 0.9),
            warmup_epochs=sw.get("warmup_epochs", 5),
            ranking=sw.get("ranking", "classwise"),
            min_weight=sw.get("min_weight", 0.4),
            max_weight=sw.get("max_weight", 1.0),
        )

    # Future: prototype, hybrid, oof_manifest, relabel_manifest
    raise ValueError(f"Unknown sample_weighting.type: {sw_type}")
