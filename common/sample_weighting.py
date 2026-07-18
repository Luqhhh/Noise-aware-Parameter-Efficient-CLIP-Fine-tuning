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

from common.manifest_loader import canonical_image_path

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

    def get_training_labels(
        self,
        sample_paths: list,
        original_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return batch training labels (may differ from original when relabel enabled)."""
        return original_labels

    def get_roles(self, sample_paths: list) -> list:
        """Return list of training roles (clean, rejected, pseudo)."""
        return ["clean"] * len(sample_paths)

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
        self._warned_missing: bool = False
        logger.info(
            "StaticManifestProvider: %d weights loaded from %s",
            len(self._weights), manifest_path,
        )

    def get_weights(self, sample_paths, labels, epoch, per_sample_loss=None):
        w_vals = []
        missing = []
        for p in sample_paths:
            entry = self._weights.get(p)
            if entry is None:
                if self._missing == "error":
                    raise KeyError(f"Sample weight missing for: {p}")
                w_vals.append(1.0)
                missing.append(p)
            else:
                w_vals.append(entry)

        if missing and not self._warned_missing:
            self._warned_missing = True
            logger.warning(
                "StaticManifestProvider: %d/%d samples missing from manifest "
                "— assigned default weight 1.0. First 5: %s",
                len(missing), len(sample_paths), missing[:5],
            )

        return torch.tensor(w_vals, device=labels.device, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────
# Prototype confidence weighting (stateless)
# ──────────────────────────────────────────────────────────────────────


class PrototypeProvider(BaseWeightProvider):
    """Loads pre-computed prototype confidence weights from a JSON manifest.

    The manifest maps image_path → {"weight": float, ...}.  The weight
    is expected to already encode the prototype confidence transformation
    (e.g. ``0.4 + 0.6 * c_i``).  This provider applies optional clamping
    to ``[min_weight, max_weight]`` and logs diagnostic statistics.
    """

    def __init__(
        self,
        manifest_path: str,
        min_weight: float = 0.4,
        max_weight: float = 1.0,
        classwise_percentile: bool = True,
        normalize_by_weight_sum: bool = True,
        missing_policy: str = "error",
    ):
        if min_weight < 0.0 or max_weight > 1.0 or min_weight >= max_weight:
            raise ValueError(
                f"Invalid weight range: min_weight={min_weight}, "
                f"max_weight={max_weight}"
            )

        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Prototype manifest not found: {path}")
        with open(path) as f:
            raw = json.load(f)

        self._weights: Dict[str, float] = {}
        margins = []
        own_sims = []
        for img_path, entry in raw.items():
            w = float(entry["weight"]) if isinstance(entry, dict) else float(entry)
            w = max(min_weight, min(max_weight, w))
            self._weights[str(img_path)] = w
            if isinstance(entry, dict):
                if "margin" in entry:
                    margins.append(float(entry["margin"]))
                if "own_similarity" in entry:
                    own_sims.append(float(entry["own_similarity"]))

        self._min_weight = min_weight
        self._max_weight = max_weight
        self._classwise_percentile = classwise_percentile
        self._normalize = normalize_by_weight_sum
        self._missing = missing_policy
        self._warned_missing: bool = False

        # Diagnostics
        w_vals = np.array(list(self._weights.values()))
        logger.info(
            "PrototypeProvider: %d weights loaded from %s | "
            "mean=%.4f median=%.4f min=%.4f max=%.4f | "
            "clamped to [%.2f, %.2f]",
            len(self._weights), manifest_path,
            float(w_vals.mean()), float(np.median(w_vals)),
            float(w_vals.min()), float(w_vals.max()),
            min_weight, max_weight,
        )
        if margins:
            logger.info(
                "PrototypeProvider: margin mean=%.4f median=%.4f (n=%d)",
                float(np.mean(margins)), float(np.median(margins)), len(margins),
            )

    def get_weights(self, sample_paths, labels, epoch, per_sample_loss=None):
        w_vals = []
        missing = []
        for p in sample_paths:
            entry = self._weights.get(p)
            if entry is None:
                if self._missing == "error":
                    raise KeyError(f"Prototype weight missing for: {p}")
                w_vals.append(1.0)
                missing.append(p)
            else:
                w_vals.append(entry)

        if missing and not self._warned_missing:
            self._warned_missing = True
            logger.warning(
                "PrototypeProvider: %d/%d samples missing from manifest "
                "— assigned default weight 1.0. First 5: %s",
                len(missing), len(sample_paths), missing[:5],
            )

        return torch.tensor(w_vals, device=labels.device, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────
# OOF manifest weighting (B-delivered CSV)
# ──────────────────────────────────────────────────────────────────────


class OOFManifestProvider(BaseWeightProvider):
    """Loads per-sample weights from a B-delivered OOF manifest CSV.

    Uses ``ManifestLoader`` for fail-closed validation.  The manifest must
    have 100% coverage and all ``sample_weight`` values in [0, 1].
    """

    def __init__(
        self,
        manifest_path: str,
        min_weight: float = 0.3,
        max_weight: float = 1.0,
        missing_policy: str = "error",
    ):
        from common.manifest_loader import ManifestLoader

        if min_weight < 0.0 or max_weight > 1.0 or min_weight >= max_weight:
            raise ValueError(
                f"Invalid weight range: min_weight={min_weight}, "
                f"max_weight={max_weight}"
            )

        self._loader = ManifestLoader(manifest_path)
        df = self._loader.load()

        self._weights: Dict[str, float] = {}
        self._training_labels: Dict[str, int] = {}
        self._warned_missing: bool = False
        self._missing = missing_policy
        for _, row in df.iterrows():
            raw_path = str(row["image_path"])
            key = canonical_image_path(raw_path)
            w = float(row["sample_weight"])
            w = max(min_weight, min(max_weight, w))
            self._weights[key] = w
            # Weight-only mode: keep original label
            self._training_labels[key] = int(row["original_label"])

        self._min_weight = min_weight
        self._max_weight = max_weight
        self._missing = missing_policy
        self._manifest_sha256 = self._loader.sha256

        w_vals = np.array(list(self._weights.values()))
        logger.info(
            "OOFManifestProvider: %d samples | "
            "weight mean=%.4f median=%.4f | "
            "manifest_sha256=%s",
            len(self._weights),
            float(w_vals.mean()), float(np.median(w_vals)),
            self._manifest_sha256[:16],
        )

    def get_weights(self, sample_paths, labels, epoch, per_sample_loss=None):
        w_vals = []
        missing = []
        for p in sample_paths:
            entry = self._weights.get(canonical_image_path(p))
            if entry is None:
                if self._missing == "error":
                    raise KeyError(f"OOF weight missing for: {p}")
                w_vals.append(1.0)
                missing.append(p)
            else:
                w_vals.append(entry)

        # One-time warning: log missing count and first few examples
        if missing and not self._warned_missing:
            self._warned_missing = True
            n_missing = len(missing)
            n_total = len(sample_paths)
            preview = missing[:5]
            logger.warning(
                "OOFManifestProvider: %d/%d samples (%.1f%%) missing from OOF "
                "manifest — assigned default weight 1.0. "
                "This is expected for validation-set images in final_fit mode, "
                "but unexpected for d3_strict training images. "
                "First 5: %s",
                n_missing, n_total, 100 * n_missing / n_total, preview,
            )
            if n_missing > 0:
                logger.info(
                    "OOFManifestProvider: total weights: min=%.4f max=%.4f "
                    "mean=%.4f (missing assigned 1.0, %.1f%% of batch)",
                    min(w_vals), max(w_vals),
                    sum(w_vals) / len(w_vals),
                    100 * n_missing / n_total,
                )

        return torch.tensor(w_vals, device=labels.device, dtype=torch.float32)

    def get_training_label(self, image_path: str, original_label: int) -> int:
        """Return the training label for a sample.

        For weight-only manifests, returns *original_label* unchanged.
        """
        return self._training_labels.get(image_path, original_label)


# ──────────────────────────────────────────────────────────────────────
# Relabel manifest weighting (B-delivered CSV)
# ──────────────────────────────────────────────────────────────────────


class RelabelManifestProvider(BaseWeightProvider):
    """Loads per-sample weights from a B-delivered relabel manifest CSV.

    Supports both weight-only (original label kept, weight reduced for
    suspicious samples) and hard relabel (training_label replaces original).
    Uses ``ManifestLoader`` for fail-closed validation.
    """

    def __init__(
        self,
        manifest_path: str,
        min_weight: float = 0.3,
        max_weight: float = 1.0,
        hard_relabel: bool = False,
        missing_policy: str = "error",
    ):
        from common.manifest_loader import ManifestLoader

        if min_weight < 0.0 or max_weight > 1.0 or min_weight >= max_weight:
            raise ValueError(
                f"Invalid weight range: min_weight={min_weight}, "
                f"max_weight={max_weight}"
            )

        self._loader = ManifestLoader(manifest_path)
        df = self._loader.load()

        self._weights: Dict[str, float] = {}
        self._training_labels: Dict[str, int] = {}
        self._roles: Dict[str, str] = {}
        self._hard_relabel = hard_relabel
        n_relabeled = 0

        for _, row in df.iterrows():
            path = canonical_image_path(str(row["image_path"]))
            w = float(row["sample_weight"])
            w = max(min_weight, min(max_weight, w))
            self._weights[path] = w

            if hard_relabel:
                self._training_labels[path] = int(row["training_label"])
            else:
                self._training_labels[path] = int(row["original_label"])

            role = str(row.get("training_role", "clean"))
            if role not in ("clean", "rejected", "pseudo"):
                role = "clean"
            self._roles[path] = role

            if int(row["training_label"]) != int(row["original_label"]):
                n_relabeled += 1

        self._min_weight = min_weight
        self._max_weight = max_weight
        self._missing = missing_policy
        self._warned_missing: bool = False
        self._manifest_sha256 = self._loader.sha256

        w_vals = np.array(list(self._weights.values()))
        logger.info(
            "RelabelManifestProvider: %d samples | hard_relabel=%s | "
            "relabeled=%d (%.2f%%) | "
            "weight mean=%.4f median=%.4f | manifest_sha256=%s",
            len(self._weights), hard_relabel,
            n_relabeled,
            100.0 * n_relabeled / max(len(self._weights), 1),
            float(w_vals.mean()), float(np.median(w_vals)),
            self._manifest_sha256[:16],
        )

    def get_weights(self, sample_paths, labels, epoch, per_sample_loss=None):
        w_vals = []
        missing = []
        for p in sample_paths:
            key = canonical_image_path(p)
            entry = self._weights.get(key)
            if entry is None:
                if self._missing == "error":
                    raise KeyError(f"Relabel weight missing for: {p}")
                w_vals.append(1.0)
                missing.append(p)
            else:
                w_vals.append(entry)

        if missing and not self._warned_missing:
            self._warned_missing = True
            logger.warning(
                "RelabelManifestProvider: %d/%d samples missing from manifest "
                "— assigned default weight 1.0. First 5: %s",
                len(missing), len(sample_paths), missing[:5],
            )

        return torch.tensor(w_vals, device=labels.device, dtype=torch.float32)

    def get_training_label(self, image_path: str, original_label: int) -> int:
        """Return the (possibly relabeled) training label for a sample."""
        return self._training_labels.get(image_path, original_label)

    def get_training_labels(
        self,
        sample_paths: list,
        original_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return batch training labels, applying relabel decisions if enabled."""
        values = []
        for path, original in zip(sample_paths, original_labels.tolist()):
            key = canonical_image_path(path)
            if key in self._training_labels:
                values.append(self._training_labels[key])
            else:
                if self._missing == "error":
                    raise KeyError(f"Relabel label missing for: {path}")
                values.append(int(original))
        return torch.tensor(
            values, device=original_labels.device, dtype=torch.long,
        )

    def get_roles(self, sample_paths: list) -> list:
        """Return training roles from manifest (clean/rejected/pseudo).

        Uses the training_role column saved at load time.
        Missing entries fail with error policy — coverage must be 100%.
        """
        roles = []
        for path in sample_paths:
            key = canonical_image_path(path)
            role = self._roles.get(key)
            if role is None:
                if self._missing == "error":
                    raise KeyError(f"Role missing for: {path}")
                role = "clean"
            roles.append(role)
        return roles


# ──────────────────────────────────────────────────────────────────────
# Prototype + EMA loss hybrid weighting (stateful)
# ──────────────────────────────────────────────────────────────────────


class PrototypeEMAProvider(BaseWeightProvider):
    """Hybrid: prototype confidence × EMA-loss confidence → per-sample weight.

    confidence_i = proto_weight * c_i^prototype + ema_weight * c_i^ema
    w_i = min_weight + (max_weight - min_weight) * confidence_i

    Stateful — EMA loss history is written to / restored from checkpoints.
    """

    def __init__(
        self,
        num_samples: int,
        prototype_manifest_path: str,
        momentum: float = 0.9,
        warmup_epochs: int = 5,
        ranking: str = "classwise",
        min_weight: float = 0.4,
        max_weight: float = 1.0,
        proto_weight: float = 0.7,
        ema_weight: float = 0.3,
        high_loss_fraction_epoch_6_15: float = 0.10,
        high_loss_fraction_epoch_16_plus: float = 0.20,
    ):
        if abs(proto_weight + ema_weight - 1.0) > 1e-9:
            raise ValueError(
                f"proto_weight + ema_weight must sum to 1.0, "
                f"got {proto_weight} + {ema_weight}"
            )
        if min_weight < 0.0 or max_weight > 1.0 or min_weight >= max_weight:
            raise ValueError(
                f"Invalid weight range: min_weight={min_weight}, "
                f"max_weight={max_weight}"
            )

        # Load prototype weights
        proto_path = Path(prototype_manifest_path)
        if not proto_path.exists():
            raise FileNotFoundError(
                f"Prototype manifest not found: {proto_path}"
            )
        with open(proto_path) as f:
            raw = json.load(f)

        self._proto_weights: Dict[str, float] = {}
        for img_path, entry in raw.items():
            w = float(entry["weight"]) if isinstance(entry, dict) else float(entry)
            self._proto_weights[str(img_path)] = w

        # EMA state
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.ranking = ranking
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.proto_weight = proto_weight
        self.ema_weight = ema_weight
        self.frac_6_15 = high_loss_fraction_epoch_6_15
        self.frac_16_plus = high_loss_fraction_epoch_16_plus

        self.register_buffer(
            "ema_loss", torch.full((num_samples,), float("nan"))
        )
        self._sample_to_idx: Dict[str, int] = {}
        self._label_of_idx: Optional[torch.Tensor] = None

        logger.info(
            "PrototypeEMAProvider: %d prototype weights loaded | "
            "momentum=%.2f warmup=%d proto_w=%.2f ema_w=%.2f",
            len(self._proto_weights), momentum, warmup_epochs,
            proto_weight, ema_weight,
        )

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def init_sample_index(self, sample_paths: list, labels: torch.Tensor):
        """Build path->index mapping (call once before training)."""
        if self._sample_to_idx:
            return
        for i, p in enumerate(sample_paths):
            if p in self._sample_to_idx:
                raise ValueError(f"Duplicate sample path: {p}")
            self._sample_to_idx[p] = i
        self._label_of_idx = labels.clone()
        logger.info(
            "PrototypeEMAProvider: initialised with %d samples",
            len(self._sample_to_idx),
        )

    def get_weights(self, sample_paths, labels, epoch, per_sample_loss=None):
        n = len(sample_paths)
        device = (
            per_sample_loss.device
            if per_sample_loss is not None
            else torch.device("cpu")
        )

        # Warmup: prototype-only weights
        if epoch <= self.warmup_epochs:
            w_vals = []
            for p in sample_paths:
                w_vals.append(self._proto_weights.get(p, 1.0))
            return torch.tensor(w_vals, device=device, dtype=torch.float32)

        # Update EMA
        if per_sample_loss is not None:
            indices = [self._sample_to_idx[p] for p in sample_paths]
            idx_t = torch.tensor(indices, dtype=torch.long)
            new_ema = torch.where(
                self.ema_loss[idx_t].isnan(),
                per_sample_loss.detach().cpu(),
                self.momentum * self.ema_loss[idx_t]
                + (1.0 - self.momentum) * per_sample_loss.detach().cpu(),
            )
            self.ema_loss[idx_t] = new_ema

        frac = self.frac_6_15 if epoch <= 15 else self.frac_16_plus
        weights = torch.ones(n)

        unique_labels = labels.unique()
        for c_label in unique_labels:
            c_mask = labels == c_label
            c_indices = c_mask.nonzero(as_tuple=True)[0]
            if len(c_indices) <= 1:
                continue

            # Prototype confidence: c = (w - 0.4) / 0.6
            c_proto = torch.tensor([
                (self._proto_weights.get(sample_paths[i.item()], 1.0)
                 - self.min_weight)
                / max(self.max_weight - self.min_weight, 1e-8)
                for i in c_indices
            ], dtype=torch.float32)
            c_proto = torch.clamp(c_proto, 0.0, 1.0)

            # EMA confidence: 1 - percentile_rank (lower loss = more confident)
            c_paths = [sample_paths[i.item()] for i in c_indices]
            c_ema_idx = torch.tensor(
                [self._sample_to_idx[p] for p in c_paths], dtype=torch.long
            )
            c_ema_raw = self.ema_loss[c_ema_idx].clone()
            c_ema_raw = torch.where(
                c_ema_raw.isnan(), torch.zeros_like(c_ema_raw), c_ema_raw
            )
            n_class = len(c_indices)
            _, sort_idx = torch.sort(c_ema_raw)
            ranks = torch.zeros(n_class)
            ranks[sort_idx] = torch.arange(n_class, dtype=torch.float32)
            c_ema = 1.0 - ranks / max(n_class - 1, 1)

            # Hybrid confidence --> weight
            hybrid = self.proto_weight * c_proto + self.ema_weight * c_ema
            hybrid = torch.clamp(hybrid, 0.0, 1.0)
            c_weights = (
                self.min_weight
                + (self.max_weight - self.min_weight) * hybrid
            )
            weights[c_indices] = c_weights

        return weights.to(device)

    def state_dict(self) -> dict:
        return {
            "ema_loss": self.ema_loss.clone(),
            "sample_to_idx_keys": list(self._sample_to_idx.keys()),
            "sample_to_idx_values": list(self._sample_to_idx.values()),
            "label_of_idx": (
                self._label_of_idx.clone()
                if self._label_of_idx is not None
                else None
            ),
        }

    def load_state_dict(self, d: dict):
        self.ema_loss = d["ema_loss"]
        self._sample_to_idx = dict(
            zip(d["sample_to_idx_keys"], d["sample_to_idx_values"])
        )
        self._label_of_idx = d.get("label_of_idx")


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
            idx_t = torch.tensor(indices, dtype=torch.long)  # CPU
            new_ema = torch.where(
                self.ema_loss[idx_t].isnan(),
                per_sample_loss.detach().cpu(),
                self.momentum * self.ema_loss[idx_t]
                + (1.0 - self.momentum) * per_sample_loss.detach().cpu(),
            )
            self.ema_loss[idx_t] = new_ema

        # Determine fraction of high-loss samples to downweight
        if epoch <= 15:
            frac = self.frac_6_15
        else:
            frac = self.frac_16_plus

        # Classwise ranking (all on CPU, return CPU tensor)
        weights = torch.ones(n)
        unique_labels = labels.unique()
        for c in unique_labels:
            c_mask = labels == c
            c_indices = c_mask.nonzero(as_tuple=True)[0]
            if len(c_indices) <= 1:
                continue
            c_paths = [sample_paths[i.item()] for i in c_indices]
            c_ema_indices = torch.tensor(
                [self._sample_to_idx[p] for p in c_paths], dtype=torch.long
            )
            c_ema = self.ema_loss[c_ema_indices].clone()
            # NaN → treat as 0 (no history yet, default to max weight)
            c_ema = torch.where(c_ema.isnan(), torch.zeros_like(c_ema), c_ema)
            n_high = max(1, int(len(c_indices) * frac))
            _, high_idx = torch.topk(c_ema, n_high)
            weights[c_indices[high_idx]] = self.min_weight

        return weights.to(device)

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

    if sw_type == "prototype":
        return PrototypeProvider(
            manifest_path=sw["manifest_path"],
            min_weight=sw.get("min_weight", 0.4),
            max_weight=sw.get("max_weight", 1.0),
            classwise_percentile=sw.get("classwise_percentile", True),
            normalize_by_weight_sum=sw.get("normalize_by_weight_sum", True),
            missing_policy=sw.get("missing_weight_policy", "error"),
        )

    if sw_type == "oof_manifest":
        return OOFManifestProvider(
            manifest_path=sw["manifest_path"],
            min_weight=sw.get("min_weight", 0.3),
            max_weight=sw.get("max_weight", 1.0),
            missing_policy=sw.get("missing_weight_policy", "error"),
        )

    if sw_type == "relabel_manifest":
        return RelabelManifestProvider(
            manifest_path=sw["manifest_path"],
            min_weight=sw.get("min_weight", 0.3),
            max_weight=sw.get("max_weight", 1.0),
            hard_relabel=sw.get("hard_relabel", False),
            missing_policy=sw.get("missing_weight_policy", "error"),
        )

    if sw_type == "prototype_ema_hybrid":
        if num_train_samples <= 0:
            raise ValueError(
                "num_train_samples is required for "
                "prototype_ema_hybrid provider"
            )
        return PrototypeEMAProvider(
            num_samples=num_train_samples,
            prototype_manifest_path=sw["manifest_path"],
            momentum=sw.get("momentum", 0.9),
            warmup_epochs=sw.get("warmup_epochs", 5),
            ranking=sw.get("ranking", "classwise"),
            min_weight=sw.get("min_weight", 0.4),
            max_weight=sw.get("max_weight", 1.0),
            proto_weight=sw.get("proto_weight", 0.7),
            ema_weight=sw.get("ema_weight", 0.3),
        )

    raise ValueError(f"Unknown sample_weighting.type: {sw_type}")
