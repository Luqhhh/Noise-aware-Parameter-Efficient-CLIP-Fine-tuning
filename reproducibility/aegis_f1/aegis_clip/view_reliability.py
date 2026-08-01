"""Protocol and cache validation for CVRG view-reliability inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch


VIEW_ORDER: tuple[str, str, str, str] = (
    "original_global",
    "original_local",
    "flipped_global",
    "flipped_local",
)
BASE_VIEW_WEIGHTS = torch.tensor(
    [0.30, 0.20, 0.30, 0.20], dtype=torch.float32
)
CVRG_NUM_CLASSES = 500
CVRG_CACHE_FORMAT_VERSION = 1
CVRG_FEATURE_SCHEMA_VERSION = 1
_TRUST_KEYS = {
    "labels",
    "clean_probability",
    "pseudo_label",
    "pseudo_labels",
    "correction_alpha",
}
_REQUIRED_VALIDATION_KEYS = {
    "labels",
    "clean_probability",
    "pseudo_label",
    "correction_alpha",
}


@dataclass(frozen=True)
class CVRGProtocol:
    crop_size: int = 160
    top_k: int = 5
    temperature: float = 1.0
    local_weight: float = 0.4
    flip_weight: float = 0.5
    prior_alignment_strength: float = 1.0


def _require_tensor(
    payload: Mapping[str, Any],
    key: str,
    *,
    rank: int,
    sample_count: int | None = None,
) -> torch.Tensor:
    if key not in payload:
        raise ValueError(f"cache missing required field: {key}")
    value = torch.as_tensor(payload[key])
    if value.ndim != rank:
        raise ValueError(f"{key} must have rank {rank}, got {value.ndim}")
    if sample_count is not None and value.shape[0] != sample_count:
        raise ValueError(f"{key} is not aligned with view_logits")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"{key} must contain only finite values")
    return value


def _validate_protocol(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("cache protocol must be a mapping")
    expected = asdict(CVRGProtocol())
    actual = {key: value.get(key) for key in expected}
    if actual != expected or set(value) != set(expected):
        raise ValueError(f"cache protocol mismatch: expected {expected}")


def validate_cvrg_cache(
    payload: Mapping[str, Any],
    *,
    require_labels: bool,
    expected_checkpoint_sha256: str | None = None,
) -> int:
    """Validate a CVRG cache and return its aligned sample count.

    Validation caches carry labels and trust metadata. Test caches must be
    explicitly label-free so that a cache cannot accidentally leak evaluation
    targets into the frozen gate or submission path.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("CVRG cache must be a mapping")
    if payload.get("format_version") != CVRG_CACHE_FORMAT_VERSION:
        raise ValueError("unsupported CVRG cache format version")
    split = payload.get("split")
    if split not in {"validation", "test"}:
        raise ValueError("cache split must be validation or test")
    if tuple(payload.get("view_order", ())) != VIEW_ORDER:
        raise ValueError(f"cache view_order must be exactly {VIEW_ORDER}")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256:
        raise ValueError("cache checkpoint_sha256 must be a non-empty string")
    if expected_checkpoint_sha256 is not None and (
        checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError("checkpoint SHA-256 mismatch")
    split_sha256 = payload.get("split_sha256")
    if not isinstance(split_sha256, str) or not split_sha256:
        raise ValueError("cache split_sha256 must be a non-empty string")
    _validate_protocol(payload.get("protocol"))

    logits = _require_tensor(payload, "view_logits", rank=3)
    sample_count, view_count, class_count = logits.shape
    if view_count != len(VIEW_ORDER) or class_count != CVRG_NUM_CLASSES:
        raise ValueError(
            f"view_logits must have shape [N,4,{CVRG_NUM_CLASSES}]"
        )
    features = _require_tensor(
        payload, "view_features", rank=3, sample_count=sample_count
    )
    if features.shape[1] != len(VIEW_ORDER) or features.shape[2] <= 0:
        raise ValueError("view_features must have shape [N,4,D] with D>0")
    feature_norm = features.float().norm(dim=-1)
    if not torch.allclose(
        feature_norm,
        torch.ones_like(feature_norm),
        atol=2.0e-3,
        rtol=2.0e-3,
    ):
        raise ValueError("view_features must be normalized")
    attention = _require_tensor(
        payload, "orientation_attention", rank=4, sample_count=sample_count
    )
    if attention.shape[1] != 2 or attention.shape[-1] <= 0:
        raise ValueError("orientation_attention must have shape [N,2,H,P]")
    boxes = _require_tensor(
        payload, "crop_boxes", rank=3, sample_count=sample_count
    )
    if boxes.shape[1:] != (2, 4):
        raise ValueError("crop_boxes must have shape [N,2,4]")

    paths = payload.get("paths")
    if not isinstance(paths, (list, tuple)):
        raise ValueError("cache paths must be a list or tuple")
    if len(paths) != sample_count:
        raise ValueError("cache paths are not aligned with view_logits")
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("cache paths must contain non-empty strings")
    if len(set(paths)) != len(paths):
        raise ValueError("cache paths must be unique")

    present_trust_keys = _TRUST_KEYS.intersection(payload)
    if require_labels:
        if split != "validation":
            raise ValueError("labels are only valid for validation caches")
        missing = _REQUIRED_VALIDATION_KEYS - present_trust_keys
        if missing:
            raise ValueError(f"validation cache missing labels/trust fields: {sorted(missing)}")
        labels = _require_tensor(
            payload, "labels", rank=1, sample_count=sample_count
        ).long()
        if (labels < 0).any() or (labels >= CVRG_NUM_CLASSES).any():
            raise ValueError("labels are outside the declared class range")
        clean = _require_tensor(
            payload, "clean_probability", rank=1, sample_count=sample_count
        ).float()
        correction = _require_tensor(
            payload, "correction_alpha", rank=1, sample_count=sample_count
        ).float()
        pseudo = _require_tensor(
            payload, "pseudo_label", rank=1, sample_count=sample_count
        ).long()
        if (clean < 0.0).any() or (clean > 1.0).any():
            raise ValueError("clean_probability must be in [0,1]")
        if (correction < 0.0).any() or (correction > 1.0).any():
            raise ValueError("correction_alpha must be in [0,1]")
        if (pseudo < -1).any() or (pseudo >= CVRG_NUM_CLASSES).any():
            raise ValueError("pseudo_label is outside the declared class range")
    elif present_trust_keys:
        raise ValueError("test caches must be label-free")

    return sample_count


__all__ = [
    "BASE_VIEW_WEIGHTS",
    "CVRG_CACHE_FORMAT_VERSION",
    "CVRG_FEATURE_SCHEMA_VERSION",
    "CVRG_NUM_CLASSES",
    "CVRGProtocol",
    "VIEW_ORDER",
    "validate_cvrg_cache",
]
