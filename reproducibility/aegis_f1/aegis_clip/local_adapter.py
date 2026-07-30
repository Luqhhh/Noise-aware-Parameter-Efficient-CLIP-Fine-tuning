"""Auditable local-view feature adapter utilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from aegis_clip.model import ResidualFeatureAdapter


LOCAL_VIEW_CACHE_KEYS = {
    "base_checkpoint_sha256",
    "split",
    "paths",
    "labels",
    "clean_probability",
    "pseudo_label",
    "correction_alpha",
    "global_features",
    "local_features",
}


def validate_local_view_cache(
    payload: dict[str, Any],
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_split: str | None = None,
) -> int:
    """Fail closed on incomplete, misaligned, or non-finite local-view caches."""
    missing = LOCAL_VIEW_CACHE_KEYS - set(payload)
    if missing:
        raise ValueError(f"Local-view cache missing keys: {sorted(missing)}")
    if (
        expected_checkpoint_sha256 is not None
        and payload["base_checkpoint_sha256"] != expected_checkpoint_sha256
    ):
        raise ValueError("Local-view cache base checkpoint hash mismatch")
    if expected_split is not None and payload["split"] != expected_split:
        raise ValueError(
            f"Expected {expected_split!r} cache, got {payload['split']!r}"
        )
    paths = [str(value) for value in payload["paths"]]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Local-view cache paths must be non-empty and unique")
    size = len(paths)
    vectors = {
        "labels": torch.as_tensor(payload["labels"]).flatten(),
        "clean_probability": torch.as_tensor(
            payload["clean_probability"]
        ).flatten(),
        "pseudo_label": torch.as_tensor(payload["pseudo_label"]).flatten(),
        "correction_alpha": torch.as_tensor(
            payload["correction_alpha"]
        ).flatten(),
    }
    for name, value in vectors.items():
        if value.numel() != size:
            raise ValueError(
                f"Local-view cache {name} has {value.numel()} rows, expected {size}"
            )
    global_features = torch.as_tensor(payload["global_features"])
    local_features = torch.as_tensor(payload["local_features"])
    if (
        global_features.ndim != 2
        or local_features.ndim != 2
        or global_features.shape != local_features.shape
        or global_features.shape[0] != size
    ):
        raise ValueError("Local-view cache feature matrices are misaligned")
    for name, value in {
        **vectors,
        "global_features": global_features,
        "local_features": local_features,
    }.items():
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"Local-view cache {name} contains non-finite values")
    if ((vectors["clean_probability"] < 0.0) | (
        vectors["clean_probability"] > 1.0
    )).any():
        raise ValueError("clean_probability must be in [0, 1]")
    if ((vectors["correction_alpha"] < 0.0) | (
        vectors["correction_alpha"] > 1.0
    )).any():
        raise ValueError("correction_alpha must be in [0, 1]")
    return size


def classifier_parameters_from_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Base checkpoint is missing model_state_dict")
    try:
        weight = torch.as_tensor(state["classifier.weight"]).float()
        bias = torch.as_tensor(state["classifier.bias"]).float()
    except KeyError as exc:
        raise ValueError("Base checkpoint must contain a linear classifier") from exc
    if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.numel():
        raise ValueError("Base checkpoint classifier tensors are malformed")
    return weight, bias


def classify_adapted_local_features(
    adapter: ResidualFeatureAdapter,
    local_features: torch.Tensor,
    classifier_weight: torch.Tensor,
    classifier_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    adapted = adapter(F.normalize(local_features.float(), dim=1))
    logits = F.linear(adapted, classifier_weight, classifier_bias)
    return logits, adapted


def build_local_adapter(
    *,
    feature_dim: int,
    bottleneck_dim: int,
    residual_scale: float,
) -> ResidualFeatureAdapter:
    return ResidualFeatureAdapter(
        feature_dim=feature_dim,
        bottleneck_dim=bottleneck_dim,
        residual_scale=residual_scale,
    )
