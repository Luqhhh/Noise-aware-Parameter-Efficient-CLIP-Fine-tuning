"""A zero-initialised learned local residual over a frozen global classifier."""

from __future__ import annotations

import torch
import torch.nn as nn


DUAL_VIEW_CACHE_KEYS = {
    "paths",
    "labels",
    "clean_probability",
    "pseudo_labels",
    "correction_alpha",
    "global_features",
    "local_features",
    "global_logits",
}


class LearnedLocalResidualHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0 or int(num_classes) <= 1:
            raise ValueError("Local residual dimensions are invalid")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.dropout_probability = float(dropout)
        self.dropout = nn.Dropout(self.dropout_probability)
        self.local_classifier = nn.Linear(self.feature_dim, self.num_classes)
        nn.init.zeros_(self.local_classifier.weight)
        nn.init.zeros_(self.local_classifier.bias)

    def forward(
        self, base_logits: torch.Tensor, local_features: torch.Tensor
    ) -> torch.Tensor:
        if base_logits.ndim != 2 or local_features.ndim != 2:
            raise ValueError("Residual inputs must be rank-2")
        if base_logits.shape != (local_features.shape[0], self.num_classes):
            raise ValueError("Base logits do not align with the residual head")
        if local_features.shape[1] != self.feature_dim:
            raise ValueError("Local feature dimension does not match the head")
        residual = self.local_classifier(self.dropout(local_features.float()))
        return base_logits.float() + residual

    def residual_parameter_norm(self) -> float:
        squared = sum(
            float(parameter.detach().float().square().sum())
            for parameter in self.parameters()
        )
        return squared**0.5


def validate_dual_view_cache(
    payload: dict[str, object],
    *,
    expected_feature_dim: int | None = None,
    expected_num_classes: int | None = None,
) -> int:
    """Fail closed on incomplete or misaligned global/local feature caches."""
    missing = DUAL_VIEW_CACHE_KEYS - set(payload)
    if missing:
        raise ValueError(f"Dual-view cache missing keys: {sorted(missing)}")
    paths = list(payload["paths"])
    if not paths or len(paths) != len(set(str(path) for path in paths)):
        raise ValueError("Dual-view cache paths must be non-empty and unique")
    samples = len(paths)
    vectors = {
        "labels": 1,
        "clean_probability": 1,
        "pseudo_labels": 1,
        "correction_alpha": 1,
        "global_features": 2,
        "local_features": 2,
        "global_logits": 2,
    }
    tensors: dict[str, torch.Tensor] = {}
    for name, rank in vectors.items():
        value = torch.as_tensor(payload[name])
        if value.ndim != rank or value.shape[0] != samples:
            raise ValueError(f"Dual-view cache tensor {name} is misaligned")
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"Dual-view cache tensor {name} is non-finite")
        tensors[name] = value
    if tensors["global_features"].shape != tensors["local_features"].shape:
        raise ValueError("Global and local feature shapes differ")
    if (
        expected_feature_dim is not None
        and tensors["local_features"].shape[1] != int(expected_feature_dim)
    ):
        raise ValueError("Dual-view cache feature dimension is unexpected")
    if (
        expected_num_classes is not None
        and tensors["global_logits"].shape[1] != int(expected_num_classes)
    ):
        raise ValueError("Dual-view cache class dimension is unexpected")
    if tensors["labels"].dtype not in {torch.int32, torch.int64}:
        raise ValueError("Dual-view labels must be integer-valued")
    return samples


def residual_prediction_metrics(
    head: LearnedLocalResidualHead,
    base_logits: torch.Tensor,
    local_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic logits and predictions for a cached split."""
    head.eval()
    with torch.no_grad():
        logits = head(base_logits, local_features)
    return logits, logits.argmax(dim=1)
