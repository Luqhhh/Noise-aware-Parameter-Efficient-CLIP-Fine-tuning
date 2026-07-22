"""A capacity-limited adapter used only by the attention-local view."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


LOCAL_ADAPTER_CACHE_KEYS = {
    "paths",
    "labels",
    "clean_probability",
    "pseudo_labels",
    "correction_alpha",
    "global_logits",
    "local_features",
    "local_logits",
    "checkpoint_sha256",
}


class BottleneckLocalFeatureAdapter(nn.Module):
    """Zero-output bottleneck residual that cannot alter the global branch."""

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int,
        *,
        residual_scale: float = 0.25,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0 or int(bottleneck_dim) <= 0:
            raise ValueError("Local adapter dimensions must be positive")
        if not 0.0 < float(residual_scale) <= 1.0:
            raise ValueError("Local adapter residual_scale must be in (0,1]")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("Local adapter dropout must be in [0,1)")
        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.residual_scale = float(residual_scale)
        self.dropout_probability = float(dropout)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.down = nn.Linear(self.feature_dim, self.bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_probability)
        self.up = nn.Linear(self.bottleneck_dim, self.feature_dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, local_features: torch.Tensor) -> torch.Tensor:
        if local_features.ndim != 2:
            raise ValueError("Local features must be rank-2")
        if local_features.shape[1] != self.feature_dim:
            raise ValueError("Local feature dimension does not match the adapter")
        values = local_features.float()
        residual = self.up(
            self.dropout(self.activation(self.down(self.norm(values))))
        )
        # Do not re-normalise here. F1 already emits normalised features and a
        # second normalisation perturbs them by a few ulps even when the
        # zero-initialised residual is exactly zero. Direct addition is what
        # makes O3 epoch zero bit-exact with the scored F1+M1 branch.
        return values + self.residual_scale * residual

    def residual_parameter_norm(self) -> float:
        squared = sum(
            float(parameter.detach().float().square().sum())
            for name, parameter in self.named_parameters()
            if name.startswith("up.")
        )
        return squared**0.5


def fuse_global_local_log_probabilities(
    global_logits: torch.Tensor,
    local_logits: torch.Tensor,
) -> torch.Tensor:
    """Return the exact M1 1:1 probability average in log space."""
    if (
        global_logits.ndim != 2
        or local_logits.ndim != 2
        or global_logits.shape != local_logits.shape
    ):
        raise ValueError("Global and local logits must have equal [N,C] shape")
    return torch.logaddexp(
        F.log_softmax(global_logits.float(), dim=1),
        F.log_softmax(local_logits.float(), dim=1),
    ) - math.log(2.0)


def validate_local_adapter_cache(
    payload: dict[str, Any],
    *,
    expected_feature_dim: int | None = None,
    expected_num_classes: int | None = None,
) -> int:
    """Fail closed on incomplete or misaligned O3 feature caches."""
    missing = LOCAL_ADAPTER_CACHE_KEYS - set(payload)
    if missing:
        raise ValueError(f"Local-adapter cache missing keys: {sorted(missing)}")
    paths = [str(path) for path in payload["paths"]]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Local-adapter cache paths must be non-empty and unique")
    samples = len(paths)
    expected_ranks = {
        "labels": 1,
        "clean_probability": 1,
        "pseudo_labels": 1,
        "correction_alpha": 1,
        "global_logits": 2,
        "local_features": 2,
        "local_logits": 2,
    }
    tensors: dict[str, torch.Tensor] = {}
    for name, rank in expected_ranks.items():
        value = torch.as_tensor(payload[name])
        if value.ndim != rank or value.shape[0] != samples:
            raise ValueError(f"Local-adapter cache tensor {name} is misaligned")
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"Local-adapter cache tensor {name} is non-finite")
        tensors[name] = value
    if tensors["labels"].dtype not in {torch.int32, torch.int64}:
        raise ValueError("Local-adapter labels must be integer-valued")
    if tensors["global_logits"].shape != tensors["local_logits"].shape:
        raise ValueError("Global and local logit shapes differ")
    if (
        expected_feature_dim is not None
        and tensors["local_features"].shape[1] != int(expected_feature_dim)
    ):
        raise ValueError("Local-adapter feature dimension is unexpected")
    if expected_num_classes is not None and (
        tensors["global_logits"].shape[1] != int(expected_num_classes)
    ):
        raise ValueError("Local-adapter class dimension is unexpected")
    clean_probability = tensors["clean_probability"].float()
    if ((clean_probability < 0.0) | (clean_probability > 1.0)).any():
        raise ValueError("Local-adapter clean probabilities must be in [0,1]")
    return samples


def load_local_feature_adapter(
    checkpoint: dict[str, Any], device: torch.device
) -> BottleneckLocalFeatureAdapter:
    payload = checkpoint.get("local_feature_adapter")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint does not contain a local feature adapter")
    spec = payload.get("spec")
    state = payload.get("state_dict")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("Local feature adapter payload is incomplete")
    adapter = BottleneckLocalFeatureAdapter(
        int(spec["feature_dim"]),
        int(spec["bottleneck_dim"]),
        residual_scale=float(spec["residual_scale"]),
        dropout=float(spec["dropout"]),
    ).to(device)
    adapter.load_state_dict(state, strict=True)
    adapter.eval()
    return adapter
