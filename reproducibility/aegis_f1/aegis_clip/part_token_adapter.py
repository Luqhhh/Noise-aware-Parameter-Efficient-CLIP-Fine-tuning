"""Auditable part-token residual adaptation for the fixed M1 local view."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aegis_clip.local_feature_adapter import (
    LOCAL_ADAPTER_CACHE_KEYS,
    validate_local_adapter_cache,
)


PART_TOKEN_CACHE_KEYS = LOCAL_ADAPTER_CACHE_KEYS | {
    "part_features",
    "part_pool_spec",
}
PART_POOL_METHOD = "cls_cosine_topk_v1"


def anchored_classifier_residual_logits(
    base_logits: torch.Tensor,
    base_features: torch.Tensor,
    adapted_features: torch.Tensor,
    classifier_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply one shared linear head as an exact logit residual around a base."""
    if (
        base_logits.ndim != 2
        or base_features.ndim != 2
        or adapted_features.ndim != 2
        or classifier_weight.ndim != 2
    ):
        raise ValueError("Classifier residual inputs must be rank-2")
    if base_features.shape != adapted_features.shape:
        raise ValueError("Base and adapted feature shapes must match")
    if (
        base_logits.shape[0] != base_features.shape[0]
        or base_logits.shape[1] != classifier_weight.shape[0]
        or base_features.shape[1] != classifier_weight.shape[1]
    ):
        raise ValueError("Classifier residual dimensions do not align")
    if not (
        base_logits.device
        == base_features.device
        == adapted_features.device
        == classifier_weight.device
    ):
        raise ValueError("Classifier residual tensors must share one device")
    with torch.autocast(device_type=base_features.device.type, enabled=False):
        residual_logits = F.linear(
            (adapted_features - base_features).float(),
            classifier_weight.float(),
            None,
        )
    return base_logits.float() + residual_logits


def pool_cls_aligned_patch_features(
    local_features: torch.Tensor,
    patch_features: torch.Tensor,
    *,
    top_patches: int = 8,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Pool the patch tokens most aligned with the same-view CLIP CLS feature.

    Sorting is stable so equal similarities are resolved by patch order instead
    of an implementation-dependent top-k tie break.  The operation is fixed and
    has no class semantics, learned parameters, or external information.
    """
    if local_features.ndim != 2 or patch_features.ndim != 3:
        raise ValueError("Expected local [N,D] and patch [N,P,D] features")
    if (
        local_features.shape[0] != patch_features.shape[0]
        or local_features.shape[1] != patch_features.shape[2]
    ):
        raise ValueError("Local and patch feature dimensions do not align")
    patch_count = int(patch_features.shape[1])
    if not 1 <= int(top_patches) <= patch_count:
        raise ValueError("top_patches is out of range")
    if not float(temperature) > 0.0:
        raise ValueError("temperature must be positive")
    local = F.normalize(local_features.float(), dim=1)
    patches = F.normalize(patch_features.float(), dim=2)
    similarities = torch.einsum("nd,npd->np", local, patches)
    indices = torch.argsort(
        similarities,
        dim=1,
        descending=True,
        stable=True,
    )[:, : int(top_patches)]
    selected = torch.gather(
        patches,
        1,
        indices[:, :, None].expand(-1, -1, patches.shape[2]),
    )
    selected_similarities = torch.gather(similarities, 1, indices)
    weights = F.softmax(selected_similarities / float(temperature), dim=1)
    pooled = (selected * weights[:, :, None]).sum(dim=1)
    return F.normalize(pooled, dim=1)


class PartTokenResidualAdapter(nn.Module):
    """Zero-output residual driven only by part-token/CLS disagreement."""

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
            raise ValueError("Part-token adapter dimensions must be positive")
        if not 0.0 < float(residual_scale) <= 1.0:
            raise ValueError("residual_scale must be in (0,1]")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
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

    def forward(
        self,
        local_features: torch.Tensor,
        part_features: torch.Tensor,
    ) -> torch.Tensor:
        if local_features.ndim != 2 or part_features.ndim != 2:
            raise ValueError("Local and part features must be rank-2")
        if local_features.shape != part_features.shape:
            raise ValueError("Local and part feature shapes must match")
        if local_features.shape[1] != self.feature_dim:
            raise ValueError("Feature dimension does not match the adapter")
        local = local_features.float()
        part_delta = part_features.float() - local
        residual = self.up(
            self.dropout(self.activation(self.down(self.norm(part_delta))))
        )
        # As in O3, avoid a second normalisation so the zero-output adapter is
        # bit-exact with the platform-scored F1+M1 local branch.
        return local + self.residual_scale * residual

    def residual_parameter_norm(self) -> float:
        squared = sum(
            float(parameter.detach().float().square().sum())
            for name, parameter in self.named_parameters()
            if name.startswith("up.")
        )
        return squared**0.5


def validate_part_token_cache(
    payload: dict[str, Any],
    *,
    expected_feature_dim: int | None = None,
    expected_num_classes: int | None = None,
) -> int:
    """Fail closed on incomplete, non-finite, or misaligned R1 caches."""
    missing = PART_TOKEN_CACHE_KEYS - set(payload)
    if missing:
        raise ValueError(f"Part-token cache missing keys: {sorted(missing)}")
    samples = validate_local_adapter_cache(
        payload,
        expected_feature_dim=expected_feature_dim,
        expected_num_classes=expected_num_classes,
    )
    local = torch.as_tensor(payload["local_features"])
    part = torch.as_tensor(payload["part_features"])
    if part.ndim != 2 or part.shape != local.shape or part.shape[0] != samples:
        raise ValueError("Part-token cache features are misaligned")
    if not torch.isfinite(part.float()).all():
        raise ValueError("Part-token cache features are non-finite")
    if (part.float().norm(dim=1) <= 0.0).any():
        raise ValueError("Part-token cache contains a zero feature")
    spec = payload["part_pool_spec"]
    if not isinstance(spec, dict):
        raise ValueError("Part-token cache pool specification is invalid")
    if spec.get("method") != PART_POOL_METHOD:
        raise ValueError("Part-token cache pool method is unsupported")
    top_patches = int(spec.get("top_patches", 0))
    temperature = float(spec.get("temperature", 0.0))
    if top_patches <= 0 or temperature <= 0.0:
        raise ValueError("Part-token cache pool parameters are invalid")
    return samples


def load_part_token_adapter(
    checkpoint: dict[str, Any], device: torch.device
) -> PartTokenResidualAdapter:
    payload = checkpoint.get("part_token_adapter")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint does not contain a part-token adapter")
    spec = payload.get("spec")
    state = payload.get("state_dict")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("Part-token adapter payload is incomplete")
    pool_spec = spec.get("part_pool_spec")
    if not isinstance(pool_spec, dict):
        raise ValueError("Part-token adapter pool specification is incomplete")
    if (
        pool_spec.get("method") != PART_POOL_METHOD
        or int(pool_spec.get("top_patches", 0)) != 8
        or float(pool_spec.get("temperature", 0.0)) != 0.07
    ):
        raise ValueError("Part-token adapter pool specification is not preregistered")
    adapter = PartTokenResidualAdapter(
        int(spec["feature_dim"]),
        int(spec["bottleneck_dim"]),
        residual_scale=float(spec["residual_scale"]),
        dropout=float(spec["dropout"]),
    ).to(device)
    adapter.load_state_dict(state, strict=True)
    adapter.eval()
    return adapter
