"""Trust-weighted class prototypes for a frozen attention-local feature view."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def trust_weighted_local_prototype_weight(
    local_features: torch.Tensor,
    labels: torch.Tensor,
    clean_probability: torch.Tensor,
    base_classifier_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.as_tensor(local_features).float()
    target = torch.as_tensor(labels).long().flatten()
    trust = torch.as_tensor(clean_probability).float().flatten()
    base_weight = torch.as_tensor(base_classifier_weight).float()
    if features.ndim != 2 or base_weight.ndim != 2:
        raise ValueError("Prototype features and base weight must be rank-2")
    if not (features.shape[0] == target.numel() == trust.numel()):
        raise ValueError("Prototype training tensors are misaligned")
    if features.shape[1] != base_weight.shape[1]:
        raise ValueError("Prototype and classifier feature dimensions differ")
    if (target < 0).any() or (target >= base_weight.shape[0]).any():
        raise ValueError("Prototype labels are outside the classifier range")
    if not torch.isfinite(features).all() or not torch.isfinite(trust).all():
        raise ValueError("Prototype inputs contain non-finite values")
    if (trust <= 0.0).any():
        raise ValueError("Prototype trust weights must be positive")
    num_classes, feature_dim = base_weight.shape
    sums = torch.zeros(num_classes, feature_dim, dtype=torch.float32)
    total_weight = torch.zeros(num_classes, dtype=torch.float32)
    sums.index_add_(0, target, features * trust.unsqueeze(1))
    total_weight.index_add_(0, target, trust)
    if (total_weight <= 0.0).any():
        missing = torch.nonzero(total_weight <= 0.0).flatten().tolist()
        raise ValueError(f"Prototype classes have no samples: {missing[:5]}")
    centroids = F.normalize(sums / total_weight.unsqueeze(1), dim=1)
    centroid_norm = centroids.norm(dim=1)
    if not torch.allclose(
        centroid_norm, torch.ones_like(centroid_norm), atol=1.0e-5, rtol=0.0
    ):
        raise RuntimeError("Prototype centroids are not unit-normalized")
    scaled_weight = centroids * base_weight.norm(dim=1, keepdim=True)
    return scaled_weight, total_weight


def local_prototype_logits(
    local_features: torch.Tensor,
    prototype_weight: torch.Tensor,
    base_classifier_bias: torch.Tensor,
) -> torch.Tensor:
    features = torch.as_tensor(local_features).float()
    weight = torch.as_tensor(prototype_weight).float()
    bias = torch.as_tensor(base_classifier_bias).float().flatten()
    if features.ndim != 2 or weight.ndim != 2:
        raise ValueError("Local features and prototype weight must be rank-2")
    if features.shape[1] != weight.shape[1] or bias.numel() != weight.shape[0]:
        raise ValueError("Local prototype classifier dimensions do not align")
    return F.linear(features, weight, bias)


def mean_global_prototype_logits(
    global_logits: torch.Tensor, prototype_logits: torch.Tensor
) -> torch.Tensor:
    global_values = torch.as_tensor(global_logits).float()
    prototype_values = torch.as_tensor(prototype_logits).float()
    if global_values.ndim != 2 or global_values.shape != prototype_values.shape:
        raise ValueError("Global and local prototype logits must have equal shape")
    return (global_values + prototype_values) / 2.0
