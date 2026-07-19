"""Multi-modal class prototypes for frozen CLIP representations."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def fit_weighted_multiprototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    num_classes: int,
    prototypes_per_class: int,
    random_state: int = 42,
    max_iter: int = 50,
) -> torch.Tensor:
    """Fit several normalized visual modes for every class."""

    if features.ndim != 2:
        raise ValueError("features must have shape [N,D]")
    if prototypes_per_class <= 0:
        raise ValueError("prototypes_per_class must be positive")
    labels = torch.as_tensor(labels, dtype=torch.long).cpu().flatten()
    weights = torch.as_tensor(sample_weights, dtype=torch.float32).cpu().flatten()
    features = F.normalize(features.detach().float().cpu(), dim=1)
    if labels.numel() != features.shape[0] or weights.numel() != features.shape[0]:
        raise ValueError("features, labels, and sample_weights must have equal length")
    if labels.numel() == 0 or int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError("labels are empty or outside [0,num_classes)")
    if not torch.isfinite(features).all() or not torch.isfinite(weights).all():
        raise ValueError("features and sample_weights must be finite")

    class_centers = []
    for class_index in range(num_classes):
        mask = labels == class_index
        class_features = features[mask]
        class_weights = weights[mask].clamp_min(0.0)
        if class_features.shape[0] < prototypes_per_class:
            raise ValueError(
                f"class {class_index} has fewer examples than requested prototypes"
            )
        if float(class_weights.sum()) <= 0.0:
            raise ValueError(f"class {class_index} has non-positive weight mass")
        if prototypes_per_class == 1:
            centers = (
                class_features * class_weights[:, None]
            ).sum(0, keepdim=True) / class_weights.sum()
        else:
            estimator = KMeans(
                n_clusters=prototypes_per_class,
                n_init=1,
                max_iter=max_iter,
                random_state=random_state + class_index,
            )
            estimator.fit(
                class_features.numpy(),
                sample_weight=class_weights.numpy(),
            )
            centers = torch.from_numpy(
                np.asarray(estimator.cluster_centers_, dtype=np.float32)
            )
        class_centers.append(F.normalize(centers, dim=1))
    return torch.stack(class_centers)


def multiprototype_logits(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    aggregation: str,
    softmax_temperature: float = 0.05,
) -> torch.Tensor:
    """Score each class by its closest or smoothly pooled visual mode."""

    if features.ndim != 2 or prototypes.ndim != 3:
        raise ValueError("expected features [N,D] and prototypes [C,K,D]")
    if features.shape[1] != prototypes.shape[2]:
        raise ValueError("feature and prototype dimensions do not match")
    queries = F.normalize(features.float(), dim=1)
    centers = F.normalize(prototypes.float(), dim=2)
    similarities = queries @ centers.flatten(0, 1).T
    similarities = similarities.reshape(
        features.shape[0], prototypes.shape[0], prototypes.shape[1]
    )
    if aggregation == "max":
        return similarities.max(dim=2).values
    if aggregation == "logmeanexp":
        if softmax_temperature <= 0.0:
            raise ValueError("softmax_temperature must be positive")
        return softmax_temperature * (
            torch.logsumexp(similarities / softmax_temperature, dim=2)
            - math.log(prototypes.shape[1])
        )
    raise ValueError(f"unsupported prototype aggregation: {aggregation}")


def match_score_scale(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Match mean per-example class-score spread for a stable logit blend."""

    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate scores must have equal [N,C] shape")
    numerator = reference.std(dim=1, unbiased=False).mean()
    denominator = candidate.std(dim=1, unbiased=False).mean().clamp_min(1.0e-12)
    value = numerator / denominator
    if not torch.isfinite(value):
        raise RuntimeError("could not determine a finite score scale")
    return float(value)


def blend_multiprototype_logits(
    base_logits: torch.Tensor,
    features: torch.Tensor,
    head: dict,
) -> torch.Tensor:
    """Apply a checkpoint-embedded prototype residual to one view's logits."""

    required = {"prototypes", "aggregation", "alpha", "candidate_scale"}
    missing = required - set(head)
    if missing:
        raise ValueError(f"multiprototype head is missing fields: {sorted(missing)}")
    prototypes = torch.as_tensor(
        head["prototypes"], device=features.device, dtype=torch.float32
    )
    candidate = multiprototype_logits(
        features,
        prototypes,
        aggregation=str(head["aggregation"]),
        softmax_temperature=float(head.get("softmax_temperature", 0.05)),
    ).to(dtype=base_logits.dtype)
    return base_logits + (
        float(head["alpha"]) * float(head["candidate_scale"]) * candidate
    )


def paired_top1_changes(
    reference_scores: torch.Tensor,
    candidate_scores: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, int]:
    """Count paired top-1 changes, including corrected and broken decisions."""

    if reference_scores.shape != candidate_scores.shape or reference_scores.ndim != 2:
        raise ValueError("score tensors must have identical [N,C] shapes")
    labels = torch.as_tensor(labels, device=reference_scores.device).long().flatten()
    if labels.numel() != reference_scores.shape[0]:
        raise ValueError("labels must match the number of score rows")
    reference = reference_scores.argmax(dim=1)
    candidate = candidate_scores.argmax(dim=1)
    fixed = (reference != labels) & (candidate == labels)
    broken = (reference == labels) & (candidate != labels)
    return {
        "changed_predictions": int((reference != candidate).sum()),
        "raw_fixed": int(fixed.sum()),
        "raw_broken": int(broken.sum()),
        "raw_net_fixed": int(fixed.sum() - broken.sum()),
    }
