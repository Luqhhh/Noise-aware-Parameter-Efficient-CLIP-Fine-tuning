"""Auditable single-model post-hoc heads and paired-view TTA utilities.

The helpers in this module deliberately fail closed.  A linear-head soup is
accepted only when every non-classifier tensor is bit-identical, so the result
remains one backbone and one checkpoint rather than a hidden ensemble.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


TTA_FUSION_MODES = {
    "mean_logits",
    "mean_probabilities",
    "entropy_weighted_probabilities",
    "standardized_logits",
    "max_margin",
}


def assert_non_classifier_state_equal(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> None:
    """Reject a head soup unless all non-classifier state is identical."""

    if set(first) != set(second):
        raise ValueError("checkpoints have different model state keys")
    exempt = {"classifier.weight", "classifier.bias"}
    for name in sorted(set(first) - exempt):
        if not torch.equal(first[name].detach().cpu(), second[name].detach().cpu()):
            raise ValueError(f"non-classifier state differs: {name}")


def interpolate_linear_heads(
    first_weight: torch.Tensor,
    first_bias: torch.Tensor,
    second_weight: torch.Tensor,
    second_bias: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate two aligned linear heads into one deployable head."""

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if first_weight.shape != second_weight.shape:
        raise ValueError("classifier weights have different shapes")
    if first_bias.shape != second_bias.shape:
        raise ValueError("classifier biases have different shapes")
    return (
        torch.lerp(first_weight, second_weight, float(alpha)),
        torch.lerp(first_bias, second_bias, float(alpha)),
    )


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
    """Fit K normalized visual modes per class with deterministic K-means."""

    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    if prototypes_per_class <= 0:
        raise ValueError("prototypes_per_class must be positive")
    labels = torch.as_tensor(labels, dtype=torch.long).cpu().flatten()
    weights = torch.as_tensor(sample_weights, dtype=torch.float32).cpu().flatten()
    features = F.normalize(features.detach().float().cpu(), dim=1)
    if labels.numel() != features.shape[0] or weights.numel() != features.shape[0]:
        raise ValueError("features, labels, and sample_weights must have equal length")
    if labels.numel() == 0 or int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError("labels are empty or outside [0, num_classes)")
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
    """Score each class by its nearest or smoothly pooled visual mode."""

    if features.ndim != 2 or prototypes.ndim != 3:
        raise ValueError("expected features [N, D] and prototypes [C, K, D]")
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
    """Match mean per-example score spread before a residual blend."""

    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate scores must have equal [N, C] shape")
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
    """Add a checkpoint-embedded multi-prototype residual to base logits."""

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


def fuse_paired_logits(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    mode: str = "mean_logits",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Fuse two same-model views into logits or normalized log-probabilities."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("paired logits must have identical [N, C] shapes")
    if mode not in TTA_FUSION_MODES:
        raise ValueError(f"unsupported TTA fusion mode: {mode}")
    if not float(temperature) > 0.0:
        raise ValueError("TTA temperature must be positive")
    if mode == "mean_logits":
        return (first + second) / 2.0
    if mode == "standardized_logits":
        return (_standardize(first) + _standardize(second)) / 2.0
    if mode == "max_margin":
        first_margin = first.topk(2, dim=1).values.diff(dim=1).abs().squeeze(1)
        second_margin = second.topk(2, dim=1).values.diff(dim=1).abs().squeeze(1)
        return torch.where((first_margin >= second_margin)[:, None], first, second)

    first_probability = F.softmax(first / float(temperature), dim=1)
    second_probability = F.softmax(second / float(temperature), dim=1)
    if mode == "mean_probabilities":
        probability = (first_probability + second_probability) / 2.0
    else:
        normalizer = math.log(first.shape[1])
        first_reliability = 1.0 - (
            -(first_probability * first_probability.clamp_min(1.0e-12).log()).sum(1)
            / normalizer
        )
        second_reliability = 1.0 - (
            -(second_probability * second_probability.clamp_min(1.0e-12).log()).sum(1)
            / normalizer
        )
        first_reliability = first_reliability.clamp_min(1.0e-6)
        second_reliability = second_reliability.clamp_min(1.0e-6)
        probability = (
            first_probability * first_reliability[:, None]
            + second_probability * second_reliability[:, None]
        ) / (first_reliability + second_reliability)[:, None]
    return probability.clamp_min(1.0e-12).log()


def _standardize(logits: torch.Tensor) -> torch.Tensor:
    centered = logits - logits.mean(dim=1, keepdim=True)
    return centered / centered.std(
        dim=1, unbiased=False, keepdim=True
    ).clamp_min(1.0e-6)
