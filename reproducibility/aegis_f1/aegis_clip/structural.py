"""Closed-form structural classifiers for frozen or adapted CLIP features."""

from __future__ import annotations

import torch


def fit_shrinkage_discriminant(
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    num_classes: int,
    shrinkage: float,
    covariance_batch_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a weighted shared-covariance discriminant as one linear head.

    The returned tensors can directly replace ``nn.Linear.weight`` and
    ``nn.Linear.bias``.  Shrinkage=1 is an isotropic nearest-centroid head;
    lower values retain progressively more within-class covariance structure.
    """

    means, covariance, isotropic_variance = weighted_class_statistics(
        features,
        labels,
        sample_weights,
        num_classes=num_classes,
        covariance_batch_size=covariance_batch_size,
    )
    return discriminant_from_statistics(
        means,
        covariance,
        isotropic_variance,
        shrinkage=shrinkage,
    )


def weighted_class_statistics(
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    num_classes: int,
    covariance_batch_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reusable weighted class means and pooled covariance."""

    if features.ndim != 2:
        raise ValueError("features must have shape [N,D]")
    labels = torch.as_tensor(labels, device=features.device, dtype=torch.long)
    sample_weights = torch.as_tensor(
        sample_weights, device=features.device, dtype=features.dtype
    ).flatten()
    if labels.numel() != features.shape[0] or sample_weights.numel() != features.shape[0]:
        raise ValueError("features, labels, and sample_weights must have equal length")
    if covariance_batch_size <= 0:
        raise ValueError("covariance_batch_size must be positive")
    if labels.numel() == 0 or int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError("labels are empty or outside [0,num_classes)")
    weights = sample_weights.clamp_min(0.0)
    if not torch.isfinite(features).all() or not torch.isfinite(weights).all():
        raise ValueError("features and sample_weights must be finite")

    class_mass = features.new_zeros(num_classes)
    class_mass.index_add_(0, labels, weights)
    if (class_mass <= 0.0).any():
        missing = torch.nonzero(class_mass <= 0.0).flatten().tolist()
        raise ValueError(f"non-positive class mass for classes: {missing[:10]}")
    weighted_sum = features.new_zeros((num_classes, features.shape[1]))
    weighted_sum.index_add_(0, labels, features * weights[:, None])
    means = weighted_sum / class_mass[:, None]

    covariance = features.new_zeros((features.shape[1], features.shape[1]))
    for start in range(0, features.shape[0], covariance_batch_size):
        stop = min(start + covariance_batch_size, features.shape[0])
        residual = features[start:stop] - means[labels[start:stop]]
        residual = residual * weights[start:stop].sqrt()[:, None]
        covariance.addmm_(residual.T, residual)
    denominator = (weights.sum() - num_classes).clamp_min(1.0)
    covariance = covariance / denominator
    isotropic_variance = covariance.diagonal().mean().clamp_min(1.0e-8)
    return means, covariance, isotropic_variance


def discriminant_from_statistics(
    means: torch.Tensor,
    covariance: torch.Tensor,
    isotropic_variance: torch.Tensor,
    *,
    shrinkage: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one shrinkage-discriminant head from reusable statistics."""

    if not 0.0 <= float(shrinkage) <= 1.0:
        raise ValueError("shrinkage must be in [0,1]")
    if means.ndim != 2 or covariance.shape != (means.shape[1], means.shape[1]):
        raise ValueError("means and covariance have incompatible shapes")
    identity = torch.eye(
        covariance.shape[0], device=covariance.device, dtype=covariance.dtype
    )
    covariance = (
        (1.0 - float(shrinkage)) * covariance
        + float(shrinkage) * isotropic_variance * identity
        + 1.0e-6 * isotropic_variance * identity
    )
    weight = torch.linalg.solve(covariance, means.T).T
    bias = -0.5 * (means * weight).sum(dim=1)
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise RuntimeError("structural head fit produced non-finite parameters")
    return weight, bias


def match_linear_logit_scale(
    features: torch.Tensor,
    reference_weight: torch.Tensor,
    reference_bias: torch.Tensor,
    candidate_weight: torch.Tensor,
    candidate_bias: torch.Tensor,
) -> float:
    """Return a constant that matches candidate and reference logit spread."""

    reference = torch.nn.functional.linear(features, reference_weight, reference_bias)
    candidate = torch.nn.functional.linear(features, candidate_weight, candidate_bias)
    reference_scale = reference.std(dim=1, unbiased=False).mean()
    candidate_scale = candidate.std(dim=1, unbiased=False).mean().clamp_min(1.0e-12)
    value = reference_scale / candidate_scale
    if not torch.isfinite(value):
        raise RuntimeError("could not determine a finite structural logit scale")
    return float(value)


def blend_linear_heads(
    reference_weight: torch.Tensor,
    reference_bias: torch.Tensor,
    candidate_weight: torch.Tensor,
    candidate_bias: torch.Tensor,
    *,
    alpha: float,
    candidate_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse a logit blend into one exactly equivalent linear head."""

    if reference_weight.shape != candidate_weight.shape:
        raise ValueError("linear head weights must have identical shapes")
    if reference_bias.shape != candidate_bias.shape:
        raise ValueError("linear head biases must have identical shapes")
    factor = float(alpha) * float(candidate_scale)
    return (
        reference_weight + factor * candidate_weight,
        reference_bias + factor * candidate_bias,
    )


def weighted_ridge_statistics(
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    num_classes: int,
    pseudo_labels: torch.Tensor | None = None,
    correction_alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reusable normal equations for a weighted multiclass ridge head."""

    if features.ndim != 2:
        raise ValueError("features must have shape [N,D]")
    device = features.device
    labels = torch.as_tensor(labels, device=device, dtype=torch.long).flatten()
    weights = torch.as_tensor(
        sample_weights, device=device, dtype=features.dtype
    ).flatten().clamp_min(0.0)
    if labels.numel() != features.shape[0] or weights.numel() != features.shape[0]:
        raise ValueError("features, labels, and sample_weights must have equal length")
    if labels.numel() == 0 or int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError("labels are empty or outside [0,num_classes)")
    if (pseudo_labels is None) != (correction_alpha is None):
        raise ValueError("pseudo_labels and correction_alpha must be provided together")
    if pseudo_labels is None:
        pseudo = labels
        correction = torch.zeros_like(weights)
    else:
        pseudo = torch.as_tensor(pseudo_labels, device=device, dtype=torch.long).flatten()
        correction = torch.as_tensor(
            correction_alpha, device=device, dtype=features.dtype
        ).flatten().clamp(0.0, 1.0)
        if pseudo.numel() != labels.numel() or correction.numel() != labels.numel():
            raise ValueError("pseudo-label vectors must match feature length")
        if int(pseudo.min()) < 0 or int(pseudo.max()) >= num_classes:
            raise ValueError("pseudo_labels are outside [0,num_classes)")
    if not torch.isfinite(features).all() or not torch.isfinite(weights).all():
        raise ValueError("features and sample_weights must be finite")

    design = torch.cat(
        [
            features,
            torch.ones(
                (features.shape[0], 1), device=device, dtype=features.dtype
            ),
        ],
        dim=1,
    )
    weighted_design = design * weights[:, None]
    gram = design.T @ weighted_design
    right = features.new_zeros((num_classes, design.shape[1]))
    right.index_add_(
        0,
        labels,
        weighted_design * (1.0 - correction)[:, None],
    )
    if bool((correction > 0.0).any()):
        right.index_add_(
            0,
            pseudo,
            weighted_design * correction[:, None],
        )
    regularization_scale = gram.diagonal()[:-1].mean().clamp_min(1.0e-8)
    return gram, right.T, regularization_scale


def ridge_head_from_statistics(
    gram: torch.Tensor,
    right: torch.Tensor,
    regularization_scale: torch.Tensor,
    *,
    ridge_strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve one regularized linear head from cached normal equations."""

    if float(ridge_strength) <= 0.0:
        raise ValueError("ridge_strength must be positive")
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    if right.ndim != 2 or right.shape[0] != gram.shape[0]:
        raise ValueError("right-hand side has incompatible shape")
    penalty = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    penalty[-1, -1] = 0.0
    parameters = torch.linalg.solve(
        gram + float(ridge_strength) * regularization_scale * penalty,
        right,
    )
    weight = parameters[:-1].T
    bias = parameters[-1]
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise RuntimeError("ridge head fit produced non-finite parameters")
    return weight, bias
