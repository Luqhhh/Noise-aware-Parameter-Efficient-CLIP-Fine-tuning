"""Label-free post-hoc prior correction for balanced evaluation."""

from __future__ import annotations

import math

import torch

from aegis_clip.evaluation import weighted_accuracy, weighted_macro_accuracy


def effective_model_prior(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    minimum_probability: float = 1.0e-8,
) -> torch.Tensor:
    """Estimate the model's learned class prior from posterior predictions."""
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] <= 1:
        raise ValueError("Prior logits must have shape [N,C] with N>0 and C>1")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 < float(minimum_probability) < 1.0:
        raise ValueError("minimum_probability must be in (0,1)")
    probabilities = torch.softmax(logits.float() / float(temperature), dim=1)
    prior = probabilities.mean(dim=0).clamp_min(float(minimum_probability))
    return prior / prior.sum()


def prior_corrected_logits(
    logits: torch.Tensor,
    source_prior: torch.Tensor,
    *,
    target_prior: torch.Tensor | None = None,
    minimum_probability: float = 1.0e-8,
) -> torch.Tensor:
    """Apply the Prior2Posterior Bayes correction in logit space.

    With a uniform target, its additive constant has no effect on argmax but is
    retained to make the requested target distribution explicit and auditable.
    """
    if logits.ndim != 2:
        raise ValueError("Inference logits must be rank-2")
    classes = logits.shape[1]
    source = torch.as_tensor(source_prior, dtype=torch.float32).flatten()
    if source.numel() != classes:
        raise ValueError("source_prior must have one value per class")
    if target_prior is None:
        target = torch.full_like(source, 1.0 / classes)
    else:
        target = torch.as_tensor(target_prior, dtype=torch.float32).flatten()
        if target.numel() != classes:
            raise ValueError("target_prior must have one value per class")
    if (source < 0.0).any() or (target < 0.0).any():
        raise ValueError("Class priors cannot be negative")
    source = source.clamp_min(float(minimum_probability))
    target = target.clamp_min(float(minimum_probability))
    source = source / source.sum()
    target = target / target.sum()
    adjustment = -source.log() + target.log()
    return logits.float() + adjustment.to(logits.device).unsqueeze(0)


def prediction_metrics(
    prediction: torch.Tensor,
    *,
    labels: torch.Tensor,
    clean_probability: torch.Tensor,
    pseudo_labels: torch.Tensor,
    correction_alpha: torch.Tensor,
    num_classes: int,
    clean_core_threshold: float,
) -> dict[str, float | int]:
    prediction = torch.as_tensor(prediction).long().flatten().cpu()
    labels = torch.as_tensor(labels).long().flatten().cpu()
    clean = torch.as_tensor(clean_probability).float().flatten().cpu()
    pseudo = torch.as_tensor(pseudo_labels).long().flatten().cpu()
    correction = torch.as_tensor(correction_alpha).float().flatten().cpu()
    if not (
        prediction.numel()
        == labels.numel()
        == clean.numel()
        == pseudo.numel()
        == correction.numel()
    ):
        raise ValueError("Prediction metric inputs must have equal length")
    proxy = torch.where(correction > 0.0, pseudo, labels)
    proxy_weight = torch.maximum(clean, correction)
    clean_core = (clean >= float(clean_core_threshold)).float()
    counts = torch.bincount(prediction, minlength=num_classes).float()
    return {
        "samples": int(prediction.numel()),
        "raw_micro": float((prediction == labels).float().mean()),
        "raw_macro": weighted_macro_accuracy(
            prediction, labels, torch.ones_like(clean), num_classes
        ),
        "trusted_micro": weighted_accuracy(prediction, labels, clean),
        "trusted_macro": weighted_macro_accuracy(
            prediction, labels, clean, num_classes
        ),
        "proxy_micro": weighted_accuracy(prediction, proxy, proxy_weight),
        "proxy_macro": weighted_macro_accuracy(
            prediction, proxy, proxy_weight, num_classes
        ),
        "clean_core_micro": weighted_accuracy(
            prediction, labels, clean_core
        ),
        "clean_core_macro": weighted_macro_accuracy(
            prediction, labels, clean_core, num_classes
        ),
        "clean_core_samples": int(clean_core.sum()),
        "prediction_count_min": int(counts.min()),
        "prediction_count_max": int(counts.max()),
        "prediction_empty_classes": int((counts == 0).sum()),
        "prediction_count_cv": float(
            counts.std(unbiased=False) / counts.mean().clamp_min(1.0e-12)
        ),
    }


def prior_diagnostics(prior: torch.Tensor) -> dict[str, float]:
    values = torch.as_tensor(prior).float().flatten()
    if values.numel() <= 1 or (values <= 0.0).any():
        raise ValueError("Prior diagnostics require a positive class distribution")
    values = values / values.sum()
    entropy = float(-(values * values.log()).sum())
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "max_min_ratio": float(values.max() / values.min()),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(values.numel()),
        "coefficient_of_variation": float(
            values.std(unbiased=False) / values.mean()
        ),
    }
