"""Deterministic same-model fusion rules for paired test-time views."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


TTA_FUSION_MODES = {
    "mean_logits",
    "mean_probabilities",
    "entropy_weighted_probabilities",
    "standardized_logits",
    "max_margin",
}


def fuse_paired_logits(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    mode: str = "mean_logits",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Fuse two views and return logits or normalized log-probabilities."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("paired logits must have identical [N,C] shapes")
    if mode not in TTA_FUSION_MODES:
        raise ValueError(f"Unsupported TTA fusion mode: {mode}")
    if not float(temperature) > 0.0:
        raise ValueError("TTA temperature must be positive")
    if mode == "mean_logits":
        return (first + second) / 2.0
    if mode == "standardized_logits":
        first = _standardize(first)
        second = _standardize(second)
        return (first + second) / 2.0
    if mode == "max_margin":
        first_margin = first.topk(2, dim=1).values.diff(dim=1).abs().squeeze(1)
        second_margin = second.topk(2, dim=1).values.diff(dim=1).abs().squeeze(1)
        choose_first = first_margin >= second_margin
        return torch.where(choose_first[:, None], first, second)

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
    return centered / centered.std(dim=1, unbiased=False, keepdim=True).clamp_min(
        1.0e-6
    )
