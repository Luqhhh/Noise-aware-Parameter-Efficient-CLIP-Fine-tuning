"""Strict single-checkpoint interpolation for compatible linear heads."""

from __future__ import annotations

import torch


def interpolate_linear_heads(
    first_weight: torch.Tensor,
    first_bias: torch.Tensor,
    second_weight: torch.Tensor,
    second_bias: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate two aligned heads; exactly equivalent to a logit blend."""

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if first_weight.shape != second_weight.shape:
        raise ValueError("classifier weights have different shapes")
    if first_bias.shape != second_bias.shape:
        raise ValueError("classifier biases have different shapes")
    return (
        torch.lerp(first_weight, second_weight, float(alpha)),
        torch.lerp(first_bias, second_bias, float(alpha)),
    )


def assert_non_classifier_state_equal(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> None:
    """Reject interpolation when it would hide a multi-backbone ensemble."""

    if set(first) != set(second):
        raise ValueError("checkpoints have different model state keys")
    exempt = {"classifier.weight", "classifier.bias"}
    for name in sorted(set(first) - exempt):
        if not torch.equal(first[name].cpu(), second[name].cpu()):
            raise ValueError(f"non-classifier state differs: {name}")
