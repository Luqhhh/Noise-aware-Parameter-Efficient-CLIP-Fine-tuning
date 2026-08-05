"""Recover and reweight nested deterministic multiscale probability fusions."""

from __future__ import annotations

import re
from typing import Any, Sequence

import torch


def parse_shared_top_k(inference_modes: Sequence[str]) -> int:
    """Extract one identical attention top-k value from nested dump modes."""
    if not inference_modes:
        raise ValueError("At least one inference mode is required")
    values: list[int] = []
    for mode in inference_modes:
        matches = re.findall(r"(?:^|:)topk=(\d+)(?::|$)", str(mode))
        if len(matches) != 1:
            raise ValueError(
                f"Inference mode must contain exactly one topk value: {mode!r}"
            )
        value = int(matches[0])
        if value <= 0:
            raise ValueError("Attention top-k must be positive")
        values.append(value)
    if len(set(values)) != 1:
        raise ValueError(f"Nested dumps use different top-k values: {values}")
    return values[0]


def parse_shared_local_adapter(inference_modes: Sequence[str]) -> str | None:
    """Require nested dumps to use the same optional local adapter marker."""
    if not inference_modes:
        raise ValueError("At least one inference mode is required")
    values: list[str | None] = []
    for mode in inference_modes:
        matches = re.findall(r"(?:^|:)adapter=([a-zA-Z0-9_-]+)(?::|$)", str(mode))
        if len(matches) > 1:
            raise ValueError(
                f"Inference mode contains multiple adapter markers: {mode!r}"
            )
        values.append(matches[0] if matches else None)
    if len(set(values)) != 1:
        raise ValueError(f"Nested dumps use different local adapters: {values}")
    return values[0]


def parse_scale_weights(
    scales_value: str,
    weights_value: str,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Parse unique scales and a normalized non-negative weight vector."""
    scales = tuple(
        int(value.strip()) for value in scales_value.split(",") if value.strip()
    )
    weights = tuple(
        float(value.strip()) for value in weights_value.split(",") if value.strip()
    )
    if len(scales) != 3:
        raise ValueError("Exactly three nested scales are required")
    if len(set(scales)) != len(scales):
        raise ValueError("Scale values must be unique")
    if tuple(sorted(scales)) != scales:
        raise ValueError("Scale values must be strictly increasing")
    if len(weights) != len(scales):
        raise ValueError("Scale and weight counts must match")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("Scale weights must be non-negative")
    total = sum(weights)
    if abs(total - 1.0) > 1.0e-8:
        raise ValueError("Scale weights must sum to one")
    return scales, weights


def _log_probabilities_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(logits).detach().float().cpu()
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] <= 1:
        raise ValueError("Each fused-logits tensor must have shape [N,C]")
    if not torch.isfinite(values).all():
        raise ValueError("Fused logits must be finite")
    probabilities = values.exp()
    normalization_error = float((probabilities.sum(dim=1) - 1.0).abs().max())
    if normalization_error > 1.0e-4:
        raise ValueError(
            "Fused logits are not normalized log probabilities: "
            f"maximum error {normalization_error:.6g}"
        )
    return probabilities / probabilities.sum(dim=1, keepdim=True)


def reconstruct_nested_scale_probabilities(
    triple_logits: torch.Tensor,
    pair_logits: torch.Tensor,
    single_logits: torch.Tensor,
    *,
    negative_tolerance: float = 1.0e-6,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, Any]]:
    """Recover small/middle/large scale fusions from nested equal means.

    The inputs must respectively be the equal probability means for
    ``(small,middle,large)``, ``(middle,large)``, and ``(large,)``.  Since each
    fused tensor already contains the same global branch, these linear
    identities recover the three single-scale global/local fusions exactly.
    """
    if negative_tolerance < 0.0:
        raise ValueError("negative_tolerance must be non-negative")
    triple = _log_probabilities_to_probabilities(triple_logits)
    pair = _log_probabilities_to_probabilities(pair_logits)
    large = _log_probabilities_to_probabilities(single_logits)
    if triple.shape != pair.shape or triple.shape != large.shape:
        raise ValueError("Nested fused-logits tensors must have identical shapes")

    middle = 2.0 * pair - large
    small = 3.0 * triple - 2.0 * pair
    raw = (small, middle, large)
    minima = [float(probabilities.min()) for probabilities in raw]
    if min(minima) < -float(negative_tolerance):
        raise ValueError(
            "Nested fusion reconstruction produced a materially negative "
            f"probability: {min(minima):.6g}"
        )

    tiny = torch.finfo(triple.dtype).tiny
    recovered = tuple(
        probabilities.clamp_min(tiny)
        / probabilities.clamp_min(tiny).sum(dim=1, keepdim=True)
        for probabilities in raw
    )
    report = {
        "minimum_reconstructed_probabilities": minima,
        "maximum_normalization_errors": [
            float((probabilities.sum(dim=1) - 1.0).abs().max())
            for probabilities in recovered
        ],
        "shape": list(triple.shape),
    }
    return recovered, report


def weighted_scale_probabilities(
    probabilities: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> torch.Tensor:
    """Return one normalized convex combination of scale probabilities."""
    if len(probabilities) != len(weights) or not probabilities:
        raise ValueError("Probability and weight counts must match and be non-empty")
    reference = probabilities[0]
    if any(value.shape != reference.shape for value in probabilities[1:]):
        raise ValueError("Scale probability tensors must have identical shapes")
    if any(float(weight) < 0.0 for weight in weights):
        raise ValueError("Scale weights must be non-negative")
    total = sum(float(weight) for weight in weights)
    if abs(total - 1.0) > 1.0e-8:
        raise ValueError("Scale weights must sum to one")
    fused = sum(float(weight) * value for weight, value in zip(weights, probabilities))
    tiny = torch.finfo(fused.dtype).tiny
    fused = fused.clamp_min(tiny)
    return fused / fused.sum(dim=1, keepdim=True)
