"""Stage-agnostic long-tail training utilities.

The preliminary stage did not emphasize class imbalance, while later stages
declare an explicit long-tail training distribution and a roughly balanced
test set.  These helpers keep every long-tail lever config-driven so the same
training entry point can be reused across stages without re-fitting any
preliminary-stage parameters.

Available levers:

* ``sampler_mode``: none / class_balanced / sqrt_class_balanced /
  balanced_oversample.
* ``loss_reweighting``: none / inverse_frequency / sqrt_inverse_frequency /
  effective_number (Cui et al., CVPR 2019 class-balanced loss).
* ``balanced_softmax_tau``: training-only log-prior additive adjustment
  (Ren et al., NeurIPS 2020 balanced softmax).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.utils.data import WeightedRandomSampler


SAMPLER_MODES = {
    "none",
    "class_balanced",
    "sqrt_class_balanced",
    "balanced_oversample",
}
REWEIGHT_MODES = {
    "none",
    "inverse_frequency",
    "sqrt_inverse_frequency",
    "effective_number",
}
WEIGHT_MODES = REWEIGHT_MODES | {"class_balanced", "sqrt_class_balanced"}


def class_counts_from_labels(
    labels: Sequence[int], num_classes: int
) -> torch.Tensor:
    """Return a strictly positive per-class count vector for a training split."""
    counts = torch.bincount(
        torch.as_tensor(list(labels), dtype=torch.long), minlength=int(num_classes)
    ).float()
    if counts.numel() != num_classes or int((counts <= 0).sum()):
        missing = torch.nonzero(counts <= 0).flatten().tolist()
        raise ValueError(f"Training labels miss classes: {missing[:10]}")
    return counts


def per_class_weights(
    counts: torch.Tensor,
    mode: str,
    *,
    effective_number_beta: float = 0.9999,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute per-class weights for the requested long-tail mode."""
    counts = torch.as_tensor(counts, dtype=torch.float32).flatten()
    if counts.numel() == 0 or (counts <= 0).any():
        raise ValueError("class counts must be non-empty and strictly positive")
    if mode not in WEIGHT_MODES:
        raise ValueError(f"Unknown weighting mode: {mode!r}")
    if mode == "none":
        weights = torch.ones_like(counts)
    elif mode in {"class_balanced", "inverse_frequency"}:
        weights = 1.0 / counts
    elif mode in {"sqrt_class_balanced", "sqrt_inverse_frequency"}:
        weights = 1.0 / counts.sqrt()
    elif mode == "effective_number":
        if not 0.0 < float(effective_number_beta) < 1.0:
            raise ValueError("effective_number_beta must be in (0,1)")
        beta = float(effective_number_beta)
        weights = (1.0 - beta) / (1.0 - beta.pow(counts))
    else:
        raise ValueError(f"Unknown weighting mode: {mode!r}")
    if normalize:
        weights = weights * (weights.numel() / weights.sum().clamp_min(1.0e-12))
    return weights


def per_sample_weights(
    labels: Sequence[int],
    counts: torch.Tensor,
    mode: str,
    *,
    effective_number_beta: float = 0.9999,
    normalize: bool = False,
) -> torch.Tensor:
    """Expand per-class weights to one weight per training sample.

    When ``normalize`` is true the expanded sample weights are rescaled to a
    mean of one, preserving the overall loss magnitude under any class size
    distribution.
    """
    labels_tensor = torch.as_tensor(list(labels), dtype=torch.long)
    weights = per_class_weights(
        counts,
        mode,
        effective_number_beta=effective_number_beta,
        normalize=False,
    )
    sample_weights = weights[labels_tensor]
    if normalize:
        sample_weights = sample_weights * (
            sample_weights.numel() / sample_weights.sum().clamp_min(1.0e-12)
        )
    return sample_weights


def build_sampler(
    labels: Sequence[int],
    counts: torch.Tensor,
    mode: str,
    num_classes: int,
    generator: torch.Generator | None = None,
) -> torch.utils.data.Sampler | None:
    """Build a long-tail training sampler, or ``None`` for plain shuffling."""
    if mode == "none":
        return None
    if mode not in SAMPLER_MODES:
        raise ValueError(f"Unknown sampler mode: {mode!r}")
    labels_tensor = torch.as_tensor(list(labels), dtype=torch.long)
    counts = torch.as_tensor(counts, dtype=torch.float32).flatten()
    if counts.numel() != num_classes or (counts <= 0).any():
        raise ValueError("counts must cover every class exactly once")
    if mode == "balanced_oversample":
        weights = 1.0 / (float(num_classes) * counts[labels_tensor])
        num_samples = int(num_classes * counts.max().long())
    elif mode == "class_balanced":
        weights = 1.0 / counts[labels_tensor]
        num_samples = len(labels_tensor)
    elif mode == "sqrt_class_balanced":
        weights = 1.0 / counts[labels_tensor].sqrt()
        num_samples = len(labels_tensor)
    else:
        raise ValueError(f"Unknown sampler mode: {mode!r}")
    return WeightedRandomSampler(
        weights=weights.double().tolist(),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


def resolve_longtail_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize the optional ``longtail`` config section with strict checks.

    ``balanced_softmax_tau`` falls back to the legacy
    ``loss.class_prior_adjustment_tau`` value so existing configs keep their
    exact behavior.
    """
    section = config.get("longtail", {}) or {}
    sampler_mode = str(section.get("sampler_mode", "none"))
    reweight_mode = str(section.get("loss_reweighting", "none"))
    if sampler_mode not in SAMPLER_MODES:
        raise ValueError(f"longtail.sampler_mode must be one of {sorted(SAMPLER_MODES)}")
    if reweight_mode not in REWEIGHT_MODES:
        raise ValueError(
            f"longtail.loss_reweighting must be one of {sorted(REWEIGHT_MODES)}"
        )
    beta = float(section.get("effective_number_beta", 0.9999))
    if not 0.0 < beta < 1.0:
        raise ValueError("longtail.effective_number_beta must be in (0,1)")
    tau = section.get("balanced_softmax_tau")
    if tau is None:
        tau = float(config["loss"].get("class_prior_adjustment_tau", 0.0))
    tau = float(tau)
    if tau < 0.0:
        raise ValueError("longtail.balanced_softmax_tau must be non-negative")
    return {
        "sampler_mode": sampler_mode,
        "loss_reweighting": reweight_mode,
        "effective_number_beta": beta,
        "balanced_softmax_tau": tau,
    }
