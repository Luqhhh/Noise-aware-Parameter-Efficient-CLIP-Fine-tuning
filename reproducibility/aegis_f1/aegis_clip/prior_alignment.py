"""Single-model logit calibration against an explicitly declared class prior."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def align_logits_to_prior(
    logits: torch.Tensor,
    *,
    target_prior: torch.Tensor | None = None,
    strength: float = 1.0,
    max_iterations: int = 50,
    tolerance: float = 1.0e-6,
    damping: float = 0.5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit one class-bias vector so the soft marginal approaches a prior.

    The model and every image logit remain fixed.  Iterative proportional
    fitting estimates a single additive class bias from the complete inference
    batch; ``strength`` interpolates between the raw model (0) and the fitted
    prior (1).  No labels or parameter updates are involved.
    """

    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] <= 1:
        raise ValueError("logits must have non-empty shape [N, C] with C > 1")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")
    if float(tolerance) <= 0.0:
        raise ValueError("tolerance must be positive")
    if not 0.0 < float(damping) <= 1.0:
        raise ValueError("damping must be in (0, 1]")

    work = logits.detach().float()
    classes = work.shape[1]
    if target_prior is None:
        prior = torch.full(
            (classes,), 1.0 / classes, device=work.device, dtype=work.dtype
        )
    else:
        prior = torch.as_tensor(
            target_prior, device=work.device, dtype=work.dtype
        ).flatten()
        if prior.numel() != classes:
            raise ValueError("target_prior length must equal the class count")
        if not torch.isfinite(prior).all() or bool((prior <= 0.0).any()):
            raise ValueError("target_prior must be finite and strictly positive")
        prior = prior / prior.sum()

    initial_marginal = F.softmax(work, dim=1).mean(dim=0)
    bias = torch.zeros_like(prior)
    iterations = 0
    fitted_error = float("inf")
    for iterations in range(1, int(max_iterations) + 1):
        marginal = F.softmax(work + bias, dim=1).mean(dim=0)
        fitted_error = float((marginal - prior).abs().max())
        if fitted_error <= float(tolerance):
            break
        update = (prior.clamp_min(1.0e-12).log() - marginal.clamp_min(1.0e-12).log())
        bias = bias + float(damping) * update
        bias = bias - bias.mean()

    aligned = work + float(strength) * bias
    final_marginal = F.softmax(aligned, dim=1).mean(dim=0)
    raw_counts = work.argmax(dim=1).bincount(minlength=classes).float()
    aligned_counts = aligned.argmax(dim=1).bincount(minlength=classes).float()
    report = {
        "strength": float(strength),
        "max_iterations": int(max_iterations),
        "iterations": int(iterations),
        "tolerance": float(tolerance),
        "damping": float(damping),
        "fitted_max_marginal_error": float(fitted_error),
        "initial_marginal_l1": float((initial_marginal - prior).abs().sum()),
        "final_marginal_l1": float((final_marginal - prior).abs().sum()),
        "final_max_marginal_error": float((final_marginal - prior).abs().max()),
        "bias_min": float(bias.min()),
        "bias_max": float(bias.max()),
        "raw_argmax_count_min": int(raw_counts.min()),
        "raw_argmax_count_max": int(raw_counts.max()),
        "aligned_argmax_count_min": int(aligned_counts.min()),
        "aligned_argmax_count_max": int(aligned_counts.max()),
        "target_prior": "uniform" if target_prior is None else "explicit",
    }
    return aligned.to(dtype=logits.dtype), report
