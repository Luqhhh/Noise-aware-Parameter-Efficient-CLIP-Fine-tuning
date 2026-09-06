"""Single-model logit calibration against an explicitly declared class prior.

The class-bias fitting step (iterative proportional fitting) is deliberately
separated from the application step.  Competition rules forbid optimising the
class prior from the test prediction distribution, so the bias must be fitted
once on the current-stage validation set and applied to the test set as a
frozen, deterministic additive offset.  ``align_logits_to_prior`` keeps the
legacy fit-and-apply-on-one-batch behavior for offline sweeps only.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def fit_prior_bias(
    logits: torch.Tensor,
    *,
    target_prior: torch.Tensor | None = None,
    max_iterations: int = 50,
    tolerance: float = 1.0e-6,
    damping: float = 0.5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit one additive class-bias vector so the soft marginal approaches a prior.

    The model and every image logit remain fixed.  Iterative proportional
    fitting estimates the bias from the provided batch.  No labels or
    parameter updates are involved.
    """

    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] <= 1:
        raise ValueError("logits must have non-empty shape [N, C] with C > 1")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
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

    final_marginal = F.softmax(work + bias, dim=1).mean(dim=0)
    report = {
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
        "target_prior": "uniform" if target_prior is None else "explicit",
    }
    return bias, report


def apply_prior_bias(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """Apply a frozen, previously fitted class bias to fresh logits.

    This is a deterministic, parameter-free transform and performs no fitting,
    so it is safe to run on the test set.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be rank-2")
    bias = torch.as_tensor(bias, dtype=torch.float32).flatten()
    if bias.numel() != logits.shape[1]:
        raise ValueError("bias length must equal the logit class count")
    if not torch.isfinite(bias).all():
        raise ValueError("bias must be finite")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    return logits.float() + float(strength) * bias.to(logits.device)


def align_logits_to_prior(
    logits: torch.Tensor,
    *,
    target_prior: torch.Tensor | None = None,
    strength: float = 1.0,
    max_iterations: int = 50,
    tolerance: float = 1.0e-6,
    damping: float = 0.5,
    return_applied_bias: bool = False,
) -> (
    tuple[torch.Tensor, dict[str, Any]]
    | tuple[torch.Tensor, dict[str, Any], torch.Tensor]
):
    """Fit and apply a prior bias on one batch (offline sweep convenience).

    Kept for backward compatibility with offline val sweeps.  Official test
    inference should use ``fit_prior_bias`` once on validation followed by
    ``apply_prior_bias`` on test logits instead.
    """
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    work = logits.detach().float()
    bias, fit_report = fit_prior_bias(
        work,
        target_prior=target_prior,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    applied_bias = float(strength) * bias
    aligned = work + applied_bias
    final_marginal = F.softmax(aligned, dim=1).mean(dim=0)
    raw_counts = work.argmax(dim=1).bincount(minlength=work.shape[1]).float()
    aligned_counts = aligned.argmax(dim=1).bincount(minlength=work.shape[1]).float()
    report = {
        **fit_report,
        "strength": float(strength),
        "raw_argmax_count_min": int(raw_counts.min()),
        "raw_argmax_count_max": int(raw_counts.max()),
        "aligned_argmax_count_min": int(aligned_counts.min()),
        "aligned_argmax_count_max": int(aligned_counts.max()),
    }
    result = aligned.to(dtype=logits.dtype)
    if return_applied_bias:
        return result, report, applied_bias.to(dtype=logits.dtype)
    return result, report
