"""A fixed, training-free shared linear-head norm alignment."""
from __future__ import annotations

import torch


def mean_norm_scale(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or not weight.is_floating_point() or min(weight.shape) == 0:
        raise ValueError("weight must be a nonempty floating [classes, features] tensor")
    if not torch.isfinite(weight).all():
        raise ValueError("weight must be finite")
    norms = weight.norm(dim=1)
    if not torch.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("classifier row norms must be finite and positive")
    scale = norms.mean() / norms
    if not torch.isfinite(scale).all():
        raise ValueError("norm alignment scale must be finite")
    return scale


def align_branch_logits(logits: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Re-express W' x + b from W x + b, before any probability fusion.

    Also applies to the existing additive dual-adapter feature residual because
    every term uses the same shared classifier weight and exactly one bias.
    """
    if logits.ndim != 2 or bias.ndim != 1 or scale.shape != bias.shape or logits.shape[1] != bias.numel():
        raise ValueError("logits, bias and scale class dimensions must agree")
    if not all(torch.isfinite(x).all() for x in (logits, bias, scale)) or (scale <= 0).any():
        raise ValueError("logits/bias must be finite and scale finite positive")
    return (logits - bias) * scale + bias
