"""Unified loss function builder.

Supports:
  - cross_entropy: standard CrossEntropyLoss
  - label_smoothing: CrossEntropyLoss with label smoothing
  - gce: Generalized Cross Entropy

All losses support reduction="none" for per-sample weighting.

Usage:
    from common.losses import build_loss

    loss_fn = build_loss({"loss": {"name": "cross_entropy"}})
    loss_fn = build_loss({"loss": {"name": "label_smoothing", "epsilon": 0.05}})
    loss_fn = build_loss({"loss": {"name": "gce", "q": 0.7}})
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with uniform label smoothing.

    Uses the standard formulation:

        q_y   = 1 - epsilon
        q_oth = epsilon / (C - 1)

    where C is the number of classes.  epsilon=0 recovers standard CE.

    Args:
        epsilon: Smoothing factor in [0, 1].
        reduction: "mean", "sum", or "none".
    """

    def __init__(self, epsilon: float = 0.05, reduction: str = "mean"):
        super().__init__()
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0,1], got {epsilon}")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got {reduction!r}"
            )
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ── Input validation ──────────────────────────────────────────
        if logits.dim() != 2:
            raise ValueError(
                f"logits must be 2-D (batch, classes), got shape {tuple(logits.shape)}"
            )
        n_classes = logits.size(-1)
        if n_classes < 2:
            raise ValueError(
                f"Label smoothing requires at least 2 classes, got {n_classes}"
            )
        if targets.dim() != 1:
            raise ValueError(
                f"targets must be 1-D, got shape {tuple(targets.shape)}"
            )
        if logits.size(0) != targets.size(0):
            raise ValueError(
                f"Batch size mismatch: logits {logits.size(0)}, targets {targets.size(0)}"
            )
        if targets.lt(0).any() or targets.ge(n_classes).any():
            raise ValueError(
                f"targets must be in [0, {n_classes - 1}], "
                f"got range [{targets.min().item()}, {targets.max().item()}]"
            )

        # ── Smooth targets ────────────────────────────────────────────
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            off_target = self.epsilon / (n_classes - 1)
            smooth_targets = torch.full_like(log_probs, off_target)
            smooth_targets.scatter_(
                1, targets.unsqueeze(1), 1.0 - self.epsilon
            )

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GCELoss(nn.Module):
    """Generalized Cross Entropy.

    L = (1 - p_y^q) / q

    When q → 0, GCE approaches standard CE.
    When q = 1, GCE becomes MAE (mean absolute error).
    0 < q < 1 gives a noise-robust compromise.

    Args:
        q: GCE parameter in (0, 1].
        probability_epsilon: Minimum probability to avoid log(0) / pow(0).
        reduction: "mean", "sum", or "none".
    """

    def __init__(
        self,
        q: float = 0.7,
        probability_epsilon: float = 1e-7,
        reduction: str = "mean",
    ):
        super().__init__()
        if not 0.0 < q <= 1.0:
            raise ValueError(f"q must be in (0, 1], got {q}")
        self.q = q
        self.eps = probability_epsilon
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        py = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        py = py.clamp_min(self.eps)
        loss = (1.0 - py.pow(self.q)) / self.q

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    """Reduce per-sample loss to a scalar via mean if needed.

    When a loss module returns per-sample values (reduction="none"),
    this helper averages them into a scalar suitable for .backward()
    and .item().

    Args:
        loss: Scalar or 1-D tensor of per-sample losses.

    Returns:
        Scalar tensor.
    """
    if loss.dim() > 0:
        return loss.mean()
    return loss


def build_loss(config: Dict) -> nn.Module:
    """Build a loss function from a configuration dictionary.

    Args:
        config: Full project config dict.  Reads the ``loss`` key.

    Returns:
        A callable nn.Module loss function.

    Raises:
        ValueError: If loss name is unknown or required parameters are missing.
    """
    loss_cfg = config.get("loss", {}).copy()
    name = loss_cfg.pop("name", "cross_entropy")
    reduction = loss_cfg.pop("reduction", "mean")

    if name == "cross_entropy":
        if loss_cfg:
            raise ValueError(f"CE loss takes no extra params, got: {list(loss_cfg)}")
        return nn.CrossEntropyLoss(reduction=reduction)

    if name == "label_smoothing":
        epsilon = loss_cfg.pop("epsilon", 0.05)
        if loss_cfg:
            raise ValueError(
                f"Unknown label_smoothing params: {list(loss_cfg)}"
            )
        return LabelSmoothingLoss(epsilon=epsilon, reduction=reduction)

    if name == "gce":
        q = loss_cfg.pop("q", 0.7)
        prob_eps = loss_cfg.pop("probability_epsilon", 1e-7)
        if loss_cfg:
            raise ValueError(f"Unknown gce params: {list(loss_cfg)}")
        return GCELoss(q=q, probability_epsilon=prob_eps, reduction=reduction)

    raise ValueError(
        f"Unknown loss name: {name!r}. "
        f"Expected one of: cross_entropy, label_smoothing, gce"
    )
