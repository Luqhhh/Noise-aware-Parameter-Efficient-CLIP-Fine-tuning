"""Noise-robust soft targets, MixUp, adaptive caps, and gradient projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def class_prior_adjusted_logits(
    logits: torch.Tensor,
    class_counts: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Apply a training-only Balanced-Softmax prior correction.

    Raw model logits remain the inference scores. During training, adding the
    empirical log prior counteracts a long-tailed training distribution when
    the evaluation prior is approximately balanced.
    """
    if tau < 0.0:
        raise ValueError("tau must be non-negative")
    counts = torch.as_tensor(
        class_counts, device=logits.device, dtype=logits.dtype
    ).flatten()
    if counts.numel() != logits.shape[1]:
        raise ValueError("class_counts must have one value per logit")
    if (counts <= 0).any():
        raise ValueError("class_counts must be strictly positive")
    if tau == 0.0:
        return logits
    log_prior = torch.log(counts / counts.sum())
    return logits + float(tau) * log_prior.unsqueeze(0)


def corrected_targets(
    noisy_labels: torch.Tensor,
    pseudo_labels: torch.Tensor,
    correction_alpha: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    noisy = F.one_hot(noisy_labels.long(), num_classes=num_classes).float()
    safe_pseudo = torch.where(
        (pseudo_labels >= 0) & (pseudo_labels < num_classes),
        pseudo_labels,
        noisy_labels,
    )
    pseudo = F.one_hot(safe_pseudo.long(), num_classes=num_classes).float()
    alpha = correction_alpha.float().clamp(0.0, 1.0).unsqueeze(1)
    return noisy * (1.0 - alpha) + pseudo * alpha


def soft_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Per-sample cross entropy for probability targets."""
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)


def soft_generalized_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    q: float,
    epsilon: float = 1.0e-7,
) -> torch.Tensor:
    """Per-sample GCE extended linearly to soft targets."""
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0,1]")
    probabilities = F.softmax(logits, dim=1).clamp_min(epsilon)
    return (targets * (1.0 - probabilities.pow(q)) / q).sum(dim=1)


def classwise_suspicion_mask(
    labels: torch.Tensor,
    clean_probabilities: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Select the lowest-trust quota independently inside every noisy class."""
    labels = torch.as_tensor(labels, dtype=torch.long).flatten().cpu()
    scores = torch.as_tensor(clean_probabilities, dtype=torch.float32).flatten().cpu()
    if labels.numel() != scores.numel():
        raise ValueError("labels and clean_probabilities must align")
    if not 0.0 < fraction < 0.5:
        raise ValueError("suspicious fraction must be in (0, 0.5)")
    mask = torch.zeros(labels.numel(), dtype=torch.bool)
    for label in labels.unique(sorted=True):
        members = torch.nonzero(labels == label, as_tuple=False).flatten()
        quota = max(1, int(round(members.numel() * float(fraction))))
        order = torch.argsort(scores[members], stable=True)
        mask[members[order[:quota]]] = True
    return mask


class EarlyLearningRegularizer(nn.Module):
    """Per-sample temporal targets with the original ELR loss direction.

    The regularizer is ``log(1 - <p, t>)``.  It is deliberately negative:
    minimising it increases agreement between the current prediction ``p``
    and its early-learning exponential moving-average target ``t``.
    """

    def __init__(
        self,
        num_examples: int,
        num_classes: int,
        momentum: float = 0.9,
        target_weight: float = 3.0,
        warmup_epochs: int = 5,
        ramp_epochs: int = 5,
        epsilon: float = 1.0e-7,
    ) -> None:
        super().__init__()
        if num_examples <= 0 or num_classes <= 1:
            raise ValueError("ELR requires positive examples and at least two classes")
        if not 0.0 < momentum < 1.0:
            raise ValueError("ELR momentum must be in (0, 1)")
        if target_weight < 0.0:
            raise ValueError("ELR target_weight must be non-negative")
        if warmup_epochs < 0 or ramp_epochs < 0:
            raise ValueError("ELR warmup and ramp epochs must be non-negative")
        self.momentum = float(momentum)
        self.target_weight = float(target_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.epsilon = float(epsilon)
        self.register_buffer(
            "targets", torch.zeros(int(num_examples), int(num_classes))
        )
        self.register_buffer("updates", torch.zeros(int(num_examples), dtype=torch.long))

    def rampup_weight(self, epoch: int) -> float:
        if epoch <= self.warmup_epochs:
            return 0.0
        if self.ramp_epochs == 0:
            return self.target_weight
        progress = min((epoch - self.warmup_epochs) / self.ramp_epochs, 1.0)
        return self.target_weight * progress

    def update_and_loss(
        self, indices: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        indices = indices.to(device=self.targets.device, dtype=torch.long).flatten()
        if indices.numel() != logits.shape[0]:
            raise ValueError("ELR indices must align one-to-one with logits")
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= self.targets.shape[0]
        ):
            raise IndexError("ELR sample index is out of range")
        probabilities = F.softmax(logits.float(), dim=1).clamp(
            self.epsilon, 1.0 - self.epsilon
        )
        detached = probabilities.detach()
        detached = detached / detached.sum(dim=1, keepdim=True)
        with torch.no_grad():
            self.targets[indices] = (
                self.momentum * self.targets[indices]
                + (1.0 - self.momentum) * detached
            )
            self.updates[indices] += 1
        agreement = (probabilities * self.targets[indices].detach()).sum(dim=1)
        return torch.log((1.0 - agreement).clamp_min(self.epsilon)).mean()


def mixup(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    alpha: float,
    probability: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Apply batch MixUp and mix both targets and trust-derived weights."""
    identity = torch.arange(inputs.shape[0], device=inputs.device)
    if alpha <= 0.0 or probability <= 0.0:
        return inputs, targets, sample_weights, 1.0, identity
    random_value = torch.rand((), generator=generator).item()
    if random_value > probability:
        return inputs, targets, sample_weights, 1.0, identity
    concentration = torch.tensor([alpha], dtype=torch.float32)
    first = torch._standard_gamma(concentration, generator=generator)
    second = torch._standard_gamma(concentration, generator=generator)
    lam = float((first / (first + second).clamp_min(1.0e-12)).item())
    permutation = torch.randperm(inputs.shape[0], generator=generator).to(inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[permutation]
    mixed_targets = lam * targets + (1.0 - lam) * targets[permutation]
    mixed_weights = lam * sample_weights + (1.0 - lam) * sample_weights[permutation]
    return mixed_inputs, mixed_targets, mixed_weights, lam, permutation


@dataclass
class AdaptiveLossCap:
    """Smoothly cap extreme sample losses using a trusted-loss quantile.

    Above the running threshold tau, ``tau * (1 + log(loss/tau))`` has gradient
    multiplier ``tau/loss``. This suppresses extreme noisy gradients without
    discarding samples or introducing a hard zero-gradient cutoff.
    """

    quantile: float = 0.90
    momentum: float = 0.90
    minimum: float = 0.05
    maximum: float = 10.0
    value: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must be in (0,1)")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0,1)")

    def __call__(
        self, losses: torch.Tensor, trusted_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            reference = losses.detach()[trusted_mask]
            if reference.numel() < 4:
                reference = losses.detach()
            observed = float(torch.quantile(reference.float(), self.quantile))
            observed = min(max(observed, self.minimum), self.maximum)
            self.value = (
                observed
                if self.value is None
                else self.momentum * self.value + (1.0 - self.momentum) * observed
            )
        tau = losses.new_tensor(float(self.value))
        safe = losses.clamp_min(1.0e-8)
        capped = tau * (1.0 + torch.log(safe / tau))
        return torch.where(losses <= tau, losses, capped)

    def state_dict(self) -> dict[str, float | None]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, float | None]) -> None:
        self.value = state.get("value")


def project_conflicting_gradients(
    parameters: Iterable[torch.nn.Parameter],
    anchor_gradients: Iterable[torch.Tensor | None],
    epsilon: float = 1.0e-12,
) -> dict[str, float | bool]:
    """Remove the component of current gradients that opposes an anchor.

    The operation is in-place on ``parameter.grad``. If the current gradient
    has a negative dot product with the trusted anchor, it is projected onto
    the half-space whose dot product with the anchor is non-negative.
    """
    pairs = [
        (parameter, anchor)
        for parameter, anchor in zip(parameters, anchor_gradients)
        if parameter.grad is not None and anchor is not None
    ]
    if not pairs:
        return {"projected": False, "dot": 0.0, "cosine": 0.0}
    dot = sum(
        (parameter.grad.detach().float() * anchor.detach().float()).sum()
        for parameter, anchor in pairs
    )
    current_norm_sq = sum(
        parameter.grad.detach().float().pow(2).sum() for parameter, _ in pairs
    )
    anchor_norm_sq = sum(
        anchor.detach().float().pow(2).sum() for _, anchor in pairs
    )
    cosine = dot / (
        current_norm_sq.clamp_min(epsilon).sqrt()
        * anchor_norm_sq.clamp_min(epsilon).sqrt()
    )
    projected = bool(dot.item() < 0.0 and anchor_norm_sq.item() > epsilon)
    if projected:
        coefficient = dot / anchor_norm_sq.clamp_min(epsilon)
        for parameter, anchor in pairs:
            parameter.grad.add_(
                anchor.to(device=parameter.grad.device, dtype=parameter.grad.dtype),
                alpha=-float(coefficient),
            )
    return {
        "projected": projected,
        "dot": float(dot),
        "cosine": float(cosine),
    }
