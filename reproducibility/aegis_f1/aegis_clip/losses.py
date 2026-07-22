"""Noise-robust soft targets, MixUp, adaptive caps, and gradient projection."""

from __future__ import annotations

import math
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


def deterministic_complementary_labels(
    noisy_labels: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    num_classes: int,
    epoch: int,
) -> torch.Tensor:
    """Choose one reproducible non-given label per sample and epoch.

    CVPR 2026 FINE samples a random complementary label for negative
    learning.  This integer hash preserves that behavior without introducing
    an unaudited RNG state into resumed or multi-worker training.
    """
    labels = torch.as_tensor(noisy_labels).long().flatten()
    indices = torch.as_tensor(sample_indices, device=labels.device).long().flatten()
    if labels.numel() != indices.numel():
        raise ValueError("noisy_labels and sample_indices must have equal length")
    if int(num_classes) <= 1:
        raise ValueError("num_classes must be greater than one")
    if int(epoch) <= 0:
        raise ValueError("epoch must be positive")
    if ((labels < 0) | (labels >= int(num_classes))).any():
        raise ValueError("noisy_labels contain an invalid class index")
    offset = (
        (indices * 1_103_515_245 + int(epoch) * 12_345)
        % (int(num_classes) - 1)
    ) + 1
    return (labels + offset) % int(num_classes)


def active_forgetting_noise_suppression_losses(
    logits: torch.Tensor,
    noisy_labels: torch.Tensor,
    suspicious_mask: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    epoch: int,
    epsilon: float = 1.0e-7,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return AFMU and NSNL losses on a preselected noisy cohort.

    AFMU is the negative cross-entropy term ``log p(y_noisy|x)`` and is
    intentionally negative.  NSNL suppresses one deterministic complementary
    class with ``-log(1-p_complement)``.  Cohort selection remains external so
    the loss cannot silently turn model predictions into labels.
    """
    if logits.ndim != 2 or logits.shape[1] <= 1:
        raise ValueError("logits must have shape [N,C] with C>1")
    labels = torch.as_tensor(noisy_labels, device=logits.device).long().flatten()
    mask = torch.as_tensor(suspicious_mask, device=logits.device).bool().flatten()
    indices = torch.as_tensor(sample_indices, device=logits.device).long().flatten()
    if not (labels.numel() == mask.numel() == indices.numel() == logits.shape[0]):
        raise ValueError("FINE inputs must share the logits batch dimension")
    if not 0.0 < float(epsilon) < 1.0:
        raise ValueError("epsilon must be in (0,1)")
    count = int(mask.sum())
    if count == 0:
        zero = logits.sum() * 0.0
        return zero, zero, 0

    complementary = deterministic_complementary_labels(
        labels,
        indices,
        num_classes=logits.shape[1],
        epoch=epoch,
    )
    selected_logits = logits[mask].float()
    selected_labels = labels[mask]
    selected_complementary = complementary[mask]
    log_probability = F.log_softmax(selected_logits, dim=1)
    probability = log_probability.exp()
    active_forgetting = log_probability.gather(
        1, selected_labels.unsqueeze(1)
    ).mean()
    complementary_probability = probability.gather(
        1, selected_complementary.unsqueeze(1)
    ).squeeze(1)
    negative_learning = -torch.log1p(
        -complementary_probability.clamp_max(1.0 - float(epsilon))
    ).mean()
    return (
        active_forgetting.to(dtype=logits.dtype),
        negative_learning.to(dtype=logits.dtype),
        count,
    )


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


def consensus_conflict_mask(
    noisy_labels: torch.Tensor,
    pseudo_labels: torch.Tensor,
    pseudo_confidence: torch.Tensor,
    correction_evidence: torch.Tensor,
    minimum_confidence: float,
) -> torch.Tensor:
    """Locate high-confidence OOF consensus that rejects the noisy label.

    ``correction_evidence`` is kept separate from the correction schedule: a
    run may deliberately disable label refurbishment while still using the
    cross-fitted disagreement as a conservative sample-exclusion signal.
    """
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0,1]")
    noisy = torch.as_tensor(noisy_labels).long().flatten()
    pseudo = torch.as_tensor(pseudo_labels).long().flatten()
    confidence = torch.as_tensor(pseudo_confidence).float().flatten()
    evidence = torch.as_tensor(correction_evidence).float().flatten()
    if not (
        noisy.numel()
        == pseudo.numel()
        == confidence.numel()
        == evidence.numel()
    ):
        raise ValueError("Consensus-conflict inputs must have equal length")
    return (
        (pseudo >= 0)
        & (pseudo != noisy)
        & (confidence >= float(minimum_confidence))
        & (evidence > 0.0)
    )


def soft_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Per-sample cross entropy for probability targets."""
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)


def double_softmax_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Per-sample DSPT loss with an inner probability normalisation.

    The first softmax converts confident mismatches into a saturated
    probability vector before the outer cross entropy is applied. Keeping
    both normalisations in fp32 is important under AMP because the intended
    gradients can be deliberately small for likely noisy samples.
    """
    values = logits.float()
    probabilities = F.softmax(values, dim=1)
    return -(targets.float() * F.log_softmax(probabilities, dim=1)).sum(dim=1)


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


def noise_tolerant_supervised_contrastive_loss(
    first_view: torch.Tensor,
    second_view: torch.Tensor,
    labels: torch.Tensor,
    trusted: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Contrast two views without propagating untrusted class labels.

    Every sample's paired stochastic view is a positive.  Same-class samples
    become additional positives only when both labels are independently
    trusted.  Untrusted examples therefore contribute label-free instance
    consistency instead of being discarded or spreading a suspected label.
    """
    if first_view.ndim != 2 or second_view.ndim != 2:
        raise ValueError("Contrastive views must be rank-2")
    if first_view.shape != second_view.shape:
        raise ValueError("Contrastive views must have identical shapes")
    if float(temperature) <= 0.0:
        raise ValueError("Contrastive temperature must be positive")
    batch_size = first_view.shape[0]
    labels = torch.as_tensor(labels, device=first_view.device).long().flatten()
    trusted = torch.as_tensor(trusted, device=first_view.device).bool().flatten()
    if labels.numel() != batch_size or trusted.numel() != batch_size:
        raise ValueError("Labels and trust flags must align with the views")
    if batch_size == 0:
        return first_view.sum() * 0.0

    # The similarity matrix and log-sum-exp must remain fp32. With CLIP views
    # already close to one another, fp16 backward can overflow the initial
    # GradScaler even though the scalar loss itself is finite.
    with torch.autocast(device_type=first_view.device.type, enabled=False):
        embeddings = F.normalize(
            torch.cat([first_view, second_view], dim=0).float(), dim=1
        )
        repeated_labels = labels.repeat(2)
        repeated_trusted = trusted.repeat(2)
        sample_ids = torch.arange(batch_size, device=first_view.device).repeat(2)
        logits = embeddings @ embeddings.T / float(temperature)
        diagonal = torch.eye(
            2 * batch_size, device=first_view.device, dtype=torch.bool
        )
        logits = logits.masked_fill(diagonal, float("-inf"))

        paired_positive = sample_ids[:, None].eq(sample_ids[None, :]) & ~diagonal
        trusted_class_positive = (
            repeated_labels[:, None].eq(repeated_labels[None, :])
            & repeated_trusted[:, None]
            & repeated_trusted[None, :]
            & ~diagonal
        )
        positives = paired_positive | trusted_class_positive
        log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_count = positives.sum(dim=1).clamp_min(1)
        return -(
            log_probability.masked_fill(~positives, 0.0).sum(dim=1)
            / positive_count
        ).mean()


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


def smoothstep_damped_loss(
    losses: torch.Tensor,
    *,
    maximum_delta: float,
    epoch_in_cycle: int,
    cycle_epochs: int,
) -> tuple[torch.Tensor, float]:
    """Apply the differentiable cyclic SmoothStep damping used by CYFLOD.

    ``exp(-loss)`` is treated as the generic confidence proxy. High-loss,
    low-confidence examples receive a smooth factor below one, while the final
    epoch of every cycle is deliberately undamped for full-data reintroduction.
    """
    if losses.ndim != 1:
        raise ValueError("Cyclic damping requires one loss per sample")
    if not 0.0 < float(maximum_delta) <= 1.0:
        raise ValueError("maximum_delta must be in (0,1]")
    if cycle_epochs < 2:
        raise ValueError("cycle_epochs must be at least 2")
    if not 1 <= int(epoch_in_cycle) <= int(cycle_epochs):
        raise ValueError("epoch_in_cycle must be in [1, cycle_epochs]")
    delta = float(maximum_delta) * math.sin(
        math.pi * int(epoch_in_cycle) / int(cycle_epochs)
    )
    if delta <= 1.0e-12:
        return losses, 0.0
    confidence = torch.exp(-losses)
    ratio = (confidence / delta).clamp(0.0, 1.0)
    smoothstep = ratio.pow(3) * (ratio * (ratio * 6.0 - 15.0) + 10.0)
    factor = torch.where(confidence < delta, smoothstep, torch.ones_like(losses))
    return losses * factor, delta


def classwise_high_loss_filter(
    losses: torch.Tensor,
    labels: torch.Tensor,
    eligible: torch.Tensor,
    *,
    remove_fraction: float,
    maximum_class_fraction: float,
    minimum_kept_per_class: int,
) -> torch.Tensor:
    """Select a deterministic global high-loss curriculum with class caps."""
    losses = torch.as_tensor(losses, dtype=torch.float32).flatten().cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).flatten().cpu()
    eligible = torch.as_tensor(eligible, dtype=torch.bool).flatten().cpu()
    if losses.numel() != labels.numel() or losses.numel() != eligible.numel():
        raise ValueError("losses, labels, and eligible must align")
    if not torch.isfinite(losses).all():
        raise ValueError("Curriculum losses must be finite")
    if not 0.0 < float(remove_fraction) < 0.5:
        raise ValueError("remove_fraction must be in (0,0.5)")
    if not 0.0 < float(maximum_class_fraction) < 0.5:
        raise ValueError("maximum_class_fraction must be in (0,0.5)")
    if int(minimum_kept_per_class) < 1:
        raise ValueError("minimum_kept_per_class must be positive")
    target = int(math.floor(int(eligible.sum()) * float(remove_fraction)))
    selected = torch.zeros_like(eligible)
    if target <= 0:
        return selected
    class_counts = torch.bincount(labels, minlength=int(labels.max()) + 1)
    class_caps = torch.minimum(
        torch.floor(class_counts.float() * float(maximum_class_fraction)).long(),
        (class_counts - int(minimum_kept_per_class)).clamp_min(0),
    )
    used = torch.zeros_like(class_counts)
    # Stable sorting makes equal-loss behavior depend only on dataset order.
    order = torch.argsort(losses, descending=True, stable=True)
    selected_count = 0
    for index in order.tolist():
        if selected_count >= target:
            break
        if not bool(eligible[index]):
            continue
        label = int(labels[index])
        if used[label] >= class_caps[label]:
            continue
        selected[index] = True
        used[label] += 1
        selected_count += 1
    return selected


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


class TrustedPrototypeBank:
    """EMA-updated per-class prototypes for contrastive feature learning.

    Only samples with ``clean_probability >= threshold`` update the prototype
    and participate in the loss, as specified in the Phase 4 P3 plan.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        momentum: float = 0.99,
        temperature: float = 0.10,
        threshold: float = 0.80,
    ) -> None:
        if not 0.0 < momentum < 1.0:
            raise ValueError("prototype momentum must be in (0, 1)")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.prototypes: torch.Tensor = torch.empty(0)
        self.initialized: torch.Tensor = torch.empty(0)

    def to(self, device: torch.device) -> "TrustedPrototypeBank":
        if self.prototypes.numel() == 0:
            self.prototypes = torch.zeros(
                self.num_classes, self.feature_dim, device=device
            )
        else:
            self.prototypes = self.prototypes.to(device)
        if self.initialized.numel() == 0:
            self.initialized = torch.zeros(
                self.num_classes, dtype=torch.bool, device=device
            )
        else:
            self.initialized = self.initialized.to(device)
        return self

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        clean_probability: torch.Tensor,
    ) -> None:
        """EMA-update prototypes using trusted samples (p >= threshold).

        Only iterates over classes *present in the current batch*, which is
        far cheaper than scanning all 500 classes for every batch.
        """
        trusted = clean_probability >= self.threshold
        features = F.normalize(features.float().detach(), dim=1)
        present = labels.unique()
        for class_idx in present.tolist():
            mask = (labels == class_idx) & trusted
            count = int(mask.sum())
            if count == 0:
                continue
            class_mean = F.normalize(
                features[mask].mean(dim=0, keepdim=True), dim=1
            ).squeeze(0)
            if not self.initialized[class_idx]:
                self.prototypes[class_idx] = class_mean
                self.initialized[class_idx] = True
            else:
                self.prototypes[class_idx] = F.normalize(
                    self.momentum * self.prototypes[class_idx]
                    + (1.0 - self.momentum) * class_mean,
                    dim=0,
                )

    def loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        clean_probability: torch.Tensor,
    ) -> torch.Tensor:
        """Prototype-contrastive loss on trusted samples only."""
        if not self.initialized.any():
            return features.new_zeros(())
        trusted = clean_probability >= self.threshold
        if trusted.sum() == 0:
            return features.new_zeros(())
        features = F.normalize(features.float(), dim=1)
        sim = features @ self.prototypes.T / self.temperature
        per_sample = F.cross_entropy(sim, labels, reduction="none")
        return per_sample[trusted].mean()
