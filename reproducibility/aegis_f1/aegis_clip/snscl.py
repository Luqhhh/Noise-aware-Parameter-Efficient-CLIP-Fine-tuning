"""Training-only SNSCL components for fine-grained noisy labels.

The final competition model never calls these modules at inference.  They are
auxiliary representation-learning machinery: a stochastic feature transform,
an independent FIFO queue for every class, and temporally smoothed anchor
labels.  Keeping them outside :class:`AegisCLIP` makes that boundary explicit.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticFeatureEmbedding(nn.Module):
    """Learn a Gaussian feature distribution and a one-layer projector.

    The mean starts as the identity and the initial standard deviation is
    deliberately small.  This preserves a strong CLIP checkpoint at step zero
    while retaining the stochastic reparameterisation used by SNSCL.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        projection_dim: int,
        *,
        initial_std: float = 0.05,
        mean_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or projection_dim <= 0:
            raise ValueError("SNSCL dimensions must be positive")
        if not 0.0 < float(initial_std) < 1.0:
            raise ValueError("initial_std must be in (0,1)")
        if not 0.0 < float(mean_residual_scale) <= 1.0:
            raise ValueError("mean_residual_scale must be in (0,1]")
        self.feature_dim = int(feature_dim)
        self.mean_residual_scale = float(mean_residual_scale)
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, feature_dim)
        self.log_variance_head = nn.Linear(hidden_dim, feature_dim)
        self.projector = nn.Linear(feature_dim, projection_dim)

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.constant_(
            self.log_variance_head.bias,
            2.0 * torch.log(torch.tensor(float(initial_std))).item(),
        )
        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected SNSCL features [N,{self.feature_dim}], got "
                f"{tuple(features.shape)}"
            )
        values = features.float()
        hidden = self.shared(values)
        mean = values + self.mean_residual_scale * self.mean_head(hidden)
        log_variance = self.log_variance_head(hidden).clamp(-12.0, 4.0)
        standard_deviation = torch.exp(0.5 * log_variance)
        sample = mean + torch.randn_like(mean) * standard_deviation
        projected = F.normalize(self.projector(sample), dim=1)
        kl = -0.5 * (
            1.0 + log_variance - mean.pow(2) - log_variance.exp()
        ).mean()
        return projected, mean, log_variance, kl


class ClasswiseFeatureQueue(nn.Module):
    """A separate fixed-length FIFO queue for every class."""

    def __init__(self, num_classes: int, queue_size: int, feature_dim: int) -> None:
        super().__init__()
        if num_classes <= 1 or queue_size <= 0 or feature_dim <= 0:
            raise ValueError("Invalid classwise queue dimensions")
        self.num_classes = int(num_classes)
        self.queue_size = int(queue_size)
        self.feature_dim = int(feature_dim)
        self.register_buffer(
            "features",
            torch.zeros(num_classes, queue_size, feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "valid", torch.zeros(num_classes, queue_size, dtype=torch.bool)
        )
        self.register_buffer("pointers", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def enqueue(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        admission_probabilities: torch.Tensor,
    ) -> int:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("Queue features have the wrong shape")
        labels = torch.as_tensor(labels, device=features.device).long().flatten()
        probabilities = torch.as_tensor(
            admission_probabilities, device=features.device
        ).float().flatten()
        if labels.numel() != features.shape[0] or probabilities.numel() != labels.numel():
            raise ValueError("Queue inputs must have equal batch length")
        if ((labels < 0) | (labels >= self.num_classes)).any():
            raise ValueError("Queue labels are out of range")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise ValueError("Queue admission probabilities must be in [0,1]")

        admitted = (probabilities >= 1.0) | (
            torch.rand_like(probabilities) < probabilities
        )
        selected = torch.nonzero(admitted, as_tuple=False).flatten().tolist()
        normalized = F.normalize(features.detach().float(), dim=1)
        for index in selected:
            label = int(labels[index])
            pointer = int(self.pointers[label])
            self.features[label, pointer].copy_(normalized[index])
            self.valid[label, pointer] = True
            self.pointers[label] = (pointer + 1) % self.queue_size
        return len(selected)

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        class_ids = torch.arange(
            self.num_classes, device=self.features.device, dtype=torch.long
        )[:, None].expand(self.num_classes, self.queue_size)
        return self.features[self.valid], class_ids[self.valid]

    @property
    def valid_count(self) -> int:
        return int(self.valid.sum())


def classwise_queue_contrastive_loss(
    queries: torch.Tensor,
    query_labels: torch.Tensor,
    queue_features: torch.Tensor,
    queue_labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    """SNSCL Eq. 6 against the valid portion of the classwise queue."""
    if queries.ndim != 2 or queue_features.ndim != 2:
        raise ValueError("Queries and queue features must be rank-2")
    if queries.shape[1] != queue_features.shape[1]:
        raise ValueError("Query and queue feature dimensions must match")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    labels = torch.as_tensor(query_labels, device=queries.device).long().flatten()
    key_labels = torch.as_tensor(
        queue_labels, device=queries.device
    ).long().flatten()
    if labels.numel() != queries.shape[0] or key_labels.numel() != queue_features.shape[0]:
        raise ValueError("Contrastive labels must align with their features")
    if queue_features.shape[0] == 0 or queries.shape[0] == 0:
        return queries.sum() * 0.0, 0

    with torch.autocast(device_type=queries.device.type, enabled=False):
        query_values = F.normalize(queries.float(), dim=1)
        key_values = F.normalize(queue_features.detach().float(), dim=1)
        logits = query_values @ key_values.T / float(temperature)
        positives = labels[:, None].eq(key_labels[None, :])
        usable = positives.any(dim=1)
        usable_count = int(usable.sum())
        if usable_count == 0:
            return queries.sum() * 0.0, 0
        usable_logits = logits[usable]
        usable_positives = positives[usable]
        positive_mean = (
            usable_logits.masked_fill(~usable_positives, 0.0).sum(dim=1)
            / usable_positives.sum(dim=1)
        )
        losses = torch.logsumexp(usable_logits, dim=1) - positive_mean
        return losses.mean(), usable_count


class StatefulSNSCL:
    """Own the trainable auxiliary, queue, and per-sample label EMA."""

    def __init__(
        self,
        *,
        num_samples: int,
        num_classes: int,
        feature_dim: int,
        hidden_dim: int,
        projection_dim: int,
        queue_size: int,
        initial_std: float,
        mean_residual_scale: float,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("SNSCL requires a non-empty training set")
        self.num_samples = int(num_samples)
        self.num_classes = int(num_classes)
        self.embedding = StochasticFeatureEmbedding(
            feature_dim,
            hidden_dim,
            projection_dim,
            initial_std=initial_std,
            mean_residual_scale=mean_residual_scale,
        )
        self.queue = ClasswiseFeatureQueue(
            num_classes, queue_size, projection_dim
        )
        # fp16 keeps the exact soft-label EMA practical for 90k x 500 labels.
        self.label_memory = torch.zeros(
            num_samples, num_classes, dtype=torch.float16, device="cpu"
        )
        self.label_initialized = torch.zeros(
            num_samples, dtype=torch.bool, device="cpu"
        )

    def to(self, device: torch.device) -> "StatefulSNSCL":
        self.embedding.to(device)
        self.queue.to(device)
        return self

    def parameters(self):
        return self.embedding.parameters()

    @torch.no_grad()
    def corrected_labels(
        self,
        *,
        indices: torch.Tensor,
        noisy_labels: torch.Tensor,
        reliability: torch.Tensor,
        logits: torch.Tensor,
        reliability_threshold: float,
        moving_average: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= float(reliability_threshold) <= 1.0:
            raise ValueError("reliability_threshold must be in [0,1]")
        if not 0.0 <= float(moving_average) < 1.0:
            raise ValueError("moving_average must be in [0,1)")
        indices_cpu = torch.as_tensor(indices).long().flatten().cpu()
        if ((indices_cpu < 0) | (indices_cpu >= self.num_samples)).any():
            raise ValueError("SNSCL sample indices are out of range")
        noisy = torch.as_tensor(noisy_labels).long().flatten()
        reliability_device = torch.as_tensor(
            reliability, device=logits.device
        ).float().flatten().clamp(0.0, 1.0)
        if noisy.numel() != indices_cpu.numel() or logits.shape != (
            indices_cpu.numel(),
            self.num_classes,
        ):
            raise ValueError("SNSCL corrected-label inputs do not align")
        weight = torch.where(
            reliability_device > float(reliability_threshold),
            torch.ones_like(reliability_device),
            reliability_device,
        )
        predictions = F.softmax(logits.detach().float(), dim=1)
        original = F.one_hot(
            noisy.to(logits.device), num_classes=self.num_classes
        ).float()
        proposed = (
            (1.0 - weight[:, None]) * predictions + weight[:, None] * original
        ).cpu()
        previous = self.label_memory[indices_cpu].float()
        was_initialized = self.label_initialized[indices_cpu]
        updated = torch.where(
            was_initialized[:, None],
            float(moving_average) * previous
            + (1.0 - float(moving_average)) * proposed,
            proposed,
        )
        updated = updated / updated.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        self.label_memory[indices_cpu] = updated.to(torch.float16)
        self.label_initialized[indices_cpu] = True
        return updated.to(logits.device), weight

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "num_samples": self.num_samples,
            "num_classes": self.num_classes,
            "embedding": self.embedding.state_dict(),
            "queue": self.queue.state_dict(),
            "label_memory": self.label_memory,
            "label_initialized": self.label_initialized,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("format_version", 0)) != 1:
            raise ValueError("Unsupported SNSCL state format")
        if int(state.get("num_samples", -1)) != self.num_samples or int(
            state.get("num_classes", -1)
        ) != self.num_classes:
            raise ValueError("SNSCL state dimensions do not match this run")
        label_memory = torch.as_tensor(state["label_memory"]).cpu()
        initialized = torch.as_tensor(state["label_initialized"]).bool().cpu()
        if label_memory.shape != self.label_memory.shape or initialized.shape != self.label_initialized.shape:
            raise ValueError("SNSCL label-memory shape mismatch")
        self.embedding.load_state_dict(state["embedding"], strict=True)
        self.queue.load_state_dict(state["queue"], strict=True)
        self.label_memory.copy_(label_memory.to(torch.float16))
        self.label_initialized.copy_(initialized)
