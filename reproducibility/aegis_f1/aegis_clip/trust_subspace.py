"""Trust-gradient subspace projection for noisy-label training.

The module deliberately keeps the mechanism small and auditable.  A bounded
FIFO basis is built only from gradients of cross-fitted trusted samples.  The
gradient contributed by uncertain labels can then be replaced by its
orthogonal projection into that basis before the optimiser step.

This is a label-only, TrustCLIP-inspired surrogate.  It is not presented as a
faithful reproduction of TrustCLIP because semantic class names and the full
paper's implementation details are unavailable in this competition setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class TrustSubspaceStep:
    """Diagnostics for one explicitly constructed optimiser gradient."""

    basis_updated: bool
    basis_rank: int
    reference_gradient_norm: float
    shared_gradient_norm: float
    uncertain_gradient_norm: float
    projected_uncertain_gradient_norm: float
    retained_uncertain_norm_ratio: float
    projection_applied: bool


class OnlineTrustGradientSubspace:
    """A deterministic, low-rank FIFO basis of trusted gradients.

    Vectors are orthonormalised with two-pass modified Gram--Schmidt.  When the
    rank cap is full, the oldest vector is tentatively removed; it is restored
    if the new direction is numerically redundant with the remaining basis.
    """

    FORMAT_VERSION = 1

    def __init__(self, max_rank: int = 8, epsilon: float = 1.0e-12) -> None:
        if int(max_rank) <= 0:
            raise ValueError("max_rank must be positive")
        if float(epsilon) <= 0.0:
            raise ValueError("epsilon must be positive")
        self.max_rank = int(max_rank)
        self.epsilon = float(epsilon)
        self._basis: list[torch.Tensor] = []
        self._dimension: int | None = None

    @property
    def rank(self) -> int:
        return len(self._basis)

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def update(self, vector: torch.Tensor) -> bool:
        """Insert one trusted direction, returning whether the basis changed."""
        value = self._validate_vector(vector)
        self._move_basis(value.device)
        original = self._basis
        candidates = original[1:] if len(original) == self.max_rank else original
        residual = value.detach().float().clone()
        # A second pass materially reduces loss of orthogonality for long,
        # nearly collinear gradients without introducing a decomposition whose
        # result can vary across CUDA library versions.
        for _ in range(2):
            for basis_vector in candidates:
                residual = residual - torch.dot(residual, basis_vector) * basis_vector
        norm = torch.linalg.vector_norm(residual)
        if not torch.isfinite(norm):
            raise FloatingPointError("Trusted reference gradient norm is non-finite")
        if float(norm) <= self.epsilon:
            return False
        direction = residual / norm
        self._basis = [*candidates, direction]
        return True

    def project(self, vector: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Project ``vector`` into the current span and return its norm ratio."""
        value = self._validate_vector(vector)
        self._move_basis(value.device)
        norm = torch.linalg.vector_norm(value.float())
        if not torch.isfinite(norm):
            raise FloatingPointError("Uncertain gradient norm is non-finite")
        if not self._basis or float(norm) <= self.epsilon:
            return torch.zeros_like(value, dtype=torch.float32), 0.0
        basis = torch.stack(self._basis, dim=0)
        coefficients = torch.mv(basis, value.float())
        projected = torch.mv(basis.transpose(0, 1), coefficients)
        projected_norm = torch.linalg.vector_norm(projected)
        ratio = float(projected_norm / norm.clamp_min(self.epsilon))
        # Round-off can produce values infinitesimally above one.
        ratio = min(max(ratio, 0.0), 1.0)
        return projected, ratio

    def state_dict(self) -> dict[str, object]:
        basis = (
            torch.stack([value.detach().cpu() for value in self._basis], dim=0)
            if self._basis
            else torch.empty((0, self._dimension or 0), dtype=torch.float32)
        )
        return {
            "format_version": self.FORMAT_VERSION,
            "max_rank": self.max_rank,
            "epsilon": self.epsilon,
            "dimension": self._dimension,
            "basis": basis,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state.get("format_version", -1)) != self.FORMAT_VERSION:
            raise ValueError("Unsupported trust-subspace state format")
        if int(state.get("max_rank", -1)) != self.max_rank:
            raise ValueError("Checkpoint trust-subspace rank differs from config")
        if float(state.get("epsilon", -1.0)) != self.epsilon:
            raise ValueError("Checkpoint trust-subspace epsilon differs from config")
        basis = state.get("basis")
        if not isinstance(basis, torch.Tensor) or basis.ndim != 2:
            raise ValueError("Checkpoint trust-subspace basis must be rank two")
        if basis.shape[0] > self.max_rank:
            raise ValueError("Checkpoint trust-subspace basis exceeds rank cap")
        dimension_value = state.get("dimension")
        dimension = None if dimension_value is None else int(dimension_value)
        if dimension is not None and basis.shape[1] != dimension:
            raise ValueError("Checkpoint trust-subspace dimension is inconsistent")
        loaded = basis.detach().float().cpu()
        if not torch.isfinite(loaded).all():
            raise FloatingPointError("Checkpoint trust-subspace basis is non-finite")
        if loaded.shape[0]:
            gram = loaded @ loaded.transpose(0, 1)
            identity = torch.eye(loaded.shape[0], dtype=loaded.dtype)
            if not torch.allclose(gram, identity, atol=1.0e-4, rtol=1.0e-4):
                raise ValueError("Checkpoint trust-subspace basis is not orthonormal")
        self._basis = [row.clone() for row in loaded]
        self._dimension = dimension

    def _validate_vector(self, vector: torch.Tensor) -> torch.Tensor:
        if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
            raise ValueError("Trust-subspace vectors must be one-dimensional tensors")
        if not torch.isfinite(vector).all():
            raise FloatingPointError("Trust-subspace vector is non-finite")
        dimension = int(vector.numel())
        if self._dimension is None:
            self._dimension = dimension
        elif dimension != self._dimension:
            raise ValueError(
                f"Trust-subspace dimension changed: {dimension} != {self._dimension}"
            )
        return vector

    def _move_basis(self, device: torch.device) -> None:
        if self._basis and self._basis[0].device != device:
            self._basis = [value.to(device=device) for value in self._basis]


def flatten_gradients(
    parameters: Sequence[torch.nn.Parameter],
    gradients: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    """Flatten a gradient list, representing unused parameters with zeros."""
    if len(parameters) != len(gradients):
        raise ValueError("Parameter and gradient sequences must have equal length")
    if not parameters:
        raise ValueError("At least one trainable parameter is required")
    chunks = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            chunks.append(
                torch.zeros(
                    parameter.numel(), device=parameter.device, dtype=torch.float32
                )
            )
        else:
            if gradient.shape != parameter.shape:
                raise ValueError("Gradient shape does not match its parameter")
            chunks.append(gradient.detach().reshape(-1).float())
    devices = {chunk.device for chunk in chunks}
    if len(devices) != 1:
        raise ValueError("All trainable parameters must reside on one device")
    return torch.cat(chunks, dim=0)


def assign_flat_gradient(
    parameters: Sequence[torch.nn.Parameter], vector: torch.Tensor
) -> None:
    """Assign one flat (scaled) gradient to a deterministic parameter order."""
    expected = sum(parameter.numel() for parameter in parameters)
    if vector.ndim != 1 or int(vector.numel()) != expected:
        raise ValueError(f"Flat gradient has {vector.numel()} values, expected {expected}")
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        chunk = vector[offset : offset + count].reshape_as(parameter)
        parameter.grad = chunk.to(
            device=parameter.device, dtype=parameter.dtype
        ).clone()
        offset += count


def construct_trust_subspace_gradient(
    *,
    parameters: Iterable[torch.nn.Parameter],
    trusted_reference_loss: torch.Tensor,
    shared_loss: torch.Tensor,
    uncertain_loss: torch.Tensor | None,
    scaler: torch.amp.GradScaler,
    subspace: OnlineTrustGradientSubspace,
    update_basis: bool,
    include_uncertain: bool,
) -> TrustSubspaceStep:
    """Construct and assign the scaled optimiser gradient for one batch.

    ``shared_loss`` is the strictly paired T0 update.  ``uncertain_loss`` is
    never differentiated into the model directly: T1 receives only its
    projection into the span of trusted-reference gradients.
    """
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("Trust-subspace projection requires trainable parameters")
    reference_gradients = torch.autograd.grad(
        scaler.scale(trusted_reference_loss),
        trainable,
        retain_graph=True,
        allow_unused=True,
    )
    shared_gradients = torch.autograd.grad(
        scaler.scale(shared_loss),
        trainable,
        retain_graph=bool(include_uncertain and uncertain_loss is not None),
        allow_unused=True,
    )
    reference_vector = flatten_gradients(trainable, reference_gradients)
    shared_vector = flatten_gradients(trainable, shared_gradients)
    basis_updated = subspace.update(reference_vector) if update_basis else False

    uncertain_vector = torch.zeros_like(shared_vector)
    projected = torch.zeros_like(shared_vector)
    retained_ratio = 0.0
    if include_uncertain and uncertain_loss is not None:
        uncertain_gradients = torch.autograd.grad(
            scaler.scale(uncertain_loss),
            trainable,
            allow_unused=True,
        )
        uncertain_vector = flatten_gradients(trainable, uncertain_gradients)
        projected, retained_ratio = subspace.project(uncertain_vector)

    assign_flat_gradient(trainable, shared_vector + projected)
    gradient_scale = max(float(scaler.get_scale()), 1.0)
    reference_norm = float(torch.linalg.vector_norm(reference_vector)) / gradient_scale
    shared_norm = float(torch.linalg.vector_norm(shared_vector)) / gradient_scale
    uncertain_norm = float(torch.linalg.vector_norm(uncertain_vector)) / gradient_scale
    projected_norm = float(torch.linalg.vector_norm(projected)) / gradient_scale
    return TrustSubspaceStep(
        basis_updated=basis_updated,
        basis_rank=subspace.rank,
        reference_gradient_norm=reference_norm,
        shared_gradient_norm=shared_norm,
        uncertain_gradient_norm=uncertain_norm,
        projected_uncertain_gradient_norm=projected_norm,
        retained_uncertain_norm_ratio=retained_ratio,
        projection_applied=bool(projected_norm > subspace.epsilon),
    )
