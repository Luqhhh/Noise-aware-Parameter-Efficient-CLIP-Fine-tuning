import pytest
import torch

from aegis_clip.trust_subspace import (
    OnlineTrustGradientSubspace,
    construct_trust_subspace_gradient,
)


def test_projection_keeps_only_the_trusted_span() -> None:
    subspace = OnlineTrustGradientSubspace(max_rank=2)
    assert subspace.update(torch.tensor([1.0, 0.0, 0.0]))
    projected, ratio = subspace.project(torch.tensor([3.0, 4.0, 0.0]))
    assert torch.allclose(projected, torch.tensor([3.0, 0.0, 0.0]))
    assert ratio == pytest.approx(0.6)


def test_fifo_basis_and_checkpoint_round_trip_are_deterministic() -> None:
    subspace = OnlineTrustGradientSubspace(max_rank=2)
    subspace.update(torch.tensor([1.0, 0.0, 0.0]))
    subspace.update(torch.tensor([0.0, 1.0, 0.0]))
    subspace.update(torch.tensor([0.0, 0.0, 1.0]))
    assert subspace.rank == 2
    projected, _ = subspace.project(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(projected, torch.tensor([0.0, 2.0, 3.0]))

    restored = OnlineTrustGradientSubspace(max_rank=2)
    restored.load_state_dict(subspace.state_dict())
    restored_projection, restored_ratio = restored.project(
        torch.tensor([1.0, 2.0, 3.0])
    )
    assert torch.equal(restored_projection, projected)
    assert restored_ratio == subspace.project(torch.tensor([1.0, 2.0, 3.0]))[1]


def test_control_gradient_is_exact_and_uncertain_component_is_projected() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    control_subspace = OnlineTrustGradientSubspace(max_rank=1)
    control = construct_trust_subspace_gradient(
        parameters=[parameter],
        trusted_reference_loss=parameter[0],
        shared_loss=2.0 * parameter[0] + 3.0 * parameter[1],
        uncertain_loss=None,
        scaler=scaler,
        subspace=control_subspace,
        update_basis=True,
        include_uncertain=False,
    )
    assert torch.equal(parameter.grad, torch.tensor([2.0, 3.0]))
    assert control.basis_rank == 1
    assert not control.projection_applied

    parameter.grad = None
    treatment_subspace = OnlineTrustGradientSubspace(max_rank=1)
    treatment = construct_trust_subspace_gradient(
        parameters=[parameter],
        trusted_reference_loss=parameter[0],
        shared_loss=2.0 * parameter[0] + 3.0 * parameter[1],
        uncertain_loss=5.0 * parameter[0] + 7.0 * parameter[1],
        scaler=scaler,
        subspace=treatment_subspace,
        update_basis=True,
        include_uncertain=True,
    )
    assert torch.equal(parameter.grad, torch.tensor([7.0, 3.0]))
    assert treatment.projection_applied
    assert treatment.projected_uncertain_gradient_norm == 5.0
    assert treatment.retained_uncertain_norm_ratio < 1.0


def test_redundant_update_does_not_drop_a_full_basis() -> None:
    subspace = OnlineTrustGradientSubspace(max_rank=2)
    subspace.update(torch.tensor([1.0, 0.0, 0.0]))
    subspace.update(torch.tensor([0.0, 1.0, 0.0]))
    assert not subspace.update(torch.tensor([0.0, 2.0, 0.0]))
    assert subspace.rank == 2
    projected, _ = subspace.project(torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(projected, torch.tensor([1.0, 1.0, 0.0]))
