import pytest
import torch

from aegis_clip.prior_alignment import align_logits_to_prior


def test_full_alignment_matches_uniform_soft_marginal() -> None:
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(2000, 5, generator=generator)
    logits[:, 0] += 2.0
    aligned, report = align_logits_to_prior(
        logits, strength=1.0, max_iterations=200, tolerance=1.0e-7
    )
    marginal = aligned.softmax(dim=1).mean(dim=0)
    assert torch.allclose(marginal, torch.full((5,), 0.2), atol=2.0e-6)
    assert report["final_marginal_l1"] < report["initial_marginal_l1"]


def test_zero_strength_preserves_logits_exactly() -> None:
    logits = torch.tensor([[4.0, 1.0], [3.0, 2.0]])
    aligned, _ = align_logits_to_prior(logits, strength=0.0)
    assert torch.equal(aligned, logits)


def test_explicit_prior_is_normalized() -> None:
    logits = torch.zeros(100, 2)
    aligned, _ = align_logits_to_prior(
        logits,
        target_prior=torch.tensor([3.0, 1.0]),
        strength=1.0,
        max_iterations=100,
    )
    assert torch.allclose(
        aligned.softmax(1).mean(0), torch.tensor([0.75, 0.25]), atol=2.0e-6
    )


def test_invalid_alignment_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="strength"):
        align_logits_to_prior(torch.zeros(2, 3), strength=1.1)
    with pytest.raises(ValueError, match="strictly positive"):
        align_logits_to_prior(
            torch.zeros(2, 3), target_prior=torch.tensor([0.5, 0.5, 0.0])
        )
