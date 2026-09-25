import math

import pytest
import torch
import torch.nn.functional as F

from aegis_clip.candidate_losses import (
    build_padded_candidate_tensors,
    flip_fusion_global_loss,
    probability_generalized_cross_entropy,
    set_generalized_cross_entropy,
)
from aegis_clip.losses import soft_generalized_cross_entropy


def _logits(rows: int, classes: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, classes, generator=generator)


def test_probability_gce_matches_logits_domain_for_hard_labels() -> None:
    logits = _logits(9, 6, seed=1)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 2, 1, 0])
    probabilities = F.softmax(logits, dim=1)
    expected = soft_generalized_cross_entropy(
        logits, F.one_hot(labels, num_classes=6).float(), 0.5
    )
    actual = probability_generalized_cross_entropy(
        probabilities, labels, q=0.5
    )
    assert actual.shape == (9,)
    assert torch.allclose(actual, expected, atol=1.0e-4)


def test_probability_gce_soft_targets_match_explicit_formula() -> None:
    probabilities = F.softmax(_logits(4, 5, seed=2), dim=1)
    soft = F.softmax(_logits(4, 5, seed=3), dim=1)
    expected = (soft * (1.0 - probabilities.clamp_min(1.0e-7).pow(0.5)) / 0.5).sum(
        dim=1
    )
    actual = probability_generalized_cross_entropy(probabilities, soft, q=0.5)
    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_flip_fusion_loss_matches_hand_computed_value() -> None:
    logits_original = torch.tensor([[math.log(3.0), 0.0]])
    logits_flipped = torch.tensor([[0.0, 0.0]])
    loss, diagnostics = flip_fusion_global_loss(
        logits_original, logits_flipped, torch.tensor([0]), temperature=1.0
    )
    # p_o=[0.75,0.25], p_f=[0.5,0.5], p_mix=[0.625,0.375], q=0.5:
    # 0.5*(l(0.75)+l(0.5))/2 + 0.5*l(0.625) = 0.42286449247241213
    assert loss.shape == (1,)
    assert loss.item() == pytest.approx(0.42286449247241213, abs=1.0e-6)
    assert set(diagnostics) == {"p_o", "p_f", "p_mix", "term_o", "term_f", "term_mix"}
    assert torch.allclose(diagnostics["p_mix"], (diagnostics["p_o"] + diagnostics["p_f"]) / 2.0)


def test_flip_fusion_both_branches_receive_gradient() -> None:
    logits_original = _logits(5, 4, seed=4).requires_grad_(True)
    logits_flipped = _logits(5, 4, seed=5).requires_grad_(True)
    target = torch.tensor([0, 1, 2, 3, 1])
    loss, _ = flip_fusion_global_loss(logits_original, logits_flipped, target)
    loss.sum().backward()
    assert logits_original.grad is not None
    assert logits_flipped.grad is not None
    assert float(logits_original.grad.abs().sum()) > 0.0
    assert float(logits_flipped.grad.abs().sum()) > 0.0


def test_set_gce_singleton_equals_probability_gce() -> None:
    probabilities = F.softmax(_logits(7, 5, seed=6), dim=1)
    labels = torch.tensor([0, 1, 2, 3, 4, 3, 1])
    expected = probability_generalized_cross_entropy(probabilities, labels, q=0.5)
    index = labels.unsqueeze(1)
    mask = torch.ones_like(index, dtype=torch.bool)
    actual = set_generalized_cross_entropy(probabilities, index, mask, q=0.5)
    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_set_gce_mask_padding_does_not_change_result() -> None:
    probabilities = F.softmax(_logits(2, 6, seed=7), dim=1)
    compact_index = torch.tensor([[0, 2], [1, 3]])
    compact_mask = torch.ones_like(compact_index, dtype=torch.bool)
    padded_index = torch.tensor([[0, 2, 99, -5], [1, 3, 17, 0]])
    padded_mask = torch.tensor(
        [[True, True, False, False], [True, True, False, False]]
    )
    compact = set_generalized_cross_entropy(probabilities, compact_index, compact_mask)
    padded = set_generalized_cross_entropy(probabilities, padded_index, padded_mask)
    assert torch.allclose(compact, padded, atol=1.0e-6)


def test_set_gce_is_differentiable_through_the_probabilities() -> None:
    logits = _logits(3, 5, seed=8).requires_grad_(True)
    probabilities = F.softmax(logits, dim=1)
    index = torch.tensor([[0, 1], [2, 3], [4, 0]])
    mask = torch.ones_like(index, dtype=torch.bool)
    loss = set_generalized_cross_entropy(probabilities, index, mask)
    loss.sum().backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_set_gce_rejects_empty_rows_and_out_of_range_ids() -> None:
    probabilities = F.softmax(_logits(1, 4, seed=9), dim=1)
    with pytest.raises(ValueError):
        set_generalized_cross_entropy(
            probabilities, torch.tensor([[0]]), torch.tensor([[False]])
        )
    with pytest.raises(ValueError):
        set_generalized_cross_entropy(
            probabilities, torch.tensor([[4]]), torch.tensor([[True]])
        )


def test_set_gce_rejects_mass_outside_unit_interval() -> None:
    probabilities = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        set_generalized_cross_entropy(
            probabilities, torch.tensor([[1]]), torch.tensor([[True]])
        )
    duplicate = torch.tensor([[0.1, 0.7, 0.1, 0.1]])
    with pytest.raises(ValueError):
        set_generalized_cross_entropy(
            duplicate, torch.tensor([[1, 1]]), torch.tensor([[True, True]])
        )


def test_build_padded_candidate_tensors_padding_and_group_weight() -> None:
    candidate_labels = {"a": [1], "b": [1, 2, 3], "c": [5, 6]}
    index, mask, weight = build_padded_candidate_tensors(
        ["a", "b", "c"], candidate_labels, num_classes=10
    )
    assert index.shape == (3, 3)
    assert mask.tolist() == [[True, False, False], [True, True, True], [True, True, False]]
    assert index[mask].tolist() == [1, 1, 2, 3, 5, 6]
    assert weight.tolist() == pytest.approx([1.0, 1.0 / 3.0, 0.5])
    assert float(weight.sum()) == pytest.approx(1.0 + 1.0 / 3.0 + 0.5)


def test_build_padded_candidate_tensors_fails_closed() -> None:
    with pytest.raises(ValueError):
        build_padded_candidate_tensors(["a", "missing"], {"a": [0]}, num_classes=4)
    with pytest.raises(ValueError):
        build_padded_candidate_tensors(["a"], {"a": []}, num_classes=4)
    with pytest.raises(ValueError):
        build_padded_candidate_tensors(["a"], {"a": [4]}, num_classes=4)
    with pytest.raises(ValueError):
        build_padded_candidate_tensors(["a"], {"a": [-1]}, num_classes=4)
    with pytest.raises(ValueError):
        build_padded_candidate_tensors(["a"], {"a": [1, 1]}, num_classes=4)
    with pytest.raises(ValueError):
        build_padded_candidate_tensors(["a"], {"a": [1.5]}, num_classes=4)


def test_probability_gce_rejects_invalid_arguments() -> None:
    probabilities = F.softmax(_logits(2, 4, seed=11), dim=1)
    with pytest.raises(ValueError):
        probability_generalized_cross_entropy(probabilities, torch.tensor([0, 1]), q=0.0)
    with pytest.raises(ValueError):
        probability_generalized_cross_entropy(probabilities, torch.tensor([0, 1]), q=1.5)
    with pytest.raises(ValueError):
        probability_generalized_cross_entropy(probabilities, torch.tensor([0, 1]), epsilon=0.0)
    with pytest.raises(ValueError):
        probability_generalized_cross_entropy(probabilities, torch.tensor([0, 9]))
    with pytest.raises(ValueError):
        probability_generalized_cross_entropy(probabilities[0], torch.tensor([0]))
