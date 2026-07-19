import pytest
import torch

from aegis_clip.multiprototype import (
    blend_multiprototype_logits,
    fit_weighted_multiprototypes,
    match_score_scale,
    multiprototype_logits,
    paired_top1_changes,
)


def test_multiprototypes_preserve_multiple_class_modes() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [-0.8, -0.2], [0.0, 1.0], [0.2, 0.8]]
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1])
    prototypes = fit_weighted_multiprototypes(
        features,
        labels,
        torch.ones(6),
        num_classes=2,
        prototypes_per_class=2,
        random_state=1,
    )
    scores = multiprototype_logits(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
        prototypes,
        aggregation="max",
    )
    assert torch.equal(scores.argmax(1), torch.tensor([0, 0, 1]))


def test_logmeanexp_one_prototype_matches_max() -> None:
    features = torch.randn(5, 3)
    prototypes = torch.randn(2, 1, 3)
    maximum = multiprototype_logits(features, prototypes, aggregation="max")
    smooth = multiprototype_logits(features, prototypes, aggregation="logmeanexp")
    assert torch.allclose(maximum, smooth, atol=1.0e-6)


def test_match_score_scale_equalizes_spread() -> None:
    reference = torch.randn(8, 4)
    assert match_score_scale(reference, 5.0 * reference) == pytest.approx(0.2)


def test_checkpoint_head_blend_matches_explicit_formula() -> None:
    features = torch.randn(4, 3)
    base = torch.randn(4, 2)
    prototypes = torch.randn(2, 2, 3)
    head = {
        "prototypes": prototypes,
        "aggregation": "max",
        "alpha": 0.25,
        "candidate_scale": 4.0,
    }
    expected = base + multiprototype_logits(
        features, prototypes, aggregation="max"
    )
    assert torch.allclose(blend_multiprototype_logits(base, features, head), expected)


def test_paired_top1_changes_separates_fixed_and_broken() -> None:
    labels = torch.tensor([0, 1, 0])
    reference = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    candidate = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert paired_top1_changes(reference, candidate, labels) == {
        "changed_predictions": 2,
        "raw_fixed": 1,
        "raw_broken": 1,
        "raw_net_fixed": 0,
    }
