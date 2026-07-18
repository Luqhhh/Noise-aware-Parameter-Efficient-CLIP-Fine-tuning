"""Unit tests for auditable single-model post-hoc inference."""

import pytest
import torch

from common.posthoc import (
    assert_non_classifier_state_equal,
    blend_multiprototype_logits,
    fit_weighted_multiprototypes,
    fuse_paired_logits,
    interpolate_linear_heads,
    match_score_scale,
    multiprototype_logits,
)


def test_soup_is_exact_logit_interpolation():
    features = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    w1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b1 = torch.tensor([0.2, -0.1])
    w2 = torch.tensor([[0.5, 1.5], [-1.0, 0.5]])
    b2 = torch.tensor([-0.3, 0.4])
    weight, bias = interpolate_linear_heads(w1, b1, w2, b2, alpha=0.4)
    actual = torch.nn.functional.linear(features, weight, bias)
    expected = 0.6 * torch.nn.functional.linear(features, w1, b1)
    expected += 0.4 * torch.nn.functional.linear(features, w2, b2)
    assert torch.allclose(actual, expected)


def test_soup_rejects_changed_backbone():
    first = {
        "visual.block": torch.tensor([1.0]),
        "classifier.weight": torch.ones(2, 2),
        "classifier.bias": torch.zeros(2),
    }
    second = {key: value.clone() for key, value in first.items()}
    second["visual.block"][0] = 2.0
    with pytest.raises(ValueError, match="non-classifier state differs"):
        assert_non_classifier_state_equal(first, second)


def test_soup_allows_only_classifier_change():
    first = {
        "visual.block": torch.tensor([1.0]),
        "classifier.weight": torch.ones(2, 2),
        "classifier.bias": torch.zeros(2),
    }
    second = {key: value.clone() for key, value in first.items()}
    second["classifier.weight"].mul_(3.0)
    assert_non_classifier_state_equal(first, second)


def test_weighted_single_prototype_is_normalized_centroid():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    weights = torch.tensor([3.0, 1.0, 1.0, 3.0])
    prototypes = fit_weighted_multiprototypes(
        features,
        labels,
        weights,
        num_classes=2,
        prototypes_per_class=1,
    )
    assert prototypes.shape == (2, 1, 2)
    assert torch.allclose(prototypes.norm(dim=2), torch.ones(2, 1))
    assert prototypes[0, 0, 0] > prototypes[0, 0, 1]
    assert prototypes[1, 0, 1] < prototypes[1, 0, 0]


def test_multiprototype_max_uses_closest_mode():
    prototypes = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    scores = multiprototype_logits(
        torch.tensor([[0.0, 2.0], [-3.0, 0.0]]),
        prototypes,
        aggregation="max",
    )
    assert scores.argmax(dim=1).tolist() == [0, 1]


def test_score_scale_matches_mean_row_spread():
    reference = torch.tensor([[0.0, 2.0], [1.0, 5.0]])
    candidate = reference / 4.0
    assert match_score_scale(reference, candidate) == pytest.approx(4.0)


def test_embedded_head_blend_matches_manual_residual():
    features = torch.tensor([[1.0, 0.0]])
    base = torch.tensor([[0.2, 0.4]])
    prototypes = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    head = {
        "prototypes": prototypes,
        "aggregation": "max",
        "alpha": 0.5,
        "candidate_scale": 2.0,
    }
    actual = blend_multiprototype_logits(base, features, head)
    expected = base + multiprototype_logits(
        features, prototypes, aggregation="max"
    )
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    "mode",
    [
        "mean_logits",
        "mean_probabilities",
        "entropy_weighted_probabilities",
        "standardized_logits",
        "max_margin",
    ],
)
def test_all_tta_modes_return_finite_scores(mode):
    first = torch.tensor([[2.0, 1.0, -1.0], [0.5, 0.1, 0.2]])
    second = torch.tensor([[1.0, 3.0, 0.0], [0.4, 0.2, 0.3]])
    fused = fuse_paired_logits(first, second, mode=mode, temperature=1.5)
    assert fused.shape == first.shape
    assert torch.isfinite(fused).all()


def test_mean_probability_fusion_matches_probability_average():
    first = torch.tensor([[2.0, 0.0]])
    second = torch.tensor([[0.0, 1.0]])
    fused = fuse_paired_logits(
        first, second, mode="mean_probabilities", temperature=1.0
    )
    expected = (first.softmax(1) + second.softmax(1)) / 2.0
    assert torch.allclose(fused.exp(), expected)


def test_tta_rejects_non_positive_temperature():
    logits = torch.ones(2, 3)
    with pytest.raises(ValueError, match="temperature must be positive"):
        fuse_paired_logits(logits, logits, temperature=0.0)
