import pytest
import torch
import torch.nn.functional as F

from aegis_clip.structural import (
    blend_linear_heads,
    discriminant_from_statistics,
    fit_shrinkage_discriminant,
    match_linear_logit_scale,
    ridge_head_from_statistics,
    weighted_class_statistics,
    weighted_ridge_statistics,
)


def test_structural_discriminant_is_finite_linear_head() -> None:
    features = torch.tensor(
        [[-2.0, 0.0], [-1.0, 0.2], [1.0, -0.1], [2.0, 0.1]]
    )
    labels = torch.tensor([0, 0, 1, 1])
    weight, bias = fit_shrinkage_discriminant(
        features,
        labels,
        torch.ones(4),
        num_classes=2,
        shrinkage=0.5,
        covariance_batch_size=2,
    )
    assert weight.shape == (2, 2)
    assert bias.shape == (2,)
    assert torch.isfinite(weight).all()
    assert torch.equal(F.linear(features, weight, bias).argmax(1), labels)


def test_structural_discriminant_rejects_missing_class_mass() -> None:
    with pytest.raises(ValueError, match="class mass"):
        fit_shrinkage_discriminant(
            torch.randn(3, 2),
            torch.tensor([0, 0, 1]),
            torch.tensor([1.0, 1.0, 0.0]),
            num_classes=2,
            shrinkage=1.0,
        )


def test_reused_statistics_match_direct_fit() -> None:
    features = torch.randn(20, 4)
    labels = torch.arange(20) % 2
    weights = torch.linspace(0.2, 1.0, 20)
    expected = fit_shrinkage_discriminant(
        features, labels, weights, num_classes=2, shrinkage=0.25
    )
    statistics = weighted_class_statistics(
        features, labels, weights, num_classes=2
    )
    actual = discriminant_from_statistics(*statistics, shrinkage=0.25)
    assert torch.allclose(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])


def test_blended_head_exactly_matches_logit_sum() -> None:
    features = torch.randn(7, 3)
    first_weight, second_weight = torch.randn(2, 3), torch.randn(2, 3)
    first_bias, second_bias = torch.randn(2), torch.randn(2)
    weight, bias = blend_linear_heads(
        first_weight,
        first_bias,
        second_weight,
        second_bias,
        alpha=0.4,
        candidate_scale=1.7,
    )
    expected = F.linear(features, first_weight, first_bias) + 0.68 * F.linear(
        features, second_weight, second_bias
    )
    assert torch.allclose(F.linear(features, weight, bias), expected, atol=1.0e-6)


def test_logit_scale_match_equalizes_mean_spread() -> None:
    features = torch.randn(20, 4)
    weight = torch.randn(3, 4)
    bias = torch.randn(3)
    scale = match_linear_logit_scale(
        features, weight, bias, 5.0 * weight, 5.0 * bias
    )
    assert scale == pytest.approx(0.2, rel=1.0e-5)


def test_weighted_ridge_head_fits_separable_classes() -> None:
    features = torch.tensor(
        [[-2.0, 0.0], [-1.0, 0.1], [1.0, -0.1], [2.0, 0.0]]
    )
    labels = torch.tensor([0, 0, 1, 1])
    statistics = weighted_ridge_statistics(
        features, labels, torch.ones(4), num_classes=2
    )
    weight, bias = ridge_head_from_statistics(
        *statistics, ridge_strength=0.01
    )
    assert torch.equal(F.linear(features, weight, bias).argmax(1), labels)


def test_weighted_ridge_correction_changes_right_hand_side() -> None:
    features = torch.tensor([[-1.0], [1.0]])
    labels = torch.tensor([0, 1])
    plain = weighted_ridge_statistics(
        features, labels, torch.ones(2), num_classes=2
    )[1]
    corrected = weighted_ridge_statistics(
        features,
        labels,
        torch.ones(2),
        num_classes=2,
        pseudo_labels=torch.tensor([1, 1]),
        correction_alpha=torch.tensor([0.5, 0.0]),
    )[1]
    assert not torch.equal(plain, corrected)
