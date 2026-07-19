import pytest
import torch
import torch.nn.functional as F

from aegis_clip.soup import (
    assert_non_classifier_state_equal,
    interpolate_linear_heads,
)


def test_interpolated_head_exactly_matches_logit_interpolation() -> None:
    features = torch.randn(6, 3)
    first_weight, second_weight = torch.randn(2, 3), torch.randn(2, 3)
    first_bias, second_bias = torch.randn(2), torch.randn(2)
    weight, bias = interpolate_linear_heads(
        first_weight, first_bias, second_weight, second_bias, alpha=0.4
    )
    expected = torch.lerp(
        F.linear(features, first_weight, first_bias),
        F.linear(features, second_weight, second_bias),
        0.4,
    )
    assert torch.allclose(F.linear(features, weight, bias), expected, atol=1.0e-6)


def test_non_classifier_mismatch_is_rejected() -> None:
    first = {
        "visual.weight": torch.tensor([1.0]),
        "classifier.weight": torch.tensor([2.0]),
        "classifier.bias": torch.tensor([3.0]),
    }
    second = {key: value.clone() for key, value in first.items()}
    assert_non_classifier_state_equal(first, second)
    second["visual.weight"] += 1.0
    with pytest.raises(ValueError, match="visual.weight"):
        assert_non_classifier_state_equal(first, second)
