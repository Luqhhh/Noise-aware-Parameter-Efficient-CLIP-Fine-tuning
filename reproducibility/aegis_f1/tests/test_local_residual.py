import pytest
import torch

from aegis_clip.local_residual import (
    LearnedLocalResidualHead,
    validate_dual_view_cache,
)


def test_zero_initialisation_preserves_base_logits() -> None:
    head = LearnedLocalResidualHead(4, 3, dropout=0.1)
    head.eval()
    base = torch.randn(5, 3)
    local = torch.randn(5, 4)

    assert torch.equal(head(base, local), base)
    assert head.residual_parameter_norm() == 0.0


def test_local_residual_receives_gradients() -> None:
    head = LearnedLocalResidualHead(4, 3, dropout=0.0)
    output = head(torch.zeros(2, 3), torch.randn(2, 4))
    output.sum().backward()

    assert head.local_classifier.weight.grad is not None
    assert head.local_classifier.bias.grad is not None


def test_local_residual_rejects_misaligned_features() -> None:
    head = LearnedLocalResidualHead(4, 3)
    with pytest.raises(ValueError, match="dimension"):
        head(torch.randn(2, 3), torch.randn(2, 5))


def _valid_cache() -> dict[str, object]:
    return {
        "paths": ["a.jpg", "b.jpg"],
        "labels": torch.tensor([0, 1]),
        "clean_probability": torch.tensor([0.8, 0.9]),
        "pseudo_labels": torch.tensor([0, 1]),
        "correction_alpha": torch.zeros(2),
        "global_features": torch.randn(2, 4),
        "local_features": torch.randn(2, 4),
        "global_logits": torch.randn(2, 3),
    }


def test_dual_view_cache_validation() -> None:
    assert validate_dual_view_cache(
        _valid_cache(), expected_feature_dim=4, expected_num_classes=3
    ) == 2


def test_dual_view_cache_rejects_alignment_error() -> None:
    payload = _valid_cache()
    payload["local_features"] = torch.randn(1, 4)
    with pytest.raises(ValueError, match="misaligned"):
        validate_dual_view_cache(payload)
