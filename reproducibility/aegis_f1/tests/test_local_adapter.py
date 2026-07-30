import pytest
import torch

from aegis_clip.local_adapter import (
    build_local_adapter,
    classifier_parameters_from_checkpoint,
    classify_adapted_local_features,
    validate_local_view_cache,
)


def _cache() -> dict:
    return {
        "base_checkpoint_sha256": "abc",
        "split": "train",
        "paths": ["0000/a.jpg", "0001/b.jpg"],
        "labels": torch.tensor([0, 1]),
        "clean_probability": torch.tensor([0.8, 0.9]),
        "pseudo_label": torch.tensor([0, 1]),
        "correction_alpha": torch.zeros(2),
        "global_features": torch.randn(2, 4),
        "local_features": torch.randn(2, 4),
    }


def test_validate_local_view_cache_checks_lineage_and_alignment() -> None:
    assert validate_local_view_cache(
        _cache(),
        expected_checkpoint_sha256="abc",
        expected_split="train",
    ) == 2
    broken = _cache()
    broken["local_features"] = torch.randn(1, 4)
    with pytest.raises(ValueError, match="misaligned"):
        validate_local_view_cache(broken)


def test_validate_local_view_cache_rejects_checkpoint_mismatch() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_local_view_cache(
            _cache(), expected_checkpoint_sha256="different"
        )


def test_classifier_parameters_require_linear_head() -> None:
    weight, bias = classifier_parameters_from_checkpoint(
        {
            "model_state_dict": {
                "classifier.weight": torch.randn(3, 4),
                "classifier.bias": torch.randn(3),
            }
        }
    )
    assert weight.shape == (3, 4)
    assert bias.shape == (3,)
    with pytest.raises(ValueError, match="linear classifier"):
        classifier_parameters_from_checkpoint({"model_state_dict": {}})


def test_zero_initialised_adapter_preserves_normalized_local_features() -> None:
    adapter = build_local_adapter(
        feature_dim=4,
        bottleneck_dim=2,
        residual_scale=0.25,
    )
    features = torch.randn(3, 4)
    weight = torch.randn(5, 4)
    bias = torch.randn(5)
    logits, adapted = classify_adapted_local_features(
        adapter, features, weight, bias
    )
    expected = torch.nn.functional.normalize(features, dim=1)
    assert torch.allclose(adapted, expected)
    assert torch.allclose(logits, torch.nn.functional.linear(expected, weight, bias))
