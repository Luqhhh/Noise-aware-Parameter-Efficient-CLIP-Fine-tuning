import math

import pytest
import torch
import torch.nn.functional as F

from aegis_clip.cli.train_local_feature_adapter import train_local_feature_adapter
from aegis_clip.local_feature_adapter import (
    BottleneckLocalFeatureAdapter,
    fuse_global_local_log_probabilities,
    load_local_feature_adapter,
    validate_local_adapter_cache,
)


def test_zero_initialisation_preserves_local_features() -> None:
    adapter = BottleneckLocalFeatureAdapter(4, 2, residual_scale=0.25, dropout=0.1)
    adapter.eval()
    features = F.normalize(torch.randn(5, 4), dim=1)

    assert torch.equal(adapter(features), features)
    assert adapter.residual_parameter_norm() == 0.0


def test_probability_fusion_matches_direct_mean() -> None:
    global_logits = torch.randn(6, 3)
    local_logits = torch.randn(6, 3)

    fused = fuse_global_local_log_probabilities(global_logits, local_logits)
    expected = (
        F.softmax(global_logits, dim=1) + F.softmax(local_logits, dim=1)
    ) / 2.0

    assert torch.allclose(fused.exp(), expected, atol=1.0e-7, rtol=1.0e-6)
    assert torch.allclose(
        fused.logsumexp(dim=1), torch.zeros(6), atol=1.0e-6, rtol=0.0
    )


def _valid_cache() -> dict[str, object]:
    return {
        "paths": ["a.jpg", "b.jpg"],
        "labels": torch.tensor([0, 1]),
        "clean_probability": torch.tensor([0.8, 0.9]),
        "pseudo_labels": torch.tensor([0, 1]),
        "correction_alpha": torch.zeros(2),
        "global_logits": torch.randn(2, 3),
        "local_features": torch.randn(2, 4),
        "local_logits": torch.randn(2, 3),
        "checkpoint_sha256": "abc",
    }


def test_local_adapter_cache_validation() -> None:
    assert validate_local_adapter_cache(
        _valid_cache(), expected_feature_dim=4, expected_num_classes=3
    ) == 2


def test_local_adapter_cache_rejects_nonfinite_values() -> None:
    payload = _valid_cache()
    payload["local_logits"] = torch.tensor(
        [[0.0, math.nan, 0.0], [0.0, 0.0, 0.0]]
    )
    with pytest.raises(ValueError, match="non-finite"):
        validate_local_adapter_cache(payload)


def test_load_local_feature_adapter_round_trip() -> None:
    source = BottleneckLocalFeatureAdapter(4, 2, residual_scale=0.25, dropout=0.1)
    checkpoint = {
        "local_feature_adapter": {
            "spec": {
                "feature_dim": 4,
                "bottleneck_dim": 2,
                "residual_scale": 0.25,
                "dropout": 0.1,
            },
            "state_dict": source.state_dict(),
        }
    }

    restored = load_local_feature_adapter(checkpoint, torch.device("cpu"))

    for expected, actual in zip(source.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)
    assert not restored.training


def test_local_adapter_training_materialises_single_composite_checkpoint(
    tmp_path,
) -> None:
    generator = torch.Generator().manual_seed(7)
    classifier_weight = torch.randn(500, 512, generator=generator)
    classifier_bias = torch.randn(500, generator=generator)
    parent_path = tmp_path / "parent.pt"
    torch.save(
        {
            "config": {"model": {"classifier_mode": "linear"}},
            "model_state_dict": {
                "classifier.weight": classifier_weight,
                "classifier.bias": classifier_bias,
            },
        },
        parent_path,
    )
    from aegis_clip.runtime import sha256_file

    parent_sha = sha256_file(parent_path)

    def cache(paths: list[str]) -> dict[str, object]:
        local_features = F.normalize(
            torch.randn(len(paths), 512, generator=generator), dim=1
        )
        return {
            "paths": paths,
            "labels": torch.arange(len(paths)).long(),
            "clean_probability": torch.full((len(paths),), 0.9),
            "pseudo_labels": torch.arange(len(paths)).long(),
            "correction_alpha": torch.zeros(len(paths)),
            "global_logits": torch.randn(len(paths), 500, generator=generator),
            "local_features": local_features,
            "local_logits": F.linear(
                local_features, classifier_weight, classifier_bias
            ),
            "checkpoint_sha256": parent_sha,
        }

    train_cache = cache(["train-a.jpg", "train-b.jpg", "train-c.jpg"])
    validation_cache = cache(["val-a.jpg", "val-b.jpg", "val-c.jpg"])
    train_path = tmp_path / "train.pt"
    validation_path = tmp_path / "validation.pt"
    center_path = tmp_path / "center.pt"
    m1_path = tmp_path / "m1.pt"
    torch.save(train_cache, train_path)
    torch.save(validation_cache, validation_path)
    torch.save(
        {
            "paths": validation_cache["paths"],
            "logits": validation_cache["global_logits"],
        },
        center_path,
    )
    torch.save(
        {
            "paths": validation_cache["paths"],
            "logits": fuse_global_local_log_probabilities(
                validation_cache["global_logits"],
                validation_cache["local_logits"],
            ),
        },
        m1_path,
    )

    result = train_local_feature_adapter(
        parent_path,
        train_path,
        validation_path,
        tmp_path / "output",
        center_reference_path=center_path,
        m1_reference_path=m1_path,
        expected_train_samples=3,
        bottleneck_dim=2,
        batch_size=2,
        max_epochs=1,
        patience=1,
        device_name="cpu",
    )

    composite = torch.load(result, map_location="cpu", weights_only=False)
    assert "local_feature_adapter" in composite
    assert composite["local_feature_adapter"]["spec"]["shared_classifier"] is True
    assert (tmp_path / "output" / "gate.json").is_file()
