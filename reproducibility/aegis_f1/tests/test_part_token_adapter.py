import math

import pytest
import torch
import torch.nn.functional as F

from aegis_clip.cli.train_part_token_adapter import train_part_token_adapter
from aegis_clip.local_feature_adapter import fuse_global_local_log_probabilities
from aegis_clip.part_token_adapter import (
    PART_POOL_METHOD,
    PartTokenResidualAdapter,
    anchored_classifier_residual_logits,
    load_part_token_adapter,
    pool_cls_aligned_patch_features,
    validate_part_token_cache,
)


def test_anchored_classifier_is_exact_for_zero_feature_residual() -> None:
    base_logits = torch.randn(5, 3).half()
    base_features = torch.randn(5, 4)
    classifier_weight = torch.randn(3, 4)

    actual = anchored_classifier_residual_logits(
        base_logits,
        base_features,
        base_features,
        classifier_weight,
    )

    assert torch.equal(actual, base_logits.float())


def test_cls_aligned_pool_selects_matching_patch() -> None:
    local = torch.tensor([[1.0, 0.0]])
    patches = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]
    )

    pooled = pool_cls_aligned_patch_features(
        local,
        patches,
        top_patches=1,
        temperature=0.07,
    )

    assert torch.equal(pooled, local)


def test_cls_aligned_pool_has_stable_tie_breaking() -> None:
    local = torch.tensor([[1.0, 0.0]])
    patches = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])

    first = pool_cls_aligned_patch_features(local, patches, top_patches=1)
    second = pool_cls_aligned_patch_features(local, patches, top_patches=1)

    assert torch.equal(first, second)


def test_zero_initialisation_preserves_local_features() -> None:
    adapter = PartTokenResidualAdapter(4, 2, residual_scale=0.25, dropout=0.1)
    adapter.eval()
    local = F.normalize(torch.randn(5, 4), dim=1)
    part = F.normalize(torch.randn(5, 4), dim=1)

    assert torch.equal(adapter(local, part), local)
    assert adapter.residual_parameter_norm() == 0.0


def _valid_cache() -> dict[str, object]:
    return {
        "paths": ["a.jpg", "b.jpg"],
        "labels": torch.tensor([0, 1]),
        "clean_probability": torch.tensor([0.8, 0.9]),
        "pseudo_labels": torch.tensor([0, 1]),
        "correction_alpha": torch.zeros(2),
        "global_logits": torch.randn(2, 3),
        "local_features": F.normalize(torch.randn(2, 4), dim=1),
        "local_logits": torch.randn(2, 3),
        "part_features": F.normalize(torch.randn(2, 4), dim=1),
        "part_pool_spec": {
            "method": PART_POOL_METHOD,
            "top_patches": 2,
            "temperature": 0.07,
        },
        "checkpoint_sha256": "abc",
    }


def test_part_token_cache_validation() -> None:
    assert validate_part_token_cache(
        _valid_cache(),
        expected_feature_dim=4,
        expected_num_classes=3,
    ) == 2


def test_part_token_cache_rejects_nonfinite_features() -> None:
    payload = _valid_cache()
    payload["part_features"] = torch.tensor(
        [[0.0, math.nan, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    with pytest.raises(ValueError, match="non-finite"):
        validate_part_token_cache(payload)


def test_load_part_token_adapter_round_trip() -> None:
    source = PartTokenResidualAdapter(4, 2, residual_scale=0.25, dropout=0.1)
    checkpoint = {
        "part_token_adapter": {
            "spec": {
                "feature_dim": 4,
                "bottleneck_dim": 2,
                "residual_scale": 0.25,
                "dropout": 0.1,
                "part_pool_spec": {
                    "method": PART_POOL_METHOD,
                    "top_patches": 8,
                    "temperature": 0.07,
                },
            },
            "state_dict": source.state_dict(),
        }
    }

    restored = load_part_token_adapter(checkpoint, torch.device("cpu"))

    for expected, actual in zip(source.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)
    assert not restored.training


def test_part_token_training_materialises_composite_checkpoint(tmp_path) -> None:
    generator = torch.Generator().manual_seed(11)
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
    pool_spec = {
        "method": PART_POOL_METHOD,
        "top_patches": 8,
        "temperature": 0.07,
    }

    def cache(paths: list[str]) -> dict[str, object]:
        local_features = F.normalize(
            torch.randn(len(paths), 512, generator=generator),
            dim=1,
        )
        part_features = F.normalize(
            torch.randn(len(paths), 512, generator=generator),
            dim=1,
        )
        return {
            "paths": paths,
            "labels": torch.arange(len(paths)).long(),
            "clean_probability": torch.full((len(paths),), 0.9),
            "pseudo_labels": torch.arange(len(paths)).long(),
            "correction_alpha": torch.zeros(len(paths)),
            "global_logits": torch.randn(
                len(paths), 500, generator=generator
            ),
            "local_features": local_features,
            "local_logits": F.linear(
                local_features,
                classifier_weight,
                classifier_bias,
            ),
            "part_features": part_features,
            "part_pool_spec": pool_spec,
            "checkpoint_sha256": parent_sha,
            "execution": {"batch_size": 128},
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

    result = train_part_token_adapter(
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
    assert "part_token_adapter" in composite
    assert composite["part_token_adapter"]["spec"]["shared_classifier"] is True
    assert composite["part_token_adapter"]["gate"]["global_path_bit_exact"]
    assert (tmp_path / "output" / "gate.json").is_file()
