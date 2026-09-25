"""Tests for the fail-closed TTA + prior cache binding."""

import pytest
import torch

from aegis_clip.tta_prior_binding import (
    FIXED_RECIPE,
    canonical_cache_paths,
    fit_bound_prior,
    resolved_recipe,
    tensor_sha256,
    verify_prior_record,
)


def _identity(**overrides):
    identity = {
        "cache": "/tmp/cache.pt",
        "cache_sha256": "cache",
        "checkpoint": "/tmp/best.pt",
        "checkpoint_sha256": "checkpoint",
        "class_mapping_sha256": "mapping",
        "dataset_manifest_sha256": "manifest",
        "dataset_fingerprint": "fingerprint",
        "split_seed": 42,
        "validation_csv_sha256": "valcsv",
        "sample_order_sha256": "order",
        "content_group_set_sha256": "groups",
        "num_samples": 8,
        "num_classes": 4,
        "input_resolution": 384,
        "preprocessing": "OpenAI CLIP resize",
        "encoder_precision": "float32",
        "feature_precision": "float32",
        "autocast": False,
    }
    identity.update(overrides)
    return identity


def _payload(rows=8, classes=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    original = torch.randn(rows, classes, generator=generator)
    flipped = torch.randn(rows, classes, generator=generator)
    return {
        "original_logits": original,
        "flip_logits": flipped,
        "labels": torch.randint(0, classes, (rows,), generator=generator),
    }


def test_resolved_recipe_freezes_the_l05_point():
    recipe = resolved_recipe(enforce_fixed=True)
    assert recipe == {
        "tta": "horizontal_flip",
        "fusion": "mean_probabilities",
        "temperature": 1.4,
        "prior_strength": 0.6,
    }
    assert FIXED_RECIPE["temperature"] == 1.4
    assert FIXED_RECIPE["prior_strength"] == 0.6


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 1.5},
        {"prior_strength": 0.5},
        {"fusion": "mean_logits"},
        {"tta": "vertical_flip"},
    ],
)
def test_resolved_recipe_rejects_drift_when_enforced(overrides):
    with pytest.raises(ValueError):
        resolved_recipe(enforce_fixed=True, **overrides)


def test_resolved_recipe_allows_declared_drift_when_not_enforced():
    recipe = resolved_recipe(temperature=1.2, prior_strength=0.3, enforce_fixed=False)
    assert recipe["temperature"] == 1.2
    assert recipe["prior_strength"] == 0.3


def test_tensor_sha256_is_deterministic_and_sensitive():
    tensor = torch.arange(8, dtype=torch.float32)
    assert tensor_sha256(tensor) == tensor_sha256(tensor.clone())
    assert tensor_sha256(tensor) != tensor_sha256(tensor + 1.0)


def test_canonical_cache_paths_normalises_prefixes():
    assert canonical_cache_paths(["train/0000/a.jpg", "./0001/b.jpg"]) == [
        "0000/a.jpg",
        "0001/b.jpg",
    ]


def test_prior_record_round_trip_and_mismatch_detection():
    identity = _identity()
    payload = _payload()
    recipe = resolved_recipe()
    bias, _, record = fit_bound_prior(payload, identity, recipe=recipe)
    verify_prior_record(record, identity, recipe=recipe, bias=bias)

    for field, value in (
        ("cache_sha256", "other"),
        ("checkpoint_sha256", "other"),
        ("class_mapping_sha256", "other"),
        ("dataset_manifest_sha256", "other"),
        ("dataset_fingerprint", "other"),
        ("split_seed", 7),
        ("validation_csv_sha256", "other"),
        ("sample_order_sha256", "other"),
        ("content_group_set_sha256", "other"),
        ("num_classes", 5),
        ("input_resolution", 224),
        ("preprocessing", "other"),
        ("encoder_precision", "float16"),
        ("feature_precision", "float16"),
        ("autocast", True),
    ):
        tampered = dict(record)
        tampered[field] = value
        with pytest.raises(ValueError):
            verify_prior_record(tampered, identity, recipe=recipe)


def test_prior_record_rejects_wrong_recipe_and_bias():
    identity = _identity()
    payload = _payload()
    recipe = resolved_recipe()
    bias, _, record = fit_bound_prior(payload, identity, recipe=recipe)
    with pytest.raises(ValueError):
        verify_prior_record(
            record,
            identity,
            recipe=resolved_recipe(temperature=1.1, enforce_fixed=False),
        )
    with pytest.raises(ValueError):
        verify_prior_record(record, identity, recipe=recipe, bias=bias + 0.01)
    with pytest.raises(ValueError):
        verify_prior_record({**record, "test_data_used": True}, identity, recipe=recipe)


def test_fold_fit_mask_never_reads_held_out_rows():
    identity = _identity(num_samples=8)
    payload = _payload()
    recipe = resolved_recipe()
    holdout = torch.tensor([True, True, True, True, False, False, False, False])
    _, _, record = fit_bound_prior(
        payload, identity, recipe=recipe, fit_mask=~holdout
    )
    assert record["fit_scope"]["mode"] == "conditional_fold_exclusion"
    assert record["fit_scope"]["fit_sample_count"] == 4
    assert record["test_data_used"] is False
