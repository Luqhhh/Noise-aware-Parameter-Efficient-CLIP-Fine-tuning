import torch

import pytest

from aegis_clip.trainer import (
    _checkpoint_is_selected,
    _select_training_forward,
    _validate_train_val_overlap,
)


def test_frozen_cached_batch_is_reported_as_cached_forward() -> None:
    features = torch.randn(4, 512)
    key, selected, cached = _select_training_forward(
        peft=False,
        input_key="features",
        mixed_inputs=features,
        mixed_reference=features,
        mix_lambda=1.0,
    )
    assert key == "features"
    assert selected is features
    assert cached is True


def test_frozen_unmixed_online_batch_reuses_frozen_features() -> None:
    images = torch.randn(4, 3, 8, 8)
    features = torch.randn(4, 512)
    key, selected, cached = _select_training_forward(
        peft=False,
        input_key="images",
        mixed_inputs=images,
        mixed_reference=features,
        mix_lambda=1.0,
    )
    assert key == "features"
    assert selected is features
    assert cached is True


def test_mixup_and_peft_never_skip_online_visual_forward() -> None:
    images = torch.randn(4, 3, 8, 8)
    features = torch.randn(4, 512)
    for peft, mix_lambda in ((False, 0.7), (True, 1.0)):
        key, selected, cached = _select_training_forward(
            peft=peft,
            input_key="images",
            mixed_inputs=images,
            mixed_reference=features,
            mix_lambda=mix_lambda,
        )
        assert key == "images"
        assert selected is images
        assert cached is False


def test_last_epoch_policy_never_uses_overlapping_validation_to_stop() -> None:
    assert _checkpoint_is_selected(
        selection_policy="last_epoch", selector=0.1, best_selector=0.9
    )


def test_best_selector_policy_keeps_strict_improvement() -> None:
    assert _checkpoint_is_selected(
        selection_policy="best_selector", selector=0.91, best_selector=0.9
    )
    assert not _checkpoint_is_selected(
        selection_policy="best_selector", selector=0.9, best_selector=0.9
    )


def test_train_val_overlap_is_canonical_and_fails_closed() -> None:
    with pytest.raises(ValueError, match=r"overlap: 1"):
        _validate_train_val_overlap(
            ["train_dedup/0001/example.jpg"],
            ["train/0001/example.jpg"],
            allow_overlap=False,
        )


def test_explicit_diagnostic_overlap_requires_validation_subset() -> None:
    _validate_train_val_overlap(
        ["train/0001/a.jpg", "train/0001/b.jpg"],
        ["train/0001/b.jpg"],
        allow_overlap=True,
    )
    with pytest.raises(ValueError, match="must be a subset"):
        _validate_train_val_overlap(
            ["train/0001/a.jpg"],
            ["train/0001/a.jpg", "train/0001/b.jpg"],
            allow_overlap=True,
        )
