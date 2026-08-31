from __future__ import annotations

import pytest
import torch

from aegis_clip.longtail import (
    build_sampler,
    per_sample_weights,
    resolve_longtail_config,
)


def _labels_and_counts() -> tuple[list[int], torch.Tensor]:
    labels = [0, 0, 0, 0, 1, 1, 2]
    counts = torch.tensor([4.0, 2.0, 1.0])
    return labels, counts


def test_inverse_frequency_reweighting_normalizes_mean() -> None:
    labels, counts = _labels_and_counts()
    weights = per_sample_weights(
        labels, counts, "inverse_frequency", normalize=True
    )
    assert weights.numel() == len(labels)
    assert torch.allclose(weights.float().mean(), torch.tensor(1.0))
    head = weights[torch.tensor([0, 1, 2, 3])]
    tail = weights[torch.tensor([6])]
    assert head.mean() < tail


def test_balanced_oversample_has_one_max_count_cycle_per_class() -> None:
    labels, counts = _labels_and_counts()
    sampler = build_sampler(
        labels, counts, "balanced_oversample", num_classes=3
    )
    assert sampler is not None
    assert sampler.num_samples == 3 * 4
    assert sampler.replacement is True


def test_none_sampler_returns_none() -> None:
    labels, counts = _labels_and_counts()
    assert build_sampler(labels, counts, "none", num_classes=3) is None


def test_resolve_longtail_config_falls_back_to_legacy_tau() -> None:
    config = {
        "loss": {"class_prior_adjustment_tau": 0.5},
    }
    resolved = resolve_longtail_config(config)
    assert resolved["sampler_mode"] == "none"
    assert resolved["loss_reweighting"] == "none"
    assert resolved["balanced_softmax_tau"] == 0.5


def test_resolve_longtail_config_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="sampler_mode"):
        resolve_longtail_config(
            {"longtail": {"sampler_mode": "not_a_mode"}, "loss": {}}
        )
    with pytest.raises(ValueError, match="loss_reweighting"):
        resolve_longtail_config(
            {"longtail": {"loss_reweighting": "bogus"}, "loss": {}}
        )
