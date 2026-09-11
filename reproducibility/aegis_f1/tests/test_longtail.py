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


@pytest.mark.parametrize('beta', [0.5, 0.9999, 1.0 - 1e-10])
def test_effective_number_matches_high_precision_formula(beta):
    from decimal import Decimal, localcontext
    from aegis_clip.longtail import per_class_weights
    counts = torch.tensor([1., 2., 2., 10000.])
    with localcontext() as context:
        context.prec = 60
        b = Decimal.from_float(beta)
        expected = torch.tensor([float((1-b)/(1-b**int(n))) for n in counts])
    actual = per_class_weights(counts, 'effective_number', effective_number_beta=beta, normalize=False)
    assert torch.isfinite(actual).all() and (actual > 0).all()
    assert actual[1] == actual[2]
    torch.testing.assert_close(actual, expected)
    normalized = per_class_weights(counts, 'effective_number', effective_number_beta=beta)
    torch.testing.assert_close(normalized.mean(), torch.tensor(1.))


def test_effective_number_sample_mean_normalization():
    labels, counts = _labels_and_counts()
    weights = per_sample_weights(labels, counts, 'effective_number', normalize=True)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.))


@pytest.mark.parametrize('beta', [0., 1., -1., float('nan'), float('inf')])
def test_effective_number_rejects_invalid_beta(beta):
    from aegis_clip.longtail import per_class_weights
    with pytest.raises(ValueError, match='beta'):
        per_class_weights(torch.ones(2), 'effective_number', effective_number_beta=beta)


@pytest.mark.parametrize('counts', [[], [0.], [-1.], [float('nan')], [float('inf')]])
def test_effective_number_rejects_invalid_counts(counts):
    from aegis_clip.longtail import per_class_weights
    with pytest.raises(ValueError, match='counts'):
        per_class_weights(torch.tensor(counts), 'effective_number')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_effective_number_preserves_cuda_device_and_matches_cpu():
    from aegis_clip.longtail import per_class_weights
    counts = torch.tensor([1., 5., 10000.])
    cpu = per_class_weights(counts, 'effective_number')
    cuda = per_class_weights(counts.cuda(), 'effective_number')
    assert cuda.device.type == 'cuda'
    torch.testing.assert_close(cuda.cpu(), cpu)
