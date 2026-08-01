from __future__ import annotations

import torch
import pytest

from aegis_clip.cvrg_crossfit import (
    CVRGFitConfig,
    cross_fit_reliability,
    make_image_folds,
    select_regularization_c,
)


def synthetic(seed: int = 4):
    generator = torch.Generator().manual_seed(seed)
    n, f, c = 30, 5, 7
    labels = torch.arange(n) % c
    logits = torch.randn(n, 4, c, generator=generator)
    logits[:, :, 0] += (labels == 0).float()[:, None] * 4
    features = torch.randn(n, 4, f, generator=generator)
    groups = [f"image-{i}" for i in range(n)]
    return features, logits, labels, groups


def test_group_folds_are_deterministic_and_complete():
    labels = torch.arange(30) % 5
    groups = [f"g-{i}" for i in range(30)]
    first = make_image_folds(labels, groups, folds=5, seed=42)
    second = make_image_folds(labels, groups, folds=5, seed=42)
    assert torch.equal(first, second)
    assert sorted(first.tolist()) == [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6


def test_nested_cross_fit_is_deterministic_and_has_oof_for_every_image():
    features, logits, labels, groups = synthetic()
    config = CVRGFitConfig(outer_folds=3, inner_folds=3, maximum_iterations=200)
    first = cross_fit_reliability(features, logits, labels, groups, config=config)
    second = cross_fit_reliability(features, logits, labels, groups, config=config)
    assert torch.equal(first.outer_fold_id, second.outer_fold_id)
    assert torch.equal(first.oof_reliability, second.oof_reliability)
    assert first.selected_c_by_outer_fold == second.selected_c_by_outer_fold
    assert first.oof_reliability.shape == (30, 4)
    assert torch.isfinite(first.oof_reliability).all()
    assert torch.all((first.oof_reliability >= 0) & (first.oof_reliability <= 1))


def test_candidate_values_and_tie_break_are_preregistered():
    assert CVRGFitConfig().c_candidates == (0.01, 0.1, 1.0)
    features, logits, labels, groups = synthetic()
    selected, scores = select_regularization_c(
        features, logits, labels, groups, torch.arange(30),
        config=CVRGFitConfig(outer_folds=3, inner_folds=3), seed_offset=1,
    )
    assert selected in (0.01, 0.1, 1.0)
    assert set(scores) == {"0.01", "0.1", "1.0"}


def test_one_class_target_fails_closed():
    features, logits, labels, groups = synthetic()
    logits.zero_()
    for index, label in enumerate(labels.tolist()):
        logits[index, :, label] = 10.0
    with pytest.raises(ValueError, match="both correctness classes"):
        cross_fit_reliability(features, logits, labels, groups)
