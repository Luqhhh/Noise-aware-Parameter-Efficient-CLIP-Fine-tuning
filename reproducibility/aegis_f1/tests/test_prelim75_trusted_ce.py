"""PRELIM75 v4 trusted-global CE and fixed-cohort invariants."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from aegis_clip.prelim75_trusted_ce import build_trusted_cohort, trusted_beta, trusted_ce_mix
from aegis_clip.losses import soft_generalized_cross_entropy


def test_beta_zero_and_empty_mask_preserve_original_tensor_path():
    logits = torch.randn(4, 3)
    targets = torch.softmax(torch.randn(4, 3), 1)
    base = soft_generalized_cross_entropy(logits, targets, q=.5)
    assert trusted_ce_mix(base, logits, targets, torch.ones(4, dtype=torch.bool), 0.0) is base
    assert trusted_ce_mix(base, logits, targets, torch.zeros(4, dtype=torch.bool), .25) is base


def test_trusted_ce_mix_is_exact_convex_per_sample_mix():
    logits = torch.tensor([[1.2, -.5, .3], [-.2, .4, 1.1]])
    targets = F.one_hot(torch.tensor([0, 2]), 3).float()
    base = soft_generalized_cross_entropy(logits, targets, q=.5)
    eligible = torch.tensor([True, False])
    actual = trusted_ce_mix(base, logits, targets, eligible, .25)
    ce = -(targets * F.log_softmax(logits, 1)).sum(1)
    torch.testing.assert_close(actual[0], .75 * base[0] + .25 * ce[0])
    torch.testing.assert_close(actual[1], base[1])


def test_one_hot_gradient_coefficient_matches_formula():
    logits_gce = torch.tensor([[0.1, -1.0, .4]], requires_grad=True)
    logits_mix = logits_gce.detach().clone().requires_grad_(True)
    targets = torch.tensor([[0., 0., 1.]])
    base_gce = soft_generalized_cross_entropy(logits_gce, targets, q=.5)
    base_mix = soft_generalized_cross_entropy(logits_mix, targets, q=.5)
    mixed = trusted_ce_mix(base_mix, logits_mix, targets, torch.tensor([True]), .25)
    base_gce.sum().backward(); mixed.sum().backward()
    probability = torch.softmax(logits_gce.detach(), 1)[0, 2]
    expected_ratio = ((.75 * probability.sqrt() + .25) / probability.sqrt())
    torch.testing.assert_close(logits_mix.grad, logits_gce.grad * expected_ratio)


def test_common_weight_denominator_makes_32_equal_two_16_microbatches():
    torch.manual_seed(3)
    logits_full = torch.randn(32, 5, requires_grad=True)
    logits_micro = logits_full.detach().clone().requires_grad_(True)
    targets = torch.softmax(torch.randn(32, 5), 1)
    weights = torch.rand(32); weights[::9] = 0
    eligible = torch.zeros(32, dtype=torch.bool); eligible[[1, 8, 19, 27]] = True
    denominator = weights.sum()

    def objective(logits, slices):
        result = logits.sum() * 0
        for item in slices:
            base = soft_generalized_cross_entropy(logits[item], targets[item], q=.5)
            mixed = trusted_ce_mix(base, logits[item], targets[item], eligible[item], .25)
            result = result + (mixed * weights[item]).sum() / denominator
        return result

    full = objective(logits_full, [slice(None)])
    micro = objective(logits_micro, [slice(0, 16), slice(16, 32)])
    full.backward(); micro.backward()
    torch.testing.assert_close(full, micro); torch.testing.assert_close(logits_full.grad, logits_micro.grad)


def test_zero_weight_row_has_zero_classification_gradient_even_when_eligible():
    logits = torch.tensor([[1., -1.], [.2, .4]], requires_grad=True)
    targets = F.one_hot(torch.tensor([0, 1]), 2).float()
    base = soft_generalized_cross_entropy(logits, targets, q=.5)
    mixed = trusted_ce_mix(base, logits, targets, torch.tensor([True, True]), .25)
    (mixed * torch.tensor([0., 1.])).sum().backward()
    assert logits.grad[0].abs().sum() == 0 and logits.grad[1].abs().sum() > 0


def test_extreme_finite_logits_remain_finite_and_inputs_are_unchanged():
    logits = torch.tensor([[1000., -1000.], [-1000., 1000.]])
    targets = F.one_hot(torch.tensor([0, 1]), 2).float(); targets_before = targets.clone()
    base = soft_generalized_cross_entropy(logits, targets, q=.5); base_before = base.clone()
    result = trusted_ce_mix(base, logits, targets, torch.tensor([True, True]), .25)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(targets, targets_before); torch.testing.assert_close(base, base_before)


def test_invalid_beta_shape_and_mask_dtype_fail_closed():
    logits = torch.randn(2, 3); targets = torch.softmax(torch.randn(2, 3), 1)
    base = soft_generalized_cross_entropy(logits, targets, q=.5)
    for beta in (-.01, .251):
        try:
            trusted_ce_mix(base, logits, targets, torch.ones(2, dtype=torch.bool), beta)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid beta accepted")
    try:
        trusted_ce_mix(base, logits, targets, torch.ones(2), .25)
    except TypeError:
        pass
    else:
        raise AssertionError("non-bool mask accepted")


def _cohort_fixture():
    paths = [f"c{i % 3}/{i}.jpg" for i in range(8)]
    labels = torch.tensor([0, 0, 1, 1, 1, 2, 1, 2])
    weights = torch.tensor([1., 1., 1., 1., 0., 1., 1., 1.])
    clean = torch.tensor([.90, .95, .99, .89, .99, .99, .99, .99])
    targets = F.one_hot(labels, 500).float(); targets[2] = 0; targets[2, 4] = 1
    groups = ["a", "a", "b", "c", "d", "conflict", "conflict", "e"]
    return {
        "paths": paths, "labels": labels, "weights": weights, "targets": targets,
        "clean_probability": clean, "group_ids": groups,
    }


def test_cohort_requires_all_fixed_conditions_and_excludes_conflict_groups():
    data = _cohort_fixture()
    mask, report, rows = build_trusted_cohort(data, minimum_groups=2, minimum_classes=2)
    np.testing.assert_array_equal(mask.numpy(), [True, True, False, False, False, False, False, True])
    assert report["eligible_rows"] == 3
    assert report["eligible_unique_groups"] == 2
    assert report["eligible_classes"] == 2
    assert report["conflicting_groups"] == 1
    assert report["gate_passed"]
    assert rows[0]["eligible_rows"] == 2 and rows[2]["eligible_rows"] == 1


def test_cohort_threshold_is_inclusive_and_gate_does_not_relax():
    data = _cohort_fixture()
    mask, report, _ = build_trusted_cohort(data)
    assert mask[0]
    assert not report["gate_passed"]
    assert report["status"] == "skipped_insufficient_trusted_support"


def test_beta_ramps_from_zero_to_quarter_during_first_epoch_only():
    assert trusted_beta(0, 4) == 0.0
    assert trusted_beta(3, 4) == .25
    assert trusted_beta(4, 4) == .25
    assert trusted_beta(100, 4) == .25
