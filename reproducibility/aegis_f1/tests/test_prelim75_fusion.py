"""PRELIM75 v5 probability-fusion loss and invariants."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from aegis_clip.losses import soft_generalized_cross_entropy
from aegis_clip.prelim75_fusion import fusion_gce_terms, probability_gce


def _fixture(batch=6, classes=5, dtype=torch.float32):
    torch.manual_seed(91)
    global_logits = torch.randn(batch, classes, dtype=dtype)
    local_logits = torch.randn(batch, classes, dtype=dtype)
    targets = torch.softmax(torch.randn(batch, classes, dtype=dtype), 1)
    return global_logits, local_logits, targets


def test_f0_matches_repository_gce_at_temperature_1p5():
    global_logits, local_logits, targets = _fixture()
    actual = fusion_gce_terms(global_logits, local_logits, targets, blend=0.0)
    expected = (
        0.6 * soft_generalized_cross_entropy(global_logits / 1.5, targets, q=0.5)
        + 0.4 * soft_generalized_cross_entropy(local_logits / 1.5, targets, q=0.5)
    )
    torch.testing.assert_close(actual["objective"], expected)


def test_f1_matches_explicit_probability_formula():
    global_logits, local_logits, targets = _fixture()
    actual = fusion_gce_terms(global_logits, local_logits, targets, blend=0.5)
    pg = torch.softmax(global_logits / 1.5, 1)
    pl = torch.softmax(local_logits / 1.5, 1)
    pm = 0.6 * pg + 0.4 * pl
    explicit_mix = (targets * (1.0 - pm.clamp_min(1e-7).sqrt()) / 0.5).sum(1)
    expected = 0.5 * actual["separate"] + 0.5 * explicit_mix
    torch.testing.assert_close(actual["fusion"], explicit_mix)
    torch.testing.assert_close(actual["objective"], expected)


def test_identical_branches_make_separate_and_fusion_equal():
    logits, _, targets = _fixture()
    terms = fusion_gce_terms(logits, logits, targets, blend=0.5)
    torch.testing.assert_close(terms["global"], terms["local"])
    torch.testing.assert_close(terms["separate"], terms["fusion"])


def test_probability_fusion_is_not_weighted_logit_softmax():
    global_logits = torch.tensor([[5.0, 0.0, -2.0]])
    local_logits = torch.tensor([[-1.0, 2.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    terms = fusion_gce_terms(global_logits, local_logits, targets, blend=0.5)
    wrong = torch.log_softmax((0.6 * global_logits + 0.4 * local_logits) / 1.5, 1)
    assert not torch.allclose(terms["log_mix"], wrong)


def test_fusion_term_has_gradient_to_both_logits():
    global_logits, local_logits, targets = _fixture()
    global_logits.requires_grad_(True); local_logits.requires_grad_(True)
    fusion_gce_terms(global_logits, local_logits, targets, blend=0.5)["fusion"].sum().backward()
    assert global_logits.grad.abs().sum() > 0
    assert local_logits.grad.abs().sum() > 0


def test_fp64_gradcheck():
    global_logits, local_logits, targets = _fixture(batch=2, classes=3, dtype=torch.float64)
    global_logits.requires_grad_(True); local_logits.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda g, l: fusion_gce_terms(g, l, targets, blend=0.5)["objective"].sum(),
        (global_logits, local_logits), eps=1e-6, atol=1e-5, rtol=1e-4,
    )


def test_zero_weight_row_has_zero_classification_gradient():
    global_logits, local_logits, targets = _fixture(batch=3)
    global_logits.requires_grad_(True); local_logits.requires_grad_(True)
    weights = torch.tensor([1.0, 0.0, 1.0])
    terms = fusion_gce_terms(global_logits, local_logits, targets, blend=0.5)
    (terms["objective"] * weights).sum().backward()
    assert global_logits.grad[1].abs().sum() == 0
    assert local_logits.grad[1].abs().sum() == 0


def test_all_zero_weight_batch_is_differentiable_zero():
    global_logits, local_logits, targets = _fixture(batch=3)
    global_logits.requires_grad_(True); local_logits.requires_grad_(True)
    weights = torch.zeros(3)
    terms = fusion_gce_terms(global_logits, local_logits, targets, blend=0.5)
    loss = (terms["objective"] * weights).sum() / weights.sum().clamp_min(1e-8)
    loss.backward()
    assert loss == 0
    assert global_logits.grad.abs().sum() == 0 and local_logits.grad.abs().sum() == 0


def test_common_denominator_makes_two_microbatches_equal_full_batch():
    global_full, local_full, targets = _fixture(batch=32)
    global_micro = global_full.clone(); local_micro = local_full.clone()
    global_full.requires_grad_(True); local_full.requires_grad_(True)
    global_micro.requires_grad_(True); local_micro.requires_grad_(True)
    weights = torch.rand(32); weights[::7] = 0; denominator = weights.sum()

    def objective(global_logits, local_logits, slices):
        result = global_logits.sum() * 0
        for item in slices:
            terms = fusion_gce_terms(global_logits[item], local_logits[item], targets[item], blend=0.5)
            result = result + (terms["objective"] * weights[item]).sum() / denominator
        return result

    full = objective(global_full, local_full, [slice(None)])
    micro = objective(global_micro, local_micro, [slice(0, 16), slice(16, 32)])
    full.backward(); micro.backward()
    torch.testing.assert_close(full, micro)
    torch.testing.assert_close(global_full.grad, global_micro.grad)
    torch.testing.assert_close(local_full.grad, local_micro.grad)


def test_extreme_finite_logits_remain_finite():
    global_logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
    local_logits = -global_logits
    targets = F.one_hot(torch.tensor([0, 1]), 2).float()
    terms = fusion_gce_terms(global_logits, local_logits, targets, blend=0.5)
    assert all(torch.isfinite(terms[key]).all() for key in ("global", "local", "fusion", "objective"))


def test_sampled_fusion_probability_expectation_matches_enumeration():
    global_views = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    local_views = torch.tensor([[0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])
    scale_weights = torch.tensor([0.2, 0.3, 0.4, 0.1])
    expected = 0.6 * global_views.mean(0) + 0.4 * (local_views * scale_weights[:, None]).sum(0)
    enumerated = sum(
        0.5 * float(scale_weights[s]) * (0.6 * global_views[o] + 0.4 * local_views[s])
        for o in range(2) for s in range(4)
    )
    torch.testing.assert_close(enumerated, expected)


def test_expected_nonlinear_loss_differs_from_loss_of_expected_probability():
    targets = torch.tensor([[1.0, 0.0]])
    probabilities = [torch.tensor([[0.9, 0.1]]), torch.tensor([[0.2, 0.8]])]
    expected_loss = sum(probability_gce(p.log(), targets) for p in probabilities) / 2
    loss_of_expected = probability_gce(((probabilities[0] + probabilities[1]) / 2).log(), targets)
    assert not torch.allclose(expected_loss, loss_of_expected)


def test_mixture_gce_obeys_convexity_for_moderate_probabilities():
    global_logits, local_logits, targets = _fixture(batch=20)
    terms = fusion_gce_terms(global_logits, local_logits, targets, blend=0.5)
    assert bool((terms["fusion"] <= terms["separate"] + 1e-6).all())


def test_targets_requiring_gradient_are_rejected():
    log_probabilities = torch.log_softmax(torch.randn(2, 3), 1)
    targets = torch.softmax(torch.randn(2, 3), 1).requires_grad_(True)
    try:
        probability_gce(log_probabilities, targets)
    except ValueError:
        pass
    else:
        raise AssertionError("learnable supervision target accepted")


def test_shape_mismatch_is_rejected():
    global_logits, local_logits, targets = _fixture()
    try:
        fusion_gce_terms(global_logits, local_logits[:, :-1], targets, blend=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched fusion shapes accepted")


def test_invalid_target_distribution_is_rejected():
    global_logits, local_logits, targets = _fixture()
    targets[0, 0] = -0.1
    try:
        fusion_gce_terms(global_logits, local_logits, targets, blend=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid target distribution accepted")


def test_nonfinite_logits_and_unregistered_controls_are_rejected():
    global_logits, local_logits, targets = _fixture()
    global_logits[0, 0] = math.inf
    failures = 0
    for kwargs in ({"blend": 0.25}, {"blend": 0.5, "temperature": 1.0}):
        try:
            fusion_gce_terms(local_logits, local_logits, targets, **kwargs)
        except ValueError:
            failures += 1
    try:
        fusion_gce_terms(global_logits, local_logits, targets, blend=0.5)
    except ValueError:
        failures += 1
    assert failures == 3
