"""Synthetic CPU tests for PRELIM75 v8 bounded training-source bias."""
from __future__ import annotations

import argparse
import math

import numpy as np
import pytest
import torch

from aegis_clip.calibration_binding import protocol_sha256
from aegis_clip.prior_alignment import apply_prior_bias
from aegis_clip.source_bias import (
    DEFAULT_BOUND,
    SCORE_SEMANTICS,
    balanced_sample_weights,
    build_inference_protocol_descriptor,
    fit_bounded_source_bias,
    objective_delta_and_gradient,
    projected_gradient_inf_norm,
)


def _source():
    scores = torch.tensor(
        [[-0.2, -1.7, -2.1], [-2.0, -0.2, -2.4], [-0.4, -1.5, -1.8],
         [-2.3, -1.2, -0.1], [-0.9, -0.8, -2.0], [-1.7, -0.4, -1.0]],
        dtype=torch.float32,
    ).log_softmax(1)
    weights = torch.tensor([1.0, 0.8, 0.7, 1.0, 0.5, 0.9])
    targets = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.3, 0.0],
         [0.0, 0.0, 1.0], [0.2, 0.2, 0.6], [0.0, 0.8, 0.2]],
        dtype=torch.float32,
    )
    return scores, weights, targets


def test_score_semantics_is_final_fused_log_probability():
    assert SCORE_SEMANTICS == "log_final_fused_probabilities"


def test_balanced_row_weights_sum_to_one():
    _, weights, targets = _source()
    _, rows = balanced_sample_weights(weights, targets, chunk_size=2)
    assert math.isclose(float(rows.sum()), 1.0, abs_tol=1e-12)


def test_balanced_cell_weights_give_equal_class_mass():
    _, weights, targets = _source()
    counts, _ = balanced_sample_weights(weights, targets, chunk_size=2)
    cells = weights[:, None].double().numpy() * targets.double().numpy()
    cells = cells / (targets.shape[1] * counts[None, :])
    np.testing.assert_allclose(cells.sum(0), np.full(3, 1 / 3), atol=1e-12)


def test_balanced_weights_reject_shape_mismatch():
    _, weights, targets = _source()
    with pytest.raises(ValueError, match="shapes"):
        balanced_sample_weights(weights[:-1], targets)


def test_balanced_weights_reject_zero_class_support():
    _, weights, targets = _source()
    targets[:, 2] = 0
    targets[:, 0] += 1 - targets.sum(1)
    with pytest.raises(ValueError, match="strictly positive"):
        balanced_sample_weights(weights, targets)


def test_balanced_weights_reject_non_normalized_targets():
    _, weights, targets = _source()
    targets[0, 0] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        balanced_sample_weights(weights, targets)


def test_balanced_weights_reject_negative_values():
    _, weights, targets = _source()
    weights[0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        balanced_sample_weights(weights, targets)


def test_objective_difference_is_zero_at_zero_bias():
    scores, weights, targets = _source()
    _, rows = balanced_sample_weights(weights, targets)
    value, _ = objective_delta_and_gradient(np.zeros(3), scores, rows)
    assert abs(value) <= 1e-14


def test_compact_objective_matches_explicit_macro_soft_ce_difference():
    scores, weights, targets = _source()
    counts, rows = balanced_sample_weights(weights, targets)
    bias = np.array([0.2, -0.3, 0.1])
    compact, _ = objective_delta_and_gradient(bias, scores, rows, regularization=0.01)
    cell = weights[:, None].double().numpy() * targets.double().numpy()
    cell /= targets.shape[1] * counts[None, :]
    original = scores.double().numpy()
    shifted = original + bias[None, :]
    log_softmax_original = original - np.logaddexp.reduce(original, axis=1)[:, None]
    log_softmax_shifted = shifted - np.logaddexp.reduce(shifted, axis=1)[:, None]
    explicit = -float((cell * log_softmax_shifted).sum())
    explicit += float((cell * log_softmax_original).sum())
    explicit += 0.01 * float(np.dot(bias, bias)) / (2 * 3)
    assert math.isclose(compact, explicit, rel_tol=0, abs_tol=1e-12)


def test_analytic_gradient_matches_finite_difference():
    scores, weights, targets = _source()
    _, rows = balanced_sample_weights(weights, targets)
    bias = np.array([0.13, -0.21, 0.08])
    _, gradient = objective_delta_and_gradient(bias, scores, rows)
    numeric = np.empty_like(gradient)
    epsilon = 1e-6
    for index in range(len(bias)):
        high = bias.copy(); high[index] += epsilon
        low = bias.copy(); low[index] -= epsilon
        high_value, _ = objective_delta_and_gradient(high, scores, rows)
        low_value, _ = objective_delta_and_gradient(low, scores, rows)
        numeric[index] = (high_value - low_value) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numeric, atol=2e-9, rtol=2e-8)


def test_objective_rejects_nonfinite_scores():
    scores, weights, targets = _source()
    _, rows = balanced_sample_weights(weights, targets)
    scores[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        objective_delta_and_gradient(np.zeros(3), scores, rows)


def test_objective_rejects_wrong_row_weight_sum():
    scores, _, _ = _source()
    with pytest.raises(ValueError, match="sum to one"):
        objective_delta_and_gradient(np.zeros(3), scores, np.ones(6))


def test_projected_gradient_keeps_interior_components():
    assert projected_gradient_inf_norm(np.zeros(2), np.array([0.2, -0.4])) == 0.4


def test_projected_gradient_zeroes_lower_bound_outward_component():
    value = projected_gradient_inf_norm(
        np.array([-DEFAULT_BOUND, 0.0]), np.array([0.7, 0.0])
    )
    assert value == 0.0


def test_projected_gradient_zeroes_upper_bound_outward_component():
    value = projected_gradient_inf_norm(
        np.array([DEFAULT_BOUND, 0.0]), np.array([-0.7, 0.0])
    )
    assert value == 0.0


def test_fixed_fit_converges_within_registered_bounds():
    scores, weights, targets = _source()
    bias, report = fit_bounded_source_bias(scores, weights, targets, chunk_size=2)
    assert report["status"] == "converged_checks_passed"
    assert report["final_objective_delta"] <= 0.0
    assert report["projected_gradient_inf_norm"] <= 1e-6
    assert float(bias.abs().max()) <= DEFAULT_BOUND + 1e-6


def test_fixed_fit_is_deterministic_for_identical_inputs():
    scores, weights, targets = _source()
    first, first_report = fit_bounded_source_bias(scores, weights, targets, chunk_size=3)
    second, second_report = fit_bounded_source_bias(scores, weights, targets, chunk_size=3)
    assert torch.equal(first, second)
    assert first_report["final_objective_delta"] == second_report["final_objective_delta"]


def test_zero_strength_exactly_preserves_frozen_scores():
    scores, weights, targets = _source()
    bias, _ = fit_bounded_source_bias(scores, weights, targets)
    assert torch.equal(apply_prior_bias(scores, bias, strength=0.0), scores)


def test_fixed_bias_application_is_batch_member_independent():
    scores, weights, targets = _source()
    bias, _ = fit_bounded_source_bias(scores, weights, targets)
    together = apply_prior_bias(scores, bias, strength=0.9)
    separate = torch.cat(
        [apply_prior_bias(row[None], bias, strength=0.9) for row in scores]
    )
    torch.testing.assert_close(together, separate, atol=0, rtol=0)


def test_fixed_bias_application_is_reorder_equivariant():
    scores, weights, targets = _source()
    bias, _ = fit_bounded_source_bias(scores, weights, targets)
    order = torch.tensor([4, 1, 5, 0, 3, 2])
    direct = apply_prior_bias(scores, bias, strength=0.9)[order]
    reordered = apply_prior_bias(scores[order], bias, strength=0.9)
    torch.testing.assert_close(direct, reordered, atol=0, rtol=0)


def _args(batch_size=64, output_dir="one", prior_config="first"):
    return argparse.Namespace(
        checkpoint="checkpoint", config="config", output_dir=output_dir,
        tta="horizontal_flip", tta_fusion="mean_probabilities",
        tta_temperature=1.5, tta_view_weight=0.5, acknowledge_tta_risk=True,
        local_view="attention_multiscale", local_crop_size=160,
        local_crop_sizes="112,128,144,160",
        local_scale_weights="0.20,0.30,0.40,0.10", local_top_k=5,
        local_weight=0.4, local_temperature=1.5, adapt_local_features=True,
        adapt_part_token_features=True, acknowledge_local_view_risk=True,
        prior_alignment_strength=0.0, prior_alignment_iterations=50,
        prior_config=prior_config, acknowledge_balanced_test_prior=False,
        overwrite=False, device="cpu", input_resize_mode="clip_center_crop",
        batch_size=batch_size, dump_logits=None, dump_branch_logits=None,
    )


def _descriptor(args):
    checkpoint = {"effective_model_spec": {"backbone": "ViT-B/32"}}
    config = {
        "model": {"backbone": "ViT-B/32"},
        "train": {"amp": False},
        "evaluation": {"batch_size": 128, "inference_batch_size": 128},
    }
    return build_inference_protocol_descriptor(
        args, checkpoint, config, "synthetic-preprocess", torch.device("cpu")
    )


def test_protocol_descriptor_excludes_paths_and_prior_record():
    first = protocol_sha256(_descriptor(_args(output_dir="one", prior_config="a")))
    second = protocol_sha256(_descriptor(_args(output_dir="two", prior_config="b")))
    assert first == second


def test_protocol_descriptor_binds_batch_size():
    assert protocol_sha256(_descriptor(_args(batch_size=64))) != protocol_sha256(
        _descriptor(_args(batch_size=32))
    )
