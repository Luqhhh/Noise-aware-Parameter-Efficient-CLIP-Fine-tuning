from __future__ import annotations

import pytest
import torch

from aegis_clip.scope_crossfit import (
    OOFThreshold,
    candidate_pair_training_mask,
    cluster_bootstrap_delta,
    fit_shared_beta,
    map_deployed_threshold,
    promotion_gate,
    select_oof_threshold,
    stratified_group_folds,
)


SOLVER = {
    "initial_upper": 1.0,
    "maximum_upper": float(2**20),
    "maximum_iterations": 100,
    "interval_tolerance": 1.0e-12,
}
POLICY = {
    "minimum_accuracy_changing_precision": 0.6,
    "minimum_wilson_lower": 0.0,
    "wilson_z": 1.959963984540054,
}


def test_stratified_group_folds_never_split_groups() -> None:
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    groups = ["a", "a", "b", "c", "d", "d", "e", "f", "g", "g", "h", "i"]
    folds = stratified_group_folds(labels, groups, folds=3, seed=42)
    assert sorted(folds.unique().tolist()) == [0, 1, 2]
    for group in set(groups):
        rows = torch.tensor([i for i, value in enumerate(groups) if value == group])
        assert folds[rows].unique().numel() == 1


def test_beta_fit_is_nonnegative_sorted_and_has_no_intercept() -> None:
    margin = torch.tensor([-0.3, -0.2, -0.1, -0.4], dtype=torch.float64)
    evidence = torch.tensor([-1.0, 1.0, -0.5, 0.8], dtype=torch.float64)
    target = torch.tensor([0, 1, 0, 1])
    rows = torch.arange(4)
    fit = fit_shared_beta(margin, evidence, target, rows, SOLVER)
    assert fit.beta > 0.0
    assert fit.intercept == 0.0
    assert abs(fit.gradient) < 1.0e-8
    with pytest.raises(ValueError, match="ascending"):
        fit_shared_beta(margin, evidence, target, torch.tensor([0, 2, 1, 3]), SOLVER)


def test_beta_boundary_and_missing_pair_target_fail_closed() -> None:
    fit = fit_shared_beta(
        torch.zeros(4), torch.tensor([-1.0, 1.0, -1.0, 1.0]),
        torch.tensor([1, 0, 1, 0]), torch.arange(4), SOLVER,
    )
    assert fit.beta == 0.0
    with pytest.raises(ValueError, match="both"):
        fit_shared_beta(torch.zeros(3), torch.ones(3), torch.ones(3), torch.arange(3), SOLVER)


def test_third_class_rows_are_excluded_only_from_beta_training() -> None:
    candidates = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
    labels = torch.tensor([0, 2, 9, 4])
    mask, target = candidate_pair_training_mask(candidates, labels)
    assert torch.equal(mask, torch.tensor([True, True, False, True]))
    assert torch.equal(target, torch.tensor([False, True, False, True]))


def test_threshold_uses_between_score_cut_and_strict_greater_than() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    eligible = torch.ones(4, dtype=torch.bool)
    candidates = torch.tensor([[0, 1]] * 4)
    labels = torch.tensor([0, 0, 1, 1])
    threshold = select_oof_threshold(scores, eligible, labels, candidates, POLICY)
    assert threshold.mode == "finite"
    assert threshold.gamma == pytest.approx(0.25)
    assert threshold.k_oof == 2
    assert torch.equal(threshold.switch_mask, torch.tensor([False, False, True, True]))


def test_threshold_precision_and_wilson_can_force_no_switch() -> None:
    policy = dict(POLICY, minimum_wilson_lower=0.95)
    result = select_oof_threshold(
        torch.tensor([0.1, 0.2], dtype=torch.float64), torch.ones(2, dtype=torch.bool),
        torch.tensor([0, 1]), torch.tensor([[0, 1], [0, 1]]), policy,
    )
    assert result.mode == "no_switch"
    assert result.no_switch_reason == "no_qualified_candidate"


def test_refit_mapping_uses_count_ratio_and_prefers_fewer_switches() -> None:
    oof = OOFThreshold(
        mode="finite", gamma=0.25, k_oof=2, n_oof=4, rho=0.5,
        no_switch_reason=None, eligible_score_hash="a" * 64,
        candidate_count=4, switch_mask=torch.tensor([False, False, True, True]),
    )
    mapped = map_deployed_threshold(
        oof,
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
        torch.ones(3, dtype=torch.bool),
    )
    assert mapped.mode == "finite"
    assert mapped.k_refit == 1
    assert mapped.gamma == pytest.approx(1.5)


def test_cluster_bootstrap_is_grouped_outer_stratified_and_deterministic() -> None:
    parent = torch.tensor([True, False, True, False, True, False])
    method = torch.tensor([True, True, True, False, False, True])
    groups = ["a", "a", "b", "c", "d", "d"]
    labels = torch.tensor([0, 0, 1, 1, 0, 0])
    outer = torch.tensor([0, 0, 0, 1, 1, 1])
    kwargs = dict(draws=200, seed=42, quantile_method="linear")
    first = cluster_bootstrap_delta(parent, method, groups, labels, outer, **kwargs)
    second = cluster_bootstrap_delta(parent, method, groups, labels, outer, **kwargs)
    assert first == second
    assert first.point == pytest.approx((method.double() - parent.double()).mean().item())


def test_promotion_gate_is_strict_and_all_conditions_are_and() -> None:
    base = {
        "raw_parent_correct": 7000, "raw_scope_correct": 7021, "raw_total": 10000,
        "clean_parent_correct": 4000, "clean_scope_correct": 4011, "clean_total": 5000,
        "corrections": 30, "regressions": 9,
        "fold_deltas": [1, 1, 0, 1, 0], "bootstrap_lower": 0.0001,
        "pace_raw_correct": 7020, "no_topology_raw_correct": 7019,
        "pace_clean_correct": 4010, "no_topology_clean_correct": 4009,
        "audits_passed": True,
    }
    decision = promotion_gate(base)
    assert decision.promoted
    assert all(decision.gates.values())
    failed = dict(base, pace_raw_correct=7021)
    assert not promotion_gate(failed).promoted
