"""Fixed v3 recovery selection, distillation, and effective-batch invariants."""
from __future__ import annotations

import numpy as np
import torch

from aegis_clip.prelim75_joint import weighted_micro_loss
from aegis_clip.prelim75_recovery import (
    fuse_teacher_probabilities,
    recovery_kd_loss,
    recovery_lambda,
    select_recovery_cohort,
)


def test_teacher_fusion_applies_temperature_per_branch_once():
    global_logits = torch.tensor([[[3., 0., -1.], [2., 1., -2.]]])
    local_logits = torch.tensor([[[
        [2., 0., -1.], [1., 0., -1.], [4., 0., -1.], [3., 1., -1.],
    ], [
        [2., 1., -1.], [2., 0., -1.], [3., 0., -1.], [1., 0., -1.],
    ]]])
    result = fuse_teacher_probabilities(global_logits, local_logits)
    scale = torch.tensor([.2, .3, .4, .1]).view(1, 1, 4, 1)
    expected_global = torch.softmax(global_logits / 1.5, -1)
    expected_local = (torch.softmax(local_logits / 1.5, -1) * scale).sum(2)
    expected = (.6 * expected_global + .4 * expected_local).mean(1)
    torch.testing.assert_close(result["probabilities"], expected)
    assert result["probabilities"].shape == (1, 3)
    assert result["branch_predictions"].shape == (1, 4)


def _selection_fixture():
    paths = []
    groups = []
    weights = []
    targets = []
    predictions = []
    branches = []
    confidence = []
    margins = []

    def add(path, group, weight, predicted, conf=.8, margin=.3, branch=None, target_class=None):
        paths.append(path); groups.append(group); weights.append(weight)
        target = np.zeros(500, dtype=np.float32)
        target[predicted if target_class is None else target_class] = 1.
        targets.append(target); predictions.append(predicted)
        branches.append([predicted] * 4 if branch is None else branch)
        confidence.append(conf); margins.append(margin)

    # Ten distinct positive groups for class 0 -> cap 2.  Five for class 1 -> cap 1.
    for i in range(10):
        add(f"positive0/{i}.jpg", f"p0-{i}", 1., 0)
    for i in range(5):
        add(f"positive1/{i}.jpg", f"p1-{i}", 1., 1)
    # A zero row in a positive group must be excluded before representative scoring.
    add("zero/contaminated.jpg", "p0-0", 0., 0, conf=.99, margin=.49)
    # Group z0 has two rows: lexicographically first is fixed even though less confident.
    add("zero/a.jpg", "z0", 0., 0, conf=.85, margin=.20)
    add("zero/z.jpg", "z0", 0., 0, conf=.99, margin=.49)
    add("zero/b.jpg", "z1", 0., 0, conf=.90, margin=.30)
    add("zero/c.jpg", "z2", 0., 0, conf=.80, margin=.25)
    # Inclusive threshold boundaries for class 1.
    add("zero/d.jpg", "z3", 0., 1, conf=.45, margin=.15)
    # Each fixed eligibility failure is represented.
    add("zero/e.jpg", "z4", 0., 1, conf=.90, margin=.30, branch=[1, 1, 0, 1])
    add("zero/f.jpg", "z5", 0., 1, conf=.449, margin=.30)
    add("zero/g.jpg", "z6", 0., 1, conf=.90, margin=.149)
    # Class 2 has G_c=0, so an otherwise eligible row must be capped out.
    add("zero/h.jpg", "z7", 0., 2, conf=.99, margin=.49)
    return (
        paths, groups, np.asarray(weights), np.stack(targets), np.asarray(predictions),
        np.asarray(branches), np.asarray(confidence), np.asarray(margins),
    )


def test_recovery_selection_fixes_representatives_thresholds_and_class_caps():
    fixture = _selection_fixture()
    selected = select_recovery_cohort(*fixture, minimum_groups=3, minimum_classes=2)
    selected_paths = {fixture[0][index] for index in selected["selected_indices"]}
    assert selected_paths == {"zero/a.jpg", "zero/b.jpg", "zero/d.jpg"}
    assert "zero/z.jpg" not in selected_paths
    report = selected["report"]
    assert report["excluded_by_positive_supervision_group_rows"] == 1
    assert report["eliminated_by_view_or_branch_disagreement"] == 1
    assert report["eliminated_by_confidence"] == 1
    assert report["eliminated_by_margin"] == 1
    assert report["eligible_before_class_cap"] == 5
    assert report["selected_after_class_cap"] == 3
    assert report["covered_prediction_classes"] == 2
    assert report["gate_passed"]
    class_rows = selected["class_rows"]
    assert class_rows[0]["class_cap"] == 2
    assert class_rows[1]["class_cap"] == 1
    assert class_rows[2]["class_cap"] == 0
    assert selected["recovery_weights"][fixture[0].index("zero/d.jpg")] == np.float32(.45)


def test_recovery_gate_is_fixed_and_caps_are_not_quotas():
    fixture = _selection_fixture()
    selected = select_recovery_cohort(*fixture)
    assert selected["report"]["status"] == "closed_insufficient_recovery_cohort"
    assert not selected["report"]["gate_passed"]
    # No class is backfilled beyond its eligible rows merely to reach a cap.
    assert selected["class_rows"][1]["selected_after_cap"] == 1


def test_zero_weight_noisy_target_cannot_change_recovery_selection_or_weights():
    fixture = list(_selection_fixture())
    baseline = select_recovery_cohort(*fixture, minimum_groups=3, minimum_classes=2)
    changed_targets = fixture[3].copy()
    zero_rows = fixture[2] == 0
    changed_targets[zero_rows] = 0
    changed_targets[zero_rows, 499] = 1
    fixture[3] = changed_targets
    changed = select_recovery_cohort(*fixture, minimum_groups=3, minimum_classes=2)
    np.testing.assert_array_equal(baseline["mask"], changed["mask"])
    np.testing.assert_array_equal(baseline["recovery_weights"], changed["recovery_weights"])


def test_kd_zero_manual_direction_and_frozen_targets():
    logits = torch.tensor([[1.2, -.4], [.1, .7]], requires_grad=True)
    same_teacher = torch.softmax(logits.detach() / 1.5, 1)
    weights = torch.tensor([.8, .6])
    zero = recovery_kd_loss(logits, same_teacher, weights, weights.sum())
    torch.testing.assert_close(zero, torch.tensor(0.), atol=1e-7, rtol=0)

    teacher = torch.tensor([[.75, .25], [.2, .8]])
    actual = recovery_kd_loss(logits, teacher, weights, weights.sum())
    manual_per = (teacher * (teacher.log() - torch.log_softmax(logits / 1.5, 1))).sum(1)
    expected = 1.5 ** 2 * (manual_per * weights).sum() / weights.sum()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert teacher.grad is None and weights.grad is None

    empty_logits = logits.detach().clone().requires_grad_(True)
    empty = recovery_kd_loss(empty_logits, teacher, torch.zeros(2), torch.tensor(0.))
    assert torch.isfinite(empty) and empty == 0
    empty.backward()
    assert empty_logits.grad is not None and empty_logits.grad.abs().sum() == 0


def test_kd_lambda_ramps_for_exactly_the_first_epoch():
    assert recovery_lambda(0, 4) == .0625
    assert recovery_lambda(3, 4) == .25
    assert recovery_lambda(4, 4) == .25
    assert recovery_lambda(100, 4) == .25


def test_full_32_objective_matches_two_microbatches_with_common_denominators():
    torch.manual_seed(7)
    batch = 32
    classes = 5
    targets = torch.softmax(torch.randn(batch, classes), 1)
    teacher = torch.softmax(torch.randn(batch, classes), 1)
    supervision_weights = torch.rand(batch)
    supervision_weights[::7] = 0
    recovery_weights = torch.zeros(batch)
    recovery_weights[[1, 8, 19, 27]] = torch.tensor([.8, .7, .6, .5])
    anchor_values = torch.rand(batch)
    config = {"name": "gce", "gce_q": .5, "epsilon": 1e-7}
    global_full = torch.randn(batch, classes, requires_grad=True)
    local_full = torch.randn(batch, classes, requires_grad=True)
    global_micro = global_full.detach().clone().requires_grad_(True)
    local_micro = local_full.detach().clone().requires_grad_(True)
    sup_denominator = supervision_weights.sum()
    kd_denominator = recovery_weights.sum()

    def objective(global_logits, local_logits, slices):
        result = global_logits.sum() * 0
        for item in slices:
            supervised_global = weighted_micro_loss(
                global_logits[item], targets[item], supervision_weights[item], sup_denominator, config, 4
            )
            supervised_local = weighted_micro_loss(
                local_logits[item], targets[item], supervision_weights[item], sup_denominator, config, 4
            )
            kd_global = recovery_kd_loss(
                global_logits[item], teacher[item], recovery_weights[item], kd_denominator
            )
            kd_local = recovery_kd_loss(
                local_logits[item], teacher[item], recovery_weights[item], kd_denominator
            )
            anchor = anchor_values[item].sum() / batch
            result = result + .6 * supervised_global + .4 * supervised_local + 2 * anchor
            result = result + .25 * (.6 * kd_global + .4 * kd_local)
        return result

    full = objective(global_full, local_full, [slice(None)])
    micro = objective(global_micro, local_micro, [slice(0, 16), slice(16, 32)])
    full.backward(); micro.backward()
    torch.testing.assert_close(full, micro)
    torch.testing.assert_close(global_full.grad, global_micro.grad)
    torch.testing.assert_close(local_full.grad, local_micro.grad)
