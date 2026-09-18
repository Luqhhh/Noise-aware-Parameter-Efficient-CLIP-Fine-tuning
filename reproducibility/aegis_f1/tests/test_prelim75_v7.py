"""CPU tests for the PRELIM75 v7 fixed cooldown scheduler semantics."""
from __future__ import annotations

import json
import math

import pytest
import torch

from aegis_clip.prelim75_cooldown import (
    CANDIDATE_HORIZONS,
    DEFAULT_STEPS_PER_EPOCH,
    FLOOR_RATIO,
    ScheduleSpec,
    audit_position,
    build_fresh_scheduler,
    scheduler_alignment_check,
    step_and_audit,
)
from aegis_clip.trainer import _warmup_cosine


def _optimizer():
    params = [torch.nn.Parameter(torch.zeros(())) for _ in range(3)]
    return torch.optim.AdamW([
        {"name": "backbone", "params": [params[0]], "lr": 1e-6, "weight_decay": 0.0},
        {"name": "head", "params": [params[1]], "lr": 1.5e-7, "weight_decay": 1e-4},
        {"name": "local_adapters", "params": [params[2]], "lr": 3e-6, "weight_decay": 0.0},
    ])


def test_schedule_boundaries_match_registered_reference_table():
    l0 = ScheduleSpec("L0", 18, DEFAULT_STEPS_PER_EPOCH)
    l1 = ScheduleSpec("L1", 3, DEFAULT_STEPS_PER_EPOCH)
    assert l0.total_steps == 18 * DEFAULT_STEPS_PER_EPOCH == 58068
    assert l1.total_steps == 3 * DEFAULT_STEPS_PER_EPOCH == 9678
    assert l0.training_steps == l1.training_steps == 9678

    assert math.isclose(l0.multiplier(0), 1.0, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(l1.multiplier(0), 1.0, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(l0.multiplier(DEFAULT_STEPS_PER_EPOCH), 0.992479837741043, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(l1.multiplier(DEFAULT_STEPS_PER_EPOCH), 0.7525, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(l0.multiplier(2 * DEFAULT_STEPS_PER_EPOCH), 0.9701478472890247, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(l1.multiplier(2 * DEFAULT_STEPS_PER_EPOCH), 0.2575000000000001, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(l0.multiplier(3 * DEFAULT_STEPS_PER_EPOCH), 0.9336825748732972, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(l1.multiplier(3 * DEFAULT_STEPS_PER_EPOCH), FLOOR_RATIO, rel_tol=0.0, abs_tol=1e-12)
    assert l0.multiplier(l0.total_steps) == FLOOR_RATIO
    assert l1.multiplier(l1.total_steps) == FLOOR_RATIO


def test_schedule_spec_matches_repository_warmup_cosine_pointwise():
    for candidate in ("L0", "L1"):
        spec = ScheduleSpec(candidate, CANDIDATE_HORIZONS[candidate], DEFAULT_STEPS_PER_EPOCH)
        for step in (0, 1, 17, spec.steps_per_epoch, 2 * spec.steps_per_epoch,
                     spec.training_steps - 1, spec.training_steps, spec.total_steps):
            assert math.isclose(
                spec.multiplier(step),
                float(_warmup_cosine(step, 0, spec.total_steps)),
                rel_tol=0.0,
                abs_tol=1e-15,
            )


def test_build_fresh_scheduler_uses_candidate_horizon_and_floor():
    optimizer = _optimizer()
    l1 = ScheduleSpec("L1", 3, DEFAULT_STEPS_PER_EPOCH)
    scheduler = build_fresh_scheduler(optimizer, l1)
    # Initial LR is exactly the base LR for each group.
    audit_position(optimizer, l1, 0)
    for step in range(l1.total_steps):
        scheduler.step()
    audit_position(optimizer, l1, l1.total_steps)
    for group in optimizer.param_groups:
        assert math.isclose(float(group["lr"]), float(group["initial_lr"]) * FLOOR_RATIO,
                            rel_tol=1e-6, abs_tol=1e-15)

    optimizer2 = _optimizer()
    l0 = ScheduleSpec("L0", 18, DEFAULT_STEPS_PER_EPOCH)
    scheduler0 = build_fresh_scheduler(optimizer2, l0)
    for step in range(l0.training_steps):
        scheduler0.step()
    # At the end of the actual 3-epoch budget L0 is still well above the floor.
    for group in optimizer2.param_groups:
        ratio = float(group["lr"]) / float(group["initial_lr"])
        assert ratio > 0.93


def test_audit_position_rejects_wrong_lr_and_unknown_positions():
    optimizer = _optimizer()
    spec = ScheduleSpec("L1", 3, DEFAULT_STEPS_PER_EPOCH)
    build_fresh_scheduler(optimizer, spec)
    optimizer.param_groups[0]["lr"] *= 0.5
    with pytest.raises(RuntimeError, match="scheduler position mismatch"):
        audit_position(optimizer, spec, 0)
    with pytest.raises(ValueError):
        spec.multiplier(-1)


def test_step_and_audit_uses_optimizer_then_scheduler_and_records_both_lrs():
    optimizer = _optimizer()
    spec = ScheduleSpec("L1", 3, DEFAULT_STEPS_PER_EPOCH)
    scheduler = build_fresh_scheduler(optimizer, spec)
    for parameter in optimizer.param_groups:
        parameter["params"][0].grad = torch.ones(())
    record = step_and_audit(optimizer, scheduler, spec, 0)
    assert record["completed_updates_before"] == 0
    assert len(record["used_lrs"]) == 3
    assert len(record["next_lrs"]) == 3
    assert math.isclose(record["next_multiplier"], spec.multiplier(1), rel_tol=0.0, abs_tol=1e-15)
    assert all(
        math.isclose(float(after), float(group["initial_lr"]) * record["next_multiplier"],
                     rel_tol=1e-6, abs_tol=1e-15)
        for after, group in zip(record["next_lrs"], optimizer.param_groups)
    )


def test_scheduler_alignment_check_is_json_serializable_and_matches_plan():
    result = scheduler_alignment_check(DEFAULT_STEPS_PER_EPOCH)
    json.dumps(result)
    assert result["status"] == "passed"
    l0 = result["candidates"]["L0"]
    l1 = result["candidates"]["L1"]
    assert l0["horizon_total_steps"] == 58068 and l0["training_steps"] == 9678
    assert l1["horizon_total_steps"] == 9678 and l1["training_steps"] == 9678
    assert math.isclose(l0["training_end_multiplier"], 0.9336825748732972, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(l1["training_end_multiplier"], FLOOR_RATIO, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(l1["boundaries"]["1M"]["multiplier"], 0.7525, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(l0["boundaries"]["1M"]["multiplier"], 0.992479837741043, rel_tol=0.0, abs_tol=1e-12)


def test_schedule_spec_rejects_unregistered_horizon_and_warmup():
    with pytest.raises(ValueError):
        ScheduleSpec("L1", 18, DEFAULT_STEPS_PER_EPOCH)
    with pytest.raises(ValueError):
        ScheduleSpec("L0", 18, DEFAULT_STEPS_PER_EPOCH, warmup_steps=10)
    with pytest.raises(ValueError):
        ScheduleSpec("L0", 18, 0)


def test_schedule_rejects_updates_beyond_registered_horizon():
    l1 = ScheduleSpec("L1", 3, DEFAULT_STEPS_PER_EPOCH)
    assert l1.multiplier(l1.total_steps) == FLOOR_RATIO
    with pytest.raises(ValueError, match="beyond schedule horizon"):
        l1.multiplier(l1.total_steps + 1)
