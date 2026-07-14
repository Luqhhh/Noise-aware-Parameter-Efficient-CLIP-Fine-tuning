"""Tests for common.loss_schedule module."""

import torch
import pytest

from common.loss_schedule import (
    ScheduledLoss,
    _validate_schedule,
    build_scheduled_loss,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def ce_gce_schedule():
    """Standard CE → GCE schedule used in Phase 3."""
    return [
        {"start_epoch": 1, "end_epoch": 5, "name": "cross_entropy"},
        {"start_epoch": 6, "end_epoch": 50, "name": "gce", "q": 0.7},
    ]


@pytest.fixture
def logits_targets():
    rng = torch.Generator().manual_seed(42)
    logits = torch.randn(8, 500, generator=rng)
    targets = torch.randint(0, 500, (8,), generator=rng)
    return logits, targets


# ── Schedule validation ───────────────────────────────────────────────


class TestValidateSchedule:
    def test_valid_schedule(self, ce_gce_schedule):
        normalised = _validate_schedule(ce_gce_schedule)
        assert len(normalised) == 2
        assert normalised[0]["start_epoch"] == 1
        assert normalised[0]["end_epoch"] == 5
        assert normalised[1]["start_epoch"] == 6

    def test_empty_schedule_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_schedule([])

    def test_gap_raises(self):
        schedule = [
            {"start_epoch": 1, "end_epoch": 5, "name": "cross_entropy"},
            {"start_epoch": 7, "end_epoch": 50, "name": "gce", "q": 0.7},
        ]
        with pytest.raises(ValueError, match="gap or overlap"):
            _validate_schedule(schedule)

    def test_overlap_raises(self):
        schedule = [
            {"start_epoch": 1, "end_epoch": 6, "name": "cross_entropy"},
            {"start_epoch": 6, "end_epoch": 50, "name": "gce", "q": 0.7},
        ]
        with pytest.raises(ValueError, match="gap or overlap"):
            _validate_schedule(schedule)

    def test_start_gt_end_raises(self):
        schedule = [
            {"start_epoch": 10, "end_epoch": 5, "name": "cross_entropy"},
        ]
        with pytest.raises(ValueError, match="start_epoch"):
            _validate_schedule(schedule)

    def test_missing_start_end_raises(self):
        with pytest.raises(ValueError, match="start_epoch"):
            _validate_schedule([{"name": "cross_entropy"}])


# ── Phase switching ───────────────────────────────────────────────────


class TestPhaseSwitching:
    def test_epoch_5_uses_ce(self, ce_gce_schedule, logits_targets):
        logits, targets = logits_targets
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(5)
        assert sl.current_phase_name == "cross_entropy"

    def test_epoch_6_uses_gce(self, ce_gce_schedule, logits_targets):
        logits, targets = logits_targets
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(6)
        assert sl.current_phase_name == "gce"

    def test_epoch_1_uses_ce(self, ce_gce_schedule, logits_targets):
        logits, targets = logits_targets
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(1)
        assert sl.current_phase_name == "cross_entropy"

    def test_epoch_50_uses_gce(self, ce_gce_schedule, logits_targets):
        logits, targets = logits_targets
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(50)
        assert sl.current_phase_name == "gce"

    def test_epoch_out_of_range_raises(self, ce_gce_schedule):
        sl = ScheduledLoss(ce_gce_schedule)
        with pytest.raises(ValueError, match="not covered"):
            sl.set_epoch(0)

    def test_forward_produces_scalar_with_reduction_none(
        self, ce_gce_schedule, logits_targets
    ):
        """With reduction='none', forward returns per-sample loss."""
        logits, targets = logits_targets
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(1)
        loss = sl(logits, targets)
        assert loss.shape == (logits.size(0),)

    def test_loss_differs_between_phases(
        self, ce_gce_schedule, logits_targets
    ):
        """CE and GCE should produce different loss values."""
        logits, targets = logits_targets
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(1)
        ce_loss = sl(logits, targets).detach().clone()
        sl.set_epoch(6)
        gce_loss = sl(logits, targets).detach().clone()
        assert not torch.allclose(ce_loss, gce_loss), (
            "CE and GCE should produce different losses"
        )


# ── Checkpoint save / restore ─────────────────────────────────────────


class TestCheckpoint:
    def test_state_dict_saves_phase(self, ce_gce_schedule):
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(6)
        sd = sl.state_dict()
        assert "_schedule_phase_idx" in sd
        assert "_schedule_epoch" in sd
        assert sd["_schedule_phase_idx"].item() == 1  # 0-based

    def test_load_state_dict_restores_phase(
        self, ce_gce_schedule, logits_targets
    ):
        logits, targets = logits_targets
        sl1 = ScheduledLoss(ce_gce_schedule)
        sl1.set_epoch(6)
        loss_before = sl1(logits, targets).detach().clone()

        sl2 = ScheduledLoss(ce_gce_schedule)
        sl2.load_state_dict(sl1.state_dict())
        loss_after = sl2(logits, targets).detach().clone()

        assert sl2.current_phase_name == "gce"
        assert sl2.current_epoch == 6
        assert torch.allclose(loss_before, loss_after)

    def test_resume_from_epoch_4(self, ce_gce_schedule):
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(4)
        sd = sl.state_dict()

        sl2 = ScheduledLoss(ce_gce_schedule)
        sl2.load_state_dict(sd)
        assert sl2.current_phase_name == "cross_entropy"
        assert sl2.current_epoch == 4

    def test_resume_from_epoch_6(self, ce_gce_schedule):
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(6)
        sd = sl.state_dict()

        sl2 = ScheduledLoss(ce_gce_schedule)
        sl2.load_state_dict(sd)
        assert sl2.current_phase_name == "gce"
        assert sl2.current_epoch == 6


# ── Factory ───────────────────────────────────────────────────────────


class TestBuildScheduledLoss:
    def test_build_from_config(self):
        config = {
            "loss": {
                "schedule": [
                    {"start_epoch": 1, "end_epoch": 3, "name": "cross_entropy"},
                    {"start_epoch": 4, "end_epoch": 10, "name": "gce", "q": 0.7},
                ]
            }
        }
        sl = build_scheduled_loss(config)
        assert isinstance(sl, ScheduledLoss)
        assert len(sl.phases) == 2

    def test_build_missing_schedule_raises(self):
        with pytest.raises(ValueError, match="schedule"):
            build_scheduled_loss({"loss": {"name": "cross_entropy"}})
