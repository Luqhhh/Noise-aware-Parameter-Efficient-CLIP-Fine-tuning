"""Scheduled loss — switches loss function at epoch boundaries.

Usage in config:

    loss:
      schedule:
        - start_epoch: 1
          end_epoch: 5
          name: cross_entropy
        - start_epoch: 6
          end_epoch: 50
          name: gce
          q: 0.7

The ``ScheduledLoss`` wrapper delegates to the active child loss.  It supports
``state_dict()`` / ``load_state_dict()`` for checkpoint resume and logs phase
transitions.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from common.losses import build_loss

logger = logging.getLogger(__name__)


def _validate_schedule(schedule: List[dict]) -> List[dict]:
    """Validate and normalise a loss schedule.

    Returns a deep-copied list with defaults filled.  Raises ``ValueError`` on
    gaps, overlaps, or unknown loss names.
    """
    if not schedule:
        raise ValueError("loss.schedule must be a non-empty list")

    normalised = []
    for i, phase in enumerate(schedule):
        p = dict(phase)
        if "start_epoch" not in p or "end_epoch" not in p:
            raise ValueError(
                f"Schedule phase {i}: must have start_epoch and end_epoch"
            )
        start = int(p["start_epoch"])
        end = int(p["end_epoch"])
        if start > end:
            raise ValueError(
                f"Schedule phase {i}: start_epoch ({start}) > end_epoch ({end})"
            )
        p["start_epoch"] = start
        p["end_epoch"] = end
        p.setdefault("name", "cross_entropy")
        normalised.append(p)

    # Check for gaps or overlaps
    normalised.sort(key=lambda x: x["start_epoch"])
    prev_end = 0
    for p in normalised:
        if p["start_epoch"] != prev_end + 1:
            raise ValueError(
                f"Schedule gap or overlap: phase starts at epoch "
                f"{p['start_epoch']} but previous phase ended at {prev_end}"
            )
        prev_end = p["end_epoch"]

    return normalised


class ScheduledLoss(nn.Module):
    """Loss wrapper that switches child loss at epoch boundaries.

    Each phase in the schedule is built via ``build_loss()`` with the phase
    dict as the ``loss`` config section.  The wrapper exposes the same
    ``forward(logits, targets)`` interface as any other loss.
    """

    def __init__(self, schedule: List[dict]):
        super().__init__()
        self.schedule = _validate_schedule(schedule)
        self._current_phase_idx: int = -1
        self._current_epoch: int = 0

        # Build a child loss for each phase
        self.phases = nn.ModuleList()
        for phase in self.schedule:
            loss_cfg = {
                k: v for k, v in phase.items()
                if k not in ("start_epoch", "end_epoch")
            }
            # Ensure reduction is 'none' so per-sample weights work uniformly
            loss_cfg.setdefault("reduction", "none")
            child = build_loss({"loss": loss_cfg})
            self.phases.append(child)

        # Start at first phase
        self._activate_phase(0)

    def _activate_phase(self, idx: int):
        """Switch to phase *idx* (0-based)."""
        self._current_phase_idx = idx
        phase = self.schedule[idx]
        self._active_loss = self.phases[idx]
        logger.info(
            "Loss schedule: phase %d/%d — %s (epochs %d–%d)",
            idx + 1, len(self.schedule),
            phase["name"], phase["start_epoch"], phase["end_epoch"],
        )

    def set_epoch(self, epoch: int):
        """Update the active loss based on the current epoch."""
        self._current_epoch = epoch
        for i, phase in enumerate(self.schedule):
            if phase["start_epoch"] <= epoch <= phase["end_epoch"]:
                if i != self._current_phase_idx:
                    self._activate_phase(i)
                return
        raise ValueError(
            f"Epoch {epoch} is not covered by any schedule phase. "
            f"Schedule range: {self.schedule[0]['start_epoch']}–"
            f"{self.schedule[-1]['end_epoch']}"
        )

    @property
    def current_phase_name(self) -> str:
        return self.schedule[self._current_phase_idx]["name"]

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._active_loss(logits, targets)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Include schedule state for checkpoint resume."""
        d = super().state_dict(destination, prefix, keep_vars)
        d[prefix + "_schedule_phase_idx"] = torch.tensor(self._current_phase_idx)
        d[prefix + "_schedule_epoch"] = torch.tensor(self._current_epoch)
        return d

    def load_state_dict(self, state_dict, strict=True):
        """Restore schedule state from checkpoint."""
        prefix = ""
        phase_key = prefix + "_schedule_phase_idx"
        epoch_key = prefix + "_schedule_epoch"
        if phase_key in state_dict:
            self._current_phase_idx = int(state_dict.pop(phase_key).item())
            self._active_loss = self.phases[self._current_phase_idx]
        if epoch_key in state_dict:
            self._current_epoch = int(state_dict.pop(epoch_key).item())
        super().load_state_dict(state_dict, strict)


def build_scheduled_loss(config: dict) -> ScheduledLoss:
    """Build a ``ScheduledLoss`` from a config dict.

    Args:
        config: Full project config.  Reads ``loss.schedule``.

    Returns:
        ScheduledLoss instance.

    Raises:
        ValueError: If schedule is missing or invalid.
    """
    schedule = config.get("loss", {}).get("schedule")
    if schedule is None:
        raise ValueError("loss.schedule is required for scheduled loss")
    return ScheduledLoss(schedule)
