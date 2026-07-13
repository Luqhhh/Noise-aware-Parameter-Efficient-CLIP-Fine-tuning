"""Training hooks — EMA and Teacher–Student interfaces."""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EMAHook:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model updated after each successful
    optimizer step.  Supports step-level warmup (direct copy for the first
    N updates) and full checkpoint save/restore.

    Usage in train.py::

        ema = EMAHook(model, decay=0.99, warmup_steps=warmup_epochs * steps_per_epoch)
        for batch in loader: ...
            optimizer.step()
            ema.update(model)  # after EACH successful step
        # Validation:
        raw_acc = validate(model, ...)
        ema_acc = validate(ema.get_ema_model(), ...)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.99,
        warmup_steps: int = 0,
    ):
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates: int = 0

        # Shadow model — never requires grad
        self._ema_model = copy.deepcopy(model)
        self._ema_model.eval()
        for p in self._ema_model.parameters():
            p.requires_grad_(False)

        logger.info(
            "EMAHook: decay=%.4f, warmup_steps=%d", decay, warmup_steps
        )

    # ── Core update ──────────────────────────────────────────────────

    def update(self, model: nn.Module):
        """Update EMA after one successful optimizer step.

        During warmup (``num_updates < warmup_steps``), EMA is replaced
        with a direct copy of the raw model.  After warmup, the standard
        EMA recurrence applies:

            ema = decay * ema + (1 - decay) * raw
        """
        self.num_updates += 1

        if self.num_updates <= self.warmup_steps:
            self._ema_model.load_state_dict(model.state_dict())
        else:
            with torch.no_grad():
                for ema_p, model_p in zip(
                    self._ema_model.parameters(), model.parameters()
                ):
                    ema_p.data.mul_(self.decay).add_(
                        model_p.data, alpha=1.0 - self.decay
                    )

    # ── Model access ─────────────────────────────────────────────────

    def get_ema_model(self) -> nn.Module:
        """Return the EMA shadow model for direct validation/inference."""
        return self._ema_model

    # ── Checkpoint serialisation ─────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "ema_model": self._ema_model.state_dict(),
            "num_updates": torch.tensor(self.num_updates, dtype=torch.long),
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, d: dict):
        self._ema_model.load_state_dict(d["ema_model"])
        self.num_updates = int(d["num_updates"].item())
        self.decay = d["decay"]
        self.warmup_steps = d["warmup_steps"]
        logger.info(
            "EMAHook restored: num_updates=%d, decay=%.4f",
            self.num_updates, self.decay,
        )


class TeacherHook:
    """EMA Teacher for consistency training.

    Maintains a teacher model updated via EMA of the student.
    Teacher never participates in backprop.
    """

    def __init__(self, student_model: nn.Module, ema_decay: float = 0.999):
        if not 0.0 < ema_decay <= 1.0:
            raise ValueError(f"ema_decay must be in (0, 1], got {ema_decay}")
        self.ema_decay = ema_decay
        self._teacher = copy.deepcopy(student_model)
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)
        logger.info("TeacherHook: ema_decay=%.4f", ema_decay)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self._teacher(images)

    def get_teacher(self) -> nn.Module:
        return self._teacher

    def state_dict(self) -> dict:
        return {
            "teacher_model": self._teacher.state_dict(),
            "ema_decay": self.ema_decay,
        }

    def load_state_dict(self, d: dict):
        self._teacher.load_state_dict(d["teacher_model"])
        self.ema_decay = d["ema_decay"]
