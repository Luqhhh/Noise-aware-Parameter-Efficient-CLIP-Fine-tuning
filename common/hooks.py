"""Training hooks — EMA and Teacher–Student interfaces.

These are thin interfaces that the training loop calls.  The actual
algorithm logic is implemented by C (Head EMA, Teacher consistency loss, etc.).
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EMAHook:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model (EMA parameters).  The training
    loop calls ``update()`` after each optimizer step and optionally uses
    EMA parameters for validation.

    C is responsible for the specific EMA logic (decay schedule, etc.).
    This class provides the hook interface that train.py calls.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.99,
        warmup_epochs: int = 5,
    ):
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self.decay = decay
        self.warmup_epochs = warmup_epochs
        self._ema_model = copy.deepcopy(model)
        self._ema_model.eval()
        for p in self._ema_model.parameters():
            p.requires_grad_(False)
        self._raw_state: Optional[Dict] = None

        logger.info(
            "EMAHook: decay=%.4f, warmup_epochs=%d", decay, warmup_epochs
        )

    def update(self, model: nn.Module, epoch: int):
        """Update EMA parameters after an optimizer step.

        During warmup, EMA is a direct copy of the model.
        After warmup: ema = decay * ema + (1 - decay) * model.
        """
        if epoch <= self.warmup_epochs:
            # Direct copy during warmup
            self._ema_model.load_state_dict(model.state_dict())
        else:
            with torch.no_grad():
                for ema_p, model_p in zip(
                    self._ema_model.parameters(), model.parameters()
                ):
                    ema_p.data.mul_(self.decay).add_(
                        model_p.data, alpha=1.0 - self.decay
                    )

    def swap_to_ema(self, model: nn.Module):
        """Replace model parameters with EMA version (for validation/inference)."""
        self._raw_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self._ema_model.state_dict())
        logger.debug("Swapped to EMA parameters for evaluation.")

    def restore_raw(self, model: nn.Module):
        """Restore original (non-EMA) model parameters."""
        if self._raw_state is None:
            logger.warning("restore_raw called but no raw state saved.")
            return
        model.load_state_dict(self._raw_state)
        self._raw_state = None
        logger.debug("Restored raw parameters.")

    def state_dict(self) -> dict:
        return {
            "ema_model": self._ema_model.state_dict(),
            "decay": self.decay,
            "warmup_epochs": self.warmup_epochs,
        }

    def load_state_dict(self, d: dict):
        self._ema_model.load_state_dict(d["ema_model"])
        self.decay = d["decay"]
        self.warmup_epochs = d["warmup_epochs"]

    def get_ema_model(self) -> nn.Module:
        """Return the EMA model for direct inference (no swap needed)."""
        return self._ema_model


class TeacherHook:
    """EMA Teacher for consistency training.

    Maintains a teacher model updated via EMA of the student.
    Teacher never participates in backprop.
    C implements the consistency loss logic.
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

    def update(self):
        """EMA update: teacher = ema_decay * teacher + (1 - ema_decay) * student."""
        # Called externally with student model reference
        pass  # C implements

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Teacher inference — no gradient."""
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
