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

    Maintains a teacher model updated via EMA of the student after each
    successful optimizer step.  Teacher never participates in backprop.

    Usage in train.py::

        teacher = TeacherHook(model, ema_decay=0.999, ramp_epochs=10)
        for batch in loader: ...
            student_logits = model(images)
            task_loss = criterion(student_logits, labels)
            # --- consistency loss ---
            with torch.no_grad():
                teacher_logits = teacher(images)
                conf_mask = teacher.confidence_mask(teacher_logits, threshold=0.8)
            if conf_mask.any():
                ramp_w = teacher.rampup_weight(epoch)
                cons_loss = F.mse_loss(student_logits[conf_mask],
                                       teacher_logits[conf_mask])
                task_loss = task_loss + ramp_w * consistency_weight * cons_loss
            # --- end consistency ---
            task_loss.backward()
            optimizer.step()
            teacher.update(model)
    """

    def __init__(
        self,
        student_model: nn.Module,
        ema_decay: float = 0.999,
        ramp_epochs: int = 10,
    ):
        if not 0.0 < ema_decay <= 1.0:
            raise ValueError(f"ema_decay must be in (0, 1], got {ema_decay}")
        if ramp_epochs < 0:
            raise ValueError(f"ramp_epochs must be >= 0, got {ramp_epochs}")

        self.ema_decay = ema_decay
        self.ramp_epochs = ramp_epochs
        self.num_updates: int = 0

        self._teacher = copy.deepcopy(student_model)
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)

        logger.info(
            "TeacherHook: ema_decay=%.4f, ramp_epochs=%d",
            ema_decay, ramp_epochs,
        )

    # ── EMA update ────────────────────────────────────────────────────

    @torch.no_grad()
    def update(self, student_model: nn.Module):
        """Update teacher via EMA after one successful optimizer step.

        teacher = ema_decay * teacher + (1 - ema_decay) * student
        """
        self.num_updates += 1
        for teacher_p, student_p in zip(
            self._teacher.parameters(), student_model.parameters()
        ):
            teacher_p.data.mul_(self.ema_decay).add_(
                student_p.data, alpha=1.0 - self.ema_decay
            )

    # ── Forward ───────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Teacher inference (no grad, eval mode)."""
        return self._teacher(images)

    def get_teacher(self) -> nn.Module:
        """Return the teacher model (for direct access)."""
        return self._teacher

    # ── Confidence mask ───────────────────────────────────────────────

    @staticmethod
    def confidence_mask(
        logits: torch.Tensor, threshold: float = 0.8
    ) -> torch.Tensor:
        """Boolean mask: True for samples where teacher max-prob >= *threshold*."""
        probs = torch.softmax(logits, dim=1)
        max_probs, _ = probs.max(dim=1)
        return max_probs >= threshold

    # ── Ramp-up ───────────────────────────────────────────────────────

    def rampup_weight(self, epoch: int) -> float:
        """Sigmoid ramp-up: 0 at epoch 0, approaching 1.0 after *ramp_epochs*."""
        if self.ramp_epochs <= 0:
            return 1.0
        if epoch <= 0:
            return 0.0
        progress = min(float(epoch) / float(self.ramp_epochs), 1.0)
        # Sigmoid: smooth transition from 0→1
        import math
        return 1.0 / (1.0 + math.exp(-10.0 * (progress - 0.5)))

    # ── Checkpoint serialisation ──────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "teacher_model": self._teacher.state_dict(),
            "ema_decay": self.ema_decay,
            "ramp_epochs": self.ramp_epochs,
            "num_updates": torch.tensor(self.num_updates, dtype=torch.long),
        }

    def load_state_dict(self, d: dict):
        self._teacher.load_state_dict(d["teacher_model"])
        self.ema_decay = d["ema_decay"]
        self.ramp_epochs = d.get("ramp_epochs", 10)
        self.num_updates = int(d.get("num_updates", torch.tensor(0)).item())
        logger.info(
            "TeacherHook restored: num_updates=%d, ema_decay=%.4f",
            self.num_updates, self.ema_decay,
        )
