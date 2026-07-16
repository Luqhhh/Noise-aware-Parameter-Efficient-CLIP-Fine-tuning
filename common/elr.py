"""Early Learning Regularization (ELR) hook.

Maintains per-sample running prediction targets and adds a temporal
consistency penalty when current predictions diverge from their
historical moving average.

Reference:
    Liu et al., "Early-Learning Regularization Prevents Memorization of
    Noisy Labels", NeurIPS 2020.

Integration pattern (follows EMAHook / TeacherHook convention)::

    elr = ELRHook(num_train_samples=92802, num_classes=500, momentum=0.9,
                  target_weight=1.0, warmup_epochs=10, ramp_epochs=10)

    for batch in loader:
        ...
        logits = model(inputs)
        task_loss = criterion(logits, labels)
        # --- ELR temporal consistency ---
        if not mixup_applied:
            elr.update(paths, logits)
            elr_weight = elr.rampup_weight(epoch)
            if elr_weight > 0:
                elr_loss = elr.compute_loss(paths, logits)
                task_loss = task_loss + elr_weight * elr_loss
        # --- end ELR ---
        task_loss.backward()
        optimizer.step()
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ELRHook:
    """Per-sample temporal ensembling for noisy-label robustness.

    For each training sample *i*, maintains an exponential moving average
    of its prediction vector::

        t_i ← β·t_i + (1 − β)·p_i

    where p_i = softmax(logits_i).  The ELR penalty discourages the
    current prediction from deviating far from its historical target::

        L_ELR = log(1 − p_i · t_i)

    The penalty is only active after *warmup_epochs* and is gradually
    ramped up over *ramp_epochs*.

    Parameters
    ----------
    num_train_samples:
        Total number of training samples.  Used to allocate the running-
        target buffer.  Each sample is identified by its image path and
        mapped to a stable slot index on first encounter.
    num_classes:
        Number of output classes.
    momentum:
        EMA decay factor β for the running targets (0 < β < 1).
    target_weight:
        Multiplier λ applied to the ELR loss term before ramp-up.
    warmup_epochs:
        Number of full epochs before the ELR penalty is activated.
        During this period ``rampup_weight`` returns 0.
    ramp_epochs:
        Number of epochs over which λ linearly increases from 0 to
        *target_weight*.
    storage_dtype:
        Dtype for the running-target buffer.  ``float32`` (default) is
        numerically safest; ``float16`` saves memory at the cost of
        potential underflow in ``log(1 − dot)``.
    """

    def __init__(
        self,
        num_train_samples: int,
        num_classes: int,
        momentum: float = 0.9,
        target_weight: float = 1.0,
        warmup_epochs: int = 10,
        ramp_epochs: int = 10,
        storage_dtype: torch.dtype = torch.float32,
    ):
        if not 0.0 < momentum < 1.0:
            raise ValueError(f"momentum must be in (0, 1), got {momentum}")
        if target_weight < 0:
            raise ValueError(f"target_weight must be >= 0, got {target_weight}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if ramp_epochs < 0:
            raise ValueError(f"ramp_epochs must be >= 0, got {ramp_epochs}")

        self.momentum = momentum
        self.target_weight = target_weight
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.num_train_samples = num_train_samples
        self.num_classes = num_classes

        # Running prediction targets — allocated lazily or up-front.
        # Shape: (num_train_samples, num_classes), init to zero.
        self.register_buffer: Optional[torch.Tensor] = None
        self._storage_dtype = storage_dtype

        # Path → slot index mapping (stable across epochs)
        self._path_to_slot: Dict[str, int] = {}
        self._next_slot: int = 0

        # Per-slot update counter (for diagnostics)
        self._slot_updates: Optional[torch.Tensor] = None

        logger.info(
            "ELRHook: momentum=%.3f, target_weight=%.2f, "
            "warmup_epochs=%d, ramp_epochs=%d, num_train_samples=%d",
            momentum, target_weight, warmup_epochs, ramp_epochs,
            num_train_samples,
        )

    # ── Lazy buffer allocation ─────────────────────────────────────────

    def _ensure_buffer(self, device: torch.device):
        """Allocate the running-target buffer on first use."""
        if self.register_buffer is None:
            self.register_buffer = torch.zeros(
                self.num_train_samples, self.num_classes,
                dtype=self._storage_dtype, device=device,
            )
            self._slot_updates = torch.zeros(
                self.num_train_samples, dtype=torch.long, device=device,
            )
            logger.info(
                "ELRHook: allocated running-target buffer "
                "(%d × %d, %s, %.1f MB)",
                self.num_train_samples, self.num_classes,
                self._storage_dtype,
                self.register_buffer.element_size() * self.num_train_samples * self.num_classes / (1024 * 1024),
            )

    def _get_slot_indices(self, paths: Tuple[str, ...]) -> torch.Tensor:
        """Map image paths to stable slot indices.

        New paths are assigned the next available slot.  Paths seen
        during previous calls return their existing slot.
        """
        indices = []
        for p in paths:
            if p not in self._path_to_slot:
                if self._next_slot >= self.num_train_samples:
                    raise RuntimeError(
                        f"ELRHook: slot overflow — saw more unique paths "
                        f"({self._next_slot + 1}) than num_train_samples "
                        f"({self.num_train_samples})"
                    )
                self._path_to_slot[p] = self._next_slot
                self._next_slot += 1
            indices.append(self._path_to_slot[p])
        return torch.tensor(indices, dtype=torch.long)

    # ── Core update ────────────────────────────────────────────────────

    @torch.no_grad()
    def update(self, paths: Tuple[str, ...], logits: torch.Tensor):
        """Update running targets for a batch of samples.

        Called once per batch (only for non-MixUp batches).  Uses the
        current softmax predictions to update the EMA targets.

        Parameters
        ----------
        paths:
            Tuple of image-path strings (batch_size,).
        logits:
            Raw logits from the model (batch_size, num_classes).
        """
        device = logits.device
        self._ensure_buffer(device)

        slot_idx = self._get_slot_indices(paths).to(device)
        probs = F.softmax(logits, dim=1).to(dtype=self._storage_dtype)

        # EMA update: t ← β·t + (1−β)·p
        beta = self.momentum
        self.register_buffer[slot_idx] = (
            beta * self.register_buffer[slot_idx] + (1.0 - beta) * probs
        )
        self._slot_updates[slot_idx] += 1

    # ── Loss computation ───────────────────────────────────────────────

    def compute_loss(
        self, paths: Tuple[str, ...], logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute the ELR penalty for a batch.

        Returns a **scalar** loss (mean over the batch).  Only call this
        when ``rampup_weight(epoch) > 0``.

        Parameters
        ----------
        paths:
            Tuple of image-path strings (batch_size,).
        logits:
            Raw logits from the model (batch_size, num_classes).

        Returns
        -------
        Scalar ELR loss.
        """
        device = logits.device
        self._ensure_buffer(device)

        slot_idx = self._get_slot_indices(paths).to(device)
        probs = F.softmax(logits, dim=1)
        targets = self.register_buffer[slot_idx].to(dtype=probs.dtype)

        # L_ELR = log(1 − p · t)
        dot = (probs * targets).sum(dim=1)                 # (batch_size,)
        # Clamp dot < 1 to avoid log(0) / log(negative)
        one_minus_dot = (1.0 - dot).clamp(min=1e-8)
        elr_per_sample = -torch.log(one_minus_dot)          # (batch_size,)
        return elr_per_sample.mean()

    # ── Diagnostics ────────────────────────────────────────────────────

    @torch.no_grad()
    def target_entropy(self, paths: Tuple[str, ...]) -> torch.Tensor:
        """Mean entropy of running targets for diagnostics."""
        device = self.register_buffer.device if self.register_buffer is not None else None
        if device is None:
            return torch.tensor(0.0)
        slot_idx = self._get_slot_indices(paths).to(device)
        t = self.register_buffer[slot_idx]
        # Avoid log(0)
        t_safe = t.clamp(min=1e-8)
        entropy = -(t_safe * torch.log(t_safe)).sum(dim=1)
        return entropy.mean()

    @property
    def slots_filled(self) -> int:
        """Number of unique samples registered so far."""
        return self._next_slot

    # ── Ramp-up scheduling ─────────────────────────────────────────────

    def rampup_weight(self, epoch: int) -> float:
        """Linearly ramp ELR weight from 0 to *target_weight*.

        Returns 0 during warmup, then increases linearly over
        *ramp_epochs*, and stays at *target_weight* thereafter.
        """
        if epoch < self.warmup_epochs:
            return 0.0
        if self.ramp_epochs <= 0:
            return self.target_weight
        ramp_progress = min(
            float(epoch - self.warmup_epochs) / float(self.ramp_epochs), 1.0
        )
        return ramp_progress * self.target_weight

    # ── Checkpoint serialisation ────────────────────────────────────────

    def state_dict(self) -> dict:
        """Return serialisable state for checkpoint save."""
        d: dict = {
            "momentum": self.momentum,
            "target_weight": self.target_weight,
            "warmup_epochs": self.warmup_epochs,
            "ramp_epochs": self.ramp_epochs,
            "num_train_samples": self.num_train_samples,
            "num_classes": self.num_classes,
            "next_slot": self._next_slot,
            "path_to_slot": dict(self._path_to_slot),
        }
        if self.register_buffer is not None:
            d["register_buffer"] = self.register_buffer.cpu().clone()
            d["slot_updates"] = self._slot_updates.cpu().clone()
        return d

    def load_state_dict(self, d: dict):
        """Restore state from a checkpoint."""
        self.momentum = d["momentum"]
        self.target_weight = d.get("target_weight", 1.0)
        self.warmup_epochs = d.get("warmup_epochs", 10)
        self.ramp_epochs = d.get("ramp_epochs", 10)
        self.num_train_samples = d["num_train_samples"]
        self.num_classes = d["num_classes"]
        self._next_slot = d.get("next_slot", 0)
        self._path_to_slot = d.get("path_to_slot", {})

        if "register_buffer" in d and d["register_buffer"] is not None:
            self.register_buffer = d["register_buffer"].clone()
            self._slot_updates = (
                d["slot_updates"].clone()
                if "slot_updates" in d
                else torch.zeros(self.num_train_samples, dtype=torch.long)
            )
        else:
            self.register_buffer = None
            self._slot_updates = None

        logger.info(
            "ELRHook restored: slots_filled=%d, momentum=%.3f",
            self._next_slot, self.momentum,
        )
