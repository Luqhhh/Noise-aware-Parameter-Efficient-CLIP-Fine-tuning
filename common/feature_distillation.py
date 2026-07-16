"""Feature distillation loss for PEFT experiments.

Prevents catastrophic feature drift when unfreezing backbone parameters
by penalising cosine distance between the student (trainable) and parent
(frozen) visual features.

Usage::

    distill = FeatureDistillation(parent_model)
    ...
    features = model.encode_image(images)
    parent_features = distill.get_parent_features(images)
    feat_loss = distill.compute_loss(features, parent_features)
    total_loss = task_loss + lambda_feat * feat_loss

The parent model is always kept in eval mode with ``requires_grad=False``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FeatureDistillation:
    """Cosine-distance penalty against a frozen parent model.

    Parameters
    ----------
    parent_model:
        Pre-trained model whose CLIP visual encoder will be kept frozen
        and used as the distillation target.  Must expose an
        ``encode_image(images) -> Tensor`` method returning
        L2-normalised features of shape ``(B, feature_dim)``.
    normalize_features:
        If True (default), both student and parent features are L2-
        normalised before computing cosine distance.
    compare_after_projection:
        If True, uses features *after* ``visual.proj``.  This is the
        default because ``encode_image`` already includes projection.
    """

    def __init__(
        self,
        parent_model: nn.Module,
        normalize_features: bool = True,
        compare_after_projection: bool = True,
    ):
        self._parent = parent_model
        self.normalize_features = normalize_features
        self.compare_after_projection = compare_after_projection

        # Freeze parent
        for p in self._parent.parameters():
            p.requires_grad_(False)
        self._parent.eval()

        logger.info(
            "FeatureDistillation: normalize=%s, after_projection=%s",
            normalize_features, compare_after_projection,
        )

    # ── Parent feature extraction ──────────────────────────────────────

    @torch.no_grad()
    def get_parent_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract frozen parent features for a batch of images.

        Returns
        -------
        Tensor of shape ``(B, feature_dim)``.  L2-normalised if
        *normalize_features* is True.
        """
        was_training = self._parent.training
        self._parent.eval()
        try:
            features = self._parent.encode_image(images)
            if self.normalize_features:
                features = F.normalize(features.float(), p=2, dim=-1)
            return features
        finally:
            if was_training:
                self._parent.train()

    # ── Loss computation ───────────────────────────────────────────────

    def compute_loss(
        self,
        student_features: torch.Tensor,
        parent_features: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine-distance loss between student and parent features.

        Parameters
        ----------
        student_features:
            Student features of shape ``(B, D)``.
        parent_features:
            Parent features of shape ``(B, D)``.

        Returns
        -------
        Scalar loss: ``mean(1 − cos(f_s, f_p))``.
        """
        if self.normalize_features:
            s = F.normalize(student_features.float(), p=2, dim=-1)
            p = F.normalize(parent_features.float(), p=2, dim=-1)
        else:
            s = student_features.float()
            p = parent_features.float()

        cos_sim = (s * p).sum(dim=1)                    # (B,)
        return (1.0 - cos_sim).mean()                   # scalar


def calibrate_feature_loss_weight(
    distill: FeatureDistillation,
    model: nn.Module,
    images: torch.Tensor,
    task_loss: torch.Tensor,
    target_ratio: float = 0.15,
) -> float:
    """Estimate the weight λ so that the feature loss is ~*target_ratio*
    of the total loss on a representative batch.

    Returns a suggested λ value.  This is a one-shot calibration —
    the caller should run it once and fix the weight for the remainder
    of training.

    Parameters
    ----------
    distill:
        Configured ``FeatureDistillation`` instance.
    model:
        The student model.
    images:
        A representative batch of images.
    task_loss:
        The scalar supervised loss on this batch (without feature term).
    target_ratio:
        Desired fraction of total loss from the feature term
        (default 0.15 = 15%).

    Returns
    -------
    Suggested λ (float).
    """
    with torch.no_grad():
        student_feat = model.encode_image(images)
        parent_feat = distill.get_parent_features(images)
        raw_feat_loss = distill.compute_loss(student_feat, parent_feat)

    if raw_feat_loss.item() == 0.0 or task_loss.item() == 0.0:
        logger.warning(
            "Feature loss or task loss is zero — using default λ=1.0"
        )
        return 1.0

    # Solve: λ·feat_loss / (task_loss + λ·feat_loss) = target_ratio
    # → λ = target_ratio · task_loss / ((1 − target_ratio) · feat_loss)
    lam = (
        target_ratio * task_loss.item()
        / ((1.0 - target_ratio) * raw_feat_loss.item())
    )
    lam = max(lam, 0.0)
    logger.info(
        "Feature distillation calibration: task_loss=%.4f, feat_loss=%.4f, "
        "suggested λ=%.4f (target ratio=%.0f%%)",
        task_loss.item(), raw_feat_loss.item(), lam, target_ratio * 100,
    )
    return lam
