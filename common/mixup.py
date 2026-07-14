"""MixUp data augmentation for image classification.

Applies MixUp (Zhang et al., 2018) to a batch of images and labels:

    x_mix = lam * x_i + (1 - lam) * x_j
    y_mix = lam * y_i + (1 - lam) * y_j  (handled via dual loss)

Usage in train.py::

    from common.mixup import mixup_batch

    images, labels_a, labels_b, lam = mixup_batch(images, labels, alpha=0.2, prob=0.2)
    logits = model(images)
    loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
    probability: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp to a batch with probability *probability*.

    When MixUp is NOT applied, returns ``(images, labels, labels, 1.0)`` so the
    standard ``lam * loss_a + (1-lam) * loss_b`` formula degenerates to a single
    loss term.

    Args:
        images: Input batch, shape (B, C, H, W).
        labels: Integer class labels, shape (B,).
        alpha: Beta distribution parameter.  Smaller values = weaker mixing.
        probability: Probability of applying MixUp to the batch.

    Returns:
        Tuple of ``(mixed_images, labels_a, labels_b, lam)`` where:
        - *mixed_images*: (B, C, H, W)
        - *labels_a*: (B,) — first component labels
        - *labels_b*: (B,) — second component labels
        - *lam*: float — weight for the first component (lam >= 0.5)
    """
    if alpha <= 0 or probability <= 0:
        return images, labels, labels, 1.0

    # Probabilistic application
    if np.random.random() > probability:
        return images, labels, labels, 1.0

    batch_size = images.size(0)
    if batch_size < 2:
        return images, labels, labels, 1.0

    # Sample lambda
    lam = float(np.random.beta(alpha, alpha))
    # Enforce lam >= 0.5 so the "primary" label is the dominant one
    lam = max(lam, 1.0 - lam)

    # Random permutation for pairing
    index = torch.randperm(batch_size, device=images.device)

    # Mix images
    mixed_images = lam * images + (1.0 - lam) * images[index]

    return mixed_images, labels, labels[index], lam
