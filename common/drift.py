"""Feature drift audit for PEFT experiments.

Computes per-epoch diagnostics comparing a fine-tuned model against its
frozen parent: cosine distance of visual features, KL divergence of
logits, and prediction change rate decomposition.

Designed to be called once per epoch on a fixed validation subset.
Output is a JSON-serialisable dictionary suitable for ``drift_epoch_N.json``.

Usage::

    from common.drift import compute_drift_audit

    drift = compute_drift_audit(
        student_model=model,
        parent_model=parent,
        dataloader=val_subset_loader,
        device=device,
    )
    # drift is a dict; save as JSON.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_drift_audit(
    student_model: torch.nn.Module,
    parent_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_list: Optional[List[int]] = None,
) -> dict:
    """Compute feature-drift and prediction-change diagnostics.

    Parameters
    ----------
    student_model:
        The fine-tuned (PEFT) model in eval mode.
    parent_model:
        The frozen parent model in eval mode.  Must expose
        ``encode_image`` and a ``forward`` that returns logits.
    dataloader:
        DataLoader yielding ``(images, labels, paths)`` over a fixed
        validation subset.
    device:
        Torch device.
    label_list:
        If provided, a list of ground-truth labels in the same order as
        the dataloader.  If None, labels are taken from the dataloader.

    Returns
    -------
    dict with keys:
        - mean_cosine_distance
        - p50_cosine_distance
        - p95_cosine_distance
        - max_cosine_distance
        - mean_logit_kl
        - prediction_change_rate
        - changed_correct_to_incorrect
        - changed_incorrect_to_correct
        - num_samples
    """
    parent_model.eval()
    student_model.eval()

    all_cos_dists: List[float] = []
    all_kls: List[float] = []
    pred_changes: int = 0
    changed_correct_to_incorrect: int = 0
    changed_incorrect_to_correct: int = 0
    total_samples: int = 0

    label_idx = 0

    for batch in dataloader:
        images, labels, _paths = _unpack_val_batch(batch, device)
        batch_size = images.size(0)

        # Features
        s_feat = student_model.encode_image(images)
        p_feat = parent_model.encode_image(images)

        s_feat_n = F.normalize(s_feat.float(), p=2, dim=-1)
        p_feat_n = F.normalize(p_feat.float(), p=2, dim=-1)
        cos_sim = (s_feat_n * p_feat_n).sum(dim=1)
        cos_dist = (1.0 - cos_sim).cpu().tolist()
        all_cos_dists.extend(cos_dist)

        # Logits
        s_logits = student_model(images)
        p_logits = parent_model(images)

        # KL(p_parent || p_student) — per sample
        s_log_probs = F.log_softmax(s_logits, dim=1)
        p_probs = F.softmax(p_logits, dim=1)
        kl = F.kl_div(s_log_probs, p_probs, reduction="none").sum(dim=1)
        all_kls.extend(kl.cpu().tolist())

        # Prediction change
        s_preds = s_logits.argmax(dim=1)
        p_preds = p_logits.argmax(dim=1)
        changed = (s_preds != p_preds)
        pred_changes += int(changed.sum().item())

        for i in range(batch_size):
            if changed[i]:
                parent_correct = (p_preds[i] == labels[i]).item()
                student_correct = (s_preds[i] == labels[i]).item()
                if parent_correct and not student_correct:
                    changed_correct_to_incorrect += 1
                elif not parent_correct and student_correct:
                    changed_incorrect_to_correct += 1

        total_samples += batch_size
        label_idx += batch_size

    # Aggregate
    cos_t = torch.tensor(all_cos_dists, dtype=torch.float32)
    kl_t = torch.tensor(all_kls, dtype=torch.float32)

    result = {
        "mean_cosine_distance": float(cos_t.mean().item()),
        "p50_cosine_distance": float(cos_t.median().item()),
        "p95_cosine_distance": float(
            cos_t.kthvalue(int(cos_t.size(0) * 0.95)).item()
        ) if cos_t.size(0) > 0 else 0.0,
        "max_cosine_distance": float(cos_t.max().item()),
        "mean_logit_kl": float(kl_t.mean().item()),
        "prediction_change_rate": (
            float(pred_changes / total_samples) if total_samples > 0 else 0.0
        ),
        "changed_correct_to_incorrect": changed_correct_to_incorrect,
        "changed_incorrect_to_correct": changed_incorrect_to_correct,
        "num_samples": total_samples,
    }

    logger.info(
        "Drift audit: cos_dist mean=%.6f p95=%.6f | KL mean=%.4f | "
        "pred_change=%.4f (%d→correct, %d→incorrect)",
        result["mean_cosine_distance"],
        result["p95_cosine_distance"],
        result["mean_logit_kl"],
        result["prediction_change_rate"],
        result["changed_incorrect_to_correct"],
        result["changed_correct_to_incorrect"],
    )

    return result


def _unpack_val_batch(batch_data, device: torch.device):
    """Unpack a validation batch — same logic as train._unpack_batch
    but always returns images (val never uses cached features)."""
    if len(batch_data) == 3:
        images, labels, paths = batch_data
    elif len(batch_data) == 2:
        images, labels = batch_data
        paths = None
    else:
        raise ValueError(f"Unexpected batch length: {len(batch_data)}")
    return images.to(device, non_blocking=True), \
        labels.to(device, non_blocking=True), \
        paths
