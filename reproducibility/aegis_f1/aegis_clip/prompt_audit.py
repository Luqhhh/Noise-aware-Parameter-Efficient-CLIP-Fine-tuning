"""Diagnostics for deciding whether numeric class names provide CLIP semantics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def numeric_prompt_diagnostics(
    *,
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    labels: torch.Tensor,
    clean_probability: torch.Tensor,
    classifier_weights: torch.Tensor,
    clean_core_threshold: float = 0.70,
) -> dict:
    """Measure numeric-prompt geometry without consulting competition test data."""
    text = F.normalize(torch.as_tensor(text_features).float(), dim=1)
    images = F.normalize(torch.as_tensor(image_features).float(), dim=1)
    classifier = F.normalize(torch.as_tensor(classifier_weights).float(), dim=1)
    labels = torch.as_tensor(labels).long().flatten()
    clean = torch.as_tensor(clean_probability).float().flatten()
    if text.ndim != 2 or images.ndim != 2 or classifier.ndim != 2:
        raise ValueError("Prompt audit features must be rank-two tensors")
    if not (
        text.shape == classifier.shape
        and images.shape[1] == text.shape[1]
        and images.shape[0] == labels.numel() == clean.numel()
    ):
        raise ValueError("Prompt audit tensor shapes are inconsistent")
    if not all(torch.isfinite(value).all() for value in (text, images, classifier, clean)):
        raise FloatingPointError("Prompt audit inputs contain NaN or Inf")
    if labels.numel() == 0 or labels.min() < 0 or labels.max() >= text.shape[0]:
        raise ValueError("Prompt audit labels are outside the class range")
    prediction = (images @ text.transpose(0, 1)).argmax(dim=1)
    clean_core = clean >= float(clean_core_threshold)
    if not clean_core.any():
        raise ValueError("Prompt audit clean core is empty")
    pairwise = text @ text.transpose(0, 1)
    off_diagonal = pairwise[
        ~torch.eye(text.shape[0], dtype=torch.bool, device=text.device)
    ]
    singular_values = torch.linalg.svdvals(text)
    energy = singular_values.square()
    cumulative = energy.cumsum(0) / energy.sum().clamp_min(1.0e-12)
    diagonal_alignment = (text * classifier).sum(dim=1)
    return {
        "validation_samples": int(labels.numel()),
        "raw_accuracy": float(prediction.eq(labels).float().mean()),
        "clean_core_accuracy": float(
            prediction[clean_core].eq(labels[clean_core]).float().mean()
        ),
        "clean_core_samples": int(clean_core.sum()),
        "unique_predicted_classes": int(torch.unique(prediction).numel()),
        "text_pairwise_off_diagonal": {
            "mean": float(off_diagonal.mean()),
            "std": float(off_diagonal.std(unbiased=False)),
            "minimum": float(off_diagonal.min()),
            "maximum": float(off_diagonal.max()),
        },
        "text_effective_rank": {
            "rank_90pct_energy": int(
                torch.searchsorted(cumulative, 0.90).item() + 1
            ),
            "rank_99pct_energy": int(
                torch.searchsorted(cumulative, 0.99).item() + 1
            ),
        },
        "same_id_alignment_with_classifier": {
            "mean": float(diagonal_alignment.mean()),
            "std": float(diagonal_alignment.std(unbiased=False)),
            "minimum": float(diagonal_alignment.min()),
            "maximum": float(diagonal_alignment.max()),
        },
    }
