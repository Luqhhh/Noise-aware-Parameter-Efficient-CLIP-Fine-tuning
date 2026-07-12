"""Class-prior logit adjustment for test-time calibration.

Given class priors π_c (estimated from training data) and a temperature τ,
adjusts logits as:

    z̃_c = z_c - τ · log(π_c + ε)

This penalizes over-confident predictions toward head classes when the
test distribution is (assumed to be) uniform.

Reference:
    Menon et al. "Long-tail learning via logit adjustment", ICLR 2021.
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F


def compute_class_priors(
    train_csv_path: str,
    num_classes: int = 500,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Compute empirical class priors from a training split CSV.

    Args:
        train_csv_path: Path to train.csv with columns
            [image_path, label, class_name].
        num_classes: Total number of classes.
        epsilon: Small constant added to every count (smoothing).

    Returns:
        Tensor of shape (num_classes,) with π_c summing to 1.
    """
    import csv

    counts = torch.zeros(num_classes, dtype=torch.float32)
    with open(train_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{train_csv_path} is empty")
        label_col = 1  # second column is the label index
        for row in reader:
            label = int(row[label_col])
            counts[label] += 1

    counts = counts + epsilon
    priors = counts / counts.sum()
    return priors


def adjust_logits(
    logits: torch.Tensor,
    priors: torch.Tensor,
    tau: float,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Apply class-prior logit adjustment.

    Args:
        logits: (N, C) tensor of raw logits.
        priors: (C,) tensor of class priors π_c.
        tau: Temperature / adjustment strength.
        epsilon: Small constant for log stability.

    Returns:
        Adjusted logits of same shape as input.
    """
    if tau == 0.0:
        return logits
    adjustment = tau * torch.log(priors.to(logits.device) + epsilon)
    return logits - adjustment.unsqueeze(0)


def sweep_logit_adjustment(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    priors: torch.Tensor,
    taus: Sequence[float],
) -> Dict[float, Dict[str, float]]:
    """Evaluate adjusted logits across a grid of tau values.

    Args:
        val_logits: (N, C) validation logits.
        val_labels: (N,) integer labels.
        priors: (C,) class priors.
        taus: Sequence of tau values to evaluate.

    Returns:
        Dict mapping tau → {micro, macro, bottom10, median}.
    """
    results = {}
    for tau in taus:
        adjusted = adjust_logits(val_logits, priors, tau)
        metrics = _compute_metrics_from_logits(adjusted, val_labels)
        results[tau] = metrics
    return results


def _compute_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """Compute micro, macro, median, bottom-10% accuracy from logits.

    Args:
        logits: (N, C) tensor.
        labels: (N,) integer labels.

    Returns:
        Dict with micro_accuracy, macro_accuracy, median_per_class_accuracy,
        bottom_10_percent_accuracy, micro_macro_gap.
    """
    num_classes = logits.size(1)
    preds = logits.argmax(dim=1)

    correct = (preds == labels).sum().item()
    total = len(labels)
    micro = correct / total

    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)

    for c in range(num_classes):
        mask = (labels == c)
        n_c = mask.sum().item()
        if n_c > 0:
            total_per_class[c] = n_c
            correct_per_class[c] = (preds[mask] == c).sum().item()

    per_class_acc = correct_per_class.float() / total_per_class.float().clamp(min=1)
    macro = per_class_acc.mean().item()
    median = per_class_acc.median().item()

    k = max(1, num_classes // 10)
    bottom10 = per_class_acc.topk(k, largest=False).values.mean().item()

    return {
        "micro_accuracy": micro,
        "macro_accuracy": macro,
        "median_per_class_accuracy": median,
        "bottom_10_percent_accuracy": bottom10,
        "micro_macro_gap": micro - macro,
        "total_samples": total,
        "correct_samples": int(correct),
    }
