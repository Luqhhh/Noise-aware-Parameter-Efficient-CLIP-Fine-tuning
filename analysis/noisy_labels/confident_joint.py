"""Class-conditional confident joint and label issue ranking.

Minimal implementation — no external ``cleanlab`` dependency.
Works with pre-computed OOF probability matrices.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def estimate_class_thresholds(
    probabilities: torch.Tensor,
    noisy_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Estimate per-class self-confidence thresholds.

    t_c = mean_{i: y_i=c} p_i(c)

    Args:
        probabilities: (N, C) softmax probabilities.
        noisy_labels: (N,) observed labels.
        num_classes: Number of classes.

    Returns:
        (C,) tensor of per-class thresholds.
    """
    thresholds = torch.zeros(num_classes)
    for c in range(num_classes):
        mask = noisy_labels == c
        if mask.any():
            thresholds[c] = probabilities[mask, c].mean()
    return thresholds


def build_confident_joint(
    probabilities: torch.Tensor,
    noisy_labels: torch.Tensor,
    thresholds: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """Build the confident joint matrix.

    For each sample i, compute suggested label from
    S_i = {c : p_i(c) >= t_c}.  If S_i non-empty, suggested = argmax_{c in S_i}
    p_i(c); otherwise suggested = argmax_c p_i(c).

    Returns:
        (C, C) numpy integer matrix where CJ[observed, suggested] is the count.
    """
    n = len(probabilities)
    joint = np.zeros((num_classes, num_classes), dtype=np.int64)
    noisy_np = noisy_labels.numpy() if isinstance(noisy_labels, torch.Tensor) else noisy_labels

    for i in range(n):
        obs = int(noisy_np[i])
        probs_i = probabilities[i]
        # Build candidate set S_i
        above = (probs_i >= thresholds).nonzero(as_tuple=False)
        if len(above) > 0:
            # argmax within S_i
            suggested = int(above[probs_i[above].argmax()].item())
        else:
            suggested = int(probs_i.argmax().item())
        joint[obs, suggested] += 1

    return joint


def rank_label_issues(
    probabilities: torch.Tensor,
    noisy_labels: torch.Tensor,
    thresholds: torch.Tensor,
    confident_joint: np.ndarray,
    *,
    max_class_reject_rate: float = 0.10,
    max_global_reject_rate: float = 0.10,
    knn_agreement: np.ndarray | None = None,
    flip_consistency: np.ndarray | None = None,
    top1_margin: np.ndarray | None = None,
) -> "pd.DataFrame":
    """Rank label issues with class caps and return a DataFrame.

    Returns a DataFrame with columns: index, observed, suggested, score, selected.
    """
    import pandas as pd

    n = len(probabilities)
    num_classes = confident_joint.shape[0]
    noisy_np = noisy_labels.numpy() if isinstance(noisy_labels, torch.Tensor) else noisy_labels

    # Count per class
    class_counts = np.bincount(noisy_np, minlength=num_classes)
    # Estimated issues per class: row_sum[c] - CJ[c,c]
    row_sum = confident_joint.sum(axis=1)
    diag = confident_joint.diagonal()
    est_issues = np.maximum(0, row_sum - diag)

    # Compute per-class cap
    class_cap = np.floor(max_class_reject_rate * class_counts).astype(np.int64)
    global_cap = int(np.floor(max_global_reject_rate * n))

    # Find label issues (suggested != observed) and compute score
    issue_rows = []
    for i in range(n):
        obs = int(noisy_np[i])
        probs_i = probabilities[i]
        above = (probs_i >= thresholds).nonzero(as_tuple=False)
        if len(above) > 0:
            suggested = int(above[probs_i[above].argmax()].item())
        else:
            suggested = int(probs_i.argmax().item())

        if suggested == obs:
            continue  # Not a label issue

        p_original = float(probs_i[obs])
        # Default signal values if not provided
        _knn = float(knn_agreement[i]) if knn_agreement is not None else 0.5
        _flip = float(flip_consistency[i]) if flip_consistency is not None else 1.0
        _margin = float(top1_margin[i]) if top1_margin is not None else 0.0

        score = (
            0.50 * (1.0 - p_original)
            + 0.25 * _margin
            + 0.15 * (1.0 - _knn)
            + 0.10 * (1.0 - _flip)
        )
        issue_rows.append({
            "index": i,
            "observed": obs,
            "suggested": int(suggested),
            "score": score,
            "p_original_label": p_original,
        })

    issues = pd.DataFrame(issue_rows)
    if len(issues) == 0:
        issues["selected"] = False
        return issues

    # Sort by score descending
    issues = issues.sort_values("score", ascending=False).reset_index(drop=True)

    # Apply per-class caps
    selected = np.zeros(len(issues), dtype=bool)
    class_used = np.zeros(num_classes, dtype=np.int64)
    global_count = 0

    for idx, row in issues.iterrows():
        obs = int(row["observed"])
        if class_used[obs] >= class_cap[obs]:
            continue
        if global_count >= global_cap:
            break
        selected[idx] = True
        class_used[obs] += 1
        global_count += 1

    issues["selected"] = selected
    return issues
