"""Multi-signal consensus drop and relabel selection.

NR_CL_KNN_DROP: confident-joint issue + OOF/kNN agreement drop.
NR_CONSENSUS_RELABEL_V2: core oof==knn hard + 3-of-5 auxiliary voting + top-k.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Float32 boundary tolerance for knn_agreement conditions
EPS = 1e-6


def select_consensus_drop(
    quality: pd.DataFrame,
    issues: pd.DataFrame,
    max_class_reject_rate: float = 0.10,
    max_global_reject_rate: float = 0.08,
    min_clean_per_class: int = 50,
) -> set:
    """Select samples for NR_CL_KNN_DROP.

    Conditions (ALL must hold):
      confident_joint marks it as a label issue
      oof_top1 != original_label
      knn_top1 != original_label
      oof_top1 == knn_top1
      top1_margin >= class 75th percentile
      knn_agreement <= 0.20 + EPS  (low agreement with noisy label)
      duplicate_conflict_flag == false

    Candidates are sorted by confidence score before per-class caps.
    """
    NUM_CLASSES = 500
    n = len(quality)
    issue_indices = set(issues[issues["selected"]]["index"].values)

    # Class-level margin thresholds
    class_margin_q75 = {}
    for c in range(NUM_CLASSES):
        mask = quality["original_label"] == c
        if mask.sum() > 0:
            class_margin_q75[c] = float(quality.loc[mask, "top1_margin"].quantile(0.75))
        else:
            class_margin_q75[c] = 0.0

    candidates = []
    for i, row in quality.iterrows():
        if i not in issue_indices:
            continue
        oof_t1 = int(row.get("oof_top1", -1))
        knn_t1 = int(row.get("knn_top1", -1))
        orig = int(row["original_label"])

        if oof_t1 == orig:
            continue
        if knn_t1 == orig:
            continue
        if oof_t1 != knn_t1:
            continue
        if float(row.get("top1_margin", 0.0)) < class_margin_q75.get(orig, 0.0):
            continue
        if float(row.get("knn_agreement", 1.0)) > 0.20 + EPS:
            continue
        if bool(row.get("duplicate_conflict_flag", False)):
            continue

        # Score: prefer low knn_agreement (strong disagreement with noisy label)
        score = (
            0.40 * (1.0 - float(row.get("knn_agreement", 0.5)))
            + 0.30 * float(row.get("top1_margin", 0.0))
            + 0.30 * float(row.get("p_top1", 0.5))
        )
        candidates.append((i, score))

    # Sort by score descending before caps
    candidates.sort(key=lambda x: x[1], reverse=True)
    scored_indices = [idx for idx, _ in candidates]

    selected = _apply_caps(
        quality, scored_indices, NUM_CLASSES,
        max_class_reject_rate, max_global_reject_rate,
        min_clean_per_class,
    )
    return selected


def select_consensus_relabel_v2(
    quality: pd.DataFrame,
    issues: pd.DataFrame,
    top_k: int = 100,
) -> set:
    """Select top-k high-confidence relabel candidates.

    Core hard conditions (ALL must hold):
      confident_joint marks it as a label issue
      oof_top1 != original_label
      knn_top1 != original_label
      oof_top1 == knn_top1
      duplicate_conflict_flag == false

    Auxiliary conditions (at least 3 of 5):
      prototype_top1 == oof_top1
      p_top1 >= class 90th percentile
      top1_margin >= class 75th percentile
      knn_top1_agreement >= 0.60  (strong kNN support for the NEW label)
      flip_consistency == 1

    Score rewards: high knn_top1_agreement, high p_top1, high margin,
    prototype match, flip consistency.
    """
    NUM_CLASSES = 500
    n = len(quality)
    issue_indices = set(issues[issues["selected"]]["index"].values)

    # Class-level percentiles
    q90_p_top1 = {}
    q75_margin = {}
    for c in range(NUM_CLASSES):
        mask = quality["original_label"] == c
        if mask.sum() > 0:
            q90_p_top1[c] = float(quality.loc[mask, "p_top1"].quantile(0.90))
            q75_margin[c] = float(quality.loc[mask, "top1_margin"].quantile(0.75))
        else:
            q90_p_top1[c] = 0.90
            q75_margin[c] = 0.50

    global_q90 = float(quality["p_top1"].quantile(0.90))
    global_q75 = float(quality["top1_margin"].quantile(0.75))

    # Percentile ranks for scoring
    p_top1_vals = quality["p_top1"].values
    margin_vals = quality["top1_margin"].values
    kta_vals = quality.get("knn_top1_agreement", pd.Series([0.5] * n)).values
    p_top1_rank = p_top1_vals.argsort().argsort() / max(n, 1)
    margin_rank = margin_vals.argsort().argsort() / max(n, 1)
    kta_rank = kta_vals.argsort().argsort() / max(n, 1)

    candidates = []
    for i, row in quality.iterrows():
        if i not in issue_indices:
            continue
        oof_t1 = int(row.get("oof_top1", -1))
        knn_t1 = int(row.get("knn_top1", -1))
        orig = int(row["original_label"])

        # Core hard conditions
        if oof_t1 == orig:
            continue
        if knn_t1 == orig:
            continue
        if oof_t1 != knn_t1:
            continue
        if bool(row.get("duplicate_conflict_flag", False)):
            continue

        # Auxiliary conditions (need at least 3 of 5)
        aux = 0
        proto_ok = int(row.get("prototype_top1", -1)) == oof_t1
        if proto_ok:
            aux += 1

        p_thresh = q90_p_top1.get(orig, global_q90)
        if float(row["p_top1"]) >= p_thresh:
            aux += 1

        m_thresh = q75_margin.get(orig, global_q75)
        if float(row["top1_margin"]) >= m_thresh:
            aux += 1

        # Use knn_top1_agreement — support for the NEW label
        kta = float(row.get("knn_top1_agreement", 0.5))
        if kta >= 0.60:
            aux += 1

        flip_ok = float(row.get("flip_consistency", 0.0)) == 1.0
        if flip_ok:
            aux += 1

        if aux < 3:
            continue

        # Score: reward high knn_top1_agreement (support for new label)
        score = (
            0.35 * float(p_top1_rank[i])
            + 0.25 * float(margin_rank[i])
            + 0.20 * float(kta_rank[i])
            + 0.10 * float(proto_ok)
            + 0.10 * float(flip_ok)
        )
        candidates.append((i, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return set(idx for idx, _ in candidates[:top_k])


def _apply_caps(
    quality: pd.DataFrame,
    scored_indices: list,
    num_classes: int,
    max_class_reject_rate: float,
    max_global_reject_rate: float,
    min_clean_per_class: int,
) -> set:
    """Apply per-class and global caps to scored candidates."""
    n = len(quality)
    class_counts = quality["original_label"].value_counts().reindex(
        range(num_classes), fill_value=0
    )
    class_cap = {
        c: min(
            int(np.floor(max_class_reject_rate * class_counts[c])),
            max(0, class_counts[c] - min_clean_per_class),
        )
        for c in range(num_classes)
    }
    global_cap = int(np.floor(max_global_reject_rate * n))

    selected = set()
    class_used = {c: 0 for c in range(num_classes)}

    for idx in scored_indices:
        if len(selected) >= global_cap:
            break
        orig = int(quality.iloc[idx]["original_label"])
        if class_used[orig] >= class_cap[orig]:
            continue
        selected.add(idx)
        class_used[orig] += 1

    return selected
