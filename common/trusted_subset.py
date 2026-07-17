"""
Trusted validation subset construction.

V1 rules are model-agnostic: they use only CLIP features, kNN topology,
robust class prototypes, and flip stability. They NEVER read D3/B2 logits,
confidence, margin, loss, or correctness.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrustedSubsetConfig:
    """Fixed V1 thresholds for trusted subset selection.

    These thresholds are pre-specified and must NOT be tuned based on
    platform scores or D3/B2 relative performance.
    """
    knn_label_agreement_min: float = 0.60
    prototype_margin_min: float = 0.02
    clip_flip_cosine_min: float = 0.90
    require_prototype_top1_matches_label: bool = True
    reject_cross_class_duplicate_conflict: bool = True

    # ── V3: OOF single-threshold trusted subset ──
    p_oof_label_min: float = 0.60  # OOF prob of original label (top 60% ≈ p≥0.598)

    # ── V2: continuous trust-weighted & class-balanced metrics ──
    class_balanced_top_k: int = 5
    prototype_margin_ref: float = 0.05  # reference for w_proto normalization


REJECTION_LABELS = {
    "low_knn_agreement": "kNN label agreement below threshold",
    "prototype_label_mismatch": "Prototype top-1 does not match noisy label",
    "low_prototype_margin": "Prototype margin below threshold",
    "low_clip_flip_cosine": "CLIP flip cosine below threshold",
    "cross_class_duplicate_conflict": "Cross-class duplicate conflict detected",
    "missing_conflict_metadata": "Conflict metadata not available — partial assessment",
    "low_oof_probability": "OOF probability of original label below threshold",
}


def build_trusted_subset(
    df: pd.DataFrame,
    config: TrustedSubsetConfig = TrustedSubsetConfig(),
) -> Tuple[pd.DataFrame, dict]:
    """Build trusted validation subset using model-agnostic V1 rules.

    Args:
        df: DataFrame with per-sample metrics. Required columns:
            knn_label_agreement, prototype_supports_noisy_label,
            prototype_margin, clip_flip_cosine,
            cross_class_duplicate_conflict (optional — if missing, defaults
            to False with warning).
        config: TrustedSubsetConfig with thresholds.

    Returns:
        manifest: Copy of df with added columns:
            trusted_v1 (bool), rejection_reasons (str).
        summary: Dict with coverage, counts, per-class stats.
    """
    df = df.copy()
    n_total = len(df)

    # Check for conflict metadata
    has_conflict = "cross_class_duplicate_conflict" in df.columns
    if not has_conflict:
        logger.warning(
            "cross_class_duplicate_conflict column not found. "
            "Defaulting to False — trusted subset marked as partial."
        )
        df["cross_class_duplicate_conflict"] = False

    # ── Build rejection reasons per sample ──
    reasons = pd.Series([[] for _ in range(n_total)], dtype=object)

    # Rule 1: kNN label agreement
    low_knn = df["knn_label_agreement"] < config.knn_label_agreement_min
    for i in df.index[low_knn]:
        reasons.iloc[i] = reasons.iloc[i] + ["low_knn_agreement"]

    # Rule 2: Prototype top-1 matches noisy label
    if config.require_prototype_top1_matches_label:
        proto_mismatch = ~df["prototype_supports_noisy_label"].astype(bool)
        for i in df.index[proto_mismatch]:
            reasons.iloc[i] = reasons.iloc[i] + ["prototype_label_mismatch"]

    # Rule 3: Prototype margin
    low_proto_margin = df["prototype_margin"] < config.prototype_margin_min
    for i in df.index[low_proto_margin]:
        reasons.iloc[i] = reasons.iloc[i] + ["low_prototype_margin"]

    # Rule 4: CLIP flip cosine
    low_flip_cos = df["clip_flip_cosine"] < config.clip_flip_cosine_min
    for i in df.index[low_flip_cos]:
        reasons.iloc[i] = reasons.iloc[i] + ["low_clip_flip_cosine"]

    # Rule 5: Cross-class duplicate conflict
    if config.reject_cross_class_duplicate_conflict:
        conflict = df["cross_class_duplicate_conflict"].astype(bool)
        for i in df.index[conflict]:
            reasons.iloc[i] = reasons.iloc[i] + ["cross_class_duplicate_conflict"]

    # Rule 6: Missing conflict metadata marker
    if not has_conflict:
        for i in df.index:
            reasons.iloc[i] = reasons.iloc[i] + ["missing_conflict_metadata"]

    # ── Determine trusted ──
    trusted = reasons.apply(lambda r: len(r) == 0)
    rejection_str = reasons.apply(lambda r: ";".join(r) if r else "")

    df["trusted_v1"] = trusted
    df["rejection_reasons"] = rejection_str

    # ── Build summary ──
    trusted_count = trusted.sum()
    coverage = trusted_count / n_total if n_total > 0 else 0.0

    represented_classes = df[trusted]["noisy_label"].nunique() if trusted_count > 0 else 0
    total_classes = df["noisy_label"].nunique()

    per_class = df.groupby("noisy_label")["trusted_v1"].agg(["sum", "count"])
    per_class.columns = ["trusted", "total"]
    per_class["coverage"] = per_class["trusted"] / per_class["total"]

    per_class_trusted = per_class["trusted"].astype(int).to_dict()

    summary = {
        "total_samples": n_total,
        "trusted_count": int(trusted_count),
        "rejected_count": int(n_total - trusted_count),
        "coverage": float(coverage),
        "represented_classes": int(represented_classes),
        "total_classes": int(total_classes),
        "missing_classes": int(total_classes - represented_classes),
        "conflict_metadata_available": has_conflict,
        "per_class_trusted": per_class_trusted,
        "min_trusted_per_class": int(per_class["trusted"].min()) if len(per_class) > 0 else 0,
        "median_trusted_per_class": float(per_class["trusted"].median()) if len(per_class) > 0 else 0.0,
        "p10_trusted_per_class": float(per_class["trusted"].quantile(0.10)) if len(per_class) > 0 else 0.0,
        "p90_trusted_per_class": float(per_class["trusted"].quantile(0.90)) if len(per_class) > 0 else 0.0,
        "rejection_reason_counts": {},
        "config": {
            "knn_label_agreement_min": config.knn_label_agreement_min,
            "prototype_margin_min": config.prototype_margin_min,
            "clip_flip_cosine_min": config.clip_flip_cosine_min,
            "require_prototype_top1_matches_label": config.require_prototype_top1_matches_label,
            "reject_cross_class_duplicate_conflict": config.reject_cross_class_duplicate_conflict,
        },
    }

    # Count rejection reasons
    all_reasons = []
    for r in reasons:
        all_reasons.extend(r)
    from collections import Counter
    summary["rejection_reason_counts"] = dict(Counter(all_reasons))

    return df, summary


# ═══════════════════════════════════════════════════════════════════════
# V3: OOF single-threshold trusted subset
# ═══════════════════════════════════════════════════════════════════════


def build_trusted_subset_oof(
    df: pd.DataFrame,
    config: TrustedSubsetConfig = TrustedSubsetConfig(),
) -> Tuple[pd.DataFrame, dict]:
    """Build trusted validation subset using OOF p(original_label) threshold.

    Uses a single model-agnostic signal: the OOF (out-of-fold) predicted
    probability of the original (noisy) label.  This is a genuinely independent
    quality estimate because OOF predictions come from models that never
    trained on the samples they scored.

    A sample is trusted when ``p_original_label >= p_oof_label_min``.

    Args:
        df: DataFrame with required column ``p_original_label`` (float in [0,1])
            plus ``noisy_label`` for per-class summary stats.
        config: TrustedSubsetConfig with ``p_oof_label_min`` threshold.

    Returns:
        manifest: Copy of df with added columns:
            trusted_v1 (bool), rejection_reasons (str).
        summary: Dict with coverage, counts, per-class stats.
    """
    df = df.copy()
    n_total = len(df)

    if "p_original_label" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'p_original_label' column for OOF-based "
            "trusted subset. Ensure the OOF quality manifest is loaded."
        )

    # ── Single rule: p_original_label >= threshold ──
    below_threshold = df["p_original_label"] < config.p_oof_label_min

    reasons = pd.Series([[] for _ in range(n_total)], dtype=object)
    for i in df.index[below_threshold]:
        reasons.iloc[i] = ["low_oof_probability"]

    trusted = reasons.apply(lambda r: len(r) == 0)
    rejection_str = reasons.apply(lambda r: ";".join(r) if r else "")

    df["trusted_v1"] = trusted
    df["rejection_reasons"] = rejection_str

    # ── Build summary ──
    trusted_count = trusted.sum()
    coverage = trusted_count / n_total if n_total > 0 else 0.0

    represented_classes = df[trusted]["noisy_label"].nunique() if trusted_count > 0 else 0
    total_classes = df["noisy_label"].nunique()

    per_class = df.groupby("noisy_label")["trusted_v1"].agg(["sum", "count"])
    per_class.columns = ["trusted", "total"]
    per_class["coverage"] = per_class["trusted"] / per_class["total"]

    per_class_trusted = per_class["trusted"].astype(int).to_dict()

    summary = {
        "method": "oof_single_threshold",
        "p_oof_label_min": config.p_oof_label_min,
        "total_samples": n_total,
        "trusted_count": int(trusted_count),
        "rejected_count": int(n_total - trusted_count),
        "coverage": float(coverage),
        "represented_classes": int(represented_classes),
        "total_classes": int(total_classes),
        "missing_classes": int(total_classes - represented_classes),
        "per_class_trusted": per_class_trusted,
        "min_trusted_per_class": int(per_class["trusted"].min()) if len(per_class) > 0 else 0,
        "median_trusted_per_class": float(per_class["trusted"].median()) if len(per_class) > 0 else 0.0,
        "p10_trusted_per_class": float(per_class["trusted"].quantile(0.10)) if len(per_class) > 0 else 0.0,
        "p90_trusted_per_class": float(per_class["trusted"].quantile(0.90)) if len(per_class) > 0 else 0.0,
        "rejection_reason_counts": {},
    }

    # Count rejection reasons
    all_reasons_list = []
    for r in reasons:
        all_reasons_list.extend(r)
    from collections import Counter
    summary["rejection_reason_counts"] = dict(Counter(all_reasons_list))

    return df, summary


# ═══════════════════════════════════════════════════════════════════════
# V2: Continuous trust-weighted and class-balanced trusted metrics
# ═══════════════════════════════════════════════════════════════════════


def _compute_composite_trust_weights(
    df: pd.DataFrame,
    margin_ref: float = 0.05,
) -> np.ndarray:
    """Compute continuous composite trust weights per sample.

    w_i = w_knn_i × w_proto_i × w_flip_i

    where:
      - w_knn_i  = knn_label_agreement          (already in [0, 1])
      - w_proto_i = clamp(proto_margin / margin_ref, 0, 1)
      - w_flip_i  = clip_flip_cosine             (already in [0, 1])

    Args:
        df: DataFrame with columns knn_label_agreement, prototype_margin,
            clip_flip_cosine.
        margin_ref: Reference margin for w_proto normalisation (default 0.05).

    Returns:
        Float ndarray of shape (n_samples,) with weights in [0, 1].
    """
    w_knn = df["knn_label_agreement"].values.astype(np.float64)
    w_flip = df["clip_flip_cosine"].values.astype(np.float64)

    proto_margin = df["prototype_margin"].values.astype(np.float64)
    w_proto = np.clip(proto_margin / margin_ref, 0.0, 1.0)

    weights = w_knn * w_proto * w_flip
    # Clamp to [0, 1] to guard against floating-point overshoot
    weights = np.clip(weights, 0.0, 1.0)
    return weights


def compute_trust_weighted_accuracy(
    df: pd.DataFrame,
    correct: np.ndarray,
    margin_ref: float = 0.05,
) -> dict:
    """Compute continuous trust-weighted accuracy.

    Uses ALL validation samples — no hard acceptance/rejection threshold.
    Each sample contributes w_i rather than a binary 1/0, giving higher
    influence to samples that multiple model-agnostic signals agree are
    trustworthy.

    score = Σ(w_i × 1[ŷ_i = y_i]) / Σ(w_i)

    Args:
        df: DataFrame with columns knn_label_agreement, prototype_margin,
            clip_flip_cosine, noisy_label (for per-class breakdown).
        correct: Boolean array of shape (n_samples,) — True if prediction
            matches the noisy label.
        margin_ref: Reference margin for w_proto normalisation (default 0.05).

    Returns:
        Dict with keys:
            accuracy:            trust-weighted accuracy
            weight_sum:          Σ w_i (total trust mass)
            effective_samples:   Σ w_i (same — interpretable as
                                 "trust-equivalent sample count")
            total_samples:       N
            mean_weight:         mean per-sample trust weight
            median_weight:       median per-sample trust weight
            per_class_accuracy:  dict class_idx → trust-weighted accuracy
                                 (NaN if zero weight in class)
    """
    n_total = len(df)
    if n_total == 0:
        return {
            "accuracy": float("nan"),
            "weight_sum": 0.0,
            "effective_samples": 0.0,
            "total_samples": 0,
            "mean_weight": float("nan"),
            "median_weight": float("nan"),
            "per_class_accuracy": {},
        }

    weights = _compute_composite_trust_weights(df, margin_ref=margin_ref)
    correct_float = np.asarray(correct, dtype=np.float64)
    weight_sum = float(weights.sum())

    if weight_sum == 0.0:
        accuracy = float("nan")
    else:
        accuracy = float((weights * correct_float).sum() / weight_sum)

    # Per-class
    labels = df["noisy_label"].values.astype(int)
    per_class = {}
    unique_classes = np.unique(labels)
    for c in unique_classes:
        c_mask = labels == c
        c_weights = weights[c_mask]
        c_weight_sum = float(c_weights.sum())
        if c_weight_sum > 0:
            per_class[str(c)] = float(
                (c_weights * correct_float[c_mask]).sum() / c_weight_sum
            )
        else:
            per_class[str(c)] = float("nan")

    return {
        "accuracy": accuracy,
        "weight_sum": weight_sum,
        "effective_samples": weight_sum,
        "total_samples": n_total,
        "mean_weight": float(weights.mean()),
        "median_weight": float(np.median(weights)),
        "per_class_accuracy": per_class,
    }


def compute_class_balanced_trusted_accuracy(
    df: pd.DataFrame,
    correct: np.ndarray,
    top_k: int = 5,
    margin_ref: float = 0.05,
) -> dict:
    """Compute class-balanced trusted accuracy via per-class Top-K selection.

    For each class:
      1. Rank samples by composite trust score (kNN × prototype × flip).
      2. Select the top-K highest-trust samples.
      3. Compute accuracy on those K samples.
    Then macro-average across all classes that have at least K candidates.

    This avoids the "ceiling effect" of V1 (where trusted accuracy ≈ 99.8%)
    and guarantees every class with sufficient candidates contributes equally.

    Args:
        df: DataFrame with columns knn_label_agreement, prototype_margin,
            clip_flip_cosine, noisy_label.
        correct: Boolean array of shape (n_samples,).
        top_k: Number of top-trust samples per class (default 5).
        margin_ref: Reference margin for w_proto normalisation.

    Returns:
        Dict with keys:
            top_k_per_class:       K value used
            macro_accuracy:         mean of per-class top-K accuracies
            num_classes_with_k:     classes with ≥ K candidates
            num_classes_total:       total classes with any candidate
            total_classes:          total unique classes in df
            coverage:               samples used / total samples
            num_samples_used:       total samples selected
            per_class_accuracy:     dict class_idx → top-K accuracy
            per_class_count:        dict class_idx → num candidates in class
    """
    n_total = len(df)
    if n_total == 0:
        return {
            "top_k_per_class": top_k,
            "macro_accuracy": float("nan"),
            "num_classes_with_k": 0,
            "num_classes_total": 0,
            "total_classes": 0,
            "coverage": 0.0,
            "num_samples_used": 0,
            "per_class_accuracy": {},
            "per_class_count": {},
        }

    weights = _compute_composite_trust_weights(df, margin_ref=margin_ref)
    correct_float = np.asarray(correct, dtype=np.float64)
    labels = df["noisy_label"].values.astype(int)

    unique_classes = np.unique(labels)
    per_class_acc = {}
    per_class_count = {}
    samples_used = 0
    n_with_k = 0
    n_with_any = 0

    for c in unique_classes:
        c_mask = labels == c
        c_count = int(c_mask.sum())
        per_class_count[str(c)] = c_count

        if c_count == 0:
            per_class_acc[str(c)] = float("nan")
            continue

        n_with_any += 1

        if c_count < top_k:
            per_class_acc[str(c)] = float("nan")
            continue

        n_with_k += 1
        # Select top-K by composite trust weight
        c_indices = np.where(c_mask)[0]
        c_weights = weights[c_indices]
        top_k_idx = c_indices[np.argsort(-c_weights)[:top_k]]
        samples_used += top_k

        top_k_correct = correct_float[top_k_idx]
        per_class_acc[str(c)] = float(top_k_correct.mean())

    # Macro-average over classes with ≥ K candidates
    valid_accs = [v for v in per_class_acc.values() if not np.isnan(v)]
    macro_accuracy = float(np.mean(valid_accs)) if valid_accs else float("nan")

    coverage = samples_used / n_total if n_total > 0 else 0.0

    return {
        "top_k_per_class": top_k,
        "macro_accuracy": macro_accuracy,
        "num_classes_with_k": n_with_k,
        "num_classes_total": n_with_any,
        "total_classes": len(unique_classes),
        "coverage": coverage,
        "num_samples_used": samples_used,
        "per_class_accuracy": per_class_acc,
        "per_class_count": per_class_count,
    }
