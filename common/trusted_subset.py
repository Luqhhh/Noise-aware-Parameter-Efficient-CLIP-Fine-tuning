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


REJECTION_LABELS = {
    "low_knn_agreement": "kNN label agreement below threshold",
    "prototype_label_mismatch": "Prototype top-1 does not match noisy label",
    "low_prototype_margin": "Prototype margin below threshold",
    "low_clip_flip_cosine": "CLIP flip cosine below threshold",
    "cross_class_duplicate_conflict": "Cross-class duplicate conflict detected",
    "missing_conflict_metadata": "Conflict metadata not available — partial assessment",
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
