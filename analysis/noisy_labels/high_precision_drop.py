"""High-precision consensus rejection for N1.

N1 deliberately migrates the preliminary ``NR_CL_KNN_DROP`` mechanism to the
750-class rematch data instead of inventing a new noise score.  A sample is
rejected only when all available high-precision evidence agrees:

* the confident-joint ranker marks it as a label issue;
* OOF top-1 and fold-safe kNN top-1 disagree with the noisy label;
* OOF top-1 and kNN top-1 agree with each other;
* the OOF margin is at least the class 75th percentile;
* kNN agreement with the noisy label is at most ``0.20``.

Cross-label exact duplicate conflicts are added as an independent hard
rejection signal.  The resulting sidecar uses ``weight = 0`` for rejected
samples and ``weight = 1`` for everything else.  Labels are never changed and
no pseudo-label or soft-repair path is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KNN_AGREEMENT_MAX = 0.20
FLOAT_EPSILON = 1.0e-6
SUSPECT_REASON = "cl_knn_strict_suspect"
DUP_REASON = "cross_label_exact_duplicate_conflict"


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"quality frame is missing columns: {sorted(missing)}")


def _duplicate_conflict_mask(quality: pd.DataFrame) -> np.ndarray:
    """Return a boolean exact-duplicate conflict mask.

    An explicit ``duplicate_conflict_flag`` wins when present.  Otherwise an
    exact content group column (``content_group``, ``sha256``, or
    ``file_sha256``) is used.  A group conflicts when it contains more than one
    original label.
    """
    count = len(quality)
    explicit_column = next(
        (
            name
            for name in ("duplicate_conflict_flag", "duplicate_conflict")
            if name in quality.columns
        ),
        None,
    )
    explicit = (
        quality[explicit_column].fillna(False).astype(bool).to_numpy(copy=True)
        if explicit_column is not None
        else np.zeros(count, dtype=bool)
    )

    group_column = next(
        (
            name
            for name in ("content_group", "sha256", "file_sha256")
            if name in quality.columns
        ),
        None,
    )
    if group_column is None:
        return explicit

    labels = quality["original_label"].astype(int)
    groups = quality.groupby(group_column, dropna=False)["original_label"].nunique()
    conflicted_groups = set(groups[groups > 1].index.tolist())
    inferred = quality[group_column].isin(conflicted_groups).to_numpy(copy=True)
    return explicit | inferred


def select_high_precision_drop(
    quality: pd.DataFrame,
    issues: pd.DataFrame,
    *,
    knn_agreement_max: float = KNN_AGREEMENT_MAX,
    class_margin_quantile: float = 0.75,
) -> dict[str, Any]:
    """Select strict consensus-drop suspects and duplicate-conflict samples.

    ``issues`` must be the output of the existing confident-joint ranker and
    contain ``index`` and ``selected`` columns aligned to ``quality.index``.
    The function returns a manifest frame plus audit metadata.
    """
    if not isinstance(quality, pd.DataFrame) or quality.empty:
        raise ValueError("quality must be a non-empty DataFrame")
    if not isinstance(issues, pd.DataFrame):
        raise ValueError("issues must be a DataFrame")
    required = {
        "image_path",
        "original_label",
        "oof_top1",
        "knn_top1",
        "top1_margin",
        "knn_agreement",
    }
    _require_columns(quality, required)
    if "index" not in issues.columns or "selected" not in issues.columns:
        raise ValueError("issues must contain index and selected columns")
    if not 0.0 <= float(knn_agreement_max) < 1.0:
        raise ValueError("knn_agreement_max must be in [0, 1)")
    if not 0.0 < float(class_margin_quantile) < 1.0:
        raise ValueError("class_margin_quantile must be in (0, 1)")

    quality = quality.reset_index(drop=True).copy()
    labels = quality["original_label"].astype(int).to_numpy()
    num_classes = int(labels.max()) + 1 if len(labels) else 0
    if num_classes <= 1:
        raise ValueError("quality must contain at least two classes")

    issue_selected = np.zeros(len(quality), dtype=bool)
    issue_index = issues["index"].astype(int).to_numpy()
    issue_flag = issues["selected"].fillna(False).astype(bool).to_numpy()
    if ((issue_index < 0) | (issue_index >= len(quality))).any():
        raise ValueError("issues.index is out of range")
    if len(set(issue_index.tolist())) != len(issue_index):
        raise ValueError("issues.index contains duplicates")
    issue_selected[issue_index] = issue_flag

    class_margin_q = {
        int(class_index): float(
            quality.loc[labels == class_index, "top1_margin"].quantile(
                float(class_margin_quantile)
            )
        )
        for class_index in range(num_classes)
        if bool((labels == class_index).any())
    }

    oof_top1 = quality["oof_top1"].astype(int).to_numpy()
    knn_top1 = quality["knn_top1"].astype(int).to_numpy()
    margins = quality["top1_margin"].astype(float).to_numpy()
    knn_agreement = quality["knn_agreement"].astype(float).to_numpy()
    class_margin = np.array(
        [class_margin_q.get(int(label), 0.0) for label in labels],
        dtype=float,
    )

    suspect = (
        issue_selected
        & (oof_top1 != labels)
        & (knn_top1 != labels)
        & (oof_top1 == knn_top1)
        & (margins >= class_margin)
        & (knn_agreement <= float(knn_agreement_max) + FLOAT_EPSILON)
    )
    duplicate_conflict = _duplicate_conflict_mask(quality)
    rejected = suspect | duplicate_conflict

    reasons = np.full(len(quality), "clean", dtype=object)
    reasons[duplicate_conflict] = DUP_REASON
    reasons[suspect] = np.where(
        duplicate_conflict[suspect],
        f"{SUSPECT_REASON}+{DUP_REASON}",
        SUSPECT_REASON,
    )
    weights = np.where(rejected, 0.0, 1.0).astype(np.float64)

    manifest = pd.DataFrame(
        {
            "image_path": quality["image_path"].astype(str),
            "weight": weights,
            "sample_id": quality["sample_id"].astype(str)
            if "sample_id" in quality.columns
            else quality["image_path"].astype(str),
            "original_label": labels,
            "training_role": np.where(rejected, "rejected", "clean"),
            "reject_reason": reasons,
            "oof_top1": oof_top1,
            "knn_top1": knn_top1,
            "cl_knn_suspect": suspect,
            "cross_label_duplicate_conflict": duplicate_conflict,
        }
    )

    class_counts = np.bincount(labels, minlength=num_classes)
    class_reject_counts = np.bincount(
        labels[rejected], minlength=num_classes
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class_rate = np.divide(
            class_reject_counts,
            class_counts,
            out=np.zeros(num_classes, dtype=float),
            where=class_counts > 0,
        )
    clean_counts = class_counts - class_reject_counts
    zero_clean_classes = sorted(
        int(index) for index in np.flatnonzero(clean_counts == 0)
    )
    if zero_clean_classes:
        raise ValueError(
            "high-precision rejection removes every clean sample from classes: "
            f"{zero_clean_classes[:10]}"
        )

    frequency_order = np.argsort(class_counts, kind="stable")
    ranked = [index for index in frequency_order if class_counts[index] > 0]
    third = max(1, len(ranked) // 3)
    head = ranked[-third:]
    middle = ranked[third:-third] if len(ranked) > 2 * third else []
    tail = ranked[:third]
    tier_report = {
        "head_reject_rate": _rate(rejected, np.isin(labels, head)),
        "middle_reject_rate": _rate(rejected, np.isin(labels, middle)),
        "tail_reject_rate": _rate(rejected, np.isin(labels, tail)),
    }

    reject_count = int(rejected.sum())
    audit = {
        "samples": int(len(quality)),
        "reject_count": reject_count,
        "reject_rate": reject_count / max(len(quality), 1),
        "suspect_count": int(suspect.sum()),
        "suspect_rate": float(suspect.mean()) if len(suspect) else 0.0,
        "duplicate_conflict_count": int(duplicate_conflict.sum()),
        "duplicate_conflict_rate": (
            float(duplicate_conflict.mean()) if len(duplicate_conflict) else 0.0
        ),
        "duplicate_conflicts_removed": int(duplicate_conflict.sum()),
        "knn_agreement_max": float(knn_agreement_max),
        "class_margin_quantile": float(class_margin_quantile),
        "per_class_reject_rate_min": (
            float(per_class_rate[class_counts > 0].min())
            if bool((class_counts > 0).any())
            else 0.0
        ),
        "per_class_reject_rate_median": (
            float(np.median(per_class_rate[class_counts > 0]))
            if bool((class_counts > 0).any())
            else 0.0
        ),
        "per_class_reject_rate_max": (
            float(per_class_rate[class_counts > 0].max())
            if bool((class_counts > 0).any())
            else 0.0
        ),
        "zero_clean_classes": zero_clean_classes,
        "classes": int((class_counts > 0).sum()),
        **tier_report,
    }
    return {"manifest": manifest, "audit": audit}


def _rate(rejected: np.ndarray, tier_mask: np.ndarray) -> float:
    if not bool(tier_mask.any()):
        return 0.0
    return float(rejected[tier_mask].astype(float).mean())


def load_quality_and_issues(
    quality_csv: str | Path,
    issues_csv: str | Path | None = None,
    *,
    oof_logits_path: str | Path | None = None,
    confident_joint_max_class_reject_rate: float = 0.10,
    confident_joint_max_global_reject_rate: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a quality table and either reuse or rebuild confident-joint issues."""
    quality = pd.read_csv(quality_csv)
    if issues_csv is not None:
        issues = pd.read_csv(issues_csv)
        return quality, issues
    if oof_logits_path is None:
        raise ValueError("either issues_csv or oof_logits_path is required")

    from analysis.noisy_labels.confident_joint import (
        build_confident_joint,
        estimate_class_thresholds,
        rank_label_issues,
    )

    payload = torch_load_cpu(oof_logits_path)
    if "logits" not in payload:
        raise ValueError(f"OOF logits payload lacks 'logits': {oof_logits_path}")
    logits = payload["logits"].float()
    labels = torch_tensor(quality["original_label"].to_numpy(copy=True))
    if logits.ndim != 2 or logits.shape[0] != len(quality):
        raise ValueError("OOF logits do not align with the quality table")
    num_classes = int(logits.shape[1])
    probabilities = logits.softmax(dim=1)
    thresholds = estimate_class_thresholds(probabilities, labels, num_classes)
    confident_joint = build_confident_joint(
        probabilities, labels, thresholds, num_classes
    )
    issues = rank_label_issues(
        probabilities,
        labels,
        thresholds,
        confident_joint,
        max_class_reject_rate=float(confident_joint_max_class_reject_rate),
        max_global_reject_rate=float(confident_joint_max_global_reject_rate),
        knn_agreement=quality["knn_agreement"].to_numpy(copy=True)
        if "knn_agreement" in quality.columns
        else None,
        flip_consistency=quality["flip_consistency"].to_numpy(copy=True)
        if "flip_consistency" in quality.columns
        else None,
        top1_margin=quality["top1_margin"].to_numpy(copy=True)
        if "top1_margin" in quality.columns
        else None,
    )
    return quality, issues


# Lazily imported torch helpers keep this module importable in environments
# that only need the pure-pandas duplicate audit.
def torch_load_cpu(path: str | Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping payload: {path}")
    return payload


def torch_tensor(values: Any):
    import torch

    return torch.as_tensor(values, dtype=torch.long)
