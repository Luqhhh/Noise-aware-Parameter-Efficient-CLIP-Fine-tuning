"""Transform OOF logits and model-agnostic signals into sample weights."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch


def _as_numpy(values: Iterable, expected_length: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if len(result) != expected_length:
        raise ValueError(
            f"{name} length {len(result)} does not match {expected_length} samples"
        )
    return result


def build_sample_quality(
    assignments: pd.DataFrame,
    logits: torch.Tensor,
    prototype_own_similarity: Iterable[float],
    prototype_margin: Iterable[float],
    prototype_top1: Iterable[int],
    knn_agreement: Iterable[float],
    knn_top1: Iterable[int],
    flip_consistency: Iterable[float],
    clip_flip_cosine: Iterable[float],
    duplicate_conflict_flag: Iterable[bool],
) -> pd.DataFrame:
    """Build the Phase-3 per-sample OOF quality table."""
    required = {"sample_id", "image_path", "label", "fold"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Missing assignment columns: {sorted(missing)}")
    if logits.ndim != 2 or logits.shape[0] != len(assignments):
        raise ValueError(
            f"Expected logits shape ({len(assignments)}, C), got {tuple(logits.shape)}"
        )

    n_samples = len(assignments)
    logits = logits.detach().float().cpu()
    probabilities = logits.softmax(dim=1)
    labels = torch.tensor(assignments["label"].to_numpy(copy=True), dtype=torch.long)
    top_values, top_indices = probabilities.topk(2, dim=1)
    row_indices = torch.arange(n_samples)
    p_original = probabilities[row_indices, labels]

    result = assignments[["sample_id", "image_path", "fold"]].copy()
    result["original_label"] = labels.numpy()
    result["oof_top1"] = top_indices[:, 0].numpy()
    result["p_original_label"] = p_original.numpy()
    result["p_top1"] = top_values[:, 0].numpy()
    result["top1_margin"] = (top_values[:, 0] - top_values[:, 1]).numpy()
    result["oof_cross_entropy"] = (-p_original.clamp_min(1e-12).log()).numpy()
    result["prototype_own_similarity"] = _as_numpy(
        prototype_own_similarity, n_samples, "prototype_own_similarity"
    )
    result["prototype_margin"] = _as_numpy(
        prototype_margin, n_samples, "prototype_margin"
    )
    result["prototype_top1"] = _as_numpy(
        prototype_top1, n_samples, "prototype_top1"
    ).astype(int)
    result["knn_agreement"] = _as_numpy(
        knn_agreement, n_samples, "knn_agreement"
    )
    result["knn_top1"] = _as_numpy(knn_top1, n_samples, "knn_top1").astype(int)
    result["flip_consistency"] = _as_numpy(
        flip_consistency, n_samples, "flip_consistency"
    )
    result["clip_flip_cosine"] = _as_numpy(
        clip_flip_cosine, n_samples, "clip_flip_cosine"
    )
    result["duplicate_conflict_flag"] = _as_numpy(
        duplicate_conflict_flag, n_samples, "duplicate_conflict_flag"
    ).astype(bool)
    class_frequency = assignments["label"].value_counts().to_dict()
    result["class_frequency"] = result["original_label"].map(class_frequency).astype(int)
    return result


def add_quality_weights(sample_quality: pd.DataFrame) -> pd.DataFrame:
    """Add protocol-defined continuous and three-level OOF weights."""
    required = {
        "original_label",
        "p_original_label",
        "prototype_margin",
        "knn_agreement",
        "flip_consistency",
    }
    missing = required - set(sample_quality.columns)
    if missing:
        raise ValueError(f"Missing quality columns: {sorted(missing)}")

    result = sample_quality.copy()
    grouped = result.groupby("original_label", sort=False)
    result["p_original_classwise_percentile"] = grouped[
        "p_original_label"
    ].rank(method="average", pct=True)
    result["prototype_margin_classwise_percentile"] = grouped[
        "prototype_margin"
    ].rank(method="average", pct=True)
    result["quality"] = (
        0.35 * result["p_original_classwise_percentile"]
        + 0.25 * result["prototype_margin_classwise_percentile"]
        + 0.25 * result["knn_agreement"].clip(0.0, 1.0)
        + 0.15 * result["flip_consistency"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    result["soft_weight"] = (0.3 + 0.7 * result["quality"]).clip(0.3, 1.0)
    result["quality_classwise_percentile"] = result.groupby(
        "original_label", sort=False
    )["quality"].rank(method="first", pct=True)
    percentile = result["quality_classwise_percentile"]
    result["discrete_weight"] = np.select(
        [percentile <= (1.0 / 3.0), percentile <= (2.0 / 3.0)],
        [0.3, 0.6],
        default=1.0,
    )
    return result
