"""Deterministic nested grouped cross-fitting for CVRG reliability gates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from .view_reliability import (
    CVRGProtocol,
    FrozenReliabilityGate,
    RELIABILITY_FEATURE_NAMES,
    predict_view_reliability,
)


@dataclass(frozen=True)
class CVRGFitConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    c_candidates: tuple[float, ...] = (0.01, 0.1, 1.0)
    seed: int = 42
    maximum_iterations: int = 1000


@dataclass(frozen=True)
class GateFitMetadata:
    checkpoint_sha256: str = ""
    validation_cache_sha256: str = ""
    feature_schema_sha256: str = ""
    protocol: CVRGProtocol = CVRGProtocol()


@dataclass(frozen=True)
class CrossFitResult:
    oof_reliability: torch.Tensor
    outer_fold_id: torch.Tensor
    selected_c_by_outer_fold: tuple[float, ...]
    inner_brier_by_outer_fold: tuple[dict[str, float], ...]
    feature_schema_sha256: str


def _validate_inputs(features, view_logits, labels, groups):
    if features.ndim != 3 or features.shape[1] != 4:
        raise ValueError("features must have shape [N,4,F]")
    if view_logits.ndim != 3 or view_logits.shape[:2] != features.shape[:2]:
        raise ValueError("view_logits and features are not aligned")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have shape [N]")
    if len(groups) != features.shape[0] or len(set(groups)) != len(groups):
        raise ValueError("groups must contain one unique path per image")
    if not torch.isfinite(features.float()).all() or not torch.isfinite(view_logits.float()).all():
        raise ValueError("cross-fitting inputs must be finite")
    if torch.unique(labels).numel() < 2:
        raise ValueError("labels must contain at least two classes")
    return int(features.shape[0]), int(features.shape[-1])


def make_image_folds(labels, groups: Sequence[str], *, folds: int, seed: int) -> torch.Tensor:
    labels = torch.as_tensor(labels).long()
    if labels.ndim != 1 or len(groups) != labels.numel():
        raise ValueError("labels and groups must be aligned")
    if folds < 2 or len(set(groups)) != len(groups):
        raise ValueError("folds must be >=2 and groups must be unique")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = torch.full((labels.numel(),), -1, dtype=torch.long)
    indices = np.arange(labels.numel())
    for fold, (_, holdout) in enumerate(splitter.split(indices, labels.numpy(), groups)):
        if (result[torch.as_tensor(holdout)] >= 0).any():
            raise RuntimeError("image received multiple outer folds")
        result[torch.as_tensor(holdout)] = fold
    if (result < 0).any():
        raise RuntimeError("incomplete fold assignment")
    return result


def _rows(features, logits, labels, image_indices):
    selected = torch.as_tensor(image_indices).long()
    x = features[selected].reshape(-1, features.shape[-1]).float()
    y = (logits[selected].argmax(-1) == labels[selected, None]).long().reshape(-1)
    if torch.unique(y).numel() < 2:
        raise ValueError("correctness target must contain both correctness classes")
    return x, y


def _fit(features, logits, labels, image_indices, *, c, seed, maximum_iterations, metadata):
    x, y = _rows(features, logits, labels, image_indices)
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1.0e-6)
    x_scaled = (x - mean) / scale
    model = LogisticRegression(
        C=float(c), penalty="l2", solver="lbfgs", max_iter=maximum_iterations,
        random_state=seed,
    )
    model.fit(x_scaled.numpy(), y.numpy())
    names = tuple(RELIABILITY_FEATURE_NAMES) if x.shape[1] == len(RELIABILITY_FEATURE_NAMES) else tuple(f"feature.{i}" for i in range(x.shape[1]))
    schema_hash = metadata.feature_schema_sha256 or hashlib.sha256(chr(10).join(names).encode()).hexdigest()
    return FrozenReliabilityGate(
        feature_names=names,
        feature_mean=mean,
        feature_scale=scale,
        coefficient=torch.from_numpy(model.coef_[0].astype(np.float32)),
        intercept=float(model.intercept_[0]),
        regularization_c=float(c),
        checkpoint_sha256=metadata.checkpoint_sha256,
        validation_cache_sha256=metadata.validation_cache_sha256,
        feature_schema_sha256=schema_hash,
        protocol=metadata.protocol,
    )


def fit_frozen_gate(features, view_logits, labels, image_indices, *, c, seed, metadata=GateFitMetadata(), maximum_iterations=1000):
    return _fit(features, view_logits, labels, image_indices, c=c, seed=seed, maximum_iterations=maximum_iterations, metadata=metadata)


def select_regularization_c(features, view_logits, labels, groups, image_indices, *, config=CVRGFitConfig(), seed_offset=0):
    selected = torch.as_tensor(image_indices).long()
    if selected.numel() < config.inner_folds:
        raise ValueError("not enough images for inner folds")
    inner_labels = torch.as_tensor(labels)[selected]
    inner_groups = [groups[int(i)] for i in selected.tolist()]
    folds = make_image_folds(inner_labels, inner_groups, folds=config.inner_folds, seed=config.seed + seed_offset)
    scores = {}
    for candidate in sorted(config.c_candidates):
        fold_scores = []
        for fold in range(config.inner_folds):
            train = selected[folds != fold]
            holdout = selected[folds == fold]
            gate = _fit(features, view_logits, labels, train, c=candidate, seed=config.seed + seed_offset + fold,
                        maximum_iterations=config.maximum_iterations, metadata=GateFitMetadata())
            predicted = predict_view_reliability(features[holdout], gate)
            target = (view_logits[holdout].argmax(-1) == labels[holdout, None]).float()
            fold_scores.append(float((predicted - target).square().mean()))
        scores[str(candidate)] = float(np.mean(fold_scores))
    best = min(sorted(config.c_candidates), key=lambda c: (scores[str(c)], c))
    return float(best), scores


def cross_fit_reliability(features, view_logits, labels, groups, *, config=CVRGFitConfig()):
    n, _ = _validate_inputs(features, view_logits, labels, groups)
    if config.outer_folds < 2 or config.inner_folds < 2:
        raise ValueError("outer_folds and inner_folds must be >=2")
    outer = make_image_folds(labels, groups, folds=config.outer_folds, seed=config.seed)
    oof = torch.empty((n, 4), dtype=torch.float32)
    selected_cs = []
    audits = []
    for fold in range(config.outer_folds):
        train = torch.nonzero(outer != fold, as_tuple=False).flatten()
        holdout = torch.nonzero(outer == fold, as_tuple=False).flatten()
        candidate, scores = select_regularization_c(
            features, view_logits, labels, groups, train,
            config=config, seed_offset=1000 + fold,
        )
        gate = _fit(features, view_logits, labels, train, c=candidate, seed=config.seed + fold,
                    maximum_iterations=config.maximum_iterations, metadata=GateFitMetadata())
        oof[holdout] = predict_view_reliability(features[holdout], gate)
        selected_cs.append(candidate)
        audits.append(scores)
    return CrossFitResult(
        oof_reliability=oof,
        outer_fold_id=outer,
        selected_c_by_outer_fold=tuple(selected_cs),
        inner_brier_by_outer_fold=tuple(audits),
        feature_schema_sha256=hashlib.sha256(chr(10).join(RELIABILITY_FEATURE_NAMES).encode()).hexdigest(),
    )
