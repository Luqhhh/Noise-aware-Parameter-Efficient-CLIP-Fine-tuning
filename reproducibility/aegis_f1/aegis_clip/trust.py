"""Cross-fitted visual trust estimation without semantic class names.

The detector deliberately combines two different visual views:

* an out-of-fold class prototype classifier, which preserves frozen CLIP geometry;
* an out-of-fold linear probe initialised from those prototypes, which adapts to
  the task while never scoring a sample it was trained on.

Class-wise two-component mixtures convert support scores into continuous clean
probabilities. A late-probe disagreement rule then recalibrates misleading
"easy" examples, inspired by Early Cutting rather than trusting early confidence.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class TrustBuildConfig:
    folds: int = 5
    seed: int = 42
    prototype_temperature: float = 0.05
    probe_temperature: float = 1.0
    probe_epochs: int = 8
    probe_batch_size: int = 2048
    probe_lr: float = 0.05
    probe_weight_decay: float = 1.0e-4
    correction_confidence: float = 0.0
    correction_confidence_quantile: float = 0.75
    maximum_correction_alpha: float = 0.45
    maximum_class_correction_rate: float = 0.20
    early_cut_strength: float = 0.70
    minimum_clean_probability: float = 0.05


@torch.no_grad()
def class_prototypes(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    features = F.normalize(features.float(), dim=1)
    labels = labels.long()
    dimension = features.shape[1]
    sums = torch.zeros(num_classes, dimension, dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.float32)
    sums.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    if (counts == 0).any():
        missing = torch.nonzero(counts == 0).flatten().tolist()
        raise ValueError(f"Prototype fold is missing classes: {missing[:10]}")
    return F.normalize(sums / counts.unsqueeze(1), dim=1)


def build_cross_fitted_trust(
    features: torch.Tensor,
    labels: torch.Tensor,
    paths: Sequence[str],
    num_classes: int,
    groups: Sequence[str] | None = None,
    config: TrustBuildConfig | None = None,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an OOF trust bundle and a compact diagnostic summary."""
    cfg = config or TrustBuildConfig()
    x = F.normalize(features.detach().cpu().float(), dim=1)
    y = labels.detach().cpu().long().flatten()
    if x.shape[0] != y.numel() or y.numel() != len(paths):
        raise ValueError("features, labels, and paths must have equal length")
    if y.min().item() < 0 or y.max().item() >= num_classes:
        raise ValueError("labels are outside the declared class range")
    if cfg.folds < 2:
        raise ValueError("At least two OOF folds are required")
    if not 0.0 <= cfg.correction_confidence <= 1.0:
        raise ValueError("correction_confidence must be in [0,1]")
    if not 0.0 < cfg.correction_confidence_quantile < 1.0:
        raise ValueError("correction_confidence_quantile must be in (0,1)")

    group_array = np.asarray(groups if groups is not None else list(paths))
    splitter = StratifiedGroupKFold(
        n_splits=cfg.folds, shuffle=True, random_state=cfg.seed
    )
    sample_count = y.numel()
    proto_label_probability = torch.zeros(sample_count)
    probe_label_probability = torch.zeros(sample_count)
    proto_top1 = torch.full((sample_count,), -1, dtype=torch.long)
    probe_top1 = torch.full((sample_count,), -1, dtype=torch.long)
    proto_confidence = torch.zeros(sample_count)
    probe_confidence = torch.zeros(sample_count)
    fold_ids = torch.full((sample_count,), -1, dtype=torch.long)

    for fold, (train_index, holdout_index) in enumerate(
        splitter.split(np.zeros(sample_count), y.numpy(), group_array)
    ):
        train_idx = torch.from_numpy(train_index).long()
        holdout_idx = torch.from_numpy(holdout_index).long()
        fold_ids[holdout_idx] = fold

        prototypes = class_prototypes(x[train_idx], y[train_idx], num_classes)
        prototype_logits = (
            x[holdout_idx] @ prototypes.T / cfg.prototype_temperature
        )
        prototype_probabilities = F.softmax(prototype_logits, dim=1)
        row = torch.arange(holdout_idx.numel())
        proto_label_probability[holdout_idx] = prototype_probabilities[
            row, y[holdout_idx]
        ]
        proto_confidence[holdout_idx], proto_top1[holdout_idx] = (
            prototype_probabilities.max(dim=1)
        )

        probe = _fit_probe(
            train_features=x[train_idx],
            train_labels=y[train_idx],
            prototypes=prototypes,
            num_classes=num_classes,
            config=cfg,
            device=torch.device(device),
            fold_seed=cfg.seed + fold * 1009,
        )
        with torch.no_grad():
            logits = _batched_probe_logits(
                probe,
                x[holdout_idx],
                batch_size=cfg.probe_batch_size,
                device=torch.device(device),
            )
            probabilities = F.softmax(logits / cfg.probe_temperature, dim=1)
        probe_label_probability[holdout_idx] = probabilities[row, y[holdout_idx]]
        probe_confidence[holdout_idx], probe_top1[holdout_idx] = probabilities.max(
            dim=1
        )

    if (fold_ids < 0).any():
        raise RuntimeError("OOF splitter did not assign every sample")

    label_support = torch.sqrt(
        proto_label_probability.clamp_min(1.0e-8)
        * probe_label_probability.clamp_min(1.0e-8)
    )
    disagreement = (proto_top1 != probe_top1).float()
    negative_evidence = -torch.log(label_support.clamp_min(1.0e-8)) + 0.35 * disagreement
    clean_probability = classwise_clean_probability(
        negative_evidence, y, num_classes, seed=cfg.seed
    )

    # Recalibrate misleading early confidence: prototype agrees with the noisy
    # label, while the adapted OOF probe confidently rejects it.
    prototype_threshold = max(
        cfg.correction_confidence,
        float(
            torch.quantile(
                proto_confidence, cfg.correction_confidence_quantile
            )
        ),
    )
    probe_threshold = max(
        cfg.correction_confidence,
        float(
            torch.quantile(
                probe_confidence, cfg.correction_confidence_quantile
            )
        ),
    )
    mislabeled_easy = (
        (proto_top1 == y)
        & (probe_top1 != y)
        & (probe_confidence >= probe_threshold)
    )
    clean_probability[mislabeled_easy] *= 1.0 - cfg.early_cut_strength
    clean_probability = clean_probability.clamp(
        cfg.minimum_clean_probability, 1.0
    )

    consensus_other = (
        (proto_top1 == probe_top1)
        & (probe_top1 != y)
        & (proto_confidence >= prototype_threshold)
        & (probe_confidence >= probe_threshold)
    )
    pseudo_label = torch.where(
        consensus_other, probe_top1, torch.full_like(probe_top1, -1)
    )
    # Raw softmax confidence is not comparable across a frozen prototype view
    # and a trainable 500-way probe. Convert each to an empirical percentile
    # before combining so correction strength is calibration-scale invariant.
    proto_calibrated = _percentile_confidence(proto_confidence)
    probe_calibrated = _percentile_confidence(probe_confidence)
    pseudo_confidence = torch.sqrt(
        proto_calibrated.clamp_min(0.0) * probe_calibrated.clamp_min(0.0)
    )
    correction_alpha = (
        (1.0 - clean_probability)
        * pseudo_confidence
        * cfg.maximum_correction_alpha
    )
    correction_alpha = torch.where(
        consensus_other, correction_alpha, torch.zeros_like(correction_alpha)
    )
    correction_alpha = cap_classwise_corrections(
        correction_alpha,
        y,
        maximum_rate=cfg.maximum_class_correction_rate,
    )

    bundle: dict[str, Any] = {
        "paths": list(paths),
        "clean_probability": clean_probability.float(),
        "pseudo_label": pseudo_label.long(),
        "pseudo_confidence": pseudo_confidence.float(),
        "correction_alpha": correction_alpha.float(),
        "metadata": {
            "method": "cross_fitted_visual_trust_v1",
            "folds": cfg.folds,
            "seed": cfg.seed,
            "num_classes": num_classes,
            "sample_count": sample_count,
            "config": cfg.__dict__,
        },
        "diagnostics": {
            "fold_id": fold_ids,
            "prototype_label_probability": proto_label_probability,
            "probe_label_probability": probe_label_probability,
            "prototype_top1": proto_top1,
            "probe_top1": probe_top1,
            "prototype_confidence": proto_confidence,
            "probe_confidence": probe_confidence,
            "mislabeled_easy": mislabeled_easy,
        },
    }
    represented = torch.bincount(y[correction_alpha > 0], minlength=num_classes) > 0
    summary = {
        "method": "cross_fitted_visual_trust_v1",
        "samples": sample_count,
        "folds": cfg.folds,
        "mean_clean_probability": float(clean_probability.mean()),
        "p10_clean_probability": float(torch.quantile(clean_probability, 0.10)),
        "median_clean_probability": float(torch.median(clean_probability)),
        "mislabeled_easy_count": int(mislabeled_easy.sum()),
        "corrected_count": int((correction_alpha > 0).sum()),
        "correction_rate": float((correction_alpha > 0).float().mean()),
        "corrected_classes": int(represented.sum()),
        "prototype_noisy_label_accuracy": float((proto_top1 == y).float().mean()),
        "probe_noisy_label_accuracy": float((probe_top1 == y).float().mean()),
        "view_agreement": float((proto_top1 == probe_top1).float().mean()),
        "prototype_correction_threshold": prototype_threshold,
        "probe_correction_threshold": probe_threshold,
    }
    return bundle, summary


def classwise_clean_probability(
    negative_evidence: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    seed: int,
) -> torch.Tensor:
    """Estimate P(clean) per noisy class with a stable rank fallback."""
    evidence = negative_evidence.detach().cpu().float()
    labels = labels.detach().cpu().long()
    result = torch.zeros_like(evidence)
    global_probability = _rank_probability(evidence)
    for class_index in range(num_classes):
        indices = torch.nonzero(labels == class_index).flatten()
        if indices.numel() < 12:
            result[indices] = global_probability[indices]
            continue
        values = evidence[indices].numpy().reshape(-1, 1)
        try:
            mixture = GaussianMixture(
                n_components=2,
                covariance_type="full",
                reg_covar=1.0e-4,
                n_init=5,
                random_state=seed + class_index,
            )
            mixture.fit(values)
            posterior = mixture.predict_proba(values)
            clean_component = int(np.argmin(mixture.means_.reshape(-1)))
            probability = torch.from_numpy(
                posterior[:, clean_component].astype(np.float32)
            )
            if not torch.isfinite(probability).all():
                raise FloatingPointError("non-finite GMM posterior")
            result[indices] = probability
        except Exception:
            local = _rank_probability(evidence[indices])
            result[indices] = local
    return result.clamp(0.0, 1.0)


def cap_classwise_corrections(
    correction_alpha: torch.Tensor,
    labels: torch.Tensor,
    maximum_rate: float,
) -> torch.Tensor:
    if not 0.0 <= maximum_rate <= 1.0:
        raise ValueError("maximum_rate must be in [0,1]")
    output = correction_alpha.clone()
    for class_index in labels.unique().tolist():
        indices = torch.nonzero(labels == class_index).flatten()
        active = indices[output[indices] > 0]
        allowed = int(math.floor(indices.numel() * maximum_rate))
        if allowed <= 0:
            output[active] = 0.0
        elif active.numel() > allowed:
            order = torch.argsort(output[active], descending=True)
            output[active[order[allowed:]]] = 0.0
    return output


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _rank_probability(evidence: torch.Tensor) -> torch.Tensor:
    count = evidence.numel()
    if count <= 1:
        return torch.ones_like(evidence)
    order = torch.argsort(evidence, descending=False)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(count, dtype=torch.float32)
    return 1.0 - ranks / float(count - 1)


def _percentile_confidence(confidence: torch.Tensor) -> torch.Tensor:
    count = confidence.numel()
    if count <= 1:
        return torch.ones_like(confidence)
    order = torch.argsort(confidence, descending=False)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(count, dtype=torch.float32)
    return ranks / float(count - 1)


def _fit_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    prototypes: torch.Tensor,
    num_classes: int,
    config: TrustBuildConfig,
    device: torch.device,
    fold_seed: int,
) -> torch.nn.Linear:
    dimension = train_features.shape[1]
    probe = torch.nn.Linear(dimension, num_classes).to(device)
    with torch.no_grad():
        probe.weight.copy_(prototypes.to(device) * 8.0)
        probe.bias.zero_()
    optimizer = torch.optim.SGD(
        probe.parameters(),
        lr=config.probe_lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=config.probe_weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(fold_seed)
    count = train_labels.numel()
    for epoch in range(config.probe_epochs):
        permutation = torch.randperm(count, generator=generator)
        cosine = 0.5 * (1.0 + math.cos(math.pi * epoch / max(config.probe_epochs, 1)))
        for group in optimizer.param_groups:
            group["lr"] = config.probe_lr * cosine
        for start in range(0, count, config.probe_batch_size):
            index = permutation[start : start + config.probe_batch_size]
            batch_x = train_features[index].to(device, non_blocking=True)
            batch_y = train_labels[index].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return probe.cpu().eval()


@torch.no_grad()
def _batched_probe_logits(
    probe: torch.nn.Linear,
    features: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    probe = probe.to(device).eval()
    chunks = []
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size].to(device, non_blocking=True)
        chunks.append(probe(batch).cpu())
    return torch.cat(chunks, dim=0)
