"""Strict reconstruction of full cross-fitted linear-head logits."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from aegis_clip.features import canonical_sample_path
from aegis_clip.runtime import (
    atomic_json_dump,
    environment_manifest,
    set_seed,
    sha256_file,
)


REQUIRED_ASSIGNMENT_COLUMNS = {"sample_id", "image_path", "label", "fold"}


@dataclass(frozen=True)
class OOFInputs:
    """Validated assignments and aligned frozen features."""

    assignments: pd.DataFrame
    features: torch.Tensor
    labels: torch.Tensor


def generalized_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    q: float,
    epsilon: float = 1.0e-7,
) -> torch.Tensor:
    """Scalar GCE matching the historical OOF implementation."""
    if not 0.0 < float(q) <= 1.0:
        raise ValueError("q must be in (0,1]")
    probabilities = F.softmax(logits, dim=1)
    selected = probabilities.gather(1, targets.long().unsqueeze(1)).squeeze(1)
    return ((1.0 - selected.clamp_min(epsilon).pow(float(q))) / float(q)).mean()


def learning_rate_factor(
    global_step: int,
    *,
    total_steps: int,
    warmup_steps: int,
) -> float:
    """Warmup plus cosine factor used in the historical three-fold run."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if global_step < 0 or global_step >= total_steps:
        raise ValueError("global_step must be in [0,total_steps)")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if global_step < warmup_steps:
        return (global_step + 1) / max(warmup_steps, 1)
    progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_oof_inputs(
    assignments_path: str | Path,
    feature_tensor_path: str | Path,
    feature_paths_path: str | Path,
    feature_labels_path: str | Path,
) -> OOFInputs:
    """Load and fail-closed align a fixed OOF assignment with a feature cache."""
    assignments = pd.read_csv(
        assignments_path,
        dtype={"sample_id": str, "image_path": str, "label": int, "fold": int},
    )
    missing = REQUIRED_ASSIGNMENT_COLUMNS - set(assignments.columns)
    if missing:
        raise ValueError(f"OOF assignments missing columns: {sorted(missing)}")
    assignments = assignments.sort_values("image_path").reset_index(drop=True)
    if assignments["sample_id"].duplicated().any():
        raise ValueError("OOF assignments contain duplicate sample IDs")
    canonical = assignments["image_path"].map(canonical_sample_path)
    if canonical.duplicated().any():
        raise ValueError("OOF assignments contain duplicate canonical paths")
    folds = sorted(assignments["fold"].unique().astype(int).tolist())
    if folds != list(range(len(folds))) or len(folds) < 2:
        raise ValueError("OOF folds must be contiguous from zero and contain >=2 folds")

    features = torch.load(feature_tensor_path, map_location="cpu", weights_only=True)
    with Path(feature_paths_path).open("r", encoding="utf-8") as handle:
        raw_paths = json.load(handle)
    with Path(feature_labels_path).open("r", encoding="utf-8") as handle:
        raw_labels = json.load(handle)
    if features.ndim != 2:
        raise ValueError("Feature tensor must be rank two")
    if len(raw_paths) != len(features) or len(raw_labels) != len(features):
        raise ValueError("Feature tensor, path index, and labels must align")

    cache_paths = [canonical_sample_path(path) for path in raw_paths]
    if len(cache_paths) != len(set(cache_paths)):
        raise ValueError("Feature cache contains duplicate canonical paths")
    path_to_index = {path: index for index, path in enumerate(cache_paths)}
    indices: list[int] = []
    for row in assignments.itertuples(index=False):
        key = canonical_sample_path(row.image_path)
        if key not in path_to_index:
            raise ValueError(f"OOF sample is missing from feature cache: {key}")
        cache_index = path_to_index[key]
        if int(raw_labels[cache_index]) != int(row.label):
            raise ValueError(f"Feature-cache label mismatch for OOF sample: {key}")
        indices.append(cache_index)

    aligned = F.normalize(features[indices].detach().float(), dim=1)
    labels = torch.tensor(assignments["label"].to_numpy(copy=True), dtype=torch.long)
    num_classes = int(labels.max()) + 1
    for fold in folds:
        train_labels = labels[assignments["fold"].to_numpy() != fold]
        if torch.bincount(train_labels, minlength=num_classes).eq(0).any():
            raise ValueError(f"Fold {fold} training partition misses at least one class")
    return OOFInputs(assignments=assignments, features=aligned, labels=labels)


def train_linear_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    q: float,
    seed: int,
    device: torch.device,
) -> tuple[nn.Linear, list[dict[str, float | int]]]:
    """Train a fixed-epoch head without observing its held-out fold."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    set_seed(seed, deterministic=True)
    features = F.normalize(features.detach().float().cpu(), dim=1)
    labels = labels.detach().long().cpu()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    head = nn.Linear(features.shape[1], num_classes)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(len(loader), 1)
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = warmup_epochs * steps_per_epoch
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        started = time.time()
        for batch_features, batch_labels in loader:
            factor = learning_rate_factor(
                global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )
            optimizer.param_groups[0]["lr"] = float(lr) * factor
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = head(batch_features)
                loss = generalized_cross_entropy(logits, batch_labels, q=q)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            count = len(batch_labels)
            total_loss += float(loss.detach()) * count
            total_correct += int((logits.argmax(dim=1) == batch_labels).sum())
            total_samples += count
            global_step += 1

        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_samples, 1),
            "train_accuracy": total_correct / max(total_samples, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"epoch={epoch:02d}/{epochs} loss={record['train_loss']:.6f} "
            f"acc={record['train_accuracy']:.4f} seconds={record['seconds']:.1f}",
            flush=True,
        )
    return head.cpu(), history


@torch.no_grad()
def infer_logits(
    head: nn.Linear,
    features: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Infer float32 logits in deterministic input order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    head = head.to(device).eval()
    outputs = []
    for start in range(0, len(features), batch_size):
        batch = F.normalize(features[start : start + batch_size].float(), dim=1).to(
            device
        )
        outputs.append(head(batch).float().cpu())
    return torch.cat(outputs, dim=0)


def audit_against_historical_quality(
    assignments: pd.DataFrame,
    logits: torch.Tensor,
    quality_path: str | Path,
) -> dict[str, float | int | bool]:
    """Compare reconstructed probabilities with retained historical OOF scalars."""
    quality = pd.read_csv(quality_path, dtype={"sample_id": str})
    required = {
        "sample_id",
        "original_label",
        "oof_top1",
        "p_original_label",
        "p_top1",
        "top1_margin",
    }
    missing = required - set(quality.columns)
    if missing:
        raise ValueError(f"Historical quality table misses columns: {sorted(missing)}")
    if quality["sample_id"].duplicated().any():
        raise ValueError("Historical quality table contains duplicate sample IDs")
    if set(quality["sample_id"]) != set(assignments["sample_id"]):
        raise ValueError("Historical quality sample IDs do not match assignments")
    quality = quality.set_index("sample_id").loc[assignments["sample_id"]].reset_index()
    labels = torch.tensor(assignments["label"].to_numpy(copy=True), dtype=torch.long)
    if not np.array_equal(
        quality["original_label"].to_numpy(dtype=np.int64), labels.numpy()
    ):
        raise ValueError("Historical quality labels do not match assignments")

    probabilities = F.softmax(logits.float(), dim=1)
    top2_probability, top2_label = probabilities.topk(2, dim=1)
    top1 = top2_label[:, 0]
    p_top1 = top2_probability[:, 0]
    p_original = probabilities.gather(1, labels[:, None]).squeeze(1)
    margin = top2_probability[:, 0] - top2_probability[:, 1]

    def error_summary(current: torch.Tensor, column: str) -> tuple[float, float, float]:
        reference = torch.tensor(quality[column].to_numpy(copy=True), dtype=torch.float32)
        error = (current.cpu() - reference).abs()
        return (
            float(error.mean()),
            float(torch.quantile(error, 0.99)),
            float(error.max()),
        )

    p_original_error = error_summary(p_original, "p_original_label")
    p_top1_error = error_summary(p_top1, "p_top1")
    margin_error = error_summary(margin, "top1_margin")
    historical_top1 = torch.tensor(
        quality["oof_top1"].to_numpy(copy=True), dtype=torch.long
    )
    return {
        "sample_count": len(assignments),
        "sample_id_coverage_exact": True,
        "all_logits_finite": bool(torch.isfinite(logits).all()),
        "top1_agreement": float((top1 == historical_top1).float().mean()),
        "reconstructed_oof_accuracy": float((top1 == labels).float().mean()),
        "historical_oof_accuracy": float((historical_top1 == labels).float().mean()),
        "p_original_mean_absolute_error": p_original_error[0],
        "p_original_p99_absolute_error": p_original_error[1],
        "p_original_max_absolute_error": p_original_error[2],
        "p_top1_mean_absolute_error": p_top1_error[0],
        "p_top1_p99_absolute_error": p_top1_error[1],
        "p_top1_max_absolute_error": p_top1_error[2],
        "top1_margin_mean_absolute_error": margin_error[0],
        "top1_margin_p99_absolute_error": margin_error[1],
        "top1_margin_max_absolute_error": margin_error[2],
    }


def rebuild_oof_logits(
    inputs: OOFInputs,
    output_dir: str | Path,
    *,
    num_classes: int,
    epochs: int,
    batch_size: int,
    infer_batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    q: float,
    seed: int,
    device: torch.device,
    input_hashes: dict[str, str],
    historical_quality_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run all fixed folds and emit a complete, lineage-tracked logits tensor."""
    if num_classes <= int(inputs.labels.max()):
        raise ValueError("num_classes does not cover every observed label")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assignments = inputs.assignments
    folds = sorted(assignments["fold"].unique().astype(int).tolist())
    merged = torch.empty(len(assignments), num_classes, dtype=torch.float16)
    filled = torch.zeros(len(assignments), dtype=torch.bool)
    fold_metrics: list[dict[str, Any]] = []

    for fold in folds:
        print(f"starting fold={fold}", flush=True)
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        holdout_mask = assignments["fold"].to_numpy() == fold
        train_rows = np.flatnonzero(~holdout_mask)
        holdout_rows = np.flatnonzero(holdout_mask)
        head, history = train_linear_head(
            inputs.features[train_rows],
            inputs.labels[train_rows],
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            q=q,
            seed=seed + fold,
            device=device,
        )
        logits = infer_logits(
            head,
            inputs.features[holdout_rows],
            batch_size=infer_batch_size,
            device=device,
        )
        accuracy = float(
            (logits.argmax(dim=1) == inputs.labels[holdout_rows]).float().mean()
        )
        record = {
            "fold": fold,
            "train_count": len(train_rows),
            "holdout_count": len(holdout_rows),
            "fixed_epochs": epochs,
            "holdout_accuracy": accuracy,
            "holdout_used_for_epoch_selection": False,
            "seed": seed + fold,
        }
        fold_metrics.append(record)
        torch.save(
            {
                "state_dict": head.state_dict(),
                "feature_dim": inputs.features.shape[1],
                "num_classes": num_classes,
                "fold": fold,
                "fixed_epochs": epochs,
                "q": q,
            },
            fold_dir / "linear_head.pt",
        )
        torch.save(
            {
                "row_indices": torch.tensor(holdout_rows),
                "sample_ids": assignments.iloc[holdout_rows]["sample_id"].tolist(),
                "logits": logits.half(),
            },
            fold_dir / "oof_logits.pt",
        )
        atomic_json_dump(history, fold_dir / "train_history.json")
        atomic_json_dump(record, fold_dir / "metrics.json")
        merged[holdout_rows] = logits.half()
        filled[holdout_rows] = True
        print(f"completed fold={fold} holdout_accuracy={accuracy:.6f}", flush=True)

    if not bool(filled.all()):
        raise RuntimeError("At least one OOF row was not filled exactly once")
    merged_path = output / "oof_logits.pt"
    torch.save(
        {
            "sample_ids": assignments["sample_id"].tolist(),
            "image_paths": assignments["image_path"].tolist(),
            "labels": inputs.labels,
            "logits": merged,
            "folds": torch.tensor(assignments["fold"].to_numpy(copy=True)),
        },
        merged_path,
    )
    audit: dict[str, Any] = {
        "sample_count": len(assignments),
        "all_samples_filled_once": bool(filled.all()),
        "all_logits_finite": bool(torch.isfinite(merged).all()),
        "holdout_used_for_epoch_selection": False,
        "fold_metrics": fold_metrics,
    }
    if historical_quality_path is not None:
        audit["historical_reproducibility"] = audit_against_historical_quality(
            assignments,
            merged.float(),
            historical_quality_path,
        )
    atomic_json_dump(audit, output / "reproducibility_audit.json")
    manifest = {
        "protocol": "fixed three-fold GCE full-logit reconstruction",
        "external_data": False,
        "test_data_used": False,
        "original_validation_used": False,
        "holdout_used_for_epoch_selection": False,
        "parameters": {
            "num_classes": num_classes,
            "epochs": epochs,
            "batch_size": batch_size,
            "infer_batch_size": infer_batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "q": q,
            "seed": seed,
        },
        "input_hashes": dict(input_hashes),
        "oof_logits_sha256": sha256_file(merged_path),
        "reproducibility_audit_sha256": sha256_file(
            output / "reproducibility_audit.json"
        ),
        "environment": environment_manifest(),
    }
    atomic_json_dump(manifest, output / "artifact_manifest.json")
    return {"audit": audit, "manifest": manifest}
