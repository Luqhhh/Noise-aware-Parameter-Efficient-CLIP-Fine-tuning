"""Cross-fitted training-dynamics reconstruction for noisy-label diagnostics."""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from aegis_clip.oof_rebuild import (
    OOFInputs,
    generalized_cross_entropy,
    infer_logits,
    learning_rate_factor,
)
from aegis_clip.runtime import (
    atomic_json_dump,
    environment_manifest,
    set_seed,
    sha256_file,
)


def first_label_wave_reversal(prediction_changes: torch.Tensor) -> int:
    """Return the epoch before the first failure of the initial decreasing wave.

    Epoch one has no previous prediction and therefore must contain zero changes.
    The returned epoch is one-based. If the wave never reverses, the final epoch
    is returned. This expresses the intended IDO checkpoint rule without keeping
    a live ``state_dict`` view that silently follows later parameter updates.
    """
    values = torch.as_tensor(prediction_changes).long().flatten()
    if len(values) < 2:
        raise ValueError("At least two epochs are required for label-wave selection")
    if int(values[0]) != 0:
        raise ValueError("Epoch-one prediction changes must be zero")
    previous = int(values[1])
    for index in range(2, len(values)):
        current = int(values[index])
        if current >= previous:
            return index
        previous = current
    return len(values)


class CrossFittedTrajectory:
    """Accumulate exactly one held-out prediction per sample and epoch."""

    def __init__(
        self,
        *,
        num_samples: int,
        epochs: int,
        num_classes: int,
        top_k: int,
    ) -> None:
        if num_samples < 1 or epochs < 2 or num_classes < 2:
            raise ValueError("Invalid trajectory dimensions")
        if top_k < 1 or top_k > num_classes:
            raise ValueError("top_k must be in [1,num_classes]")
        if max(epochs, num_classes) >= torch.iinfo(torch.int16).max:
            raise ValueError("epochs and num_classes must fit in int16")
        self.num_samples = int(num_samples)
        self.epochs = int(epochs)
        self.num_classes = int(num_classes)
        self.top_k = int(top_k)
        self.seen = torch.zeros(num_samples, epochs, dtype=torch.bool)
        self.wrong_event_count = torch.zeros(num_samples, dtype=torch.int16)
        self.prediction_change_count = torch.zeros(num_samples, dtype=torch.int16)
        self.previous_prediction = torch.full(
            (num_samples,), -1, dtype=torch.int16
        )
        self.original_label_probability = torch.empty(
            num_samples, epochs, dtype=torch.float16
        )
        self.topk_indices = torch.empty(
            num_samples, epochs, top_k, dtype=torch.int16
        )
        self.topk_probabilities = torch.empty(
            num_samples, epochs, top_k, dtype=torch.float16
        )
        self.final_logits = torch.empty(
            num_samples, num_classes, dtype=torch.float16
        )
        self.epoch_correct = torch.zeros(epochs, dtype=torch.int64)
        self.epoch_prediction_change = torch.zeros(epochs, dtype=torch.int64)
        self.epoch_class_correct = torch.zeros(
            epochs, num_classes, dtype=torch.int64
        )
        self.epoch_class_count = torch.zeros(
            epochs, num_classes, dtype=torch.int64
        )

    @torch.no_grad()
    def update(
        self,
        *,
        epoch_index: int,
        row_indices: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, int | float]:
        """Record a disjoint held-out batch for one epoch."""
        if epoch_index < 0 or epoch_index >= self.epochs:
            raise ValueError("epoch_index is out of range")
        rows = torch.as_tensor(row_indices).long().cpu().flatten()
        targets = torch.as_tensor(labels).long().cpu().flatten()
        values = torch.as_tensor(logits).float().cpu()
        if values.ndim != 2 or values.shape != (len(rows), self.num_classes):
            raise ValueError("logits shape does not match rows and num_classes")
        if len(targets) != len(rows):
            raise ValueError("labels must align with row_indices")
        if len(rows) != len(torch.unique(rows)):
            raise ValueError("row_indices contain duplicates")
        if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= self.num_samples):
            raise ValueError("row_indices are out of range")
        if len(targets) and (
            int(targets.min()) < 0 or int(targets.max()) >= self.num_classes
        ):
            raise ValueError("labels are out of range")
        if bool(self.seen[rows, epoch_index].any()):
            raise ValueError("A held-out row was recorded twice for one epoch")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Trajectory logits contain non-finite values")

        probabilities = F.softmax(values, dim=1)
        top_probability, top_index = probabilities.topk(self.top_k, dim=1)
        prediction = top_index[:, 0]
        previous = self.previous_prediction[rows].long()
        changed = previous.ge(0) & prediction.ne(previous)
        wrong = prediction.ne(targets)
        correct = ~wrong

        self.seen[rows, epoch_index] = True
        self.wrong_event_count[rows] += wrong.to(torch.int16)
        self.prediction_change_count[rows] += changed.to(torch.int16)
        self.previous_prediction[rows] = prediction.to(torch.int16)
        self.original_label_probability[rows, epoch_index] = probabilities[
            torch.arange(len(rows)), targets
        ].half()
        self.topk_indices[rows, epoch_index] = top_index.to(torch.int16)
        self.topk_probabilities[rows, epoch_index] = top_probability.half()
        if epoch_index == self.epochs - 1:
            self.final_logits[rows] = values.half()

        self.epoch_correct[epoch_index] += int(correct.sum())
        self.epoch_prediction_change[epoch_index] += int(changed.sum())
        ones = torch.ones(len(rows), dtype=torch.int64)
        self.epoch_class_count[epoch_index].scatter_add_(0, targets, ones)
        self.epoch_class_correct[epoch_index].scatter_add_(
            0, targets, correct.to(torch.int64)
        )
        return {
            "count": len(rows),
            "accuracy": float(correct.float().mean()) if len(rows) else math.nan,
            "wrong_events": int(wrong.sum()),
            "prediction_changes": int(changed.sum()),
        }

    def finalize(self) -> dict[str, torch.Tensor | int]:
        """Validate complete coverage and return a serializable payload."""
        if not bool(self.seen.all()):
            missing = int((~self.seen).sum())
            raise RuntimeError(f"Trajectory is incomplete: {missing} cells are missing")
        if bool(self.topk_indices.lt(0).any()):
            raise RuntimeError("Trajectory contains unset top-k indices")
        if not bool(torch.isfinite(self.original_label_probability).all()):
            raise RuntimeError("Trajectory contains non-finite original probabilities")
        if not bool(torch.isfinite(self.topk_probabilities).all()):
            raise RuntimeError("Trajectory contains non-finite top-k probabilities")
        if not bool(torch.isfinite(self.final_logits).all()):
            raise RuntimeError("Trajectory contains non-finite final logits")
        class_retention = (
            self.epoch_class_correct.float()
            / self.epoch_class_count.clamp_min(1).float()
        )
        epoch_accuracy = self.epoch_correct.float() / float(self.num_samples)
        return {
            "wrong_event_count": self.wrong_event_count,
            "prediction_change_count": self.prediction_change_count,
            "original_label_probability": self.original_label_probability,
            "topk_indices": self.topk_indices,
            "topk_probabilities": self.topk_probabilities,
            "final_logits": self.final_logits,
            "epoch_oof_accuracy": epoch_accuracy,
            "epoch_prediction_change": self.epoch_prediction_change,
            "epoch_class_retention": class_retention.half(),
            "selected_base_epoch": first_label_wave_reversal(
                self.epoch_prediction_change
            ),
        }


def audit_final_against_reference(
    *,
    sample_ids: list[str],
    labels: torch.Tensor,
    folds: torch.Tensor,
    final_logits: torch.Tensor,
    reference_path: str | Path,
    batch_size: int = 4096,
) -> dict[str, float | int | bool]:
    """Fail-closed comparison to the already audited I0 final OOF artifact."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not sample_ids:
        raise ValueError("At least one OOF sample is required")
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    required = {"sample_ids", "labels", "folds", "logits"}
    missing = required - set(reference)
    if missing:
        raise ValueError(f"Reference OOF artifact misses keys: {sorted(missing)}")
    if reference["sample_ids"] != sample_ids:
        raise ValueError("Reference OOF sample IDs do not align")
    if not torch.equal(torch.as_tensor(reference["labels"]).long(), labels.long()):
        raise ValueError("Reference OOF labels do not align")
    if not torch.equal(torch.as_tensor(reference["folds"]).long(), folds.long()):
        raise ValueError("Reference OOF folds do not align")
    current = torch.as_tensor(final_logits).cpu()
    expected = torch.as_tensor(reference["logits"]).cpu()
    if current.shape != expected.shape:
        raise ValueError("Reference OOF logits shape does not align")
    if current.ndim != 2 or current.shape[0] != len(labels):
        raise ValueError("Current OOF logits do not align with labels")

    probability_error = torch.empty(len(labels), dtype=torch.float32)
    top1_matches = 0
    logit_max_absolute_error = 0.0
    targets = labels.long().cpu()
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        current_batch = current[start:stop].float()
        expected_batch = expected[start:stop].float()
        current_probability = F.softmax(current_batch, dim=1)
        expected_probability = F.softmax(expected_batch, dim=1)
        batch_targets = targets[start:stop]
        rows = torch.arange(stop - start)
        probability_error[start:stop] = (
            current_probability[rows, batch_targets]
            - expected_probability[rows, batch_targets]
        ).abs()
        current_top1 = current_probability.topk(1, dim=1).indices[:, 0]
        expected_top1 = expected_probability.topk(1, dim=1).indices[:, 0]
        top1_matches += int(current_top1.eq(expected_top1).sum())
        logit_max_absolute_error = max(
            logit_max_absolute_error,
            float((current_batch - expected_batch).abs().max()),
        )
    return {
        "sample_count": len(sample_ids),
        "all_logits_finite": bool(torch.isfinite(current).all()),
        "reference_all_logits_finite": bool(torch.isfinite(expected).all()),
        "top1_agreement": top1_matches / len(sample_ids),
        "p_original_mean_absolute_error": float(probability_error.mean()),
        "p_original_p99_absolute_error": float(
            torch.quantile(probability_error, 0.99)
        ),
        "p_original_max_absolute_error": float(probability_error.max()),
        "logit_max_absolute_error": logit_max_absolute_error,
    }


def trajectory_reference_gate(audit: dict[str, Any]) -> dict[str, Any]:
    """Apply preregistered final-epoch reproduction thresholds."""
    checks = {
        "all_logits_finite": bool(audit["all_logits_finite"]),
        "reference_all_logits_finite": bool(audit["reference_all_logits_finite"]),
        "top1_agreement_at_least_0_9999": float(audit["top1_agreement"]) >= 0.9999,
        "p_original_mae_at_most_0_002": float(
            audit["p_original_mean_absolute_error"]
        )
        <= 0.002,
        "p_original_p99_at_most_0_01": float(
            audit["p_original_p99_absolute_error"]
        )
        <= 0.01,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def train_fold_with_trajectory(
    *,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    holdout_features: torch.Tensor,
    holdout_labels: torch.Tensor,
    holdout_rows: torch.Tensor,
    accumulator: CrossFittedTrajectory,
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
) -> tuple[
    nn.Linear,
    list[dict[str, float | int]],
    dict[str, torch.Tensor],
]:
    """Reproduce one fixed I0 fold while recording held-out dynamics."""
    set_seed(seed, deterministic=True)
    features = F.normalize(train_features.detach().float().cpu(), dim=1)
    targets = train_labels.detach().long().cpu()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(features, targets),
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
    epoch_weights: list[torch.Tensor] = []
    epoch_biases: list[torch.Tensor] = []
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
            total_correct += int(logits.argmax(dim=1).eq(batch_labels).sum())
            total_samples += count
            global_step += 1

        holdout_logits = infer_logits(
            head,
            holdout_features,
            batch_size=infer_batch_size,
            device=device,
        )
        trajectory_record = accumulator.update(
            epoch_index=epoch - 1,
            row_indices=holdout_rows,
            logits=holdout_logits,
            labels=holdout_labels,
        )
        epoch_weights.append(head.weight.detach().float().cpu().clone())
        epoch_biases.append(head.bias.detach().float().cpu().clone())
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_samples, 1),
            "train_accuracy": total_correct / max(total_samples, 1),
            "holdout_accuracy": float(trajectory_record["accuracy"]),
            "holdout_wrong_events": int(trajectory_record["wrong_events"]),
            "holdout_prediction_changes": int(
                trajectory_record["prediction_changes"]
            ),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"epoch={epoch:02d}/{epochs} loss={record['train_loss']:.6f} "
            f"train_acc={record['train_accuracy']:.4f} "
            f"holdout_acc={record['holdout_accuracy']:.4f} "
            f"changes={record['holdout_prediction_changes']} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
    snapshots = {
        "weight": torch.stack(epoch_weights),
        "bias": torch.stack(epoch_biases),
        "epoch": torch.arange(1, epochs + 1, dtype=torch.int16),
    }
    return head.cpu(), history, snapshots


def rebuild_oof_trajectory(
    inputs: OOFInputs,
    output_dir: str | Path,
    *,
    reference_oof_path: str | Path,
    num_classes: int,
    epochs: int,
    batch_size: int,
    infer_batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    q: float,
    seed: int,
    top_k: int,
    device: torch.device,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    """Rebuild cross-fitted trajectories without using validation or test data."""
    assignments = inputs.assignments
    if not (
        len(assignments) == len(inputs.features) == len(inputs.labels)
        and len(assignments) > 0
    ):
        raise ValueError("Assignments, features, and labels must align and be non-empty")
    if inputs.features.ndim != 2 or not bool(torch.isfinite(inputs.features).all()):
        raise ValueError("Features must be a finite rank-two tensor")
    if inputs.labels.ndim != 1 or int(inputs.labels.min()) < 0:
        raise ValueError("Labels must be a non-negative rank-one tensor")
    if num_classes <= int(inputs.labels.max()):
        raise ValueError("num_classes does not cover every observed label")
    if epochs < 2 or batch_size < 1 or infer_batch_size < 1:
        raise ValueError("epochs must be >=2 and batch sizes must be positive")
    if lr <= 0.0 or weight_decay < 0.0 or warmup_epochs < 0:
        raise ValueError("Invalid optimizer or warmup settings")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fold_values = assignments["fold"].to_numpy(copy=True)
    folds = sorted(np.unique(fold_values).astype(int).tolist())
    if folds != list(range(len(folds))) or len(folds) < 2:
        raise ValueError("OOF folds must be contiguous from zero and contain >=2 folds")
    for fold in folds:
        fold_train_labels = inputs.labels[torch.from_numpy(fold_values != fold)]
        if torch.bincount(fold_train_labels, minlength=num_classes).eq(0).any():
            raise ValueError(f"Fold {fold} training partition misses at least one class")
    accumulator = CrossFittedTrajectory(
        num_samples=len(assignments),
        epochs=epochs,
        num_classes=num_classes,
        top_k=top_k,
    )
    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        print(f"starting trajectory fold={fold}", flush=True)
        holdout_mask = fold_values == fold
        train_rows = torch.from_numpy(np.flatnonzero(~holdout_mask)).long()
        holdout_rows = torch.from_numpy(np.flatnonzero(holdout_mask)).long()
        head, history, epoch_heads = train_fold_with_trajectory(
            train_features=inputs.features[train_rows],
            train_labels=inputs.labels[train_rows],
            holdout_features=inputs.features[holdout_rows],
            holdout_labels=inputs.labels[holdout_rows],
            holdout_rows=holdout_rows,
            accumulator=accumulator,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            infer_batch_size=infer_batch_size,
            lr=lr,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            q=q,
            seed=seed + fold,
            device=device,
        )
        fold_dir = output / f"fold_{fold}"
        _atomic_torch_save(
            {
                "state_dict": {
                    key: value.clone() for key, value in head.state_dict().items()
                },
                "feature_dim": inputs.features.shape[1],
                "num_classes": num_classes,
                "fold": fold,
                "fixed_epochs": epochs,
                "q": q,
            },
            fold_dir / "linear_head.pt",
        )
        epoch_heads_path = fold_dir / "epoch_heads.pt"
        _atomic_torch_save(
            {
                **epoch_heads,
                "feature_dim": inputs.features.shape[1],
                "num_classes": num_classes,
                "fold": fold,
                "q": q,
            },
            epoch_heads_path,
        )
        atomic_json_dump(history, fold_dir / "train_history.json")
        fold_records.append(
            {
                "fold": fold,
                "train_count": len(train_rows),
                "holdout_count": len(holdout_rows),
                "seed": seed + fold,
                "fixed_epochs": epochs,
                "holdout_used_for_epoch_selection": False,
                "epoch_heads_sha256": sha256_file(epoch_heads_path),
            }
        )

    trajectory = accumulator.finalize()
    selected_base_epoch = int(trajectory["selected_base_epoch"])
    for record in fold_records:
        fold_dir = output / f"fold_{record['fold']}"
        epoch_heads = torch.load(
            fold_dir / "epoch_heads.pt",
            map_location="cpu",
            weights_only=True,
        )
        selected_head_path = fold_dir / "selected_base_head.pt"
        index = selected_base_epoch - 1
        _atomic_torch_save(
            {
                "state_dict": {
                    "weight": epoch_heads["weight"][index].clone(),
                    "bias": epoch_heads["bias"][index].clone(),
                },
                "feature_dim": int(epoch_heads["feature_dim"]),
                "num_classes": int(epoch_heads["num_classes"]),
                "fold": int(epoch_heads["fold"]),
                "selected_base_epoch": selected_base_epoch,
                "selection_signal": "global OOF prediction-change first reversal",
                "selection_uses_label_correctness": False,
                "q": float(epoch_heads["q"]),
            },
            selected_head_path,
        )
        record["selected_base_epoch"] = selected_base_epoch
        record["selected_base_head_sha256"] = sha256_file(selected_head_path)
    sample_ids = assignments["sample_id"].tolist()
    fold_tensor = torch.tensor(fold_values, dtype=torch.long)
    reference_audit = audit_final_against_reference(
        sample_ids=sample_ids,
        labels=inputs.labels,
        folds=fold_tensor,
        final_logits=torch.as_tensor(trajectory["final_logits"]),
        reference_path=reference_oof_path,
    )
    reference_gate = trajectory_reference_gate(reference_audit)
    payload = {
        "sample_ids": sample_ids,
        "image_paths": assignments["image_path"].tolist(),
        "labels": inputs.labels,
        "folds": fold_tensor,
        **trajectory,
    }
    trajectory_path = output / "trajectory.pt"
    _atomic_torch_save(payload, trajectory_path)
    audit = {
        "sample_count": len(assignments),
        "epochs": epochs,
        "top_k": top_k,
        "all_samples_seen_once_per_epoch": bool(accumulator.seen.all()),
        "holdout_used_for_epoch_selection": False,
        "selected_base_epoch_uses_labels": False,
        "selected_base_epoch": selected_base_epoch,
        "fold_records": fold_records,
        "reference_audit": reference_audit,
        "reference_gate": reference_gate,
    }
    audit_path = output / "audit.json"
    atomic_json_dump(audit, audit_path)
    manifest = {
        "protocol": "Q1 cross-fitted wrong-event and top-k trajectory reconstruction",
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
            "top_k": top_k,
        },
        "execution": {
            "device_type": device.type,
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "amp_enabled": device.type == "cuda",
        },
        "input_hashes": dict(input_hashes),
        "trajectory_sha256": sha256_file(trajectory_path),
        "audit_sha256": sha256_file(audit_path),
        "environment": environment_manifest(),
        "gate_passed": bool(reference_gate["passed"]),
    }
    atomic_json_dump(manifest, output / "artifact_manifest.json")
    return {"audit": audit, "manifest": manifest}
