"""
Training script for the CLIP Linear Classifier baseline.

Supports three modes:
  dev       - Train/val split, track best epoch, save best.pt and eval_results.json
  confirm   - Train for --frozen-epochs on full split, evaluate on val split
  final_fit - Train for --frozen-epochs on the FULL dataset (no val split)

Additional features:
  - Feature caching via --use-cached-features (E0/E1 experiments)
  - Augmentation presets (a0/a1/a2/a3) via --augmentation-preset
  - Guard enforcement for invalid combinations (B0+cached, etc.)
  - Rich checkpoint metadata for downstream reproducibility

Usage:
    python -m experiments.baseline.train --config configs/baseline.yaml
    python -m experiments.baseline.train --config configs/baseline.yaml \\
        --mode confirm --experiment-id E0 --use-cached-features \\
        --source-dev-best-epoch 5 --frozen-epochs 10
"""

import argparse
import json
import logging
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.cache import CachedFeatureDataset
from common.class_mapping import load_or_generate_mapping
from common.dataset import TrainImageDataset, seed_worker
from common.losses import build_loss, reduce_loss
from common.mixup import mixup_batch
from common.peft import apply_peft, build_peft_param_groups
from common.resolved_config import resolve_config, write_resolved_config
from common.runtime_config import resolve_runtime_args
from common.transforms import build_train_transform, VALID_PRESETS
from common.artifact_manifest import build_artifact_manifest, write_artifact_manifest
from common.utils import (count_parameters, ensure_dir, format_time,
                          load_config, save_config_snapshot, set_train_seed,
                          setup_logging)

logger = logging.getLogger(__name__)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CLIP Linear Classifier Baseline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., outputs/checkpoints/last.pt).",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for model weight initialization ONLY. "
             "Does NOT restore optimizer, scheduler, or epoch state. "
             "Use this (not --resume) for F0-F3 partial unfreeze experiments.",
    )
    # Mode and experiment identity
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment identifier (e.g., B0, E0) for guard enforcement.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["dev", "confirm", "final_fit"],
        help="Training mode: dev (train+val, best epoch), "
             "confirm (frozen epoch count, eval), "
             "final_fit (frozen epoch count, full dataset, no val).",
    )
    # Feature caching
    parser.add_argument(
        "--use-cached-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use pre-computed CLIP features from cache (E0/E1 experiments).",
    )
    # Epoch freezing for confirm/final_fit
    parser.add_argument(
        "--frozen-epochs",
        type=int,
        default=None,
        help="Number of frozen training epochs (for confirm/final_fit modes).",
    )
    # Augmentation preset
    parser.add_argument(
        "--augmentation-preset",
        type=str,
        default=None,
        choices=sorted(VALID_PRESETS),
        help="Augmentation preset: a0 (none/deterministic), a1, a2, a3.",
    )
    # Source dev best epoch (carried forward from dev stage)
    parser.add_argument(
        "--source-dev-best-epoch",
        type=int,
        default=None,
        help="Best epoch from the dev stage, carried forward to confirm/final_fit.",
    )
    # Head type
    parser.add_argument(
        "--head-type",
        type=str,
        default=None,
        choices=["linear", "cosine"],
        help="Classifier head type: linear (default) or cosine. "
             "Overrides experiment.head_type in config if provided.",
    )
    # Cosine head options (overrides config model section)
    parser.add_argument(
        "--cos-init-scale",
        type=float,
        default=None,
        help="Initial logit scale for cosine head (overrides model.cos_init_scale).",
    )
    parser.add_argument(
        "--cos-learnable-scale",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Whether logit scale is learnable for cosine head "
             "(overrides model.cos_learnable_scale).",
    )
    # Multi-seed override
    parser.add_argument(
        "--seed-override",
        type=int,
        default=None,
        help="Override data.seed, data.split_seed, data.train_seed "
             "and adjust output paths to seed{N}/ subdirectories. "
             "Used for multi-seed paired delta experiments.",
    )
    # Fresh-run artifact guard
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help=(
            "Allow removal of generated artifacts in the configured output "
            "directories before a fresh non-resume run. Historical artifacts "
            "must be archived by the caller before using this flag."
        ),
    )
    return parser.parse_args()


def _enforce_guards(
    experiment_id: Optional[str],
    use_cached_features: bool,
    augmentation_preset: str,
    freeze_clip: bool,
) -> None:
    """Hard enforcement of feature caching rules.

    Raises:
        ValueError: If any guard rule is violated.
    """
    if experiment_id == "B0" and use_cached_features:
        raise ValueError(
            "B0 regression must use the original online encoding path. "
            "Remove --use-cached-features for B0."
        )
    if use_cached_features and augmentation_preset != "a0":
        raise ValueError(
            "Cached features only valid for deterministic A0 preprocessing. "
            "Use --augmentation-preset a0 with --use-cached-features."
        )
    if use_cached_features and not freeze_clip:
        raise ValueError(
            "Cached features require freeze_clip=True. "
            "Set model.freeze_clip=true in config."
        )


def _build_checkpoint_metadata(
    model: nn.Module,
    config: Dict[str, Any],
    mode: str,
    args: argparse.Namespace,
    best_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the unified checkpoint metadata dictionary.

    Args:
        model: The trained model (with .head_type, .class_to_idx_, .idx_to_class_).
        config: Full configuration dictionary.
        mode: Training mode (dev/confirm/final_fit).
        args: Parsed CLI arguments.
        best_epoch: Best validation epoch (dev mode only).

    Returns:
        Dictionary of checkpoint metadata fields.
    """
    meta: Dict[str, Any] = {
        "class_to_idx": getattr(model, "class_to_idx_", None),
        "idx_to_class": getattr(model, "idx_to_class_", None),
        "head_type": getattr(model, "head_type", "linear"),
        "augmentation_preset": args.augmentation_preset,
        "training_mode": mode,
        "split_seed": config["data"].get("split_seed", None),
    }

    if args.init_checkpoint:
        meta["init_checkpoint"] = args.init_checkpoint

    if mode == "dev":
        meta.update({
            "dev_best_epoch": best_epoch,
            "frozen_train_epochs": best_epoch,
            "trained_epochs": config["train"]["epochs"],
            "epoch_selection_policy": "dev_best_epoch_frozen_before_confirm",
            "epoch_selection_split": config["data"].get("split_seed", 42),
        })
    elif mode in ("confirm", "final_fit"):
        meta.update({
            "source_dev_best_epoch": args.source_dev_best_epoch,
            "frozen_train_epochs": args.frozen_epochs,
            "trained_epochs": args.frozen_epochs,
            "epoch_selection_policy": "dev_best_epoch_frozen_before_confirm",
            "epoch_selection_split": config["data"].get("split_seed", 42),
        })

    return meta


def _check_splits_exist(split_dir: str) -> bool:
    """Check if train/val split files exist."""
    split_dir = Path(split_dir)
    required = ["train.csv", "val.csv", "class_to_idx.json", "idx_to_class.json"]
    return all((split_dir / f).exists() for f in required)


def _prepare_fresh_run_artifacts(
    save_dir: Path,
    log_dir: Path,
    resume_path: Optional[str],
    allow_overwrite: bool,
) -> None:
    """Guard fresh (non-resume) runs against clobbering existing artifacts.

    Raises FileExistsError when generated files already exist and
    --allow-overwrite was not passed.  With --allow-overwrite the stale
    files are removed so the fresh run starts from a clean slate.
    """
    generated_files = [
        save_dir / "best.pt",
        save_dir / "best_raw.pt",
        save_dir / "best_ema.pt",
        save_dir / "last.pt",
        save_dir / "eval_results.json",
        save_dir / "config_snapshot.yaml",
        log_dir / "train_log.csv",
    ]

    existing = [p for p in generated_files if p.exists()]

    if resume_path:
        if not Path(resume_path).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        return

    if existing and not allow_overwrite:
        formatted = "\n".join(f"  - {p}" for p in existing)
        raise FileExistsError(
            "Fresh run refused because generated artifacts already exist:\n"
            f"{formatted}\n"
            "Archive the old run or pass --allow-overwrite explicitly."
        )

    if allow_overwrite:
        for path in existing:
            path.unlink()


def _cosine_factor(
    step: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    """Cosine decay factor preserving LR ratios across parameter groups.

    Returns a factor in [min_lr_ratio, 1.0] following cosine annealing.
    All parameter groups are scaled by the same factor at each step,
    ensuring backbone_lr / head_lr stays constant throughout training.

    Args:
        step: Current step (0-indexed, after warmup).
        total_steps: Total steps for cosine annealing.
        min_lr_ratio: Minimum LR as fraction of initial LR.

    Returns:
        Float in [min_lr_ratio, 1.0].
    """
    progress = min(step / max(total_steps, 1), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _build_optimizer_and_scheduler(
    model: nn.Module, config: Dict[str, Any], cosine_steps: int,
    peft_cfg: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Build optimizer and learning rate scheduler.

    Supports four modes:
      - PEFT-driven: uses build_peft_param_groups for non-default PEFT types.
      - Linear head, frozen CLIP: uniform LR from train.lr.
      - Linear head, partial unfreeze: discriminative LRs via get_param_groups.
      - Cosine head: uses its own get_param_groups (with scale handling).

    Scheduler uses proportional LambdaLR to preserve backbone_lr/head_lr ratio
    across all parameter groups.
    """
    train_cfg = config["train"]
    model_cfg = config.get("model", {})

    peft_type = peft_cfg.get("type", "linear_head_only") if peft_cfg else "linear_head_only"

    if peft_type != "linear_head_only":
        # PEFT-driven param groups
        param_groups = build_peft_param_groups(
            model, peft_cfg,
            head_lr=train_cfg["lr"],
            head_weight_decay=train_cfg["weight_decay"],
        )
        optimizer = torch.optim.AdamW(param_groups)
    elif hasattr(model, "get_param_groups"):
        optimizer = torch.optim.AdamW(
            model.get_param_groups(train_cfg["lr"], train_cfg["weight_decay"]),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )

    # Save per-group initial_lr before any warmup/scheduler modification
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    # LambdaLR with proportional cosine decay preserves LR ratios
    min_lr_ratio = train_cfg.get("min_lr_ratio", 0.01)
    lr_lambda = lambda step: _cosine_factor(
        step=step,
        total_steps=cosine_steps,
        min_lr_ratio=min_lr_ratio,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lr_lambda] * len(optimizer.param_groups),
    )

    return optimizer, scheduler


def _warmup_lr(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    current_step: int,
) -> None:
    """Apply linear warmup using per-group initial_lr."""
    if current_step >= warmup_steps:
        return
    scale = (current_step + 1) / max(warmup_steps, 1)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * scale


def _unpack_batch(
    batch_data,
    device: torch.device,
):
    """Return inputs, labels, is_cached, paths.

    paths is None for cached-feature batches (2-tuple) and a tuple of
    image-path strings for online-image batches (3-tuple).
    """
    if len(batch_data) == 3:
        images, labels, paths = batch_data
        inputs = images.to(device, non_blocking=True)
        is_cached = False
    elif len(batch_data) == 2:
        features, labels = batch_data
        inputs = features.to(device, non_blocking=True)
        is_cached = True
        paths = None
    else:
        raise ValueError(
            f"Unexpected batch tuple length: {len(batch_data)}"
        )

    labels = labels.to(device, non_blocking=True)
    return inputs, labels, is_cached, paths


def _apply_sample_weights(
    loss_per_sample: torch.Tensor,
    paths: tuple,
    sample_weights,
    normalize_by_weight_sum: bool,
    missing_policy: str,
    device: torch.device,
    epoch: int = 0,
    labels: torch.Tensor = None,
) -> torch.Tensor:
    """Apply per-sample weights to a loss vector.

    ``sample_weights`` may be:
      - ``None`` → unweighted mean
      - ``dict`` (legacy) → lookup by image path
      - ``BaseWeightProvider`` (new) → ``get_weights()`` per batch

    When *paths* is None, returns the unweighted mean.
    """
    if sample_weights is None or paths is None:
        return loss_per_sample.mean()

    # New provider path
    if hasattr(sample_weights, "get_weights"):
        w = sample_weights.get_weights(
            list(paths), labels, epoch, loss_per_sample
        )
        w = w.to(device)
        if normalize_by_weight_sum:
            return (loss_per_sample * w).sum() / (w.sum() + 1e-8)
        return (loss_per_sample * w).mean()

    # Legacy dict path
    w_vals = []
    for p in paths:
        entry = sample_weights.get(p)
        if entry is None:
            if missing_policy == "error":
                raise KeyError(
                    f"Sample weight missing for image: {p}"
                )
            w_vals.append(1.0)
        else:
            w_vals.append(float(entry["weight"]))

    w = torch.tensor(w_vals, device=device, dtype=loss_per_sample.dtype)

    if normalize_by_weight_sum:
        return (loss_per_sample * w).sum() / (w.sum() + 1e-8)
    return (loss_per_sample * w).mean()


def _forward_inputs(
    model: nn.Module,
    inputs: torch.Tensor,
    is_cached: bool,
) -> torch.Tensor:
    if is_cached:
        if not hasattr(model, "forward_features"):
            raise TypeError(
                f"{type(model).__name__} does not implement "
                "forward_features() required by cached training."
            )
        return model.forward_features(inputs)

    return model(inputs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    warmup_steps: int,
    global_step: int,
    sample_weights=None,
    weight_provider=None,
    ema_hook=None,
    teacher_hook=None,
    mixup_cfg: Optional[Dict[str, Any]] = None,
    elr_hook=None,
) -> tuple:
    """Train for one epoch.

    Args:
        sample_weights: Optional dict[image_path -> {"weight": float}].

    Returns:
        Tuple of (avg_loss, accuracy, global_step, head_grad_norm,
                  backbone_grad_norm).
    """
    model.train()
    train_cfg = config["train"]
    use_amp = train_cfg.get("amp", False)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    normalize_by_weight_sum = (
        config.get("sample_weighting", {}).get("normalize_by_weight_sum", True)
    )
    missing_policy = (
        config.get("sample_weighting", {}).get("missing_weight_policy", "error")
    )

    total_loss = 0.0
    correct = 0
    total = 0
    head_grad_sum = 0.0
    backbone_grad_sum = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [Train]", dynamic_ncols=True)

    for batch_idx, batch_data in enumerate(pbar):
        inputs, labels, is_cached, paths = _unpack_batch(batch_data, device)

        # ── MixUp data augmentation ──
        mixup_applied = False
        labels_a = labels_b = labels
        if mixup_cfg is not None and mixup_cfg.get("enabled", False):
            mixup_alpha = mixup_cfg.get("alpha", 0.2)
            mixup_prob = mixup_cfg.get("probability", 0.2)
            if not is_cached:  # MixUp only for online image batches
                inputs, labels_a, labels_b, lam = mixup_batch(
                    inputs, labels, alpha=mixup_alpha, probability=mixup_prob,
                )
                mixup_applied = lam < 1.0

        # Warmup
        if global_step < warmup_steps:
            _warmup_lr(optimizer, warmup_steps, global_step)

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type=device.type, enabled=use_amp):
                logits = _forward_inputs(model, inputs, is_cached)

                if mixup_applied:
                    loss_per_sample_a = criterion(logits, labels_a)
                    loss_per_sample_b = criterion(logits, labels_b)
                    loss_per_sample = (
                        lam * loss_per_sample_a + (1.0 - lam) * loss_per_sample_b
                    )
                else:
                    loss_per_sample = criterion(logits, labels)

            if mixup_applied:
                loss = reduce_loss(loss_per_sample)
            else:
                loss = _apply_sample_weights(
                    loss_per_sample, paths, weight_provider or sample_weights,
                    normalize_by_weight_sum, missing_policy, device,
                    epoch=epoch, labels=labels,
                )

                # ── Teacher–Student consistency loss (A-INFRA-7) ──
                if teacher_hook is not None:
                    teacher_cfg = config.get("teacher", {})
                    conf_thresh = teacher_cfg.get("confidence_threshold", 0.8)
                    cons_weight = teacher_cfg.get("consistency_weight", 1.0)
                    ramp_w = teacher_hook.rampup_weight(epoch)
                    flip_consistency = teacher_cfg.get("flip_consistency", False)
                    with torch.no_grad():
                        teacher_logits = teacher_hook.forward(inputs)
                        conf_mask = teacher_hook.confidence_mask(
                            teacher_logits, threshold=conf_thresh
                        )
                    if conf_mask.any():
                        if flip_consistency:
                            inputs_flip = torch.flip(inputs, dims=[3])
                            with autocast(device_type=device.type, enabled=use_amp):
                                student_flip_logits = model(inputs_flip)
                            cons_loss = torch.nn.functional.mse_loss(
                                student_flip_logits[conf_mask], teacher_logits[conf_mask],
                            )
                        else:
                            cons_loss = torch.nn.functional.mse_loss(
                                logits[conf_mask], teacher_logits[conf_mask],
                            )
                        loss = loss + ramp_w * cons_weight * cons_loss

                # ── ELR temporal consistency ──
                # §5.4: MixUp batches use mixed inputs — targets don't
                # correspond to single real samples, so skip ELR update.
                if elr_hook is not None and not mixup_applied:
                    elr_hook.update(paths, logits.detach())
                    elr_w = elr_hook.rampup_weight(epoch)
                    if elr_w > 0:
                        elr_loss = elr_hook.compute_loss(paths, logits)
                        loss = loss + elr_w * elr_loss

            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()

            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            # EMA: only update if optimizer step was NOT skipped (no overflow)
            if ema_hook is not None and scaler.get_scale() >= old_scale:
                ema_hook.update(model)
            # Teacher: same gate as EMA (no update on overflow)
            if teacher_hook is not None and scaler.get_scale() >= old_scale:
                teacher_hook.update(model)
        else:
            logits = _forward_inputs(model, inputs, is_cached)

            if mixup_applied:
                loss_per_sample_a = criterion(logits, labels_a)
                loss_per_sample_b = criterion(logits, labels_b)
                loss_per_sample = (
                    lam * loss_per_sample_a + (1.0 - lam) * loss_per_sample_b
                )
            else:
                loss_per_sample = criterion(logits, labels)

            if mixup_applied:
                loss = reduce_loss(loss_per_sample)
            else:
                loss = _apply_sample_weights(
                    loss_per_sample, paths, weight_provider or sample_weights,
                    normalize_by_weight_sum, missing_policy, device,
                    epoch=epoch, labels=labels,
                )

                # ── Teacher–Student consistency loss (A-INFRA-7) ──
                if teacher_hook is not None:
                    teacher_cfg = config.get("teacher", {})
                    conf_thresh = teacher_cfg.get("confidence_threshold", 0.8)
                    cons_weight = teacher_cfg.get("consistency_weight", 1.0)
                    ramp_w = teacher_hook.rampup_weight(epoch)
                    flip_consistency = teacher_cfg.get("flip_consistency", False)
                    with torch.no_grad():
                        teacher_logits = teacher_hook.forward(inputs)
                        conf_mask = teacher_hook.confidence_mask(
                            teacher_logits, threshold=conf_thresh
                        )
                    if conf_mask.any():
                        if flip_consistency:
                            inputs_flip = torch.flip(inputs, dims=[3])
                            student_flip_logits = model(inputs_flip)
                            cons_loss = torch.nn.functional.mse_loss(
                                student_flip_logits[conf_mask], teacher_logits[conf_mask],
                            )
                        else:
                            cons_loss = torch.nn.functional.mse_loss(
                                logits[conf_mask], teacher_logits[conf_mask],
                            )
                        loss = loss + ramp_w * cons_weight * cons_loss

                # ── ELR temporal consistency ──
                # §5.4: MixUp batches use mixed inputs — targets don't
                # correspond to single real samples, so skip ELR update.
                if elr_hook is not None and not mixup_applied:
                    elr_hook.update(paths, logits.detach())
                    elr_w = elr_hook.rampup_weight(epoch)
                    if elr_w > 0:
                        elr_loss = elr_hook.compute_loss(paths, logits)
                        loss = loss + elr_w * elr_loss

            loss.backward()

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            # EMA: always update in non-AMP path
            if ema_hook is not None:
                ema_hook.update(model)
            # Teacher: always update in non-AMP path
            if teacher_hook is not None:
                teacher_hook.update(model)

        # Compute per-group gradient norms (before optimizer step is too late
        # since step already happened; compute after clipping, before step would
        # be ideal but we compute post-step for logging purposes)
        for group in optimizer.param_groups:
            gn_sq = sum(
                p.grad.norm().item() ** 2
                for p in group["params"]
                if p.grad is not None
            )
            gn = gn_sq ** 0.5
            if group.get("name") == "head":
                head_grad_sum += gn
            elif group.get("name") == "backbone":
                backbone_grad_sum += gn
            else:
                head_grad_sum += gn  # Default group → head

        # Cosine head: clamp logit_scale after each optimizer step
        if hasattr(model, "clamp_scale"):
            model.clamp_scale()

        # Only step scheduler after warmup
        if global_step >= warmup_steps:
            scheduler.step()

        # Statistics
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        # Use primary labels (labels_a) for accuracy — with lam >= 0.5 this is
        # the dominant label; for non-MixUp batches labels_a == labels.
        correct += (preds == labels_a).sum().item()
        total += batch_size
        global_step += 1

        # Update progress bar
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / total:.4f}",
                "lr": f"{current_lr:.2e}",
            }
        )

    avg_loss = total_loss / total
    accuracy = correct / total
    n_batches = max(len(loader), 1)
    head_grad_norm = head_grad_sum / n_batches
    backbone_grad_norm = backbone_grad_sum / n_batches

    return avg_loss, accuracy, global_step, head_grad_norm, backbone_grad_norm


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    val_weights: Optional[torch.Tensor] = None,
) -> tuple:
    """Run validation.

    Args:
        val_weights: Optional per-sample weight tensor of shape (N,).
            Samples with weight == 0 are excluded from loss and accuracy.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.eval()
    use_amp = config["train"].get("amp", False)

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=" " * 16 + "[Val]  ", dynamic_ncols=True)

    for batch_idx, batch_data in enumerate(pbar):
        inputs, labels, is_cached, _paths = _unpack_batch(batch_data, device)

        if use_amp:
            with autocast(device_type=device.type, enabled=use_amp):
                logits = _forward_inputs(model, inputs, is_cached)
                loss = criterion(logits, labels)
        else:
            logits = _forward_inputs(model, inputs, is_cached)
            loss = criterion(logits, labels)

        # Handle reduction='none' (sample-weighted training uses per-sample loss)
        if loss.ndim > 0:
            loss = loss.mean()

        # Apply val weights: mask out zero-weight samples
        if val_weights is not None:
            batch_size = inputs.size(0)
            start_idx = batch_idx * loader.batch_size
            end_idx = min(start_idx + batch_size, len(val_weights))
            batch_weights = val_weights[start_idx:end_idx].to(device)
            mask = batch_weights > 0
            n_kept = mask.sum().item()
            if n_kept > 0:
                total_loss += loss.item() * n_kept
                preds = logits.argmax(dim=1)
                correct += (preds[mask] == labels[mask]).sum().item()
                total += n_kept
        else:
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / total:.4f}",
            }
        )

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_val_acc: float,
    config: Dict[str, Any],
    filepath: str,
    extra_meta: Optional[Dict[str, Any]] = None,
    ema_hook=None,
    weight_provider=None,
    teacher_hook=None,
    elr_hook=None,
    best_raw_acc: float = 0.0,
    best_raw_epoch: int = 0,
    best_ema_acc: float = 0.0,
    best_ema_epoch: int = 0,
    selection_source: str = "raw",
) -> None:
    """Save a training checkpoint with optional extra metadata."""
    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_acc": best_val_acc,
        "config": config,
        # EMA fields
        "best_raw_val_acc": best_raw_acc,
        "best_raw_epoch": best_raw_epoch,
        "best_ema_val_acc": best_ema_acc,
        "best_ema_epoch": best_ema_epoch,
        "selection_source": selection_source,
    }
    if ema_hook is not None:
        checkpoint["ema_state_dict"] = ema_hook.state_dict()
        checkpoint["ema_num_updates"] = ema_hook.num_updates
    if teacher_hook is not None:
        checkpoint["teacher_state_dict"] = teacher_hook.state_dict()
        checkpoint["teacher_num_updates"] = teacher_hook.num_updates
    if elr_hook is not None:
        checkpoint["elr_state_dict"] = elr_hook.state_dict()
    if hasattr(weight_provider, "state_dict") and callable(weight_provider.state_dict):
        checkpoint["weight_provider_state"] = weight_provider.state_dict()
    if extra_meta:
        checkpoint.update(extra_meta)
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    scaler: GradScaler = None,
    device: torch.device = None,
) -> dict:
    """Load a training checkpoint.

    Returns:
        Dictionary with checkpoint metadata (epoch, global_step, best_val_acc).
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    logger.info(f"Checkpoint loaded from {filepath} (epoch {checkpoint['epoch']})")

    return {
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "best_val_acc": checkpoint.get("best_val_acc", 0.0),
        "best_raw_acc": checkpoint.get("best_raw_val_acc", 0.0),
        "best_raw_epoch": checkpoint.get("best_raw_epoch", 0),
        "best_ema_acc": checkpoint.get("best_ema_val_acc", 0.0),
        "best_ema_epoch": checkpoint.get("best_ema_epoch", 0),
        "ema_state_dict": checkpoint.get("ema_state_dict"),
        "selection_source": checkpoint.get("selection_source", "raw"),
        "weight_provider_state": checkpoint.get("weight_provider_state"),
        "teacher_state_dict": checkpoint.get("teacher_state_dict"),
    }


def _build_dataloaders_online(
    config: Dict[str, Any],
    train_transform: callable,
    val_transform: callable,
    class_to_idx: Dict[str, int],
    split_dir: str,
) -> tuple:
    """Build train and validation DataLoaders from online images."""
    train_cfg = config["train"]
    train_csv = str(Path(split_dir) / "train.csv")
    val_csv = str(Path(split_dir) / "val.csv")

    train_dataset = TrainImageDataset(
        data_root=config["data"]["train_dir"],
        split_csv=train_csv,
        class_to_idx=class_to_idx,
        transform=train_transform,
        return_path=True,
    )

    val_dataset = TrainImageDataset(
        data_root=config["data"]["train_dir"],
        split_csv=val_csv,
        class_to_idx=class_to_idx,
        transform=val_transform,
        return_path=True,
    )

    # Use a dedicated Generator for deterministic DataLoader shuffling
    train_seed = config["data"].get("train_seed", config["data"].get("seed", 42))
    g = torch.Generator()
    g.manual_seed(train_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg.get("pin_memory", True), timeout=120,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg.get("pin_memory", True), timeout=120,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    logger.info(
        f"Train loader: {len(train_dataset)} samples, {len(train_loader)} batches"
    )
    logger.info(f"Val loader:   {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader


def _run_init_checkpoint_audit(
    args: argparse.Namespace,
    config: Dict[str, Any],
    init_ckpt_path: str,
    checkpoint: dict,
    train_logger: logging.Logger,
) -> None:
    """Run parent-child split lineage audit for --init-checkpoint experiments.

    Determines the parent's split directory from the checkpoint path and
    compares it against the child's split directory.  Writes
    ``split_lineage_audit.json`` into the child experiment output directory.

    Raises:
        SplitAuditError: If any integrity rule is violated.
    """
    from common.split_audit import SplitAuditError, run_split_audit

    # Determine parent experiment id from checkpoint path
    # Convention: outputs/<parent_exp>/.../best.pt  → parent_exp
    ckpt_path = Path(init_ckpt_path)
    # Walk up from the checkpoint to find experiment root
    # Typical: outputs/<exp>/<seed>/checkpoints/best.pt
    #         outputs/<exp>/checkpoints/best.pt
    parent_exp = "unknown"
    for ancestor in ckpt_path.parents:
        if ancestor.parent and ancestor.parent.name == "outputs":
            parent_exp = str(ancestor.relative_to(ancestor.parent.parent))
            break
    if parent_exp == "unknown":
        parent_exp = ckpt_path.parent.parent.name

    # Determine parent split directory
    # Try: checkpoint contains metadata about its split
    parent_split_dir = None
    if "config" in checkpoint:
        parent_cfg = checkpoint["config"]
        parent_split_dir = parent_cfg.get("data", {}).get("split_dir")

    if not parent_split_dir:
        # Fallback: infer from checkpoint path structure
        # outputs/<exp>/checkpoints/best.pt → outputs/<exp>/splits
        # outputs/<exp>/seed42/checkpoints/best.pt → outputs/<exp>/seed42
        parent_exp_dir = ckpt_path.parent.parent
        if (parent_exp_dir / "train.csv").exists():
            parent_split_dir = str(parent_exp_dir)
        elif (parent_exp_dir / "splits" / "train.csv").exists():
            parent_split_dir = str(parent_exp_dir / "splits")

    if not parent_split_dir:
        raise SplitAuditError(
            "Cannot determine parent split directory from checkpoint "
            f"'{init_ckpt_path}'. The split lineage audit is a mandatory "
            "integrity check — refusing to train without it. "
            "Ensure the parent checkpoint's config includes data.split_dir, "
            "or that the parent experiment directory contains train.csv."
        )

    child_split_dir = Path(config["data"]["split_dir"])

    train_logger.info("Running split lineage audit...")
    train_logger.info("  Parent: %s", parent_exp)
    train_logger.info("  Parent split: %s", parent_split_dir)
    train_logger.info("  Child split:  %s", child_split_dir)

    run_split_audit(
        parent_experiment_id=parent_exp,
        parent_checkpoint_path=init_ckpt_path,
        parent_train_csv=Path(parent_split_dir) / "train.csv",
        parent_val_csv=Path(parent_split_dir) / "val.csv",
        child_train_csv=child_split_dir / "train.csv",
        child_val_csv=child_split_dir / "val.csv",
        output_dir=Path(config["train"]["save_dir"]).parent,
    )
    train_logger.info("Split lineage audit: protocol_valid = True")


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # --seed-override: apply multi-seed path and seed adjustments
    if args.seed_override is not None:
        import re
        new_seed = args.seed_override
        config["data"]["seed"] = new_seed
        config["data"]["split_seed"] = new_seed
        config["data"]["train_seed"] = new_seed

        # Replace seed{N} or seed_N in all path-like config values
        def _replace_seed(path_str: str) -> str:
            path_str = re.sub(r'/seed\d+/', f'/seed{new_seed}/', path_str)
            path_str = re.sub(r'/seed_\d+/', f'/seed_{new_seed}/', path_str)
            return path_str

        config["data"]["split_dir"] = _replace_seed(
            config["data"]["split_dir"])
        config["train"]["save_dir"] = _replace_seed(
            config["train"]["save_dir"])
        config["output"]["log_dir"] = _replace_seed(
            config["output"]["log_dir"])
        config["output"]["submission_dir"] = _replace_seed(
            config["output"].get("submission_dir",
                                 config["output"]["log_dir"]))
        # Also adjust class_mapping_path if present
        if "class_mapping_path" in config["data"]:
            config["data"]["class_mapping_path"] = _replace_seed(
                config["data"]["class_mapping_path"])

    # Resolve runtime args: explicit CLI > YAML > hard default
    args = resolve_runtime_args(args, config)

    # ── Config schema validation (A-INFRA-1) ──
    from common.config_schema import validate_config
    schema_warnings = validate_config(config)
    for w in schema_warnings:
        logger.warning("Config schema: %s", w)

    mode = args.mode
    experiment_id = args.experiment_id
    use_cached = args.use_cached_features
    aug_preset = args.augmentation_preset
    head_type = args.head_type

    # Set random seed for training (does NOT enable cudnn.deterministic)
    train_seed = config["data"].get("train_seed", config["data"].get("seed", 42))
    set_train_seed(train_seed)

    # Setup logging first
    log_dir = ensure_dir(config["output"]["log_dir"])
    train_logger = setup_logging(str(log_dir), name="train")

    # Setup device
    device = torch.device(
        config["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    train_logger.info(f"Using device: {device}")
    train_logger.info(f"Configuration: {args.config}")
    train_logger.info(f"Train seed: {train_seed}")

    # Validate resolved runtime values
    if aug_preset not in VALID_PRESETS:
        raise ValueError(
            f"Unknown augmentation preset: {aug_preset}. "
            f"Expected one of {sorted(VALID_PRESETS)}"
        )
    if mode not in {"dev", "confirm", "final_fit"}:
        raise ValueError(f"Unsupported training mode: {mode}")
    if head_type not in {"linear", "cosine"}:
        raise ValueError(f"Unsupported head type: {head_type}")

    train_logger.info(
        "Resolved runtime: "
        f"experiment_id={experiment_id}, "
        f"mode={mode}, "
        f"head_type={head_type}, "
        f"augmentation={aug_preset}, "
        f"cached={use_cached}"
    )

    # Guard enforcement
    freeze_clip = config["model"].get("freeze_clip", True)
    _enforce_guards(experiment_id, use_cached, aug_preset, freeze_clip)

    # Build model based on head type
    if head_type == "cosine":
        from experiments.cosine.model import build_cosine_model

        # CLI args override config values for cosine head options
        if args.cos_init_scale is not None:
            config["model"]["cos_init_scale"] = args.cos_init_scale
        if args.cos_learnable_scale is not None:
            config["model"]["cos_learnable_scale"] = (
                args.cos_learnable_scale.lower() == "true"
            )

        model, preprocess = build_cosine_model(config, device)
    else:
        from .model import build_model
        model, preprocess = build_model(config, device)

    total_params, trainable_params = count_parameters(model)
    train_logger.info(f"Total parameters:     {total_params:,}")
    train_logger.info(f"Trainable parameters: {trainable_params:,}")

    # ── PEFT configuration ──
    # Apply after model building; overrides manual freeze/unfreeze.
    # Only non-default PEFT types trigger reconfiguration.
    peft_cfg = config.get("peft", {})
    peft_type = peft_cfg.get("type", "linear_head_only")
    peft_info = None
    if peft_type != "linear_head_only":
        peft_info = apply_peft(model, peft_cfg)
        # Re-count after PEFT reconfiguration
        total_params, trainable_params = count_parameters(model)
        train_logger.info(
            f"After PEFT ({peft_type}): "
            f"total={total_params:,}, trainable={trainable_params:,}"
        )

    # ── EMA Hook (A-INFRA-5) ──
    ema_cfg = config.get("head_ema", {})
    ema_enabled = ema_cfg.get("enabled", False)
    ema_hook = None
    ema_decay = ema_cfg.get("decay", 0.99)
    ema_warmup_epochs = ema_cfg.get("warmup_epochs", 5)
    ema_selection_source = ema_cfg.get("selection_source", "ema")

    # ── Teacher Hook (A-INFRA-7) ──
    teacher_cfg = config.get("teacher", {})
    teacher_enabled = teacher_cfg.get("enabled", False)
    teacher_hook = None
    teacher_ema_decay = teacher_cfg.get("ema_decay", 0.999)
    teacher_confidence_threshold = teacher_cfg.get("confidence_threshold", 0.8)
    teacher_consistency_weight = teacher_cfg.get("consistency_weight", 1.0)
    teacher_ramp_epochs = teacher_cfg.get("ramp_epochs", 10)

    # Canonical class mapping via common.class_mapping
    class_mapping_path = config["data"].get("class_mapping_path", config["data"]["split_dir"])
    expected_num_classes = config["model"]["num_classes"]
    class_to_idx, idx_to_class = load_or_generate_mapping(
        metadata_dir=class_mapping_path,
        train_dir=config["data"]["train_dir"],
        expected_num_classes=expected_num_classes,
    )

    # Store mapping on model for checkpoint metadata
    model.class_to_idx_ = class_to_idx
    model.idx_to_class_ = idx_to_class

    # Build train transform with augmentation preset
    if aug_preset != "a0" and not use_cached:
        train_transform = build_train_transform(aug_preset, preprocess)
    else:
        train_transform = preprocess

    val_transform = preprocess

    # Dataset and DataLoader construction
    split_dir = config["data"]["split_dir"]
    loader_pin_memory = config["train"].get("pin_memory", True)

    if use_cached:
        # Use cached features instead of online encoding
        train_logger.info("Using cached features for training.")
        cache_dir = config["cache"]["cache_dir"]
        train_split_csv = (
            None
            if mode == "final_fit"
            else str(Path(split_dir) / "train.csv")
        )
        class_to_idx_path = str(Path(class_mapping_path) / "class_to_idx.json")
        verification = config["cache"].get("verification", "full")

        train_dataset = CachedFeatureDataset(
            cache_dir=cache_dir,
            split_csv=train_split_csv,
            class_to_idx_path=class_to_idx_path,
            dataset_root=config["data"]["train_dir"],
            verification=verification,
        )

        train_seed_val = config["data"].get("train_seed", config["data"].get("seed", 42))
        g = torch.Generator()
        g.manual_seed(train_seed_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=0,  # Cached features: no need for image loading workers
            pin_memory=loader_pin_memory, timeout=120,
            drop_last=False,
            generator=g,
        )

        # A0 preprocessing and a frozen CLIP encoder make cached validation
        # exactly equivalent to deterministic online feature extraction.
        val_loader = None
        if mode in {"dev", "confirm"}:
            if mode == "confirm" and not _check_splits_exist(split_dir):
                train_logger.error(
                    f"Train/val splits not found in {split_dir}.\n"
                    f"Please run: python scripts/split_data.py --config {args.config}"
                )
                raise FileNotFoundError(
                    f"Splits not found in {split_dir}. "
                    f"Run: python scripts/split_data.py --config {args.config}"
                )

            val_dataset = CachedFeatureDataset(
                cache_dir=cache_dir,
                split_csv=str(Path(split_dir) / "val.csv"),
                class_to_idx_path=class_to_idx_path,
                dataset_root=config["data"]["train_dir"],
                verification=verification,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["eval"]["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=loader_pin_memory, timeout=120,
                drop_last=False,
                generator=g,
            )
            train_logger.info(
                f"Val loader (cached): {len(val_dataset)} samples, "
                f"{len(val_loader)} batches"
            )

        train_logger.info(
            f"Train loader (cached): {len(train_dataset)} samples, "
            f"{len(train_loader)} batches"
        )
    elif mode == "final_fit":
        # final_fit: no split, use full dataset, no validation
        train_logger.info("final_fit mode: loading full dataset (no val split).")
        train_dataset = TrainImageDataset(
            data_root=config["data"]["train_dir"],
            split_csv=None,
            class_to_idx=class_to_idx,
            transform=train_transform,
            return_path=True,
        )

        train_seed_val = config["data"].get("train_seed", config["data"].get("seed", 42))
        g = torch.Generator()
        g.manual_seed(train_seed_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=config["train"]["num_workers"],
            pin_memory=loader_pin_memory, timeout=120,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_loader = None
        train_logger.info(
            f"Train loader (final_fit): {len(train_dataset)} samples, "
            f"{len(train_loader)} batches"
        )
    else:
        # Standard online training (B0, dev mode by default)
        if not _check_splits_exist(split_dir):
            train_logger.error(
                f"Train/val splits not found in {split_dir}.\n"
                f"Please run: python scripts/split_data.py --config {args.config}"
            )
            raise FileNotFoundError(
                f"Splits not found in {split_dir}. "
                f"Run: python scripts/split_data.py --config {args.config}"
            )

        train_loader, val_loader = _build_dataloaders_online(
            config, train_transform, val_transform, class_to_idx, split_dir
        )

    # Training setup
    train_cfg = config["train"]

    # Determine epoch count based on mode
    if mode in ("confirm", "final_fit"):
        if args.frozen_epochs is None:
            raise ValueError(
                f"--frozen-epochs is required for {mode} mode. "
                f"Specify the number of training epochs."
            )
        if args.source_dev_best_epoch is None:
            raise ValueError(
                f"--source-dev-best-epoch is required for {mode} mode. "
                f"Specify the dev best epoch to carry forward."
            )
        epochs = args.frozen_epochs
        train_logger.info(f"Using frozen epochs: {epochs} ({mode} mode)")
    else:
        epochs = train_cfg["epochs"]
        train_logger.info(f"Using config epochs: {epochs} ({mode} mode)")

    warmup_steps = train_cfg["warmup_epochs"] * len(train_loader)
    total_steps = epochs * len(train_loader)
    cosine_steps = max(total_steps - warmup_steps, 1)

    # ── Build loss function ────────────────────────────────────────
    from common.losses import build_loss
    from common.sample_weighting import (
        build_weight_provider, NoneWeightProvider,
    )

    # Backward-compat: old sample_weighting.enabled → translate to new provider
    sw_cfg = config.get("sample_weighting", {})
    if sw_cfg.get("enabled", False) and "type" not in sw_cfg:
        config.setdefault("sample_weighting", {})
        config["sample_weighting"]["type"] = "static_manifest"
        config["sample_weighting"]["manifest_path"] = sw_cfg["weights_path"]
        train_logger.info(
            "Translated legacy sample_weighting.enabled to type=static_manifest"
        )

    # n_train from the already-constructed train_loader
    n_train = len(train_loader.dataset) if train_loader is not None else 0
    weight_provider = build_weight_provider(config, num_train_samples=n_train)

    # For stateful providers, initialise the path→index mapping
    if hasattr(weight_provider, "init_sample_index") and train_loader is not None:
        ds = train_loader.dataset
        all_paths = [str(p) for p in ds.samples]
        all_labels = torch.tensor(ds.labels, dtype=torch.long)
        weight_provider.init_sample_index(all_paths, all_labels)

    use_sample_weights = not isinstance(weight_provider, NoneWeightProvider)

    # Build val weights from the same manifest (for OOF zero-weight filtering)
    val_weights = None
    if use_sample_weights and val_loader is not None:
        sw_cfg = config.get("sample_weighting", {})
        manifest_path = sw_cfg.get("manifest_path")
        if manifest_path and Path(manifest_path).suffix == ".csv":
            import pandas as pd
            try:
                manifest_df = pd.read_csv(manifest_path)
                # Match by stable key (class/filename) — paths may differ
                # between train_dedup/ (manifest) and /home/.../train/ (val).
                def _stable_key(p: str) -> str:
                    parts = str(p).replace("\\", "/").split("/")
                    return "/".join(parts[-2:])
                w_map = dict(zip(
                    manifest_df["image_path"].apply(_stable_key),
                    manifest_df["sample_weight"],
                ))
                val_paths = [_stable_key(str(p)) for p in val_loader.dataset.samples]
                val_weights = torch.tensor(
                    [w_map.get(p, 1.0) for p in val_paths],
                    dtype=torch.float32,
                )
                n_zero_val = (val_weights == 0).sum().item()
                train_logger.info(
                    "Val weights from manifest: %d/%d zeroed (%.1f%%)",
                    n_zero_val, len(val_weights),
                    100.0 * n_zero_val / max(len(val_weights), 1),
                )
            except Exception as e:
                train_logger.warning("Failed to build val weights from manifest: %s", e)

    if use_sample_weights:
        loss_cfg = config.get("loss", {}).copy()
        loss_cfg["reduction"] = "none"
        criterion = build_loss({"loss": loss_cfg})
    else:
        criterion = build_loss(config)

    train_logger.info(
        "Loss: %s (reduction=%s)",
        config.get("loss", {}).get("name", "cross_entropy"),
        config.get("loss", {}).get("reduction", "mean"),
    )
    optimizer, scheduler = _build_optimizer_and_scheduler(
        model, config, cosine_steps, peft_cfg=peft_cfg,
    )
    scaler = GradScaler(device=device.type, enabled=train_cfg.get("amp", False))

    # Diagnostic: log optimizer parameter group configuration
    train_logger.info("Optimizer parameter groups:")
    total_opt_params = 0
    for group in optimizer.param_groups:
        n = sum(p.numel() for p in group["params"])
        total_opt_params += n
        train_logger.info(
            f"  {group.get('name', 'default'):12s}: "
            f"{n:>10,} params, "
            f"lr={group['lr']:.2e}, "
            f"wd={group['weight_decay']:.2e}"
        )
    train_logger.info(
        f"  {'TOTAL':12s}: {total_opt_params:>10,} optimizer params"
    )

    # Per-component trainable param breakdown
    head_trainable = sum(
        p.numel() for p in model.classifier.parameters() if p.requires_grad
    )
    visual_trainable = sum(
        p.numel() for p in model.visual.parameters() if p.requires_grad
    )
    train_logger.info(f"  Trainable head params:    {head_trainable:>10,}")
    train_logger.info(f"  Trainable visual params:  {visual_trainable:>10,}")

    # Early stopping config
    early_stop_patience = train_cfg.get("early_stop_patience", 0)
    early_stop_counter = 0
    early_stopped = False

    # --resume and --init-checkpoint are mutually exclusive
    if args.resume and args.init_checkpoint:
        raise ValueError(
            "--resume and --init-checkpoint are mutually exclusive. "
            "Use --init-checkpoint to load model weights for a new "
            "training run; use --resume to continue a crashed run."
        )

    # Declare epoch0 tracking variables (initialized to None for non-init-checkpoint runs;
    # overwritten with computed values inside the init_checkpoint block below)
    epoch0_val_acc = None
    epoch0_val_loss = None
    epoch0_parent_acc = None
    epoch0_delta = None

    # Initialize model weights from checkpoint (no optimizer/scheduler/epoch)
    if args.init_checkpoint:
        init_ckpt_path = args.init_checkpoint
        train_logger.info(
            f"Initializing model weights from: {init_ckpt_path}"
        )
        checkpoint = torch.load(init_ckpt_path, map_location=device)
        model_state = checkpoint.get("model_state_dict", checkpoint)

        missing_keys, unexpected_keys = model.load_state_dict(
            model_state, strict=False
        )

        if missing_keys:
            train_logger.warning(f"Missing keys ({len(missing_keys)}): {missing_keys}")
        if unexpected_keys:
            train_logger.warning(
                f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}"
            )

        if not missing_keys and not unexpected_keys:
            train_logger.info(
                "Model weights loaded with exact key match."
            )
        else:
            train_logger.warning(
                f"Model weight load had {len(missing_keys)} missing, "
                f"{len(unexpected_keys)} unexpected keys."
            )

        # Do NOT load optimizer state — fresh training from epoch 1
        # Do NOT load scheduler state
        # Do NOT restore epoch

        # ── Parent-child split lineage audit ─────────────────────
        _run_init_checkpoint_audit(args, config, init_ckpt_path, checkpoint,
                                   train_logger)

        # ── Epoch-0 validation gate ──────────────────────────────
        if val_loader is not None:
            train_logger.info("=" * 60)
            train_logger.info("Epoch-0 validation gate: verifying loaded checkpoint")

            val_loss_0, val_acc_0 = validate(
                model, val_loader, criterion, device, config
            )

            parent_expected_acc = checkpoint.get("best_val_acc", None)

            # Store for eval_results.json
            epoch0_val_acc = float(val_acc_0)
            epoch0_val_loss = float(val_loss_0)
            epoch0_parent_acc = float(parent_expected_acc) if parent_expected_acc is not None else None
            epoch0_delta = float(abs(val_acc_0 - parent_expected_acc)) if parent_expected_acc is not None else None

            train_logger.info(
                "Epoch 0   | Val Loss: %.4f | Val Acc: %.4f",
                val_loss_0, val_acc_0,
            )

            if parent_expected_acc is not None:
                delta = abs(val_acc_0 - parent_expected_acc)
                if delta > 0.0005:  # 0.05pp threshold
                    train_logger.error(
                        "EPOCH-0 VALIDATION MISMATCH: "
                        "loaded=%.4f, expected=%.4f, delta=%.6f (> 0.0005). "
                        "Check model loading, transforms, class mapping.",
                        val_acc_0, parent_expected_acc, delta,
                    )
                    raise RuntimeError(
                        f"Epoch-0 validation mismatch: delta={delta:.6f} > 0.0005"
                    )
                train_logger.info(
                    "Epoch-0 validation gate PASSED: delta=%.6f <= 0.0005",
                    delta,
                )
            else:
                train_logger.warning(
                    "No best_val_acc in checkpoint metadata; "
                    "skipping epoch-0 gate."
                )
            train_logger.info("=" * 60)
        else:
            train_logger.info(
                "No val_loader available; skipping epoch-0 validation gate."
            )

    # ── Create EMA hook (after model weights are finalised) ──
    if ema_enabled:
        from common.hooks import EMAHook
        steps_per_epoch = len(train_loader)
        ema_warmup_steps = ema_warmup_epochs * steps_per_epoch
        ema_hook = EMAHook(model, decay=ema_decay, warmup_steps=ema_warmup_steps)
        train_logger.info(
            "EMA enabled: decay=%.4f, warmup_epochs=%d, warmup_steps=%d, selection=%s",
            ema_decay, ema_warmup_epochs, ema_warmup_steps, ema_selection_source,
        )

    # ── Create Teacher hook (after model weights are finalised) ──
    if teacher_enabled:
        from common.hooks import TeacherHook
        teacher_hook = TeacherHook(
            model,
            ema_decay=teacher_ema_decay,
            ramp_epochs=teacher_ramp_epochs,
        )
        train_logger.info(
            "Teacher-Student enabled: ema_decay=%.4f, confidence_threshold=%.2f, "
            "consistency_weight=%.2f, ramp_epochs=%d",
            teacher_ema_decay, teacher_confidence_threshold,
            teacher_consistency_weight, teacher_ramp_epochs,
        )

    # ── Create ELR hook (after model weights are finalised) ──
    elr_cfg = config.get("elr", {})
    elr_enabled = elr_cfg.get("enabled", False)
    elr_hook = None
    if elr_enabled:
        from common.elr import ELRHook

        # Determine number of training samples
        train_split_csv = (
            None
            if mode == "final_fit"
            else str(Path(split_dir) / "train.csv")
        )
        if train_split_csv is not None:
            import pandas as pd
            num_train_samples = len(pd.read_csv(train_split_csv))
        else:
            num_train_samples = len(train_dataset)

        elr_hook = ELRHook(
            num_train_samples=num_train_samples,
            num_classes=config["model"]["num_classes"],
            momentum=elr_cfg.get("momentum", 0.9),
            target_weight=elr_cfg.get("target_weight", 1.0),
            warmup_epochs=elr_cfg.get("warmup_epochs", 10),
            ramp_epochs=elr_cfg.get("ramp_epochs", 10),
            storage_dtype=torch.float32,
        )
        train_logger.info(
            "ELR enabled: momentum=%.3f, target_weight=%.2f, "
            "warmup_epochs=%d, ramp_epochs=%d, num_samples=%d",
            elr_cfg.get("momentum", 0.9),
            elr_cfg.get("target_weight", 1.0),
            elr_cfg.get("warmup_epochs", 10),
            elr_cfg.get("ramp_epochs", 10),
            num_train_samples,
        )

    # Resume if requested
    start_epoch = 1
    global_step = 0
    best_val_acc = 0.0
    best_raw_acc = 0.0
    best_raw_epoch = 0
    best_ema_acc = 0.0
    best_ema_epoch = 0
    dev_best_epoch = None

    if args.resume:
        resume_info = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )
        start_epoch = resume_info["epoch"] + 1
        global_step = resume_info["global_step"]
        best_val_acc = resume_info["best_val_acc"]
        # Restore EMA state if present
        best_raw_acc = resume_info.get("best_raw_acc", best_val_acc)
        best_raw_epoch = resume_info.get("best_raw_epoch", 0)
        best_ema_acc = resume_info.get("best_ema_acc", 0.0)
        best_ema_epoch = resume_info.get("best_ema_epoch", 0)
        if ema_enabled and ema_hook is not None:
            if "ema_state_dict" in resume_info:
                ema_hook.load_state_dict(resume_info["ema_state_dict"])
            elif getattr(args, "ema_reset_on_resume", False):
                train_logger.warning(
                    "Resuming with EMA enabled but no EMA state in checkpoint. "
                    "Re-initialising EMA from raw weights."
                )
            else:
                raise ValueError(
                    "EMA enabled and --resume used, but checkpoint has no EMA "
                    "state. Use --ema-reset-on-resume to re-initialise EMA "
                    "from raw weights."
                )
        # Restore sample weight provider state (e.g. ema_loss history)
        if "weight_provider_state" in resume_info and resume_info["weight_provider_state"] is not None:
            if hasattr(weight_provider, "load_state_dict"):
                weight_provider.load_state_dict(resume_info["weight_provider_state"])
                train_logger.info("Restored weight provider state.")
        # Restore teacher state
        if teacher_enabled and teacher_hook is not None:
            if "teacher_state_dict" in resume_info:
                teacher_hook.load_state_dict(resume_info["teacher_state_dict"])
                train_logger.info(
                    "Restored teacher state: num_updates=%d",
                    teacher_hook.num_updates,
                )
            else:
                train_logger.warning(
                    "Teacher enabled and --resume used, but checkpoint has no "
                    "teacher state. Teacher will be re-initialised from current "
                    "student weights."
                )
        # Restore ELR state
        if elr_enabled and elr_hook is not None:
            if "elr_state_dict" in resume_info:
                elr_hook.load_state_dict(resume_info["elr_state_dict"])
                train_logger.info(
                    "Restored ELR state: slots_filled=%d",
                    elr_hook.slots_filled,
                )
            else:
                train_logger.warning(
                    "ELR enabled and --resume used, but checkpoint has no "
                    "ELR state. ELR memory will be initialised from scratch."
                )
        train_logger.info(
            f"Resumed from epoch {resume_info['epoch']}, "
            f"best val acc: {best_val_acc:.4f}"
        )

    # Save config snapshot
    save_dir = ensure_dir(train_cfg["save_dir"])

    # Guard: refuse fresh runs when generated artifacts already exist
    _prepare_fresh_run_artifacts(
        save_dir=save_dir,
        log_dir=log_dir,
        resume_path=args.resume,
        allow_overwrite=args.allow_overwrite,
    )

    save_config_snapshot(config, str(save_dir))

    # Write resolved config with explicit defaults (A-INFRA-1)
    resolved = resolve_config(config)
    write_resolved_config(resolved, str(save_dir))

    # Training log CSV
    log_file = Path(config["output"]["log_dir"]) / "train_log.csv"
    log_header = not log_file.exists()

    # Training loop
    train_logger.info(
        f"Starting training: {epochs} epochs, {len(train_loader)} batches/epoch"
    )
    train_logger.info(
        f"Warmup steps: {warmup_steps}, Total steps: {total_steps}, Cosine steps: {cosine_steps}"
    )
    train_logger.info(f"AMP: {train_cfg.get('amp', False)}")
    train_logger.info("=" * 60)

    train_start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # Train
        # ScheduledLoss: set active phase for this epoch
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)

        train_loss, train_acc, global_step, head_grad_norm, backbone_grad_norm = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            scaler,
            device,
            epoch,
            config,
            warmup_steps,
            global_step,
            weight_provider=weight_provider if use_sample_weights else None,
            ema_hook=ema_hook,
            teacher_hook=teacher_hook,
            mixup_cfg=config.get("mixup") if config.get("mixup", {}).get("enabled", False) else None,
            elr_hook=elr_hook,
        )

        # Validate (skip if no val_loader, e.g., final_fit)
        val_loss = None
        val_acc = None
        ema_val_loss = None
        ema_val_acc = None
        if val_loader is not None:
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, config, val_weights=val_weights
            )
            # EMA validation (separate path — no swap, uses get_ema_model)
            if ema_hook is not None:
                ema_model = ema_hook.get_ema_model()
                ema_val_loss, ema_val_acc = validate(
                    ema_model, val_loader, criterion, device, config, val_weights=val_weights
                )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        head_lr_val = optimizer.param_groups[0]["lr"]
        backbone_lr_val = (
            optimizer.param_groups[1]["lr"]
            if len(optimizer.param_groups) > 1
            else 0.0
        )

        if val_acc is not None:
            ema_str = ""
            if ema_val_acc is not None:
                ema_str = (
                    f" | EMA Val Loss: {ema_val_loss:.4f} | EMA Val Acc: {ema_val_acc:.4f}"
                    f" | EMA updates: {ema_hook.num_updates}"
                )
            log_msg = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                f"{ema_str} | "
                f"head_lr: {head_lr_val:.2e} | "
                f"bb_lr: {backbone_lr_val:.2e} | "
                f"Time: {format_time(epoch_time)}"
            )
        else:
            log_msg = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"head_lr: {head_lr_val:.2e} | "
                f"bb_lr: {backbone_lr_val:.2e} | "
                f"Time: {format_time(epoch_time)} [no val]"
            )
        train_logger.info(log_msg)

        # Save to CSV
        with open(log_file, "a") as f:
            if log_header:
                if val_acc is not None:
                    f.write(
                        "epoch,train_loss,train_acc,raw_val_loss,raw_val_acc,"
                        "ema_val_loss,ema_val_acc,ema_num_updates,"
                        "head_lr,backbone_lr,head_grad_norm,backbone_grad_norm,"
                        "epoch_time\n"
                    )
                else:
                    f.write(
                        "epoch,train_loss,train_acc,"
                        "head_lr,backbone_lr,head_grad_norm,backbone_grad_norm,"
                        "epoch_time\n"
                    )
                log_header = False
            if val_acc is not None:
                ema_updates = ema_hook.num_updates if ema_hook is not None else 0
                ema_vl_str = f"{ema_val_loss:.6f}" if ema_val_loss is not None else ""
                ema_va_str = f"{ema_val_acc:.6f}" if ema_val_acc is not None else ""
                f.write(
                    f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                    f"{val_loss:.6f},{val_acc:.6f},"
                    f"{ema_vl_str},{ema_va_str},{ema_updates},"
                    f"{head_lr_val:.8f},{backbone_lr_val:.8f},"
                    f"{head_grad_norm:.6f},{backbone_grad_norm:.6f},"
                    f"{epoch_time:.2f}\n"
                )
            else:
                f.write(
                    f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                    f"{head_lr_val:.8f},{backbone_lr_val:.8f},"
                    f"{head_grad_norm:.6f},{backbone_grad_norm:.6f},"
                    f"{epoch_time:.2f}\n"
                )

        # Track best epoch (raw + EMA)
        is_best_raw = False
        is_best_ema = False
        if val_acc is not None:
            if val_acc > best_raw_acc:
                best_raw_acc = val_acc
                best_raw_epoch = epoch
                is_best_raw = True
            if ema_val_acc is not None and ema_val_acc > best_ema_acc:
                best_ema_acc = ema_val_acc
                best_ema_epoch = epoch
                is_best_ema = True

            # Compat best_val_acc: track based on selection_source
            if ema_hook is not None and ema_selection_source == "ema":
                current_best = best_ema_acc
            else:
                current_best = best_raw_acc

            is_best = False
            if epoch == 1 or (ema_hook is not None and ema_selection_source == "ema" and is_best_ema):
                is_best = True
                best_val_acc = current_best
                dev_best_epoch = epoch
            elif ema_hook is None and is_best_raw:
                is_best = True
                best_val_acc = current_best
                dev_best_epoch = epoch

            if is_best_raw:
                train_logger.info(f"  >> New best RAW! Val Acc: {best_raw_acc:.4f}")
            if is_best_ema:
                train_logger.info(f"  >> New best EMA! Val Acc: {best_ema_acc:.4f}")

            if is_best:
                early_stop_counter = 0
            elif early_stop_patience > 0:
                early_stop_counter += 1
                train_logger.info(
                    f"  >> No improvement for {early_stop_counter}/{early_stop_patience} epochs"
                )

        # Build checkpoint metadata for this save
        extra_meta = _build_checkpoint_metadata(
            model, config, mode, args, best_epoch=dev_best_epoch
        )

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            best_val_acc,
            config,
            str(save_dir / "last.pt"),
            extra_meta=extra_meta,
            ema_hook=ema_hook,
            teacher_hook=teacher_hook,
            elr_hook=elr_hook,
            best_raw_acc=best_raw_acc,
            best_raw_epoch=best_raw_epoch,
            best_ema_acc=best_ema_acc,
            best_ema_epoch=best_ema_epoch,
            selection_source=ema_selection_source if ema_enabled else "raw",
                weight_provider=weight_provider,
        )

        # Save best_raw.pt and best_ema.pt independently
        if is_best_raw:
            save_checkpoint(
                model,
                optimizer, scheduler, scaler,
                epoch, global_step, best_raw_acc, config,
                str(save_dir / "best_raw.pt"),
                extra_meta=extra_meta,
                ema_hook=ema_hook,
                teacher_hook=teacher_hook,
                elr_hook=elr_hook,
                best_raw_acc=best_raw_acc, best_raw_epoch=best_raw_epoch,
                best_ema_acc=best_ema_acc, best_ema_epoch=best_ema_epoch,
                selection_source="raw",
                weight_provider=weight_provider,
            )
        if is_best_ema and ema_hook is not None:
            # Save EMA weights as best_ema.pt
            ema_model = ema_hook.get_ema_model()
            ema_state = model.state_dict()  # save raw model state for resume
            model.load_state_dict(ema_model.state_dict())  # swap to EMA for save
            save_checkpoint(
                model,
                optimizer, scheduler, scaler,
                epoch, global_step, best_ema_acc, config,
                str(save_dir / "best_ema.pt"),
                extra_meta=extra_meta,
                ema_hook=ema_hook,
                teacher_hook=teacher_hook,
                elr_hook=elr_hook,
                best_raw_acc=best_raw_acc, best_raw_epoch=best_raw_epoch,
                best_ema_acc=best_ema_acc, best_ema_epoch=best_ema_epoch,
                selection_source="ema",
                weight_provider=weight_provider,
            )
            model.load_state_dict(ema_state)  # restore raw weights

        if is_best:
            best_ckpt_path = str(save_dir / "best.pt")
            if ema_enabled and ema_selection_source == "ema":
                # best.pt = best_ema.pt (copy EMA weights)
                ema_model = ema_hook.get_ema_model()
                ema_state_bk = model.state_dict()
                model.load_state_dict(ema_model.state_dict())
                save_checkpoint(
                    model,
                    optimizer, scheduler, scaler,
                    epoch, global_step, best_val_acc, config,
                    best_ckpt_path,
                    extra_meta=extra_meta,
                    ema_hook=ema_hook,
                    teacher_hook=teacher_hook,
                    elr_hook=elr_hook,
                    best_raw_acc=best_raw_acc, best_raw_epoch=best_raw_epoch,
                    best_ema_acc=best_ema_acc, best_ema_epoch=best_ema_epoch,
                    selection_source="ema",
                weight_provider=weight_provider,
                )
                model.load_state_dict(ema_state_bk)
            else:
                save_checkpoint(
                    model,
                    optimizer, scheduler, scaler,
                    epoch, global_step, best_val_acc, config,
                    best_ckpt_path,
                    extra_meta=extra_meta,
                    ema_hook=ema_hook,
                    teacher_hook=teacher_hook,
                    elr_hook=elr_hook,
                    best_raw_acc=best_raw_acc, best_raw_epoch=best_raw_epoch,
                    best_ema_acc=best_ema_acc, best_ema_epoch=best_ema_epoch,
                    selection_source="raw",
                weight_provider=weight_provider,
                )

        # Early stopping check
        if early_stop_patience > 0 and early_stop_counter >= early_stop_patience:
            train_logger.info(
                f"Early stopping triggered at epoch {epoch} "
                f"(no improvement for {early_stop_patience} epochs). "
                f"Best val acc: {best_val_acc:.4f} at epoch {dev_best_epoch}."
            )
            early_stopped = True
            break

    total_time = time.time() - train_start_time
    train_logger.info("=" * 60)
    train_logger.info(f"Training complete! Total time: {format_time(total_time)}")
    train_logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    train_logger.info(f"Best model saved to: {save_dir / 'best.pt'}")
    train_logger.info(f"Training log saved to: {log_file}")

    # Save eval_results.json for dev/confirm modes
    if mode in ("dev", "confirm"):
        # ── Reload best checkpoint for post-training evaluation ──
        # The in-memory model may hold the final-epoch weights, not the
        # best-epoch weights.  We must reload best.pt so that every metric
        # in eval_results.json is derived from the same checkpoint.
        best_checkpoint_path = save_dir / "best.pt"
        post_eval_ckpt_epoch = None
        post_eval_ckpt_best_val_acc = None

        if val_loader is not None:
            if not best_checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Best checkpoint missing before post-training evaluation: "
                    f"{best_checkpoint_path}"
                )

            best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(
                best_checkpoint["model_state_dict"],
                strict=True,
            )
            model.to(device)
            model.eval()

            post_eval_ckpt_epoch = best_checkpoint.get("epoch")
            post_eval_ckpt_best_val_acc = float(
                best_checkpoint.get("best_val_acc", -1.0)
            )

            train_logger.info(
                "Reloaded best checkpoint for post-training evaluation: %s "
                "(epoch=%s, best_val_acc=%.8f)",
                best_checkpoint_path,
                post_eval_ckpt_epoch,
                post_eval_ckpt_best_val_acc,
            )

            # ── Per-class evaluation from best.pt ──
            train_logger.info("Running post-training per-class evaluation...")
            from .evaluate import evaluate as evaluate_full

            per_class_results = evaluate_full(
                model, val_loader, criterion, device,
                use_amp=train_cfg.get("amp", False),
            )

            post_eval_micro = float(per_class_results["accuracy"])
            post_eval_macro = float(per_class_results["macro_accuracy"])
            post_eval_gap = post_eval_micro - post_eval_macro
            reported_gap = float(per_class_results["micro_macro_gap"])

            # ── Consistency hard-checks ──
            if abs(post_eval_micro - float(best_val_acc)) > 1e-8:
                raise RuntimeError(
                    "Best-checkpoint post evaluation does not reproduce "
                    "best_val_acc: "
                    f"post_eval_micro={post_eval_micro:.10f}, "
                    f"best_val_acc={best_val_acc:.10f}"
                )

            if abs(reported_gap - post_eval_gap) > 1e-10:
                raise RuntimeError(
                    "micro_macro_gap is inconsistent: "
                    f"reported={reported_gap:.10f}, "
                    f"expected={post_eval_gap:.10f}"
                )

            per_class_metrics = {
                "macro_accuracy": post_eval_macro,
                "median_per_class_accuracy": float(
                    per_class_results["median_per_class_accuracy"]
                ),
                "bottom_10_percent_accuracy": float(
                    per_class_results["bottom_10_percent_accuracy"]
                ),
                "micro_macro_gap": reported_gap,
                "post_eval_checkpoint": str(best_checkpoint_path),
                "post_eval_checkpoint_epoch": post_eval_ckpt_epoch,
                "post_eval_micro_accuracy": post_eval_micro,
                "post_eval_macro_accuracy": post_eval_macro,
            }
        else:
            per_class_metrics = {}

        eval_results = {
            "experiment_id": experiment_id,
            "mode": mode,
            "config_path": args.config,
            "split_seed": config["data"].get("split_seed"),
            "train_seed": train_seed,
            "best_val_acc": float(best_val_acc),
            "dev_best_epoch": dev_best_epoch,
            "trained_epochs": epochs,
            "max_epochs": epochs,
            "actual_epochs_run": epoch,
            "head_type": model.head_type,
            "augmentation_preset": aug_preset,
            "use_cached_features": use_cached,
            "learning_rate": config["train"]["lr"],
            "weight_decay": config["train"]["weight_decay"],
            "batch_size": config["train"]["batch_size"],
            "freeze_clip": config["model"].get("freeze_clip", True),
            "clip_model_name": config["model"]["clip_model_name"],
            "git_commit": get_git_commit(),
            "early_stop_patience": early_stop_patience,
            "early_stopped": early_stopped,
            "stopped_at_epoch": epoch if early_stopped else None,
            "init_checkpoint": args.init_checkpoint,
            "epoch0_val_acc": epoch0_val_acc,
            "epoch0_val_loss": epoch0_val_loss,
            "parent_best_val_acc": epoch0_parent_acc,
            "epoch0_delta": epoch0_delta,
            # EMA fields
            "ema_enabled": ema_enabled,
            "ema_decay": ema_decay if ema_enabled else None,
            "ema_warmup_epochs": ema_warmup_epochs if ema_enabled else None,
            "selection_source": ema_selection_source if ema_enabled else "raw",
            "best_raw_val_acc": float(best_raw_acc),
            "best_raw_epoch": best_raw_epoch,
            "best_ema_val_acc": float(best_ema_acc) if ema_enabled else None,
            "best_ema_epoch": best_ema_epoch if ema_enabled else None,
            "post_eval_weight_source": ema_selection_source if ema_enabled else "raw",
            # Teacher-Student fields
            "teacher_enabled": teacher_enabled,
            "teacher_ema_decay": teacher_ema_decay if teacher_enabled else None,
            "teacher_confidence_threshold": teacher_confidence_threshold if teacher_enabled else None,
            "teacher_consistency_weight": teacher_consistency_weight if teacher_enabled else None,
            "teacher_ramp_epochs": teacher_ramp_epochs if teacher_enabled else None,
            **per_class_metrics,
        }
        eval_path = save_dir / "eval_results.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        train_logger.info(f"Eval results saved to: {eval_path}")

        # ── Write artifact manifest (A-INFRA-9) ──
        best_ckpt_path = str(save_dir / "best.pt")
        train_csv_path = str(Path(split_dir) / "train.csv")
        val_csv_path = str(Path(split_dir) / "val.csv")

        manifest = build_artifact_manifest(
            experiment_id=experiment_id,
            parent_experiment=config.get("experiment", {}).get("parent"),
            config=resolved,
            checkpoint_path=best_ckpt_path,
            train_csv=train_csv_path,
            val_csv=val_csv_path,
            extra={
                "best_val_acc": float(best_val_acc),
                "best_raw_acc": float(best_raw_acc),
                "best_ema_acc": (
                    float(best_ema_acc) if ema_enabled else None
                ),
                "best_epoch": dev_best_epoch,
                "sample_weighting_type": config.get(
                    "sample_weighting", {}
                ).get("type", "none"),
            },
        )
        artifact_path = write_artifact_manifest(manifest, str(save_dir))
        train_logger.info(
            "Artifact manifest saved to: %s", artifact_path
        )

        # ── Write per_class_metrics.csv (A-INFRA-9) ──
        if val_loader is not None and per_class_results:
            import csv as _csv
            per_class_path = save_dir / "per_class_metrics.csv"
            with open(per_class_path, "w", newline="") as f:
                writer = _csv.writer(f)
                writer.writerow(["class_idx", "accuracy", "n_samples"])
                pca = per_class_results.get("per_class_accuracy", [])
                pcc = per_class_results.get("per_class_counts", [])
                for i in range(len(pca)):
                    count = pcc[i] if i < len(pcc) else 0
                    writer.writerow([i, f"{pca[i]:.6f}", count])
            train_logger.info(
                "Per-class metrics saved to: %s", per_class_path
            )

        # ── Write prediction_records.csv (A-INFRA-9) ──
        if val_loader is not None:
            pred_path = save_dir / "prediction_records.csv"
            model.eval()
            with open(pred_path, "w", newline="") as f:
                writer = _csv.writer(f)
                writer.writerow(
                    ["image_path", "true_label", "pred_label", "pred_conf"]
                )
                with torch.no_grad():
                    for batch_data in val_loader:
                        inputs, labels, is_cached, paths = _unpack_batch(
                            batch_data, device
                        )
                        if is_cached:
                            logits = model.forward_features(inputs)
                        else:
                            logits = model(inputs)
                        probs = torch.softmax(logits, dim=1)
                        confs, preds = probs.max(dim=1)
                        for path, tl, pl, cf in zip(
                            paths, labels.cpu(), preds.cpu(), confs.cpu()
                        ):
                            writer.writerow([
                                path, int(tl.item()),
                                int(pl.item()), f"{cf.item():.6f}",
                            ])
            train_logger.info(
                "Prediction records saved to: %s", pred_path
            )


if __name__ == "__main__":
    main()
