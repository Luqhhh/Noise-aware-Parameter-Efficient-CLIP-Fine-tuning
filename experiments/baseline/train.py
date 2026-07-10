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
from common.runtime_config import resolve_runtime_args
from common.transforms import build_train_transform, VALID_PRESETS
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


def _build_optimizer_and_scheduler(
    model: nn.Module, config: Dict[str, Any], cosine_steps: int
) -> tuple:
    """Build optimizer and learning rate scheduler.

    Supports both linear and cosine heads:
      - Linear: uses model.get_trainable_parameters() with uniform LR + WD.
      - Cosine: uses model.get_param_groups(lr, wd) for separate scale handling.
    """
    train_cfg = config["train"]

    # Use model.get_param_groups() if available (cosine head handles scale separately),
    # otherwise use get_trainable_parameters() (linear head with uniform config).
    if hasattr(model, "get_param_groups"):
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

    # Cosine annealing scheduler: T_max is the number of steps after warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=train_cfg["lr"] * 0.01,
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
    """Return inputs, labels, is_cached."""
    if len(batch_data) == 3:
        images, labels, _paths = batch_data
        inputs = images.to(device, non_blocking=True)
        is_cached = False
    elif len(batch_data) == 2:
        features, labels = batch_data
        inputs = features.to(device, non_blocking=True)
        is_cached = True
    else:
        raise ValueError(
            f"Unexpected batch tuple length: {len(batch_data)}"
        )

    labels = labels.to(device, non_blocking=True)
    return inputs, labels, is_cached


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
) -> tuple:
    """Train for one epoch.

    Returns:
        Tuple of (avg_loss, accuracy, global_step).
    """
    model.train()
    train_cfg = config["train"]
    use_amp = train_cfg.get("amp", False)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [Train]", dynamic_ncols=True)

    for batch_idx, batch_data in enumerate(pbar):
        inputs, labels, is_cached = _unpack_batch(batch_data, device)

        # Warmup
        if global_step < warmup_steps:
            _warmup_lr(optimizer, warmup_steps, global_step)

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type=device.type, enabled=use_amp):
                logits = _forward_inputs(model, inputs, is_cached)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = _forward_inputs(model, inputs, is_cached)
            loss = criterion(logits, labels)
            loss.backward()

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

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
        correct += (preds == labels).sum().item()
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

    return avg_loss, accuracy, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
) -> tuple:
    """Run validation.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.eval()
    use_amp = config["train"].get("amp", False)

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=" " * 16 + "[Val]  ", dynamic_ncols=True)

    for batch_data in pbar:
        inputs, labels, is_cached = _unpack_batch(batch_data, device)

        if use_amp:
            with autocast(device_type=device.type, enabled=use_amp):
                logits = _forward_inputs(model, inputs, is_cached)
                loss = criterion(logits, labels)
        else:
            logits = _forward_inputs(model, inputs, is_cached)
            loss = criterion(logits, labels)

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
    }
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
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    logger.info(
        f"Train loader: {len(train_dataset)} samples, {len(train_loader)} batches"
    )
    logger.info(f"Val loader:   {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Resolve runtime args: explicit CLI > YAML > hard default
    args = resolve_runtime_args(args, config)

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
            pin_memory=True,
            drop_last=False,
            generator=g,
        )

        # Build val loader online (validation always uses online images)
        val_loader = None
        if mode == "dev":
            val_csv = str(Path(split_dir) / "val.csv")
            val_dataset = TrainImageDataset(
                data_root=config["data"]["train_dir"],
                split_csv=val_csv,
                class_to_idx=class_to_idx,
                transform=val_transform,
                return_path=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["eval"]["batch_size"],
                shuffle=False,
                num_workers=config["train"]["num_workers"],
                pin_memory=True,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=g,
            )
            train_logger.info(
                f"Val loader (online): {len(val_dataset)} samples, "
                f"{len(val_loader)} batches"
            )
        elif mode == "confirm":
            if not _check_splits_exist(split_dir):
                train_logger.error(
                    f"Train/val splits not found in {split_dir}.\n"
                    f"Please run: python scripts/split_data.py --config {args.config}"
                )
                raise FileNotFoundError(
                    f"Splits not found in {split_dir}. "
                    f"Run: python scripts/split_data.py --config {args.config}"
                )

            val_csv = str(Path(split_dir) / "val.csv")
            val_dataset = TrainImageDataset(
                data_root=config["data"]["train_dir"],
                split_csv=val_csv,
                class_to_idx=class_to_idx,
                transform=val_transform,
                return_path=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["eval"]["batch_size"],
                shuffle=False,
                num_workers=config["train"]["num_workers"],
                pin_memory=True,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=g,
            )
            train_logger.info(
                f"Val loader (online): {len(val_dataset)} samples, "
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
            pin_memory=True,
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

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = _build_optimizer_and_scheduler(
        model, config, cosine_steps
    )
    scaler = GradScaler(device=device.type, enabled=train_cfg.get("amp", False))

    # Resume if requested
    start_epoch = 1
    global_step = 0
    best_val_acc = 0.0
    dev_best_epoch = None

    if args.resume:
        resume_info = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )
        start_epoch = resume_info["epoch"] + 1
        global_step = resume_info["global_step"]
        best_val_acc = resume_info["best_val_acc"]
        train_logger.info(
            f"Resumed from epoch {resume_info['epoch']}, "
            f"best val acc: {best_val_acc:.4f}"
        )

    # Save config snapshot
    save_dir = ensure_dir(train_cfg["save_dir"])
    save_config_snapshot(config, str(save_dir))

    # Training log CSV
    log_file = Path(config["output"]["log_dir"]) / "train_log.csv"
    log_header = not log_file.exists() or args.resume is None

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
        train_loss, train_acc, global_step = train_one_epoch(
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
        )

        # Validate (skip if no val_loader, e.g., final_fit)
        val_loss = None
        val_acc = None
        if val_loader is not None:
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, config
            )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        if val_acc is not None:
            log_msg = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e} | Time: {format_time(epoch_time)}"
            )
        else:
            log_msg = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"LR: {current_lr:.2e} | Time: {format_time(epoch_time)} [no val]"
            )
        train_logger.info(log_msg)

        # Save to CSV
        with open(log_file, "a") as f:
            if log_header:
                if val_acc is not None:
                    f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,epoch_time\n")
                else:
                    f.write("epoch,train_loss,train_acc,lr,epoch_time\n")
                log_header = False
            if val_acc is not None:
                f.write(
                    f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                    f"{val_loss:.6f},{val_acc:.6f},{current_lr:.8f},{epoch_time:.2f}\n"
                )
            else:
                f.write(
                    f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                    f"{current_lr:.8f},{epoch_time:.2f}\n"
                )

        # Track best epoch and save checkpoints (dev mode only for tracking)
        is_best = False
        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            dev_best_epoch = epoch
            is_best = True
            train_logger.info(f"  >> New best model! Val Acc: {best_val_acc:.4f}")

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
        )

        if is_best:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_val_acc,
                config,
                str(save_dir / "best.pt"),
                extra_meta=extra_meta,
            )

    total_time = time.time() - train_start_time
    train_logger.info("=" * 60)
    train_logger.info(f"Training complete! Total time: {format_time(total_time)}")
    train_logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    train_logger.info(f"Best model saved to: {save_dir / 'best.pt'}")
    train_logger.info(f"Training log saved to: {log_file}")

    # Save eval_results.json for dev/confirm modes
    if mode in ("dev", "confirm"):
        eval_results = {
            "experiment_id": experiment_id,
            "mode": mode,
            "config_path": args.config,
            "split_seed": config["data"].get("split_seed"),
            "train_seed": train_seed,
            "best_val_acc": float(best_val_acc),
            "dev_best_epoch": dev_best_epoch,
            "trained_epochs": epochs,
            "head_type": model.head_type,
            "augmentation_preset": aug_preset,
            "use_cached_features": use_cached,
            "learning_rate": config["train"]["lr"],
            "weight_decay": config["train"]["weight_decay"],
            "batch_size": config["train"]["batch_size"],
            "freeze_clip": config["model"].get("freeze_clip", True),
            "clip_model_name": config["model"]["clip_model_name"],
            "git_commit": get_git_commit(),
        }
        eval_path = save_dir / "eval_results.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        train_logger.info(f"Eval results saved to: {eval_path}")


if __name__ == "__main__":
    main()
