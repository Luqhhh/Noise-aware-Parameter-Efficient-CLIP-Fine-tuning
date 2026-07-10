"""
Training script for augmentation experiments (E2/E3/E4).

Reuses the CLIPLinearClassifier from experiments.baseline.model but applies
augmentation presets (A1/A2/A3) during training. Uses online encoding
(use_cached_features=False, enforced) because random augmentations must be
recomputed each epoch rather than pre-cached.

Usage:
    python -m experiments.augmentation.train --config configs/augmentation.yaml --preset a1
    python -m experiments.augmentation.train --config configs/augmentation.yaml --preset a2
    python -m experiments.augmentation.train --config configs/augmentation.yaml --preset a3
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.dataset import TrainImageDataset
from common.utils import (count_parameters, ensure_dir, format_time,
                          load_config, save_config_snapshot, set_seed,
                          setup_logging)

from experiments.baseline.model import build_model

logger = logging.getLogger(__name__)

VALID_PRESETS = {"a1", "a2", "a3"}

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CLIP Linear Classifier with Augmentation"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        choices=sorted(VALID_PRESETS),
        help=f"Augmentation preset. One of: {sorted(VALID_PRESETS)}.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., outputs/checkpoints/last.pt).",
    )
    return parser.parse_args()


def build_augmentation_transform(preset: str, image_size: int = 224) -> T.Compose:
    """Build a training transform for the given augmentation preset.

    Each preset composes CLIP-compatible normalization with an increasing
    level of stochastic augmentation:
      A1: RandomResizedCrop + RandomHorizontalFlip
      A2: A1 + ColorJitter
      A3: A2 + RandomErasing (applied AFTER Normalize)

    Args:
        preset: One of "a1", "a2", "a3".
        image_size: Target spatial size (default 224 for CLIP ViT-B/32).

    Returns:
        Composed torchvision transform.

    Raises:
        ValueError: If preset is not in VALID_PRESETS.
    """
    if preset not in VALID_PRESETS:
        raise ValueError(
            f"Unknown augmentation preset: {preset!r}. "
            f"Valid presets: {sorted(VALID_PRESETS)}"
        )

    if preset == "a1":
        # A1: Light — RandomResizedCrop + RandomHorizontalFlip
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    elif preset == "a2":
        # A2: Medium — A1 + ColorJitter
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    elif preset == "a3":
        # A3: Heavy — A2 + RandomErasing (after Normalize)
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])


def _check_splits_exist(split_dir: str) -> bool:
    """Check if train/val split files exist."""
    split_dir = Path(split_dir)
    required = ["train.csv", "val.csv", "class_to_idx.json", "idx_to_class.json"]
    return all((split_dir / f).exists() for f in required)


def _build_dataloaders(
    config: Dict[str, Any], train_transform: callable, val_transform: callable,
    class_to_idx: Dict[str, int], train_seed: int,
) -> tuple:
    """Build train and validation DataLoaders.

    Args:
        config: Configuration dictionary.
        train_transform: Augmented transform for training data.
        val_transform: Deterministic transform for validation data.
        class_to_idx: Class name to index mapping.
        train_seed: Seed for deterministic DataLoader shuffling.

    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset).
    """
    train_cfg = config["train"]
    data_cfg = config["data"]

    split_dir = data_cfg["split_dir"]
    train_csv = str(Path(split_dir) / "train.csv")
    val_csv = str(Path(split_dir) / "val.csv")

    train_dataset = TrainImageDataset(
        data_root=data_cfg["train_dir"],
        split_csv=train_csv,
        class_to_idx=class_to_idx,
        transform=train_transform,
        return_path=True,
    )

    val_dataset = TrainImageDataset(
        data_root=data_cfg["train_dir"],
        split_csv=val_csv,
        class_to_idx=class_to_idx,
        transform=val_transform,
        return_path=True,
    )

    # Deterministic DataLoader with drop_last=False
    g = torch.Generator()
    g.manual_seed(train_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        f"Train loader: {len(train_dataset)} samples, {len(train_loader)} batches"
    )
    logger.info(f"Val loader:   {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader, train_dataset, val_dataset


def _build_optimizer_and_scheduler(
    model: nn.Module, config: Dict[str, Any], num_training_steps: int
) -> tuple:
    """Build optimizer and learning rate scheduler."""
    train_cfg = config["train"]

    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
        eta_min=train_cfg["lr"] * 0.01,
    )

    return optimizer, scheduler


def _warmup_lr(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    current_step: int,
    base_lr: float,
) -> None:
    """Apply linear warmup to learning rate."""
    if current_step < warmup_steps:
        lr_scale = (current_step + 1) / warmup_steps
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr * lr_scale


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
    base_lr = train_cfg["lr"]

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [Train]", dynamic_ncols=True)

    for batch_idx, (images, labels, _paths) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Warmup
        if global_step < warmup_steps:
            _warmup_lr(optimizer, warmup_steps, global_step, base_lr)

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        # Only step scheduler after warmup
        if global_step >= warmup_steps:
            scheduler.step()

        # Statistics
        batch_size = images.size(0)
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

    for images, labels, _paths in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = images.size(0)
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
    preset: str,
) -> None:
    """Save a training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_acc": best_val_acc,
        "augmentation_preset": preset,
        "head_type": "linear",
        "config": config,
    }
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
        Dictionary with checkpoint metadata.
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


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Enforce: augmentation experiments do NOT use cached features
    model_cfg = config.get("model", {})
    if model_cfg.get("use_cached_features", False):
        raise ValueError(
            f"Augmentation experiments require use_cached_features=False, "
            f"but config specifies use_cached_features=True. "
            f"Random augmentations (preset={args.preset}) must be "
            f"recomputed each epoch and cannot use cached features."
        )

    # Set random seed
    seed = config["data"]["seed"]
    set_seed(seed)

    # Setup logging
    log_dir = ensure_dir(config["output"]["log_dir"])
    train_logger = setup_logging(str(log_dir), name="train")

    # Setup device
    device = torch.device(
        config["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    train_logger.info(f"Using device: {device}")
    train_logger.info(f"Configuration: {args.config}")
    train_logger.info(f"Augmentation preset: {args.preset}")
    train_logger.info(f"Random seed: {seed}")
    train_logger.info("use_cached_features: False (enforced for augmentation)")

    # Check that splits exist
    split_dir = config["data"]["split_dir"]
    if not _check_splits_exist(split_dir):
        train_logger.error(
            f"Train/val splits not found in {split_dir}.\n"
            f"Please run: python scripts/split_data.py --config {args.config}"
        )
        raise FileNotFoundError(
            f"Splits not found in {split_dir}. "
            f"Run: python scripts/split_data.py --config {args.config}"
        )

    # Load class mapping
    with open(Path(split_dir) / "class_to_idx.json", "r") as f:
        class_to_idx = json.load(f)

    # Build model and get CLIP preprocess for validation
    model, preprocess = build_model(config, device)

    total_params, trainable_params = count_parameters(model)
    train_logger.info(f"Total parameters:     {total_params:,}")
    train_logger.info(f"Trainable parameters: {trainable_params:,}")

    # Build transforms
    image_size = config["train"].get("image_size", 224)
    train_transform = build_augmentation_transform(args.preset, image_size)
    val_transform = preprocess  # Deterministic CLIP preprocess for validation

    train_logger.info(f"Train transform: {args.preset} augmentation")
    train_logger.info(f"Val transform:   CLIP deterministic preprocess")

    # Build dataloaders with augmentation transforms
    train_seed = config["data"].get("train_seed", config["data"].get("seed", 42))
    train_loader, val_loader, train_dataset, val_dataset = _build_dataloaders(
        config, train_transform, val_transform, class_to_idx, train_seed,
    )

    # Training setup
    train_cfg = config["train"]
    epochs = train_cfg["epochs"]
    num_training_steps = epochs * len(train_loader)
    warmup_steps = train_cfg["warmup_epochs"] * len(train_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = _build_optimizer_and_scheduler(
        model, config, num_training_steps
    )
    scaler = GradScaler(device=device.type, enabled=train_cfg.get("amp", False))

    # Resume if requested
    start_epoch = 1
    global_step = 0
    best_val_acc = 0.0

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
        f"Warmup steps: {warmup_steps}, Total steps: {num_training_steps}"
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

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, config)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        log_msg = (
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | Time: {format_time(epoch_time)}"
        )
        train_logger.info(log_msg)

        # Save to CSV
        with open(log_file, "a") as f:
            if log_header:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,epoch_time\n")
                log_header = False
            f.write(
                f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                f"{val_loss:.6f},{val_acc:.6f},{current_lr:.8f},{epoch_time:.2f}\n"
            )

        # Save checkpoints
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            train_logger.info(f"  >> New best model! Val Acc: {best_val_acc:.4f}")

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
            args.preset,
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
                args.preset,
            )

    total_time = time.time() - train_start_time
    train_logger.info("=" * 60)
    train_logger.info(f"Training complete! Total time: {format_time(total_time)}")
    train_logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    train_logger.info(f"Best model saved to: {save_dir / 'best.pt'}")
    train_logger.info(f"Training log saved to: {log_file}")


if __name__ == "__main__":
    main()
