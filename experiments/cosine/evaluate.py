"""
Evaluation script: evaluate a trained CosineClassifier checkpoint on the validation set.

Usage:
    python -m experiments.cosine.evaluate --config configs/cosine.yaml \
        --ckpt outputs/cosine/checkpoints/best.pt
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.dataset import TrainImageDataset
from common.utils import load_config, set_seed, setup_logging

from .model import build_cosine_model

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a CosineClassifier checkpoint on the validation set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cosine.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file).",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> dict:
    """Evaluate model on the given dataloader.

    Args:
        model: The CosineClassifier model.
        loader: DataLoader for evaluation data.
        criterion: Loss function.
        device: torch device.
        use_amp: Whether to use AMP autocast.

    Returns:
        Dictionary with keys: loss, accuracy, total_samples, correct_samples.
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Evaluating", dynamic_ncols=True)

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

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "total_samples": total,
        "correct_samples": correct,
    }


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    set_seed(config["data"]["seed"])

    # Device
    device = torch.device(
        config["train"]["device"] if torch.cuda.is_available() else "cpu"
    )

    # Setup logging
    log_dir = Path(config["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_dir), name="evaluate")

    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Device: {device}")

    # Load class mapping
    split_dir = Path(config["data"]["split_dir"])
    with open(split_dir / "class_to_idx.json", "r") as f:
        class_to_idx = json.load(f)

    # Build model
    model, preprocess = build_cosine_model(config, device)

    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Checkpoint best val acc: {checkpoint.get('best_val_acc', 'N/A')}")

    model = model.to(device)

    # Load validation dataset
    val_csv = split_dir / "val.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation split not found: {val_csv}")

    val_dataset = TrainImageDataset(
        data_root=config["data"]["train_dir"],
        split_csv=str(val_csv),
        class_to_idx=class_to_idx,
        transform=preprocess,
        return_path=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )

    logger.info(
        f"Validation set: {len(val_dataset)} samples, {len(val_loader)} batches"
    )

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    results = evaluate(
        model,
        val_loader,
        criterion,
        device,
        use_amp=config["train"].get("amp", False),
    )

    # Print results
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"  Total samples:    {results['total_samples']}")
    logger.info(f"  Correct samples:  {results['correct_samples']}")
    logger.info(
        f"  Top-1 Accuracy:   {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)"
    )
    logger.info(f"  Loss:             {results['loss']:.4f}")
    logger.info("=" * 50)

    return results


if __name__ == "__main__":
    main()
