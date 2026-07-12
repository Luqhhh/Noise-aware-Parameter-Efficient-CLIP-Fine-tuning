"""
Evaluation script: evaluate a trained checkpoint on the validation set.

Loads the class mapping from common.class_mapping (canonical) and saves
evaluation results as JSON alongside the checkpoint.

Usage:
    python -m experiments.baseline.evaluate --config configs/baseline.yaml \
        --ckpt outputs/baseline/checkpoints/best.pt
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.class_mapping import load_or_generate_mapping
from common.dataset import TrainImageDataset
from common.losses import build_loss, reduce_loss
from common.utils import load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def _sha256_hex(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (1 MiB chunks by default)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the validation set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--head-type",
        type=str,
        default=None,
        choices=["linear", "cosine"],
        help="Classifier head type: linear (default) or cosine. "
             "Overrides experiment.head_type in config if provided.",
    )
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
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write evaluation results JSON. "
             "Defaults to reeval_best.json alongside the checkpoint.",
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
        model: The CLIPLinearClassifier model.
        loader: DataLoader for evaluation data.
        criterion: Loss function.
        device: torch device.
        use_amp: Whether to use AMP autocast.

    Returns:
        Dictionary with keys: loss, accuracy, macro_accuracy,
        median_per_class_accuracy, bottom_10_percent_accuracy,
        micro_macro_gap, per_class_accuracy, total_samples, correct_samples.
    """
    model.eval()
    num_classes = getattr(model, 'num_classes', None)
    if num_classes is None:
        num_classes = model.classifier.out_features

    total_loss = 0.0
    correct = 0
    total = 0
    correct_per_class = torch.zeros(num_classes, device=device, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, device=device, dtype=torch.long)

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

        loss = reduce_loss(loss)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        # Per-class accumulation
        for c in range(num_classes):
            mask = (labels == c)
            n_c = mask.sum().item()
            if n_c > 0:
                total_per_class[c] += n_c
                correct_per_class[c] += (preds[mask] == c).sum().item()

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / total:.4f}",
            }
        )

    avg_loss = total_loss / total
    micro_acc = correct / total

    # Per-class and macro metrics
    per_class_acc = correct_per_class.float() / total_per_class.float().clamp(min=1)
    macro_acc = per_class_acc.mean().item()
    median_per_class = per_class_acc.median().item()

    # Bottom-10%: mean of worst 10% classes (50 out of 500)
    k = max(1, num_classes // 10)
    bottom_10_percent_acc = per_class_acc.topk(k, largest=False).values.mean().item()

    micro_macro_gap = micro_acc - macro_acc

    return {
        "loss": avg_loss,
        "accuracy": micro_acc,
        "macro_accuracy": macro_acc,
        "per_class_accuracy": per_class_acc.cpu().tolist(),
        "median_per_class_accuracy": median_per_class,
        "bottom_10_percent_accuracy": bottom_10_percent_acc,
        "micro_macro_gap": micro_macro_gap,
        "total_samples": total,
        "correct_samples": correct,
    }


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Flexible seed access: try train_seed → split_seed → seed → 42
    seed = config["data"].get(
        "train_seed",
        config["data"].get("split_seed", config["data"].get("seed", 42)),
    )
    set_seed(seed)

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
    logger.info(f"Seed: {seed}")
    logger.info(f"Device: {device}")

    # Load class mapping from canonical source (common.class_mapping)
    class_mapping_path = config["data"].get(
        "class_mapping_path", config["data"]["split_dir"]
    )
    class_to_idx, _ = load_or_generate_mapping(
        metadata_dir=class_mapping_path,
        train_dir=config["data"]["train_dir"],
        expected_num_classes=config["model"]["num_classes"],
    )

    # Determine head type: CLI arg overrides config
    head_type = args.head_type
    if head_type is None:
        head_type = config.get("experiment", {}).get("head_type", "linear")
    logger.info(f"Head type: {head_type}")

    # Build model based on head type
    if head_type == "cosine":
        from experiments.cosine.model import build_cosine_model

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

    # Load checkpoint
    ckpt_path = Path(args.ckpt)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Checkpoint best val acc: {checkpoint.get('best_val_acc', 'N/A')}")

    model = model.to(device)

    # Compute checkpoint SHA-256 (streaming, never loads whole file into memory)
    ckpt_sha256 = _sha256_hex(ckpt_path)
    logger.info(f"Checkpoint SHA-256: {ckpt_sha256}")

    # Load validation dataset
    split_dir = Path(config["data"]["split_dir"])
    val_csv = split_dir / "val.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation split not found: {val_csv}")

    # Compute val CSV SHA-256
    val_csv_sha256 = _sha256_hex(val_csv)
    logger.info(f"Val CSV SHA-256: {val_csv_sha256}")

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
    criterion = build_loss(config)
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
    logger.info(f"  Total samples:              {results['total_samples']}")
    logger.info(f"  Correct samples:            {results['correct_samples']}")
    logger.info(
        f"  Top-1 Accuracy (micro):     {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)"
    )
    logger.info(
        f"  Macro Accuracy:             {results['macro_accuracy']:.4f} ({results['macro_accuracy']*100:.2f}%)"
    )
    logger.info(
        f"  Median Per-Class Accuracy:  {results['median_per_class_accuracy']:.4f} ({results['median_per_class_accuracy']*100:.2f}%)"
    )
    logger.info(
        f"  Bottom-10%% Accuracy:        {results['bottom_10_percent_accuracy']:.4f} ({results['bottom_10_percent_accuracy']*100:.2f}%)"
    )
    logger.info(
        f"  Micro-Macro Gap:            {results['micro_macro_gap']:+.4f} ({results['micro_macro_gap']*100:+.2f}%)"
    )
    logger.info(f"  Loss:                       {results['loss']:.4f}")
    logger.info("=" * 50)

    # Determine output path: --output-json > default reeval_best.json
    if args.output_json:
        eval_results_path = Path(args.output_json)
    else:
        eval_results_path = ckpt_path.parent / "reeval_best.json"

    eval_results = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha256,
        "val_csv": str(val_csv),
        "val_csv_sha256": val_csv_sha256,
        "config": str(Path(args.config).resolve()),
        "micro_accuracy": float(results["accuracy"]),
        "macro_accuracy": float(results["macro_accuracy"]),
        "median_per_class_accuracy": float(results["median_per_class_accuracy"]),
        "bottom_10_percent_accuracy": float(results["bottom_10_percent_accuracy"]),
        "micro_macro_gap": float(results["micro_macro_gap"]),
        "loss": float(results["loss"]),
        "total_samples": results["total_samples"],
        "correct_samples": results["correct_samples"],
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val_acc": float(checkpoint.get("best_val_acc", -1.0)),
    }
    eval_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Eval results saved to: {eval_results_path}")

    return results


if __name__ == "__main__":
    main()
