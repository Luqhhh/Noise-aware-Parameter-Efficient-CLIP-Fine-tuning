"""
2-view horizontal-flip TTA evaluation script.

Evaluates a checkpoint on the validation set with and without test-time
augmentation (horizontal flip), reporting per-class metrics and the
prediction change rate between baseline and TTA.

Usage:
    python scripts/evaluate_tta.py \
        --config configs/baseline.yaml \
        --checkpoint outputs/baseline/checkpoints/best.pt \
        --output-dir outputs/baseline/tta_eval
"""

import argparse
import csv
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
from common.utils import load_config, set_seed, setup_logging
from experiments.baseline.model import build_model

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint with 2-view horizontal-flip TTA on the validation set."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--tta",
        type=str,
        default="horizontal_flip",
        choices=["horizontal_flip"],
        help="TTA strategy (default: horizontal_flip).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write results.json and per_class_delta.csv.",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_tta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> dict:
    """Evaluate model with 2-view horizontal-flip TTA on the given dataloader.

    For each batch:
        1. Forward pass with normal preprocess -> logits_orig
        2. Horizontally flip each image, forward pass -> logits_flip
        3. Fused logits = (logits_orig + logits_flip) / 2.0

    Computes metrics for both baseline (view 1 only) and TTA (fused) outputs.

    Args:
        model: The CLIPLinearClassifier model.
        loader: DataLoader for evaluation data.
        device: torch device.
        use_amp: Whether to use AMP autocast.

    Returns:
        Dictionary with baseline and TTA metrics, prediction_change_rate,
        and per-class accuracy arrays.
    """
    model.eval()
    num_classes = getattr(model, "num_classes", None)
    if num_classes is None:
        num_classes = model.classifier.out_features

    correct = 0
    total = 0
    correct_per_class = torch.zeros(num_classes, device=device, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, device=device, dtype=torch.long)

    correct_tta = 0
    correct_per_class_tta = torch.zeros(num_classes, device=device, dtype=torch.long)

    n_pred_changed = 0

    pbar = tqdm(loader, desc="TTA Evaluating", dynamic_ncols=True)

    for images, labels, _paths in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with autocast(device_type=device.type, enabled=use_amp):
                logits_orig = model(images)
                # Horizontal flip: flip along the width dimension (dim=3)
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
        else:
            logits_orig = model(images)
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)

        logits_tta = (logits_orig + logits_flip) / 2.0

        batch_size = images.size(0)
        preds = logits_orig.argmax(dim=1)
        preds_tta = logits_tta.argmax(dim=1)

        correct += (preds == labels).sum().item()
        correct_tta += (preds_tta == labels).sum().item()
        total += batch_size

        # Per-class accumulation (baseline)
        for c in range(num_classes):
            mask = (labels == c)
            n_c = mask.sum().item()
            if n_c > 0:
                total_per_class[c] += n_c
                correct_per_class[c] += (preds[mask] == c).sum().item()

        # Per-class accumulation (TTA)
        for c in range(num_classes):
            mask = (labels == c)
            n_c = mask.sum().item()
            if n_c > 0:
                correct_per_class_tta[c] += (preds_tta[mask] == c).sum().item()

        # Prediction change rate
        n_pred_changed += (preds != preds_tta).sum().item()

        pbar.set_postfix(
            {
                "baseline_acc": f"{correct / total:.4f}",
                "tta_acc": f"{correct_tta / total:.4f}",
            }
        )

    micro_acc = correct / total
    micro_tta_acc = correct_tta / total

    # Per-class and macro metrics (baseline)
    per_class_acc = correct_per_class.float() / total_per_class.float().clamp(min=1)
    macro_acc = per_class_acc.mean().item()
    median_per_class = per_class_acc.median().item()
    k = max(1, num_classes // 10)
    bottom_10_percent_acc = per_class_acc.topk(k, largest=False).values.mean().item()

    # Per-class and macro metrics (TTA)
    per_class_tta_acc = correct_per_class_tta.float() / total_per_class.float().clamp(min=1)
    macro_tta_acc = per_class_tta_acc.mean().item()
    median_tta_per_class = per_class_tta_acc.median().item()
    bottom_10_tta_percent_acc = per_class_tta_acc.topk(k, largest=False).values.mean().item()

    # Prediction change rate
    prediction_change_rate = n_pred_changed / total

    # Per-class delta
    per_class_delta = (per_class_tta_acc - per_class_acc).cpu().tolist()

    return {
        "baseline_micro": micro_acc,
        "baseline_macro": macro_acc,
        "baseline_median": median_per_class,
        "baseline_bottom10": bottom_10_percent_acc,
        "baseline_per_class_acc": per_class_acc.cpu().tolist(),
        "tta_micro": micro_tta_acc,
        "tta_macro": macro_tta_acc,
        "tta_median": median_tta_per_class,
        "tta_bottom10": bottom_10_tta_percent_acc,
        "tta_per_class_acc": per_class_tta_acc.cpu().tolist(),
        "per_class_delta": per_class_delta,
        "prediction_change_rate": prediction_change_rate,
        "total_samples": total,
        "correct_samples_baseline": correct,
        "correct_samples_tta": correct_tta,
        "num_classes": num_classes,
    }


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Flexible seed access
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(output_dir), name="evaluate_tta")

    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"TTA strategy: {args.tta}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Device: {device}")

    # Load class mapping
    class_mapping_path = config["data"].get(
        "class_mapping_path", config["data"]["split_dir"]
    )
    class_to_idx, _ = load_or_generate_mapping(
        metadata_dir=class_mapping_path,
        train_dir=config["data"]["train_dir"],
        expected_num_classes=config["model"]["num_classes"],
    )

    # Build model → apply PEFT → strict-load checkpoint
    from common.model_loader import build_and_load_model

    ckpt_path = Path(args.checkpoint)
    model, preprocess, load_info = build_and_load_model(
        config, args.checkpoint, device,
        build_model_fn=build_model, strict=True,
    )
    logger.info(f"Loaded checkpoint from epoch {load_info.get('checkpoint_epoch', 'unknown')}")
    logger.info(f"Checkpoint best val acc: {load_info.get('parent_best_val_acc', 'N/A')}")

    model = model.to(device)

    # Load validation dataset
    split_dir = Path(config["data"]["split_dir"])
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

    # Evaluate with TTA
    use_amp = config["train"].get("amp", False)
    results = evaluate_tta(model, val_loader, device, use_amp=use_amp)

    # Log results
    logger.info("=" * 50)
    logger.info("TTA Evaluation Results:")
    logger.info(f"  Total samples:                {results['total_samples']}")
    logger.info(f"  Baseline correct:             {results['correct_samples_baseline']}")
    logger.info(f"  TTA correct:                  {results['correct_samples_tta']}")
    logger.info(
        f"  Baseline Top-1 (micro):       {results['baseline_micro']:.4f} ({results['baseline_micro'] * 100:.2f}%)"
    )
    logger.info(
        f"  TTA Top-1 (micro):            {results['tta_micro']:.4f} ({results['tta_micro'] * 100:.2f}%)"
    )
    logger.info(
        f"  Baseline Macro:               {results['baseline_macro']:.4f} ({results['baseline_macro'] * 100:.2f}%)"
    )
    logger.info(
        f"  TTA Macro:                    {results['tta_macro']:.4f} ({results['tta_macro'] * 100:.2f}%)"
    )
    logger.info(
        f"  Baseline Median:              {results['baseline_median']:.4f} ({results['baseline_median'] * 100:.2f}%)"
    )
    logger.info(
        f"  TTA Median:                   {results['tta_median']:.4f} ({results['tta_median'] * 100:.2f}%)"
    )
    logger.info(
        f"  Baseline Bottom-10%%:          {results['baseline_bottom10']:.4f} ({results['baseline_bottom10'] * 100:.2f}%)"
    )
    logger.info(
        f"  TTA Bottom-10%%:               {results['tta_bottom10']:.4f} ({results['tta_bottom10'] * 100:.2f}%)"
    )
    logger.info(
        f"  Prediction Change Rate:       {results['prediction_change_rate']:.4f} ({results['prediction_change_rate'] * 100:.2f}%)"
    )
    logger.info("=" * 50)

    # Write results.json
    summary = {
        "baseline_micro": float(results["baseline_micro"]),
        "baseline_macro": float(results["baseline_macro"]),
        "baseline_median": float(results["baseline_median"]),
        "baseline_bottom10": float(results["baseline_bottom10"]),
        "tta_micro": float(results["tta_micro"]),
        "tta_macro": float(results["tta_macro"]),
        "tta_median": float(results["tta_median"]),
        "tta_bottom10": float(results["tta_bottom10"]),
        "prediction_change_rate": float(results["prediction_change_rate"]),
        "total_samples": results["total_samples"],
        "correct_samples_baseline": results["correct_samples_baseline"],
        "correct_samples_tta": results["correct_samples_tta"],
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": load_info.get("checkpoint_epoch"),
        "checkpoint_best_val_acc": float(load_info.get("parent_best_val_acc", -1.0)),
        "config": str(Path(args.config).resolve()),
        "tta_strategy": args.tta,
    }

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to: {results_json_path}")

    # Write per_class_delta.csv
    per_class_delta_csv = output_dir / "per_class_delta.csv"
    with open(per_class_delta_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "baseline_acc", "tta_acc", "delta"])
        for idx in range(results["num_classes"]):
            writer.writerow([
                idx,
                f"{results['baseline_per_class_acc'][idx]:.6f}",
                f"{results['tta_per_class_acc'][idx]:.6f}",
                f"{results['per_class_delta'][idx]:.6f}",
            ])
    logger.info(f"Per-class delta CSV saved to: {per_class_delta_csv}")


if __name__ == "__main__":
    main()
