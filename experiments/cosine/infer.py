"""
Inference script: generate predictions on the test set using a CosineClassifier.

Loads a trained checkpoint, runs inference on all test images, and produces
a raw prediction file (pred_raw.csv) with columns: image_name, pred_idx, pred_label.

Usage:
    python -m experiments.cosine.infer --config configs/cosine.yaml \
        --ckpt outputs/cosine/checkpoints/best.pt
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.dataset import TestImageDataset
from common.utils import ensure_dir, load_config, set_seed, setup_logging

from .model import build_cosine_model

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on test set using a trained CosineClassifier."
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
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    idx_to_class: dict = None,
) -> list:
    """Run inference on the test set.

    Args:
        model: Trained CosineClassifier.
        loader: DataLoader for test data.
        device: torch device.
        use_amp: Whether to use AMP autocast.
        idx_to_class: Mapping from index string to class name string.

    Returns:
        List of dicts with keys: image_name, pred_idx, pred_label.
    """
    model.eval()
    predictions = []

    pbar = tqdm(loader, desc="Inference", dynamic_ncols=True)

    for images, image_names, _paths in pbar:
        images = images.to(device, non_blocking=True)

        if use_amp:
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
        else:
            logits = model(images)

        pred_indices = logits.argmax(dim=1).cpu().numpy()

        for img_name, pred_idx in zip(image_names, pred_indices):
            predictions.append(
                {
                    "image_name": img_name,
                    "pred_idx": int(pred_idx),
                    "pred_label": str(idx_to_class[str(int(pred_idx))]).zfill(4),
                }
            )

    return predictions


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
    log_dir = ensure_dir(config["output"]["log_dir"])
    logger = setup_logging(str(log_dir), name="infer")

    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Device: {device}")

    # Load idx_to_class mapping for pred_label formatting
    split_dir = Path(config["data"]["split_dir"])
    with open(split_dir / "idx_to_class.json", "r") as f:
        idx_to_class = json.load(f)

    # Build model
    model, preprocess = build_cosine_model(config, device)

    # Load checkpoint (only model weights needed for inference)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Checkpoint best val acc: {checkpoint.get('best_val_acc', 'N/A')}")

    model = model.to(device)

    # Build test dataset
    test_dataset = TestImageDataset(
        data_root=config["data"]["test_dir"],
        transform=preprocess,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )

    logger.info(f"Test set: {len(test_dataset)} images, {len(test_loader)} batches")

    # Run inference
    predictions = run_inference(
        model,
        test_loader,
        device,
        use_amp=config["train"].get("amp", False),
        idx_to_class=idx_to_class,
    )

    # Save raw predictions
    submission_dir = ensure_dir(config["output"]["submission_dir"])
    raw_csv_path = submission_dir / "pred_raw.csv"

    df = pd.DataFrame(predictions)
    df.to_csv(raw_csv_path, index=False)
    logger.info(f"Raw predictions saved to: {raw_csv_path}")
    logger.info(f"Total predictions: {len(predictions)}")

    # Quick sanity check
    unique_preds = df["pred_idx"].nunique()
    logger.info(
        f"Unique predicted classes: {unique_preds} / {config['model']['num_classes']}"
    )

    # Check that all pred_labels are 4-digit strings
    label_lengths = df["pred_label"].str.len().value_counts().to_dict()
    logger.info(f"Prediction label lengths: {label_lengths}")

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
