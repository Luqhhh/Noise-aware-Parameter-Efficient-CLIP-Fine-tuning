"""
Inference script: generate predictions on the test set.

Loads a trained checkpoint, runs inference on all test images, and produces
a raw prediction file (pred_raw.csv) with columns: image_name, pred_idx, pred_label.

The idx_to_class mapping is loaded from the checkpoint metadata (not from split_dir),
ensuring the mapping used at inference time matches what was used during training.

Usage:
    python -m experiments.baseline.infer --config configs/baseline.yaml \
        --ckpt outputs/baseline/checkpoints/best.pt
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

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on test set using a trained classifier."
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
        model: Trained CLIPLinearClassifier.
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

        def _build_fn(cfg, dev):
            return build_cosine_model(cfg, dev)
    else:
        from .model import build_model as _linear_build

        def _build_fn(cfg, dev):
            return _linear_build(cfg, dev)

    # Unified model build → PEFT → strict checkpoint load
    from common.model_loader import build_and_load_model

    model, preprocess, load_info = build_and_load_model(
        config, args.ckpt, device, build_model_fn=_build_fn, strict=True,
    )
    logger.info(f"Loaded checkpoint from epoch {load_info.get('checkpoint_epoch', 'unknown')}")
    logger.info(f"Checkpoint best val acc: {load_info.get('parent_best_val_acc', 'N/A')}")

    model = model.to(device)

    # Load checkpoint metadata for idx_to_class
    checkpoint = torch.load(args.ckpt, map_location=device)
    idx_to_class = checkpoint.get("idx_to_class")
    if idx_to_class is None:
        logger.warning(
            "Checkpoint does not contain idx_to_class metadata. "
            "Falling back to split_dir."
        )
        split_dir = Path(config["data"]["split_dir"])
        with open(split_dir / "idx_to_class.json", "r") as f:
            idx_to_class = json.load(f)
    else:
        logger.info("Loaded idx_to_class mapping from checkpoint metadata.")

    # Verify the loaded mapping
    if not isinstance(idx_to_class, dict):
        raise ValueError(
            f"idx_to_class must be a dict, got {type(idx_to_class)}"
        )
    logger.info(f"idx_to_class mapping has {len(idx_to_class)} entries.")

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
