"""
Cache validation and test set logits from a trained checkpoint.

Loads a checkpoint, runs the model (without argmax) on both the validation
split and the full test set, and saves the raw logits alongside metadata.

Usage:
    python scripts/cache_val_test_logits.py \
        --config configs/baseline.yaml \
        --checkpoint outputs/baseline/checkpoints/best.pt \
        --output-dir outputs/phase2/d3_logits \
        --batch-size 256
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
from common.dataset import TestImageDataset, TrainImageDataset
from common.utils import ensure_dir, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def _sha256_hex(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file via streaming reads (1 MiB chunks)."""
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
        description="Cache val and test logits from a trained checkpoint."
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
        "--output-dir",
        type=str,
        default="outputs/phase2/d3_logits",
        help="Directory to save logits and metadata.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference (overrides eval.batch_size in config).",
    )
    return parser.parse_args()


@torch.no_grad()
def cache_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    desc: str = "Caching",
) -> torch.Tensor:
    """Run model inference and collect raw logits for every sample.

    Args:
        model: Trained classifier model.
        loader: DataLoader yielding (images, *_) tuples.  The first element
            must be the image tensor.
        device: torch device.
        use_amp: Whether to use AMP autocast.
        desc: tqdm progress bar description.

    Returns:
        Float32 tensor of shape (N, num_classes) with raw logits,
        where N is the total number of samples in the loader.
    """
    model.eval()
    all_logits = []

    pbar = tqdm(loader, desc=desc, dynamic_ncols=True)
    for batch in pbar:
        images = batch[0].to(device, non_blocking=True)

        if use_amp:
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
        else:
            logits = model(images)

        all_logits.append(logits.cpu().float())

    return torch.cat(all_logits, dim=0)


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Flexible seed access: train_seed -> split_seed -> seed -> 42
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
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logging(str(output_dir), name="cache_logits")

    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Device: {device}")

    # ---- Build model ----
    from experiments.baseline.model import build_model

    model, preprocess = build_model(config, device)

    # Load checkpoint weights
    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Checkpoint best val acc: {checkpoint.get('best_val_acc', 'N/A')}")

    model = model.to(device)

    # Compute streaming SHA-256 of the checkpoint file
    ckpt_sha256 = _sha256_hex(ckpt_path)
    logger.info(f"Checkpoint SHA-256: {ckpt_sha256}")

    # ---- Batch size ----
    batch_size = args.batch_size

    # ---- Validation set ----
    split_dir = Path(config["data"]["split_dir"])
    val_csv = split_dir / "val.csv"

    # Load class mapping from canonical source
    class_to_idx, _ = load_or_generate_mapping(
        metadata_dir=str(split_dir),
        train_dir=config["data"]["train_dir"],
        expected_num_classes=config["model"]["num_classes"],
    )

    val_dataset = TrainImageDataset(
        data_root=config["data"]["train_dir"],
        split_csv=str(val_csv),
        class_to_idx=class_to_idx,
        transform=preprocess,
        return_path=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )

    logger.info(f"Validation set: {len(val_dataset)} samples, {len(val_loader)} batches")

    # Cache val logits
    use_amp = config["train"].get("amp", False)
    val_logits = cache_logits(
        model, val_loader, device, use_amp=use_amp, desc="Val"
    )
    logger.info(f"Val logits shape: {val_logits.shape}")

    # Collect val labels and paths
    val_labels = torch.tensor(val_dataset.labels, dtype=torch.long)
    val_paths = [str(p) for p in val_dataset.samples]

    # Sanity: logits count matches labels count
    assert val_logits.size(0) == val_labels.size(0), (
        f"Val logits count {val_logits.size(0)} != labels count {val_labels.size(0)}"
    )

    # ---- Test set ----
    test_dataset = TestImageDataset(
        data_root=config["data"]["test_dir"],
        transform=preprocess,
    )

    # The dataset's _safe_load_image already returns torch.zeros(3, 224, 224)
    # on failure for both TrainImageDataset and TestImageDataset, so the
    # fallback requirement is satisfied by default.  We log a warning below
    # for any sample where the image is a zero tensor to make the user aware.

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )

    logger.info(f"Test set: {len(test_dataset)} images, {len(test_loader)} batches")

    # Check test count up front
    expected_test_count = 24967
    if len(test_dataset) != expected_test_count:
        logger.warning(
            f"Test dataset has {len(test_dataset)} images, "
            f"expected {expected_test_count}. Continuing anyway."
        )

    # Cache test logits
    test_logits = cache_logits(
        model, test_loader, device, use_amp=use_amp, desc="Test"
    )
    logger.info(f"Test logits shape: {test_logits.shape}")

    # Collect test image filenames
    test_names = [str(p.name) for p in test_dataset.images]

    # Sanity: logits count matches test count
    assert test_logits.size(0) == len(test_names), (
        f"Test logits count {test_logits.size(0)} != images count {len(test_names)}"
    )

    # Warn about any zero-tensor fallback images in the test set
    # We re-run through the dataset to detect fallbacks
    logger.info("Checking for zero-tensor fallback images in test set...")
    fallback_count = 0
    for i, (img_tensor, img_name, img_path) in enumerate(test_dataset):
        if img_tensor.sum().item() == 0.0:
            logger.warning(
                f"Fallback (zero tensor) for test image {i}: "
                f"{img_name} at {img_path}"
            )
            fallback_count += 1

    if fallback_count > 0:
        logger.warning(
            f"Total fallback (zero tensor) test images: {fallback_count} / "
            f"{len(test_dataset)}"
        )
    else:
        logger.info("No zero-tensor fallback images found in test set.")

    # ---- Save outputs ----
    # Validation
    torch.save(val_logits, str(output_dir / "val_logits.pt"))
    torch.save(val_labels, str(output_dir / "val_labels.pt"))
    with open(output_dir / "val_paths.json", "w") as f:
        json.dump(val_paths, f, indent=2)
    logger.info(f"Saved val_logits.pt ({val_logits.shape}), "
                f"val_labels.pt ({val_labels.shape}), val_paths.json")

    # Test
    torch.save(test_logits, str(output_dir / "test_logits.pt"))
    with open(output_dir / "test_names.json", "w") as f:
        json.dump(test_names, f, indent=2)
    logger.info(f"Saved test_logits.pt ({test_logits.shape}), test_names.json")

    # Manifest
    manifest = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha256,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val_acc": float(checkpoint.get("best_val_acc", -1.0)),
        "config": str(Path(args.config).resolve()),
        "output_dir": str(output_dir),
        "batch_size": batch_size,
        "val_count": val_logits.size(0),
        "test_count": test_logits.size(0),
        "num_classes": test_logits.size(1),
        "fallback_test_images": fallback_count,
        "device": str(device),
        "amp": use_amp,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest.json")

    # Summary
    logger.info("=" * 50)
    logger.info("Caching complete.")
    logger.info(f"  Validation logits: {val_logits.shape}")
    logger.info(f"  Test logits:       {test_logits.shape}")
    logger.info(f"  Fallback images:   {fallback_count}")
    logger.info(f"  Output directory:  {output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
