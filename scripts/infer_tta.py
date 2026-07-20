#!/usr/bin/env python3
"""TTA inference on test set — generates pred_raw.csv with fused logits.

Usage:
    PYTHONPATH=. python scripts/infer_tta.py \
        --config configs/ref.yaml \
        --checkpoint outputs/ref/seed42/checkpoints/best.pt \
        --tta horizontal_flip \
        --output-dir outputs/phase2/ta1_tta_flip
"""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.dataset import TestImageDataset
from common.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def infer_tta(model, preprocess, test_dir, batch_size, num_workers, device, use_amp, idx_to_class, output_dir):
    """Run TTA inference on test set, save pred_raw.csv and pred_results.csv."""
    dataset = TestImageDataset(data_root=test_dir, transform=preprocess)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    logger.info(f"Test set: {len(dataset)} images, {len(loader)} batches")

    results = []
    fallback = 0

    for images, names, paths in tqdm(loader, desc="TTA Test"):
        # Handle corrupted images via zero-tensor fallback
        is_fallback = images.abs().sum(dim=[1, 2, 3]) == 0
        fallback += is_fallback.sum().item()

        images = images.to(device, non_blocking=True)
        images_flip = torch.flip(images, dims=[3])

        # Get model dtype for casting
        try:
            target_dtype = model.visual.conv1.weight.dtype
        except AttributeError:
            target_dtype = torch.float32

        images = images.to(target_dtype)
        images_flip = images_flip.to(target_dtype)

        if use_amp:
            with torch.amp.autocast(device_type=device.type, enabled=True):
                logits_orig = model(images)
                logits_flip = model(images_flip)
        else:
            logits_orig = model(images)
            logits_flip = model(images_flip)

        logits_fused = (logits_orig + logits_flip) / 2.0
        preds = logits_fused.argmax(dim=1).cpu().tolist()

        for i, name in enumerate(names):
            pred_idx = preds[i]
            pred_label = str(idx_to_class[str(pred_idx)]).zfill(4)
            results.append([name, pred_idx, pred_label])

    logger.info(f"Fallback images (zero tensor): {fallback}")
    logger.info(f"Total predictions: {len(results)}")

    # Save pred_raw.csv (with header: image_name, pred_idx, pred_label)
    raw_path = output_dir / "pred_raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_idx", "pred_label"])
        writer.writerows(results)
    logger.info(f"Saved pred_raw.csv: {len(results)} predictions")

    # Save pred_results.csv (no header, format: image_name.jpg, 0001)
    results_path = output_dir / "pred_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in results:
            writer.writerow([row[0], f" {row[2]}"])
    logger.info(f"Saved pred_results.csv")

    return raw_path, results_path, fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tta", default="horizontal_flip")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    setup_logging(str(output_dir), name="infer_tta")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Build model → apply PEFT → strict-load checkpoint
    from experiments.baseline.model import build_model
    from common.model_loader import build_and_load_model

    model, preprocess, load_info = build_and_load_model(
        config, args.checkpoint, device,
        build_model_fn=build_model, strict=True,
    )
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint epoch {load_info.get('checkpoint_epoch')}")

    # Load checkpoint metadata for idx_to_class (lightweight re-read)
    _ckpt_meta = torch.load(args.checkpoint, map_location=device)
    idx_to_class = _ckpt_meta.get("idx_to_class")
    if idx_to_class is None:
        import json as _json
        with open(Path(config["data"]["split_dir"]) / "idx_to_class.json") as f:
            idx_to_class = _json.load(f)
    logger.info(f"Checkpoint epoch: {_ckpt_meta.get('epoch')}")

    use_amp = config["train"].get("amp", False)
    test_dir = config["data"]["test_dir"]

    raw_path, results_path, fallback = infer_tta(
        model, preprocess, test_dir, args.batch_size,
        config["train"].get("num_workers", 8), device, use_amp,
        idx_to_class, output_dir,
    )

    # Generate submission.zip
    import zipfile
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(results_path, arcname="pred_results.csv")
    logger.info(f"Saved submission.zip")

    # Validate
    from common.submission import validate_submission_coverage
    try:
        validate_submission_coverage(test_dir, str(results_path))
        logger.info("Submission validation PASSED")
    except ValueError as e:
        logger.error(f"Submission validation FAILED: {e}")

    # Build manifest
    ckpt_sha = hashlib.sha256(open(args.checkpoint, "rb").read()).hexdigest()
    manifest = {
        "experiment_id": "TA1_TTA_FLIP",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": _ckpt_meta.get("epoch"),
        "tta_strategy": args.tta,
        "num_predictions": len(list(open(results_path))),
        "fallback_images": fallback,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest.json")


if __name__ == "__main__":
    main()
