#!/usr/bin/env python3
"""
Export frozen CLIP ViT-B/32 feature banks for train and validation splits.

Produces:
    train_feature_bank.pt — dict with features, labels, paths, hashes, metadata
    val_feature_bank.pt   — same, plus flip_features (horizontal mirror)

Usage:
    python tools/export_feature_bank.py \\
        --train-csv outputs/baseline/splits/train.csv \\
        --val-csv outputs/baseline/splits/val.csv \\
        --output-dir outputs/baseline/feature_banks \\
        --batch-size 256 --num-workers 8 --device cuda
"""

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.clip_utils import encode_frozen_clip_features, load_openai_clip
from common.utils import setup_logging

logger = logging.getLogger(__name__)


class _FeatureExportDataset(Dataset):
    """Load and preprocess images for feature bank export.

    Optionally applies horizontal flip (via PIL.ImageOps.mirror) BEFORE
    the CLIP preprocessing transform — never flip post-normalization.
    """

    def __init__(self, image_paths, preprocess, flip=False):
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.flip = flip

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.flip:
                    img = ImageOps.mirror(img)
                return self.preprocess(img)
        except Exception:
            logger.warning("Using a zero image for %s: loading failed", path)
            return torch.zeros(3, 224, 224)


def _sha256_hex(file_path):
    """Compute SHA-256 hex digest of a file, reading in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1_048_576)  # 1 MiB
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _csv_sha256(csv_path):
    """Compute SHA-256 of the entire CSV file."""
    return _sha256_hex(csv_path)


@torch.no_grad()
def _encode_features(clip_model, preprocess, image_paths, device,
                     batch_size, num_workers, flip=False):
    """Encode a list of image paths through frozen CLIP.

    Returns FloatTensor[N, 512] of L2-normalized features.
    """
    dataset = _FeatureExportDataset(image_paths, preprocess, flip=flip)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    all_features = []
    for images in loader:
        images = images.to(device, non_blocking=(device.type == "cuda"))
        features = encode_frozen_clip_features(
            clip_model, images, device, use_amp=False
        )
        all_features.append(features.cpu())
    return torch.cat(all_features, dim=0)


def _verify_features(features, label="features"):
    """Verify feature tensor integrity and normalization.

    Checks:
        - 2D shape
        - feature dim == 512
        - no NaN or Inf values
        - all L2 norms within 1e-5 of 1.0
    """
    assert features.ndim == 2, \
        f"{label}: expected 2D tensor, got shape {features.shape}"
    assert features.shape[1] == 512, \
        f"{label}: feature dim {features.shape[1]} != 512"
    assert torch.isfinite(features).all().item(), \
        f"{label}: contains NaN or Inf values"
    norms = features.norm(dim=1)
    max_dev = (norms - 1.0).abs().max().item()
    assert max_dev < 1e-5, \
        f"{label}: max norm deviation {max_dev:.2e} exceeds 1e-5"
    logger.info(
        "%s: shape=%s, norms within %.2e of 1.0, finite=True",
        label, list(features.shape), max_dev,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export frozen CLIP ViT-B/32 feature banks"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to experiment config YAML (optional, for logging setup)",
    )
    parser.add_argument(
        "--train-csv",
        required=True,
        help="Path to train split CSV (columns: image_path, label, class_name)",
    )
    parser.add_argument(
        "--val-csv",
        required=True,
        help="Path to val split CSV (columns: image_path, label, class_name)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save train_feature_bank.pt and val_feature_bank.pt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="DataLoader batch size (default: 256)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader num_workers (default: 8)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Torch device string, e.g. "cuda", "cpu" (default: cuda)',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging — use config's log_dir if available, else stdout only
    if args.config:
        from common.utils import load_config
        config = load_config(args.config)
        log_dir = config.get("output", {}).get("log_dir", None)
        if log_dir:
            setup_logging(log_dir, name="export_feature_bank")
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    logger.info("=== Feature Bank Export ===")
    logger.info("Train CSV: %s", args.train_csv)
    logger.info("Val CSV:   %s", args.val_csv)
    logger.info("Output:    %s", args.output_dir)
    logger.info("Batch size: %d, num_workers: %d, device: %s",
                args.batch_size, args.num_workers, args.device)

    # Resolve paths
    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load CLIP model ---
    logger.info("Loading CLIP ViT-B/32...")
    device = torch.device(args.device)
    clip_model, preprocess = load_openai_clip(device)
    clip_model.eval()
    logger.info("CLIP model loaded.")

    # --- Step 2: Read train CSV ---
    logger.info("Reading train CSV: %s", train_csv)
    train_df = pd.read_csv(train_csv)
    # Resolve paths: CWD-relative (if path is absolute, Path.cwd() / path
    # yields the absolute path since Python's pathlib discards the first
    # operand when the second is absolute).
    train_paths = [str(Path.cwd() / p) for p in train_df["image_path"].tolist()]
    train_labels = torch.tensor(train_df["label"].tolist(), dtype=torch.long)
    train_class_names = train_df["class_name"].tolist()
    logger.info("Train samples: %d", len(train_paths))

    # --- Step 3: SHA-256 hash train images ---
    logger.info("Hashing %d train images (SHA-256, 1 MiB chunks)...",
                len(train_paths))
    t0 = time.time()
    train_sha256 = [_sha256_hex(p) for p in train_paths]
    logger.info("Hashing done in %.1fs", time.time() - t0)

    # --- Step 4: Source CSV hash ---
    train_csv_sha256 = _csv_sha256(train_csv)

    # --- Step 5: Encode train features ---
    logger.info("Encoding train features (batch_size=%d, num_workers=%d)...",
                args.batch_size, args.num_workers)
    t0 = time.time()
    train_features = _encode_features(
        clip_model, preprocess, train_paths, device,
        args.batch_size, args.num_workers, flip=False,
    )
    logger.info("Train encoding done in %.1fs", time.time() - t0)

    # --- Step 6: Save train feature bank ---
    train_bank = {
        "features": train_features,
        "labels": train_labels,
        "paths": train_paths,
        "image_sha256": train_sha256,
        "class_names": train_class_names,
        "source_csv_sha256": train_csv_sha256,
        "clip_model_name": "ViT-B/32",
        "normalized": True,
    }
    train_out = output_dir / "train_feature_bank.pt"
    torch.save(train_bank, train_out)
    logger.info("Saved train feature bank: %s", train_out)

    # --- Step 7: Read val CSV ---
    logger.info("Reading val CSV: %s", val_csv)
    val_df = pd.read_csv(val_csv)
    val_paths = [str(Path.cwd() / p) for p in val_df["image_path"].tolist()]
    val_labels = torch.tensor(val_df["label"].tolist(), dtype=torch.long)
    val_class_names = val_df["class_name"].tolist()
    logger.info("Val samples: %d", len(val_paths))

    # --- Step 8: SHA-256 hash val images ---
    logger.info("Hashing %d val images (SHA-256, 1 MiB chunks)...",
                len(val_paths))
    t0 = time.time()
    val_sha256 = [_sha256_hex(p) for p in val_paths]
    logger.info("Hashing done in %.1fs", time.time() - t0)

    # --- Step 9: Source CSV hash ---
    val_csv_sha256 = _csv_sha256(val_csv)

    # --- Step 10: Encode val features (normal) ---
    logger.info("Encoding val features (normal)...")
    t0 = time.time()
    val_features = _encode_features(
        clip_model, preprocess, val_paths, device,
        args.batch_size, args.num_workers, flip=False,
    )
    logger.info("Val encoding done in %.1fs", time.time() - t0)

    # --- Step 11: Encode val features (horizontal flip) ---
    logger.info("Encoding val features (horizontal flip)...")
    t0 = time.time()
    val_flip_features = _encode_features(
        clip_model, preprocess, val_paths, device,
        args.batch_size, args.num_workers, flip=True,
    )
    logger.info("Val flip encoding done in %.1fs", time.time() - t0)

    # --- Step 12: Save val feature bank ---
    val_bank = {
        "features": val_features,
        "flip_features": val_flip_features,
        "labels": val_labels,
        "paths": val_paths,
        "image_sha256": val_sha256,
        "class_names": val_class_names,
        "source_csv_sha256": val_csv_sha256,
        "clip_model_name": "ViT-B/32",
        "normalized": True,
    }
    val_out = output_dir / "val_feature_bank.pt"
    torch.save(val_bank, val_out)
    logger.info("Saved val feature bank: %s", val_out)

    # --- Step 13: Verification ---
    logger.info("=== Verification ===")
    _verify_features(train_features, "train_features")
    _verify_features(val_features, "val_features")
    _verify_features(val_flip_features, "val_flip_features")
    logger.info("Train samples: %d", len(train_paths))
    logger.info("Val samples:   %d", len(val_paths))
    logger.info("All verifications passed.")
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
