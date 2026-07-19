#!/usr/bin/env python3
"""A0 smoke acceptance: 20 batches of real training with manifest audit.

Loads a real A0 config, overrides manifest/split with small fixtures,
runs the runtime manifest audit, then trains 20 batches through the
full pipeline (CLIP encode → MixUp → weighted loss → backward → step).

Usage:
    python scripts/run_a0_acceptance.py \
        --config configs/nr_ctrl_oof_zero_0001_fixed.yaml \
        --max-batches 20 \
        --output-log logs/a0_real_20batch.log
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--output-log", default="logs/a0_real_20batch.log")
    parser.add_argument("--manifest", default="fixtures/a0_manifest_small.csv")
    parser.add_argument("--split-csv", default="fixtures/train_split_small.csv")
    args = parser.parse_args()

    # Setup logging
    log_path = REPO / args.output_log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("a0_acceptance")
    logger.info("=== A0 Smoke Acceptance ===")
    logger.info("Config: %s", args.config)
    logger.info("Max batches: %d", args.max_batches)

    # Load config
    with open(REPO / args.config) as f:
        config = yaml.safe_load(f)

    # Override paths to use small fixtures
    config["sample_weighting"]["manifest_path"] = args.manifest
    config["data"]["split_dir"] = str(REPO / "fixtures")
    # Create train.csv symlink-style: we override the dataset builder
    config["train"]["num_workers"] = 0  # single-process for smoke test
    config["train"]["epochs"] = 1
    config["train"]["warmup_epochs"] = 0
    logger.info("Overrides: manifest=%s, split=%s, workers=0, epochs=1",
                 args.manifest, config["data"]["split_dir"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load CLIP model ──
    logger.info("Loading CLIP ViT-B/32...")
    from common.clip_utils import load_openai_clip
    clip_model, preprocess = load_openai_clip(device)
    clip_model.visual = clip_model.visual.float()
    clip_model.eval()
    logger.info("CLIP loaded.")

    # ── Build Dataset from fixture split ──
    from common.dataset import TrainImageDataset, seed_worker

    split_csv = REPO / args.split_csv
    split_df = pd.read_csv(split_csv)
    logger.info("Split CSV: %d rows from %s", len(split_df), split_csv)

    train_dataset = TrainImageDataset(
        data_root=str(REPO),  # dataset resolves paths relative to repo root
        split_csv=str(split_csv),
        class_to_idx={"0": 0},  # all samples are class 0
        transform=preprocess,
        return_path=True,
    )
    logger.info("Dataset: %d samples", len(train_dataset))
    logger.info("Dataset samples[:3]: %s", train_dataset.samples[:3])
    logger.info("Dataset labels[:3]: %s", train_dataset.labels[:3])

    g = torch.Generator().manual_seed(42)
    loader = DataLoader(
        train_dataset,
        batch_size=min(2, len(train_dataset)),  # small batches to get >=20 steps
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=g,
        drop_last=True,
    )
    logger.info("DataLoader: %d batches (batch_size=%d)", len(loader),
                 min(2, len(train_dataset)))

    # ── Build weight provider ──
    from common.sample_weighting import build_weight_provider

    config.setdefault("sample_weighting", {})
    config["sample_weighting"]["type"] = "relabel_manifest"
    config["sample_weighting"]["manifest_path"] = args.manifest
    config["sample_weighting"]["hard_relabel"] = False
    config["sample_weighting"]["min_weight"] = 0.0
    config["sample_weighting"]["max_weight"] = 1.0
    config["sample_weighting"]["missing_weight_policy"] = "error"

    weight_provider = build_weight_provider(config, num_train_samples=len(train_dataset))
    logger.info("Weight provider: %s", type(weight_provider).__name__)

    # ── Runtime manifest audit ──
    logger.info("=== Runtime Manifest Audit ===")
    from experiments.baseline.train import _runtime_manifest_audit

    save_dir = REPO / "outputs" / "a0_smoke_audit"
    _runtime_manifest_audit(
        train_dataset, weight_provider, "dev",
        save_dir, logger,
    )

    # Read back audit
    audit_path = save_dir / "manifest_runtime_audit.json"
    audit = json.loads(audit_path.read_text())
    logger.info("Audit results:")
    for k, v in audit.items():
        logger.info("  %s: %s", k, v)

    # Verify audit invariants
    assert audit["dataset_sample_count"] == len(train_dataset), \
        f"dataset_sample_count mismatch: {audit['dataset_sample_count']} != {len(train_dataset)}"
    assert audit["manifest_row_count"] == len(split_df), \
        f"manifest_row_count mismatch: {audit['manifest_row_count']} != {len(split_df)}"
    assert audit["coverage"] == 1.0, f"Coverage {audit['coverage']} != 1.0"
    assert audit["missing_in_manifest"] == 0, f"Missing: {audit['missing_in_manifest']}"
    assert audit["extra_in_manifest"] == 0, f"Extra: {audit['extra_in_manifest']}"
    assert audit["original_label_mismatches"] == 0, \
        f"Label mismatches: {audit['original_label_mismatches']}"
    logger.info("Audit PASSED: coverage=1.0, missing=0, extra=0, mismatches=0")

    # ── Build model head ──
    from experiments.baseline.model import build_model
    model, _ = build_model(config, device)
    logger.info("Model head built: %d trainable params",
                 sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ── Loss ──
    from common.losses import build_loss
    config["loss"] = {"name": "gce", "q": 0.5, "reduction": "none"}
    criterion = build_loss({"loss": config["loss"]})

    # ── Optimizer ──
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.005, weight_decay=1e-4)

    # ── Training loop ──
    from common.mixup import mixup_batch
    from experiments.baseline.train import _reduce_weighted_mixup

    logger.info("=== Training: %d batches ===", args.max_batches)
    model.train()
    n_batches = 0
    total_loss = 0.0
    nan_count = 0
    inf_count = 0

    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        images, labels, paths = batch_data
        images = images.to(device)
        labels = labels.to(device)

        # Relabel from manifest
        old_labels = labels.clone()
        labels = weight_provider.get_training_labels(list(paths), labels)
        n_relabeled = (labels != old_labels).sum().item()

        # CLIP encode
        with torch.no_grad():
            features = clip_model.encode_image(images).float()

        # MixUp
        inputs, labels_a, labels_b, lam, mix_perm = mixup_batch(
            features, labels, alpha=0.2, probability=0.2,
        )
        mixup_applied = lam < 1.0

        optimizer.zero_grad()
        logits = model.forward_features(features)

        if mixup_applied:
            loss_per_a = criterion(logits, labels_a)
            loss_per_b = criterion(logits, labels_b)
            w = weight_provider.get_weights(list(paths), labels, 1)
            loss = _reduce_weighted_mixup(
                loss_per_a, loss_per_b, w, mix_perm, lam,
                normalize_by_weight_sum=True,
            )
        else:
            loss_per_sample = criterion(logits, labels)
            w = weight_provider.get_weights(list(paths), labels, 1)
            loss = (w * loss_per_sample).sum() / w.sum().clamp_min(1e-8)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        loss_val = float(loss.detach())
        total_loss += loss_val
        n_batches += 1

        if np.isnan(loss_val):
            nan_count += 1
        if np.isinf(loss_val):
            inf_count += 1

        if batch_idx < 3 or batch_idx >= args.max_batches - 1:
            logger.info(
                "batch %2d: loss=%.6f mixup=%s relabeled=%d",
                batch_idx + 1, loss_val, mixup_applied, n_relabeled,
            )

    logger.info("=== Training Complete ===")
    logger.info("Completed batches: %d", n_batches)
    logger.info("Average loss: %.6f", total_loss / max(n_batches, 1))
    logger.info("Non-finite loss count: %d (nan=%d, inf=%d)", nan_count + inf_count, nan_count, inf_count)
    logger.info("Optimizer step count: %d", n_batches)

    # ── Checkpoint round-trip ──
    ckpt_path = save_dir / "smoke_checkpoint.pt"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "batch": n_batches,
    }, ckpt_path)
    logger.info("Checkpoint saved: %s", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Checkpoint loaded successfully: batch=%d", ckpt["batch"])

    # ── Final assertions ──
    assert n_batches == args.max_batches, f"Expected {args.max_batches} batches, got {n_batches}"
    assert nan_count == 0, f"NaN loss in {nan_count} batches"
    assert inf_count == 0, f"Inf loss in {inf_count} batches"

    logger.info("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
