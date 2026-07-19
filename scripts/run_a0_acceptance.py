#!/usr/bin/env python3
"""A0/A1/A2/A3 smoke acceptance: N batches of real training with manifest audit.

Usage:
    python scripts/run_a0_acceptance.py \\
        --config configs/nr_cl_classwise_drop.yaml \\
        --max-batches 20 \\
        --output-dir acceptance/a1
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
    parser.add_argument("--output-dir", default="acceptance/a1")
    args = parser.parse_args()

    output_dir = REPO / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "smoke.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("smoke")
    logger.info("=== Smoke Acceptance: %s ===", args.config)

    # Load config
    with open(REPO / args.config) as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config: %s", args.config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load CLIP ──
    from common.clip_utils import load_openai_clip
    clip_model, preprocess = load_openai_clip(device)
    clip_model.visual = clip_model.visual.float()
    clip_model.eval()
    logger.info("CLIP loaded")

    # ── Dataset ──
    from common.class_mapping import load_or_generate_mapping
    from common.dataset import TrainImageDataset, seed_worker

    class_to_idx, idx_to_class = load_or_generate_mapping(
        config["data"].get("class_mapping_path", config["data"]["split_dir"]),
        config["data"]["train_dir"], config["model"]["num_classes"],
    )
    split_dir = Path(config["data"]["split_dir"])
    train_csv = split_dir / "train.csv"

    train_dataset = TrainImageDataset(
        data_root=config["data"]["train_dir"],
        split_csv=str(train_csv),
        class_to_idx=class_to_idx,
        transform=preprocess,
        return_path=True,
    )
    logger.info("Dataset: %d samples, %d classes", len(train_dataset), len(class_to_idx))

    g = torch.Generator().manual_seed(42)
    loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=min(4, config["train"].get("num_workers", 8)),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    logger.info("DataLoader: %d batches", len(loader))

    # ── Weight provider ──
    from common.sample_weighting import build_weight_provider
    weight_provider = build_weight_provider(config, num_train_samples=len(train_dataset))
    logger.info("Weight provider: %s", type(weight_provider).__name__)

    # ── Reject policy: mirror production path ──
    sw_cfg_pre = config.get("sample_weighting", {})
    reject_policy = sw_cfg_pre.get("reject_policy", "weight_zero")
    if reject_policy == "drop":
        import pandas as _pd
        from common.manifest_loader import canonical_image_path as _canon
        _manifest_path = sw_cfg_pre.get("manifest_path")
        _mf = _pd.read_csv(_manifest_path)
        if "training_role" in _mf.columns:
            _rejected_mask = _mf["training_role"] == "rejected"
        elif "sample_weight" in _mf.columns:
            _rejected_mask = _mf["sample_weight"] == 0.0
        else:
            raise ValueError("reject_policy=drop requires training_role or sample_weight")
        _rejected_paths = set(_mf[_rejected_mask]["image_path"].astype(str).map(_canon))
        _old_n = len(train_dataset)
        _keep = [i for i, p in enumerate(train_dataset.samples) if _canon(str(p)) not in _rejected_paths]
        train_dataset.samples = [train_dataset.samples[i] for i in _keep]
        train_dataset.labels = [train_dataset.labels[i] for i in _keep]
        logger.info("Reject policy 'drop': %d → %d samples (%d rejected removed)",
                     _old_n, len(_keep), _old_n - len(_keep))
        # Rebuild loader
        _g = torch.Generator().manual_seed(42)
        loader = DataLoader(
            train_dataset, batch_size=config["train"]["batch_size"],
            shuffle=True, num_workers=min(4, config["train"].get("num_workers", 8)),
            pin_memory=True, worker_init_fn=seed_worker, generator=_g,
        )
        logger.info("DataLoader rebuilt: %d batches", len(loader))

    # ── Runtime audit ──
    from experiments.baseline.train import _runtime_manifest_audit
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    _runtime_manifest_audit(train_dataset, weight_provider, "dev", audit_dir, logger,
                            reject_policy=reject_policy)

    audit = json.loads((audit_dir / "manifest_runtime_audit.json").read_text())
    logger.info("Audit: coverage=%.4f missing=%d extra=%d mismatches=%d rejected_left=%d",
                 audit["coverage"], audit["missing_in_manifest"],
                 audit["extra_in_manifest"], audit["original_label_mismatches"],
                 audit.get("rejected_left_in_dataset", 0))
    assert audit["coverage"] == 1.0
    assert audit["missing_in_manifest"] == 0
    assert audit["extra_in_manifest"] == 0
    assert audit.get("rejected_left_in_dataset", 0) == 0, \
        f"Rejected samples left in dataset: {audit.get('rejected_left_in_dataset', 0)}"

    # ── Model head ──
    from experiments.baseline.model import build_model
    model, _ = build_model(config, device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d trainable params", n_trainable)

    # ── Loss + optimizer ──
    from common.losses import build_loss
    loss_cfg = config.get("loss", {}).copy()
    loss_cfg["reduction"] = "none"
    criterion = build_loss({"loss": loss_cfg})
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    # ── Training ──
    from common.mixup import mixup_batch
    from experiments.baseline.train import _reduce_weighted_mixup

    mixup_cfg = config.get("mixup", {})
    normalize_by_weight_sum = config.get("sample_weighting", {}).get(
        "normalize_by_weight_sum", True,
    )

    logger.info("=== Training: %d batches ===", args.max_batches)
    model.train()
    n_batches = 0
    first_loss = None
    last_loss = None
    nan_count = 0
    inf_count = 0
    relabeled_total = 0
    rejected_seen = 0
    pseudo_seen = 0

    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        images, labels, paths = batch_data
        images = images.to(device)
        labels = labels.to(device)

        # Relabel
        if weight_provider is not None:
            old_labels = labels.clone()
            labels = weight_provider.get_training_labels(list(paths), labels)
            relabeled_total += (labels != old_labels).sum().item()

        # Count roles
        if weight_provider is not None:
            w = weight_provider.get_weights(list(paths), labels, 1)
            rejected_seen += (w == 0.0).sum().item()
            roles = weight_provider.get_roles(list(paths)) if hasattr(weight_provider, "get_roles") else []
            pseudo_seen += sum(1 for r in roles if r == "pseudo")

        # CLIP encode
        with torch.no_grad():
            features = clip_model.encode_image(images).float()

        # MixUp on features
        mixup_applied = False
        labels_a = labels_b = labels
        mix_perm = None
        lam = 1.0
        mixed_features = features
        if mixup_cfg.get("enabled", False):
            mixed_features, labels_a, labels_b, lam, mix_perm = mixup_batch(
                features, labels,
                alpha=mixup_cfg.get("alpha", 0.2),
                probability=mixup_cfg.get("probability", 0.2),
            )
            mixup_applied = lam < 1.0

        optimizer.zero_grad()
        logits = model.forward_features(mixed_features)

        if mixup_applied:
            loss_per_a = criterion(logits, labels_a)
            loss_per_b = criterion(logits, labels_b)
            w = weight_provider.get_weights(list(paths), labels, 1)
            loss = _reduce_weighted_mixup(
                loss_per_a, loss_per_b, w, mix_perm, lam,
                normalize_by_weight_sum=normalize_by_weight_sum,
            )
        else:
            loss_per_sample = criterion(logits, labels)
            w = weight_provider.get_weights(list(paths), labels, 1)
            loss = (w * loss_per_sample).sum() / w.sum().clamp_min(1e-8)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        loss_val = float(loss.detach())
        n_batches += 1
        if first_loss is None:
            first_loss = loss_val
        last_loss = loss_val

        if np.isnan(loss_val):
            nan_count += 1
        if np.isinf(loss_val):
            inf_count += 1

    # Gradient check from last batch
    head_grad_norm = sum(
        p.grad.norm().item() ** 2 for p in trainable if p.grad is not None
    ) ** 0.5
    grad_finite = not (np.isnan(head_grad_norm) or np.isinf(head_grad_norm))

    logger.info("=== Training Complete ===")
    logger.info("completed_batches: %d", n_batches)
    logger.info("optimizer_steps: %d", n_batches)
    logger.info("first_batch_loss: %.6f", first_loss or -1)
    logger.info("last_batch_loss: %.6f", last_loss or -1)
    logger.info("non_finite_loss_count: %d (nan=%d, inf=%d)", nan_count + inf_count, nan_count, inf_count)
    logger.info("non_finite_gradient_count: %d", 0 if grad_finite else 1)
    logger.info("relabeled_total: %d", relabeled_total)
    logger.info("rejected_seen: %d", rejected_seen)
    logger.info("pseudo_seen: %d", pseudo_seen)

    # ── Checkpoint ──
    ckpt_path = output_dir / "smoke_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "batch": n_batches,
    }, ckpt_path)
    logger.info("checkpoint_save_pass: true")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("checkpoint_load_pass: true")

    # ── Smoke summary ──
    smoke = {
        "config": args.config,
        "completed_batches": n_batches,
        "optimizer_steps": n_batches,
        "first_batch_loss": first_loss,
        "last_batch_loss": last_loss,
        "non_finite_loss_count": nan_count + inf_count,
        "non_finite_gradient_count": 0 if grad_finite else 1,
        "relabeled_total": relabeled_total,
        "rejected_seen": rejected_seen,
        "pseudo_seen": pseudo_seen,
        "coverage": audit["coverage"],
        "missing_in_manifest": audit["missing_in_manifest"],
        "extra_in_manifest": audit["extra_in_manifest"],
        "label_mismatches": audit["original_label_mismatches"],
        "clean_count": audit["clean_count"],
        "rejected_count": audit["rejected_count"],
        "pseudo_count": audit["pseudo_count"],
        "checkpoint_save_pass": True,
        "checkpoint_load_pass": True,
    }
    (output_dir / "smoke_summary.json").write_text(json.dumps(smoke, indent=2))

    # ── Final assertions ──
    assert n_batches == args.max_batches, f"Expected {args.max_batches} batches, got {n_batches}"
    assert nan_count == 0, f"NaN in {nan_count} batches"
    assert inf_count == 0, f"Inf in {inf_count} batches"
    assert grad_finite, "Non-finite gradient"

    logger.info("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
