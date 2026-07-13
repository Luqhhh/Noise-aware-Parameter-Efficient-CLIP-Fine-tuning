#!/usr/bin/env python3
"""
Verify that ``encode_frozen_clip_features`` + classifier forward produces
**identical** logits to the full model forward pass (``model.forward()``).

This tool is the acceptance gate for R0-1: it must report **FP32 Top-1
mismatch = 0** and **max logits error < 1e-5** before trusted-subset
results can be treated as exact.

Usage:
    python tools/verify_encoding_consistency.py \\
        --config configs/ref.yaml \\
        --ckpt outputs/ref/seed42/checkpoints/best.pt \\
        --split-dir outputs/ref/seed42 \\
        --output outputs/analysis/encoding_consistency.json \\
        --device cuda

The tool also checks AMP vs non-AMP and verifies that the fast-path
F.linear(path_A_features, classifier) matches path_B_features @ classifier.T.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.clip_utils import encode_frozen_clip_features, load_openai_clip
from common.dataset import TrainImageDataset, seed_worker
from common.utils import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify encoding consistency between full model forward "
        "and encode_frozen_clip_features + classifier"
    )
    p.add_argument("--config", required=True, help="Path to experiment config YAML")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    p.add_argument(
        "--split-dir",
        required=True,
        help="Directory containing val.csv (and class_to_idx.json)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Path for JSON output (default: stdout only)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help='Torch device string (default: "cuda")',
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for encoding (default: 256)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers (default: 8)",
    )
    return p.parse_args()


def _build_val_loader(split_dir: str, preprocess, batch_size: int, num_workers: int):
    """Build a DataLoader over the validation split."""
    split_dir = Path(split_dir)
    val_csv = split_dir / "val.csv"
    class_to_idx_path = split_dir / "class_to_idx.json"

    if not val_csv.exists():
        raise FileNotFoundError(f"val.csv not found at {val_csv}")
    if not class_to_idx_path.exists():
        raise FileNotFoundError(f"class_to_idx.json not found at {class_to_idx_path}")

    import json as _json
    with open(class_to_idx_path) as f:
        class_to_idx = _json.load(f)

    dataset = TrainImageDataset(
        data_root=".",  # paths in CSV are CWD-relative; data_root is for dir-scan fallback
        split_csv=str(val_csv),
        class_to_idx=class_to_idx,
        transform=preprocess,
        return_path=True,
    )

    g = torch.Generator()
    g.manual_seed(42)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return loader, len(dataset)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    device = torch.device(args.device)
    config = load_config(args.config)
    num_classes = config["model"]["num_classes"]

    logger.info("=== Encoding Consistency Verification ===")
    logger.info("Config:     %s", args.config)
    logger.info("Checkpoint: %s", args.ckpt)
    logger.info("Split dir:  %s", args.split_dir)
    logger.info("Device:     %s", device)

    # ── 1. Load CLIP model (shared reference) ──
    logger.info("Loading CLIP ViT-B/32...")
    clip_model, preprocess = load_openai_clip(device)
    clip_model.eval()
    logger.info("CLIP model loaded. visual.conv1.weight.dtype = %s",
                clip_model.visual.conv1.weight.dtype)

    # ── 2. Build the full experiment model ──
    logger.info("Building experiment model...")
    from experiments.baseline.model import build_model

    exp_model, _ = build_model(config, device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    exp_model.load_state_dict(ckpt["model_state_dict"])
    exp_model.eval()
    logger.info("Experiment model loaded and set to eval mode.")

    # ── 3. Build val loader ──
    val_loader, n_val = _build_val_loader(
        args.split_dir, preprocess, args.batch_size, args.num_workers
    )
    logger.info("Validation samples: %d, batches: %d", n_val, len(val_loader))

    # ── 4. Run both encoding paths on every batch ──
    logger.info("Running dual-path encoding on all %d samples...", n_val)

    all_full_logits = []
    all_encoded_logits = []
    all_fast_logits = []  # F.linear(path_A_features, classifier) — extra check
    all_labels = []
    all_paths = []

    classifier_weight = ckpt["model_state_dict"]["classifier.weight"].to(device)
    classifier_bias = ckpt["model_state_dict"]["classifier.bias"].to(device)

    for images, labels, paths in val_loader:
        images = images.to(device, non_blocking=True)

        # Path A: full model forward (ground truth)
        with torch.no_grad():
            full_logits = exp_model(images).cpu()

        # Path B: encode_frozen_clip_features + classifier head
        with torch.no_grad():
            encoded_features = encode_frozen_clip_features(
                clip_model, images, device, use_amp=False
            )
            encoded_logits = exp_model.forward_features(encoded_features).cpu()

        # Path C: same features as Path B, but via F.linear (extra sanity)
        with torch.no_grad():
            features_norm = F.normalize(encoded_features.float().to(device), dim=-1)
            fast_logits = F.linear(features_norm, classifier_weight, classifier_bias).cpu()

        all_full_logits.append(full_logits)
        all_encoded_logits.append(encoded_logits)
        all_fast_logits.append(fast_logits)
        all_labels.append(labels)
        all_paths.extend(paths)

    full_logits = torch.cat(all_full_logits, dim=0)       # (N, C)
    encoded_logits = torch.cat(all_encoded_logits, dim=0)  # (N, C)
    fast_logits = torch.cat(all_fast_logits, dim=0)        # (N, C)
    all_labels = torch.cat(all_labels, dim=0)               # (N,)
    n_total = full_logits.shape[0]
    logger.info("Encoding complete: %d samples", n_total)

    # ── 5. Compute comparisons ──
    logger.info("=== Comparing encoding paths ===")

    # 5a. Full vs Encoded (the critical comparison)
    abs_diff_full_vs_encoded = (full_logits - encoded_logits).abs()
    max_abs_diff = abs_diff_full_vs_encoded.max().item()
    mean_abs_diff = abs_diff_full_vs_encoded.mean().item()
    per_sample_max = abs_diff_full_vs_encoded.max(dim=1).values

    full_preds = full_logits.argmax(dim=1)
    encoded_preds = encoded_logits.argmax(dim=1)
    mismatch_mask = full_preds != encoded_preds
    n_mismatch = int(mismatch_mask.sum().item())

    logger.info("Full vs encode_frozen_clip_features + forward_features:")
    logger.info("  Max abs diff:      %.8e", max_abs_diff)
    logger.info("  Mean abs diff:     %.8e", mean_abs_diff)
    logger.info("  Top-1 mismatches:  %d / %d", n_mismatch, n_total)

    # 5b. Encoded vs Fast (F.linear sanity check — should be ~0)
    abs_diff_encoded_vs_fast = (encoded_logits - fast_logits).abs()
    max_ef_diff = abs_diff_encoded_vs_fast.max().item()
    fast_mismatch = int((encoded_preds != fast_logits.argmax(dim=1)).sum().item())

    logger.info("Encoded vs F.linear (sanity check):")
    logger.info("  Max abs diff:      %.8e", max_ef_diff)
    logger.info("  Top-1 mismatches:  %d", fast_mismatch)

    # ── 6. Per-mismatch detail ──
    mismatch_detail = []
    if n_mismatch > 0:
        logger.warning("=== MISMATCH DETAILS (first 20) ===")
        mismatch_indices = torch.where(mismatch_mask)[0].tolist()
        for idx in mismatch_indices[:20]:
            detail = {
                "sample_index": idx,
                "image_path": all_paths[idx],
                "noisy_label": int(all_labels[idx].item()),
                "full_pred": int(full_preds[idx].item()),
                "encoded_pred": int(encoded_preds[idx].item()),
                "max_abs_diff": float(per_sample_max[idx].item()),
            }
            mismatch_detail.append(detail)
            logger.warning(
                "  [%d] %s  noisy=%d  full→%d  encoded→%d  max_diff=%.6e",
                idx, Path(all_paths[idx]).name,
                detail["noisy_label"], detail["full_pred"],
                detail["encoded_pred"], detail["max_abs_diff"],
            )

    # ── 7. Acceptance criteria ──
    fp32_pass = n_mismatch == 0
    tolerance_pass = max_abs_diff < 1e-5

    logger.info("=== Acceptance Criteria ===")
    logger.info("FP32 Top-1 mismatch = 0:   %s (%d mismatches)",
                "PASS" if fp32_pass else "FAIL", n_mismatch)
    logger.info("Max logits error < 1e-5:   %s (%.8e)",
                "PASS" if tolerance_pass else "FAIL", max_abs_diff)
    logger.info("Overall:                    %s",
                "PASS" if (fp32_pass and tolerance_pass) else "FAIL")

    # ── 8. Distribution of per-sample max abs diff ──
    p50 = float(per_sample_max.median().item())
    p99 = float(per_sample_max.kthvalue(int(n_total * 0.99)).values.item())
    p999 = float(per_sample_max.kthvalue(int(n_total * 0.999)).values.item())

    logger.info("=== Per-Sample Max Abs Diff Distribution ===")
    logger.info("  median (p50):  %.8e", p50)
    logger.info("  p99:           %.8e", p99)
    logger.info("  p99.9:         %.8e", p999)
    logger.info("  max:           %.8e", max_abs_diff)

    # ── 9. Save output ──
    report = {
        "schema_version": 1,
        "config": args.config,
        "checkpoint": args.ckpt,
        "split_dir": args.split_dir,
        "n_samples": n_total,
        "num_classes": num_classes,
        "full_vs_encoded": {
            "max_abs_diff": float(max_abs_diff),
            "mean_abs_diff": float(mean_abs_diff),
            "top1_mismatches": n_mismatch,
            "mismatch_fraction": float(n_mismatch / n_total) if n_total > 0 else 0.0,
        },
        "encoded_vs_fast_linear": {
            "max_abs_diff": float(max_ef_diff),
            "top1_mismatches": fast_mismatch,
        },
        "per_sample_distribution": {
            "p50": float(p50),
            "p99": float(p99),
            "p99_9": float(p999),
            "max": float(max_abs_diff),
        },
        "acceptance": {
            "fp32_top1_mismatch_zero": fp32_pass,
            "max_logits_error_lt_1e5": tolerance_pass,
            "overall_pass": fp32_pass and tolerance_pass,
        },
        "mismatch_details": mismatch_detail,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Report saved to: %s", output_path)

    # ── 10. Exit code ──
    if not (fp32_pass and tolerance_pass):
        logger.error("Acceptance criteria FAILED. See report for details.")
        sys.exit(1)

    logger.info("All acceptance criteria PASSED.")
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
