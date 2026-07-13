#!/usr/bin/env python3
"""
Evaluate a checkpoint on both raw noisy labels and the trusted validation subset.

Loads a pre-computed val feature bank, applies the classifier head from a
checkpoint via the fast-path F.linear, and produces dual metrics:

  - raw_noisy_validation: metrics on all validation samples using noisy labels
  - trusted_validation:   metrics masked by trusted_v1 + coverage stats
  - rejected_subset_diagnostic: metrics on the complement (samples rejected
    by the trusted-subset rules)

Usage:
    python tools/evaluate_dual_validation.py \\
        --name my_experiment \\
        --config configs/baseline.yaml \\
        --ckpt outputs/baseline/checkpoints/best.pt \\
        --val-feature-bank outputs/baseline/feature_banks/val_feature_bank.pt \\
        --trusted-manifest outputs/baseline/sample_metrics/trusted_manifest.csv \\
        --output outputs/my_experiment/dual_validation.json \\
        --device cuda
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────


def _sha256_hex(file_path: str) -> str:
    """Compute SHA-256 hex digest of a file, reading in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1_048_576)  # 1 MiB
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_feature_bank_path(config, override_path: str | None) -> str:
    """Resolve val feature bank path.

    Precedence:
      1. Explicit ``--val-feature-bank`` CLI argument.
      2. Config-based default (``data.split_dir`` / ``val_feature_bank.pt``).
      3. Fallback relative to config's ``output.save_dir``.

    Raises ``FileNotFoundError`` if none exists.
    """
    if override_path:
        return override_path

    # Try config-based defaults
    split_dir = config.get("data", {}).get("split_dir", None)
    save_dir = config.get("output", {}).get("save_dir", None)

    candidates = []
    if split_dir:
        candidates.append(Path(split_dir) / "val_feature_bank.pt")
        candidates.append(Path(split_dir) / ".." / "val_feature_bank.pt")
    if save_dir:
        candidates.append(Path(save_dir) / "val_feature_bank.pt")
        candidates.append(Path(save_dir).parent / "val_feature_bank.pt")

    for p in candidates:
        resolved = p.resolve()
        if resolved.exists():
            return str(resolved)

    raise FileNotFoundError(
        "Could not locate val_feature_bank.pt. "
        "Provide --val-feature-bank explicitly."
    )


def _resolve_trusted_manifest_path(config, override_path: str | None) -> str:
    """Resolve trusted manifest CSV path.

    Precedence:
      1. Explicit ``--trusted-manifest`` CLI argument.
      2. Config-based default (``output.save_dir`` / ``trusted_manifest.csv``).

    Raises ``FileNotFoundError`` if none exists.
    """
    if override_path:
        return override_path

    save_dir = config.get("output", {}).get("save_dir", None)

    candidates = []
    if save_dir:
        candidates.append(Path(save_dir) / "trusted_manifest.csv")
        candidates.append(Path(save_dir).parent / "trusted_manifest.csv")

    # Also check submission_dir
    sub_dir = config.get("output", {}).get("submission_dir", None)
    if sub_dir:
        candidates.append(Path(sub_dir) / "trusted_manifest.csv")

    for p in candidates:
        resolved = p.resolve()
        if resolved.exists():
            return str(resolved)

    raise FileNotFoundError(
        "Could not locate trusted_manifest.csv. "
        "Provide --trusted-manifest explicitly."
    )


# ──────────────────────────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────────────────────────


def compute_per_class_accuracy(
    correct: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> tuple:
    """Compute per-class accuracy statistics.

    Returns
    -------
    per_class_acc : dict mapping class index -> float accuracy (NaN if no samples)
    per_class_total : dict mapping class index -> int sample count
    """
    per_class_acc = {}
    per_class_total = {}
    for c in range(num_classes):
        mask = labels == c
        total = int(mask.sum())
        per_class_total[c] = total
        if total > 0:
            per_class_acc[c] = float(correct[mask].mean())
        else:
            per_class_acc[c] = float("nan")
    return per_class_acc, per_class_total


def compute_raw_metrics(
    correct: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> dict:
    """Compute raw validation metrics.

    Metrics:
      - micro_accuracy: overall fraction correct
      - macro_accuracy_present_classes: mean per-class accuracy over classes
        with at least 2 samples (per_class_total > 1)
      - macro_accuracy_all: mean per-class accuracy over all num_classes
        (classes with zero samples treated as 0.0)
      - median_class_accuracy: median of per-class accuracies (present classes)
      - bottom_10pct_class_accuracy: mean of per-class accuracies in the
        worst-decile (bottom 10% of present classes by accuracy)
    """
    per_class_acc, per_class_total = compute_per_class_accuracy(
        correct, labels, num_classes
    )

    # Classes with at least 2 samples (per_class_total > 1, per brief spec).
    present_classes = [c for c in range(num_classes) if per_class_total[c] > 1]
    present_accs = np.array([per_class_acc[c] for c in present_classes])

    micro_accuracy = float(correct.mean())

    if len(present_accs) > 0:
        macro_present = float(present_accs.mean())
        median_class = float(np.median(present_accs))
        # Bottom 10%
        sorted_accs = np.sort(present_accs)
        n_bottom = max(1, int(np.ceil(len(sorted_accs) * 0.10)))
        bottom_10pct = float(sorted_accs[:n_bottom].mean())
    else:
        macro_present = float("nan")
        median_class = float("nan")
        bottom_10pct = float("nan")

    # Macro over all 500 classes: missing classes → 0.0
    all_accs = np.array([
        0.0 if per_class_total[c] == 0 else per_class_acc[c]
        for c in range(num_classes)
    ])
    macro_all = float(all_accs.mean())

    return {
        "micro_accuracy": micro_accuracy,
        "macro_accuracy_present_classes": macro_present,
        "macro_accuracy_all": macro_all,
        "median_class_accuracy": median_class,
        "bottom_10pct_class_accuracy": bottom_10pct,
        "num_classes": num_classes,
        "num_samples": int(len(correct)),
        "per_class_accuracy": {
            str(c): per_class_acc[c] for c in range(num_classes)
        },
        "per_class_total": {
            str(c): per_class_total[c] for c in range(num_classes)
        },
    }


def compute_trusted_metrics(
    correct: np.ndarray,
    labels: np.ndarray,
    trusted_mask: np.ndarray,
    num_classes: int,
) -> dict:
    """Compute trusted-subset metrics.

    Metrics mirror ``compute_raw_metrics`` but only include samples where
    ``trusted_mask`` is True.  Additionally reports coverage and
    represented_classes.
    """
    n_total = len(trusted_mask)
    n_trusted = int(trusted_mask.sum())
    coverage = n_trusted / n_total if n_total > 0 else 0.0

    trusted_correct = correct[trusted_mask]
    trusted_labels = labels[trusted_mask]

    # Count how many classes have at least one trusted sample
    represented_classes = int(np.unique(trusted_labels).size) if n_trusted > 0 else 0

    metrics = compute_raw_metrics(trusted_correct, trusted_labels, num_classes)
    metrics["coverage"] = coverage
    metrics["num_trusted_samples"] = n_trusted
    metrics["num_total_samples"] = n_total
    metrics["represented_classes"] = represented_classes
    metrics["missing_classes"] = num_classes - represented_classes

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Dual validation: evaluate checkpoint on raw noisy labels "
        "and trusted subset"
    )
    p.add_argument(
        "--name",
        required=True,
        help="Experiment name for the output report",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to experiment config YAML (used for resolving default paths)",
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="Path to checkpoint .pt file",
    )
    p.add_argument(
        "--val-feature-bank",
        default=None,
        help="Path to val_feature_bank.pt (optional if resolvable from config)",
    )
    p.add_argument(
        "--trusted-manifest",
        default=None,
        help="Path to trusted_manifest.csv (optional if resolvable from config)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Path to output JSON (default: based on checkpoint location)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help='Torch device string, e.g. "cuda", "cpu" (default: cuda)',
    )
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    logger.info("=== Dual Validation Evaluation ===")
    logger.info("Experiment name: %s", args.name)
    logger.info("Checkpoint:      %s", args.ckpt)
    logger.info("Val feature bank: %s", args.val_feature_bank or "(resolve from config)")
    logger.info("Trusted manifest: %s", args.trusted_manifest or "(resolve from config)")
    logger.info("Device:          %s", args.device)

    # ── Load config (optional) ──
    config = None
    if args.config:
        from common.utils import load_config

        config = load_config(args.config)
        num_classes = config["model"]["num_classes"]
        logger.info("Loaded config: %s (num_classes=%d)", args.config, num_classes)
    else:
        num_classes = 500
        logger.info("No config provided, using num_classes=%d", num_classes)

    device = torch.device(args.device)

    # ── Resolve file paths ──
    val_bank_path = _resolve_feature_bank_path(config, args.val_feature_bank)
    trusted_manifest_path = _resolve_trusted_manifest_path(
        config, args.trusted_manifest
    )

    logger.info("Val feature bank:  %s", val_bank_path)
    logger.info("Trusted manifest:  %s", trusted_manifest_path)

    # ── Compute SHA-256 hashes ──
    logger.info("Computing SHA-256 hashes...")
    ckpt_sha256 = _sha256_hex(args.ckpt)
    val_bank_sha256 = _sha256_hex(val_bank_path)
    trusted_manifest_sha256 = _sha256_hex(trusted_manifest_path)
    logger.info("Checkpoint SHA-256:         %s", ckpt_sha256[:16] + "...")
    logger.info("Val feature bank SHA-256:   %s", val_bank_sha256[:16] + "...")
    logger.info("Trusted manifest SHA-256:   %s", trusted_manifest_sha256[:16] + "...")

    # ── Load val feature bank ──
    logger.info("Loading val feature bank...")
    val_bank = torch.load(val_bank_path, map_location="cpu", weights_only=True)
    val_features = val_bank["features"]  # (N, 512) L2-normalized
    val_labels = val_bank["labels"]  # (N,)
    val_paths = val_bank["paths"]  # list of str
    n_val = val_features.shape[0]
    logger.info("Val samples: %d", n_val)

    # ── Load checkpoint ──
    logger.info("Loading checkpoint...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    weight = sd["classifier.weight"]  # (C, D)
    bias = sd["classifier.bias"]  # (C,)

    # ── Fast-path logit computation ──
    logger.info("Computing logits via fast-path F.linear...")
    with torch.no_grad():
        features_gpu = F.normalize(
            val_features.float().to(device), dim=-1
        )
        weight_gpu = weight.to(device)
        bias_gpu = bias.to(device)
        logits = F.linear(features_gpu, weight_gpu, bias_gpu).cpu()

    # ── Predictions and correctness ──
    preds = logits.argmax(dim=1).numpy()
    val_labels_np = val_labels.numpy()
    correct = (preds == val_labels_np).astype(np.float64)
    logger.info(
        "Raw accuracy: %.4f (%d/%d)",
        correct.mean(),
        int(correct.sum()),
        n_val,
    )

    # ── Load trusted manifest and verify paths ──
    logger.info("Loading trusted manifest: %s", trusted_manifest_path)
    manifest = pd.read_csv(trusted_manifest_path)

    # The manifest should have image_path and trusted_v1 columns
    if "trusted_v1" not in manifest.columns:
        logger.error(
            "trusted_v1 column not found in manifest. "
            "Available columns: %s",
            list(manifest.columns),
        )
        sys.exit(1)

    if "image_path" not in manifest.columns:
        logger.error(
            "image_path column not found in manifest. "
            "Available columns: %s",
            list(manifest.columns),
        )
        sys.exit(1)

    manifest_paths = manifest["image_path"].tolist()
    trusted_v1 = manifest["trusted_v1"].values.astype(bool)

    # Verify manifest paths match val paths exactly
    if len(manifest_paths) != n_val:
        logger.error(
            "Manifest has %d samples but val feature bank has %d. "
            "Cannot align.",
            len(manifest_paths),
            n_val,
        )
        sys.exit(1)

    mismatches = []
    for i, (mp, vp) in enumerate(zip(manifest_paths, val_paths)):
        # Normalize: use Path to resolve / compare suffixes
        mp_norm = str(Path(mp))
        vp_norm = str(Path(vp))
        if mp_norm != vp_norm:
            mismatches.append((i, mp_norm, vp_norm))

    if mismatches:
        logger.error(
            "Found %d path mismatches between manifest and val feature bank. "
            "First 5 shown below. Aborting.",
            len(mismatches),
        )
        for idx, mp, vp in mismatches[:5]:
            logger.error("  [%d] manifest=%s  vs  val_bank=%s", idx, mp, vp)
        sys.exit(1)

    logger.info(
        "Path verification passed: all %d paths match.",
        n_val,
    )

    # ── Compute raw metrics ──
    logger.info("Computing raw noisy validation metrics...")
    raw_metrics = compute_raw_metrics(correct, val_labels_np, num_classes)

    # ── Compute trusted metrics (V1) ──
    logger.info("Computing trusted validation metrics (V1)...")
    trusted_metrics = compute_trusted_metrics(
        correct, val_labels_np, trusted_v1, num_classes
    )

    # ── Compute rejected subset diagnostic ──
    logger.info("Computing rejected subset diagnostic...")
    rejected_mask = ~trusted_v1
    rejected_metrics = compute_raw_metrics(
        correct[rejected_mask],
        val_labels_np[rejected_mask],
        num_classes,
    )
    rejected_metrics["num_rejected_samples"] = int(rejected_mask.sum())
    rejected_metrics["num_total_samples"] = n_val

    # ── Compute V2 trusted metrics (continuous + class-balanced) ──
    logger.info("Computing trusted validation metrics (V2)...")
    from common.trusted_subset import (
        TrustedSubsetConfig,
        compute_class_balanced_trusted_accuracy,
        compute_trust_weighted_accuracy,
    )

    v2_config = TrustedSubsetConfig()

    # Verify required columns are present in manifest
    v2_required_cols = [
        "knn_label_agreement", "prototype_margin",
        "clip_flip_cosine", "noisy_label",
    ]
    v2_missing_cols = [
        c for c in v2_required_cols if c not in manifest.columns
    ]
    if v2_missing_cols:
        logger.warning(
            "V2 metrics skipped: manifest missing columns: %s",
            v2_missing_cols,
        )
        trusted_v2_metrics = None
    else:
        # Verify manifest row order matches val bank
        # (already verified paths match above)
        trust_weighted = compute_trust_weighted_accuracy(
            manifest,
            correct.astype(bool),
            margin_ref=v2_config.prototype_margin_ref,
        )
        class_balanced = compute_class_balanced_trusted_accuracy(
            manifest,
            correct.astype(bool),
            top_k=v2_config.class_balanced_top_k,
            margin_ref=v2_config.prototype_margin_ref,
        )
        trusted_v2_metrics = {
            "trust_weighted": trust_weighted,
            "class_balanced": class_balanced,
            "config": {
                "class_balanced_top_k": v2_config.class_balanced_top_k,
                "prototype_margin_ref": v2_config.prototype_margin_ref,
            },
        }
        logger.info(
            "V2 trust-weighted accuracy: %.4f (eff_samples=%.1f)",
            trust_weighted["accuracy"],
            trust_weighted["effective_samples"],
        )
        logger.info(
            "V2 class-balanced accuracy (Top-%d): %.4f (%d/%d classes)",
            class_balanced["top_k_per_class"],
            class_balanced["macro_accuracy"],
            class_balanced["num_classes_with_k"],
            class_balanced["num_classes_total"],
        )

    # ── Build output report ──
    report = {
        "schema_version": 2,
        "experiment_name": args.name,
        "checkpoint_sha256": ckpt_sha256,
        "val_feature_bank_sha256": val_bank_sha256,
        "trusted_manifest_sha256": trusted_manifest_sha256,
        "raw_noisy_validation": raw_metrics,
        "trusted_validation": trusted_metrics,
        "rejected_subset_diagnostic": rejected_metrics,
    }
    if trusted_v2_metrics is not None:
        report["trusted_validation_v2"] = trusted_v2_metrics

    # ── Save output ──
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: same directory as the checkpoint
        ckpt_dir = Path(args.ckpt).resolve().parent
        output_path = ckpt_dir / "dual_validation.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Output saved to: %s", output_path)

    # ── Print summary ──
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Raw noisy validation:")
    logger.info("  micro_accuracy:                 %.4f", raw_metrics["micro_accuracy"])
    logger.info("  macro_accuracy (present):       %.4f", raw_metrics["macro_accuracy_present_classes"])
    logger.info("  macro_accuracy (all 500):       %.4f", raw_metrics["macro_accuracy_all"])
    logger.info("  median_class_accuracy:          %.4f", raw_metrics["median_class_accuracy"])
    logger.info("  bottom_10pct_class_accuracy:    %.4f", raw_metrics["bottom_10pct_class_accuracy"])
    logger.info("Trusted validation:")
    logger.info("  micro_accuracy:                 %.4f", trusted_metrics["micro_accuracy"])
    logger.info("  macro_accuracy (present):       %.4f", trusted_metrics["macro_accuracy_present_classes"])
    logger.info("  coverage:                       %.4f", trusted_metrics["coverage"])
    logger.info("  represented_classes:            %d / %d",
                trusted_metrics["represented_classes"], num_classes)
    logger.info("Rejected subset:")
    logger.info("  micro_accuracy:                 %.4f", rejected_metrics["micro_accuracy"])
    logger.info("  num_rejected_samples:           %d", rejected_metrics["num_rejected_samples"])
    if trusted_v2_metrics is not None:
        tw = trusted_v2_metrics["trust_weighted"]
        cb = trusted_v2_metrics["class_balanced"]
        logger.info("Trusted V2 — continuous trust-weighted:")
        logger.info("  accuracy:                       %.4f", tw["accuracy"])
        logger.info("  effective_samples:              %.1f", tw["effective_samples"])
        logger.info("  mean_weight:                    %.4f", tw["mean_weight"])
        logger.info("Trusted V2 — class-balanced (Top-%d):", cb["top_k_per_class"])
        logger.info("  macro_accuracy:                 %.4f", cb["macro_accuracy"])
        logger.info("  classes with ≥%d:               %d / %d",
                    cb["top_k_per_class"], cb["num_classes_with_k"],
                    cb["num_classes_total"])
        logger.info("  samples used:                   %d (%.1f%%)",
                    cb["num_samples_used"], 100.0 * cb["coverage"])
    logger.info("=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()
