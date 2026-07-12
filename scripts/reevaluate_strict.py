#!/usr/bin/env python3
"""
Re-evaluate all strict experiment checkpoints with per-class metrics.

Loads the 4 best checkpoints (E0-strict, D3-strict, F0-strict, F1-strict),
runs enhanced evaluate() that computes micro, macro, median per-class,
bottom-10% accuracy, and micro-macro gap.

Saves results as reeval_results.json alongside each checkpoint and prints
a summary comparison table.

Usage:
    python3 scripts/reevaluate_strict.py
"""

import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.class_mapping import load_or_generate_mapping
from common.dataset import TrainImageDataset
from common.utils import load_config
from experiments.baseline.evaluate import evaluate  # enhanced evaluate()

logger = logging.getLogger(__name__)

# ── Checkpoint definitions ──────────────────────────────────────────
EXPERIMENTS = [
    {
        "id": "E0_STRICT",
        "config": "configs/e0_strict.yaml",
        "ckpt": "outputs/e0_strict/seed42/checkpoints/best.pt",
        "head_type": "linear",
    },
    {
        "id": "D3_STRICT",
        "config": "configs/d3_strict.yaml",
        "ckpt": "outputs/d3_strict/seed42/checkpoints/best.pt",
        "head_type": "linear",
    },
    {
        "id": "F0_STRICT",
        "config": "configs/f0_strict.yaml",
        "ckpt": "outputs/f0_strict/seed42/checkpoints/best.pt",
        "head_type": "linear",
    },
    {
        "id": "F1_STRICT",
        "config": "configs/f1_strict.yaml",
        "ckpt": "outputs/f1_strict/seed42/checkpoints/best.pt",
        "head_type": "linear",
    },
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    results = {}

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        config_path = exp["config"]
        ckpt_path = exp["ckpt"]
        head_type = exp["head_type"]

        logger.info("=" * 60)
        logger.info(f"Evaluating: {exp_id}")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Checkpoint: {ckpt_path}")

        # Load config
        config = load_config(config_path)

        # Load class mapping
        class_mapping_path = config["data"].get(
            "class_mapping_path", config["data"]["split_dir"]
        )
        class_to_idx, _ = load_or_generate_mapping(
            metadata_dir=class_mapping_path,
            train_dir=config["data"]["train_dir"],
            expected_num_classes=config["model"]["num_classes"],
        )

        # Build model
        if head_type == "cosine":
            from experiments.cosine.model import build_cosine_model
            model, preprocess = build_cosine_model(config, device)
        else:
            from experiments.baseline.model import build_model
            model, preprocess = build_model(config, device)

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}"
        )
        model = model.to(device)

        # Load validation dataset
        split_dir = Path(config["data"]["split_dir"])
        val_csv = split_dir / "val.csv"
        if not val_csv.exists():
            logger.error(f"Validation split not found: {val_csv}")
            continue

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

        logger.info(f"  Validation set: {len(val_dataset)} samples, "
                     f"{len(val_loader)} batches")

        # Evaluate with enhanced metrics
        criterion = nn.CrossEntropyLoss()
        eval_results = evaluate(
            model, val_loader, criterion, device,
            use_amp=config["train"].get("amp", False),
        )

        # Save reeval results
        ckpt_dir = Path(ckpt_path).parent
        reeval_path = ckpt_dir / "reeval_results.json"
        with open(reeval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"  Saved: {reeval_path}")

        results[exp_id] = eval_results

    # ── Print summary table ──────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Strict Experiment Re-Evaluation Summary (Per-Class Metrics)")
    print("=" * 100)
    header = (
        f"{'Experiment':<14} {'Micro':>8} {'Macro':>8} {'Median':>8} "
        f"{'Bot-10%':>8} {'μ-Macro':>9} {'Loss':>8}"
    )
    print(header)
    print("-" * 100)

    for exp_id, r in results.items():
        row = (
            f"{exp_id:<14} "
            f"{r['accuracy']*100:>7.2f}% "
            f"{r['macro_accuracy']*100:>7.2f}% "
            f"{r['median_per_class_accuracy']*100:>7.2f}% "
            f"{r['bottom_10_percent_accuracy']*100:>7.2f}% "
            f"{r['micro_macro_gap']*100:+>8.2f}% "
            f"{r['loss']:>8.4f}"
        )
        print(row)

    print("-" * 100)

    # Paired comparisons
    if "E0_STRICT" in results and "D3_STRICT" in results:
        d3_vs_e0 = (results["D3_STRICT"]["accuracy"]
                    - results["E0_STRICT"]["accuracy"]) * 100
        print(f"\nD3 vs E0 (micro): {d3_vs_e0:+.4f}pp")

    if "D3_STRICT" in results and "F0_STRICT" in results:
        f0_vs_d3 = (results["F0_STRICT"]["accuracy"]
                    - results["D3_STRICT"]["accuracy"]) * 100
        print(f"F0 vs D3 (micro): {f0_vs_d3:+.4f}pp")

    if "D3_STRICT" in results and "F1_STRICT" in results:
        f1_vs_d3 = (results["F1_STRICT"]["accuracy"]
                    - results["D3_STRICT"]["accuracy"]) * 100
        print(f"F1 vs D3 (micro): {f1_vs_d3:+.4f}pp")

    # Macro gap analysis
    print("\nMicro-Macro Gap Analysis:")
    for exp_id, r in results.items():
        gap = r["micro_macro_gap"] * 100
        flag = " ⚠ MAJORITY BIAS" if abs(gap) > 1.0 else ""
        print(f"  {exp_id}: micro - macro = {gap:+.2f}pp{flag}")

    print("\nDone. Re-evaluation results saved to each checkpoint directory.")


if __name__ == "__main__":
    main()
