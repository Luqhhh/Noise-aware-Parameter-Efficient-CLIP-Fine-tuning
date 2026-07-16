#!/usr/bin/env python3
"""Epoch-0 Gate — verify that a PEFT init checkpoint reproduces parent predictions.

Usage:
    python3 -m common.epoch0_gate \
        --config configs/s_peft_e1_ln_1e6.yaml \
        --init-checkpoint outputs/w1_gce05_mixup/seed42/checkpoints/best.pt

Exit code 0 = gate passed.  Exit code 1 = gate failed (mismatch detected).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("epoch0_gate")


def build_model_from_config(config: dict, device: torch.device) -> torch.nn.Module:
    """Build a CLIPLinearClassifier from a project config."""
    from experiments.baseline.model import CLIPLinearClassifier

    model_cfg = config["model"]
    model = CLIPLinearClassifier(
        clip_model_name=model_cfg["clip_model_name"],
        num_classes=model_cfg["num_classes"],
        freeze_clip=model_cfg.get("freeze_clip", True),
        feature_dim=model_cfg.get("feature_dim", 512),
        dropout=model_cfg.get("dropout", 0.0),
        unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 0),
        train_ln_post=model_cfg.get("train_ln_post", False),
        train_visual_proj=model_cfg.get("train_visual_proj", False),
    )
    model.to(device)
    return model


def load_checkpoint_weights(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
) -> dict:
    """Load model weights from a checkpoint.  Returns the full checkpoint dict."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt


def gate_prediction_identity(
    model: torch.nn.Module,
    parent_model: torch.nn.Module,
    val_loader,
    device: torch.device,
    tolerance: float = 1e-5,
) -> dict:
    """Compare predictions between model (post-init) and parent.

    Returns a dict with keys:
        prediction_mismatch: int  (must be 0)
        max_logit_diff: float
        val_accuracy_diff: float  (must be <= 0.02pp = 0.0002)
    """
    model.eval()
    parent_model.eval()

    mismatches = 0
    total = 0
    max_logit_diff = 0.0
    model_correct = 0
    parent_correct = 0

    with torch.no_grad():
        for batch in val_loader:
            images, labels, _paths = _unpack_batch(batch, device)
            logits = model(images)
            p_logits = parent_model(images)

            preds = logits.argmax(dim=1)
            p_preds = p_logits.argmax(dim=1)

            mismatches += int((preds != p_preds).sum().item())
            total += images.size(0)

            diff = (logits - p_logits).abs().max().item()
            if diff > max_logit_diff:
                max_logit_diff = diff

            model_correct += int((preds == labels).sum().item())
            parent_correct += int((p_preds == labels).sum().item())

    model_acc = model_correct / total if total > 0 else 0.0
    parent_acc = parent_correct / total if total > 0 else 0.0

    return {
        "prediction_mismatch": mismatches,
        "total_samples": total,
        "max_logit_diff": float(max_logit_diff),
        "model_accuracy": float(model_acc),
        "parent_accuracy": float(parent_acc),
        "val_accuracy_diff": float(abs(model_acc - parent_acc)),
    }


def gate_classifier_weight_match(
    model: torch.nn.Module,
    parent_model: torch.nn.Module,
    tolerance: float = 1e-8,
) -> dict:
    """Verify classifier weights are identical between model and parent."""
    w = model.classifier.weight.data
    p_w = parent_model.classifier.weight.data
    max_diff = (w - p_w).abs().max().item()

    b = model.classifier.bias.data
    p_b = parent_model.classifier.bias.data
    max_bias_diff = (b - p_b).abs().max().item()

    return {
        "classifier_weight_max_diff": float(max_diff),
        "classifier_bias_max_diff": float(max_bias_diff),
        "classifier_weight_match": bool(max_diff <= tolerance),
    }


def gate_feature_identity(
    model: torch.nn.Module,
    parent_model: torch.nn.Module,
    val_loader,
    device: torch.device,
    tolerance: float = 1e-6,
    max_samples: int = 1000,
) -> dict:
    """Verify features from model and parent are near-identical."""
    model.eval()
    parent_model.eval()

    all_cos_sims = []
    samples_seen = 0

    with torch.no_grad():
        for batch in val_loader:
            if samples_seen >= max_samples:
                break
            images, _labels, _paths = _unpack_batch(batch, device)
            f = F.normalize(model.encode_image(images).float(), p=2, dim=-1)
            p_f = F.normalize(parent_model.encode_image(images).float(), p=2, dim=-1)
            cos_sim = (f * p_f).sum(dim=1)
            all_cos_sims.extend(cos_sim.cpu().tolist())
            samples_seen += images.size(0)

    cos_t = torch.tensor(all_cos_sims)
    min_cos = float(cos_t.min().item())
    mean_cos = float(cos_t.mean().item())

    return {
        "feature_min_cosine": min_cos,
        "feature_mean_cosine": mean_cos,
        "feature_samples_checked": samples_seen,
        "feature_identity_ok": bool(min_cos >= (1.0 - tolerance)),
    }


# ── Helpers ────────────────────────────────────────────────────────────

def _unpack_batch(batch_data, device: torch.device):
    if len(batch_data) == 3:
        images, labels, paths = batch_data
    elif len(batch_data) == 2:
        images, labels = batch_data
        paths = None
    else:
        raise ValueError(f"Unexpected batch length: {len(batch_data)}")
    return images.to(device, non_blocking=True), \
        labels.to(device, non_blocking=True), \
        paths


def build_val_loader(config: dict):
    """Build a validation DataLoader from a project config."""
    from common.dataset import TrainImageDataset, seed_worker
    from common.class_mapping import load_or_generate_mapping
    from torch.utils.data import DataLoader
    import clip

    data_cfg = config["data"]
    eval_cfg = config["eval"]
    train_cfg = config["train"]

    split_dir = data_cfg["split_dir"]
    class_mapping_path = data_cfg.get(
        "class_mapping_path", split_dir
    )

    _, preprocess = clip.load(
        config["model"]["clip_model_name"], device="cpu", jit=False
    )

    class_to_idx, _ = load_or_generate_mapping(
        metadata_dir=class_mapping_path,
        train_dir=data_cfg["train_dir"],
        expected_num_classes=config["model"]["num_classes"],
    )

    val_dataset = TrainImageDataset(
        data_root=data_cfg["train_dir"],
        split_csv=str(Path(split_dir) / "val.csv"),
        class_to_idx=class_to_idx,
        transform=preprocess,
        return_path=True,
    )

    g = torch.Generator()
    g.manual_seed(data_cfg.get("train_seed", data_cfg.get("seed", 42)))

    return DataLoader(
        val_dataset,
        batch_size=eval_cfg.get("batch_size", 256),
        shuffle=False,
        num_workers=min(train_cfg.get("num_workers", 4), 4),
        pin_memory=train_cfg.get("pin_memory", True),
        worker_init_fn=seed_worker,
        generator=g,
    )


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Epoch-0 Gate — verify PEFT init checkpoint fidelity"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment config YAML (defines model architecture)."
    )
    parser.add_argument(
        "--init-checkpoint", required=True,
        help="Path to parent checkpoint to verify."
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write gate result as JSON."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load config
    from common.utils import load_config
    config = load_config(args.config)

    # Build parent model and load checkpoint
    logger.info("Loading parent checkpoint: %s", args.init_checkpoint)
    parent = build_model_from_config(config, device)
    _ckpt = load_checkpoint_weights(parent, args.init_checkpoint, device)

    # Build child model and load SAME checkpoint
    logger.info("Building child model from same checkpoint...")
    child = build_model_from_config(config, device)
    load_checkpoint_weights(child, args.init_checkpoint, device)

    # Apply PEFT if configured (this should NOT change predictions at epoch 0)
    peft_cfg = config.get("peft", {})
    if peft_cfg:
        from common.peft import apply_peft
        logger.info("Applying PEFT: %s", peft_cfg.get("strategy", "unknown"))
        apply_peft(child, peft_cfg, config["train"].get("lr", 5e-3))
        # After PEFT, reload classifier weights from parent to ensure identity
        child.classifier.load_state_dict(parent.classifier.state_dict())

    # Build val loader
    val_loader = build_val_loader(config)

    # ── Run gates ──
    all_passed = True

    # Gate 1: Prediction identity
    logger.info("Gate 1/3: Prediction identity...")
    pred_result = gate_prediction_identity(child, parent, val_loader, device)
    pred_ok = pred_result["prediction_mismatch"] == 0
    all_passed &= pred_ok
    logger.info(
        "  mismatches=%d/%d | max_logit_diff=%.2e | acc_diff=%.6f | %s",
        pred_result["prediction_mismatch"], pred_result["total_samples"],
        pred_result["max_logit_diff"], pred_result["val_accuracy_diff"],
        "PASS" if pred_ok else "FAIL",
    )

    # Gate 2: Classifier weight match
    logger.info("Gate 2/3: Classifier weight match...")
    weight_result = gate_classifier_weight_match(child, parent)
    weight_ok = weight_result["classifier_weight_match"]
    all_passed &= weight_ok
    logger.info(
        "  weight_max_diff=%.2e | bias_max_diff=%.2e | %s",
        weight_result["classifier_weight_max_diff"],
        weight_result["classifier_bias_max_diff"],
        "PASS" if weight_ok else "FAIL",
    )

    # Gate 3: Feature identity
    logger.info("Gate 3/3: Feature identity (checking %d samples)...", 1000)
    feat_result = gate_feature_identity(child, parent, val_loader, device)
    feat_ok = feat_result["feature_identity_ok"]
    all_passed &= feat_ok
    logger.info(
        "  min_cosine=%.8f | mean_cosine=%.8f | %s",
        feat_result["feature_min_cosine"],
        feat_result["feature_mean_cosine"],
        "PASS" if feat_ok else "FAIL",
    )

    # ── Final verdict ──
    result = {
        "gate_passed": all_passed,
        "prediction": pred_result,
        "classifier_weight": weight_result,
        "feature": feat_result,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Gate result written to %s", output_path)

    if all_passed:
        logger.info("EPOCH-0 GATE: PASSED")
        sys.exit(0)
    else:
        logger.error("EPOCH-0 GATE: FAILED — do not proceed with this experiment")
        sys.exit(1)


if __name__ == "__main__":
    main()
