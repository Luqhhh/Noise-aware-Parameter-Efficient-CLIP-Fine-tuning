#!/usr/bin/env python3
"""
Analyze checkpoint disagreement between two experiments on the validation set.

Computes per-sample metrics for both checkpoints, partitions into four groups
(both_correct, ref_only_correct, cand_only_correct, both_wrong), and saves
detailed CSVs plus summary statistics.

Usage:
    python tools/analyze_checkpoint_disagreement.py \\
        --audit outputs/audit/d3_vs_b2/audit.json \\
        --reference-name d3 --candidate-name b2 \\
        --reference-config configs/ref.yaml \\
        --candidate-config configs/gce_q07.yaml \\
        --reference-ckpt outputs/ref/seed42/checkpoints/best.pt \\
        --candidate-ckpt outputs/gce_q07/seed42/checkpoints/best.pt \\
        --train-feature-bank outputs/ref/seed42/feature_banks/train_feature_bank.pt \\
        --val-feature-bank outputs/ref/seed42/feature_banks/val_feature_bank.pt \\
        --output-dir outputs/d3_vs_b2_disagreement \\
        --device cuda
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.diagnostic_metrics import (
    build_trimmed_class_prototypes,
    chunked_topk_cosine,
    jensen_shannon_divergence,
    knn_label_metrics,
    per_sample_cross_entropy,
    per_sample_gce,
    prototype_metrics,
    softmax_confidence_margin_entropy,
)
from common.utils import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Analyze checkpoint disagreement between two experiments"
    )
    p.add_argument("--audit", required=True, help="Path to audit JSON")
    p.add_argument(
        "--reference-name",
        default=None,
        help="Reference experiment name (e.g., d3). Used in column names.",
    )
    p.add_argument(
        "--candidate-name",
        default=None,
        help="Candidate experiment name (e.g., b2). Used in column names.",
    )
    p.add_argument("--reference-config", required=True)
    p.add_argument("--candidate-config", required=True)
    p.add_argument("--reference-ckpt", required=True)
    p.add_argument("--candidate-ckpt", required=True)
    p.add_argument("--train-feature-bank", required=True)
    p.add_argument("--val-feature-bank", required=True)
    p.add_argument(
        "--knn-k", type=int, default=20, help="Number of kNN neighbors (default: 20)"
    )
    p.add_argument(
        "--prototype-trim",
        type=float,
        default=0.10,
        help="Trim fraction for robust prototypes (default: 0.10)",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--device", default="cuda", help='Torch device string (default: "cuda")'
    )
    return p.parse_args()


def load_audit(audit_path: str) -> dict:
    """Load audit JSON file."""
    with open(audit_path) as f:
        return json.load(f)


def load_feature_bank(path: str) -> dict:
    """Load a feature bank .pt file.

    Returns dict with keys: features, labels, paths, image_sha256, class_names,
    flip_features (val only), etc.
    """
    return torch.load(path, map_location="cpu", weights_only=True)


def compute_logits_fast(
    features: torch.Tensor,
    ckpt: dict,
    device: torch.device,
) -> torch.Tensor:
    """Compute logits via the classifier head directly (fast path).

    Args:
        features: (N, D) L2-normalized features on CPU.
        ckpt: Checkpoint dict with 'model_state_dict' containing
              'classifier.weight' and 'classifier.bias'.
        device: Torch device.

    Returns:
        (N, C) logits on CPU.
    """
    weight = ckpt["model_state_dict"]["classifier.weight"].to(device)
    bias = ckpt["model_state_dict"]["classifier.bias"].to(device)
    features_gpu = F.normalize(features.float().to(device), dim=-1)
    logits = F.linear(features_gpu, weight, bias)
    return logits.cpu()


def verify_fast_path(
    val_features: torch.Tensor,
    ref_ckpt: dict,
    ref_config: dict,
    device: torch.device,
    n_samples: int = 32,
) -> tuple:
    """Verify fast-path logits match full model ``forward_features``.

    Args:
        val_features: (N, D) validation features on CPU.
        ref_ckpt: Reference checkpoint dict.
        ref_config: Reference experiment config.
        device: Torch device.
        n_samples: Number of samples to verify (default: 32).

    Returns:
        (passed, max_diff) tuple.
    """
    from experiments.baseline.model import build_model

    model, _ = build_model(ref_config, device)
    model.load_state_dict(ref_ckpt["model_state_dict"])
    model.eval()

    features_subset = val_features[:n_samples].float().to(device)

    # Fast path
    weight = ref_ckpt["model_state_dict"]["classifier.weight"].to(device)
    bias = ref_ckpt["model_state_dict"]["classifier.bias"].to(device)
    features_norm = F.normalize(features_subset, dim=-1)
    fast_logits = F.linear(features_norm, weight, bias)

    # Full model
    with torch.no_grad():
        full_logits = model.forward_features(features_subset).cpu()

    max_diff = (fast_logits.cpu() - full_logits).abs().max().item()
    return max_diff < 1e-6, max_diff


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    # Resolve reference/candidate names
    ref_name = args.reference_name or "reference"
    cand_name = args.candidate_name or "candidate"

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading audit: %s", args.audit)
    audit = load_audit(args.audit)
    max_visual_abs_diff = audit.get("max_visual_abs_diff", -1.0)
    use_fast_path = max_visual_abs_diff == 0.0
    logger.info(
        "max_visual_abs_diff=%.6f, use_fast_path=%s",
        max_visual_abs_diff,
        use_fast_path,
    )

    logger.info("Loading train feature bank: %s", args.train_feature_bank)
    train_bank = load_feature_bank(args.train_feature_bank)
    train_features = train_bank["features"]  # (N_train, 512)
    train_labels = train_bank["labels"]  # (N_train,)

    logger.info("Loading val feature bank: %s", args.val_feature_bank)
    val_bank = load_feature_bank(args.val_feature_bank)
    val_features = val_bank["features"]  # (N_val, 512)
    val_flip_features = val_bank["flip_features"]  # (N_val, 512)
    val_labels = val_bank["labels"]  # (N_val,)
    val_paths = val_bank["paths"]  # list of str
    val_class_names = val_bank.get("class_names", None)
    val_sha256 = val_bank.get("image_sha256", None)
    train_sha256 = train_bank.get("image_sha256", None)

    n_val = val_features.shape[0]
    logger.info("Validation samples: %d", n_val)

    logger.info("Loading reference checkpoint: %s", args.reference_ckpt)
    ref_ckpt = torch.load(
        args.reference_ckpt, map_location="cpu", weights_only=False
    )

    logger.info("Loading candidate checkpoint: %s", args.candidate_ckpt)
    cand_ckpt = torch.load(
        args.candidate_ckpt, map_location="cpu", weights_only=False
    )

    # Load configs
    ref_config = load_config(args.reference_config)
    cand_config = load_config(args.candidate_config)
    num_classes = ref_config["model"]["num_classes"]

    # Load idx_to_class for class summary
    class_mapping_path = ref_config["data"].get(
        "class_mapping_path", ref_config["data"]["split_dir"]
    )
    idx_to_class_path = Path(class_mapping_path) / "idx_to_class.json"
    if idx_to_class_path.exists():
        with open(idx_to_class_path) as f:
            idx_to_class = json.load(f)
    else:
        idx_to_class = None
        logger.warning("idx_to_class.json not found at %s", idx_to_class_path)

    # ------------------------------------------------------------------
    # 2. Compute logits
    # ------------------------------------------------------------------
    device = torch.device(args.device)

    ref_logits = None
    cand_logits = None
    shared_feature_fast_path = False

    if use_fast_path:
        logger.info("Using fast-path logit computation...")
        ref_logits = compute_logits_fast(val_features, ref_ckpt, device)
        cand_logits = compute_logits_fast(val_features, cand_ckpt, device)

        # Verify fast path
        n_verify = min(32, n_val)
        logger.info("Verifying fast path on %d samples...", n_verify)
        ok, max_diff = verify_fast_path(
            val_features, ref_ckpt, ref_config, device, n_verify
        )
        if ok:
            logger.info("Fast path verified: max_diff=%.2e (OK)", max_diff)
            shared_feature_fast_path = True
        else:
            logger.warning(
                "Fast path verification FAILED: max_diff=%.2e >= 1e-6. "
                "Falling back to full model.",
                max_diff,
            )
            shared_feature_fast_path = False
            ref_logits = None
            cand_logits = None

    if not use_fast_path or not shared_feature_fast_path:
        logger.info("Using full model inference in batches...")
        from experiments.baseline.model import build_model

        batch_size = ref_config.get("eval", {}).get("batch_size", 256)

        for name, ckpt, config in [
            ("reference", ref_ckpt, ref_config),
            ("candidate", cand_ckpt, cand_config),
        ]:
            model, _ = build_model(config, device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            all_logits = []
            for i in range(0, n_val, batch_size):
                end = min(i + batch_size, n_val)
                batch_feats = val_features[i:end].float().to(device)
                with torch.no_grad():
                    logits_out = model.forward_features(batch_feats).cpu()
                all_logits.append(logits_out)

            logits_tensor = torch.cat(all_logits, dim=0)
            if name == "reference":
                ref_logits = logits_tensor
            else:
                cand_logits = logits_tensor

    # ------------------------------------------------------------------
    # 3. Compute all per-sample metrics
    # ------------------------------------------------------------------
    logger.info("Computing per-sample metrics...")

    # -- Predictions and correctness --
    _, _, _, ref_preds = softmax_confidence_margin_entropy(ref_logits)
    _, _, _, cand_preds = softmax_confidence_margin_entropy(cand_logits)

    ref_preds_np = ref_preds.numpy()
    cand_preds_np = cand_preds.numpy()
    val_labels_np = val_labels.numpy()

    ref_correct = ref_preds_np == val_labels_np
    cand_correct = cand_preds_np == val_labels_np

    # -- Confidence, margin, entropy --
    ref_confidence, ref_margin, ref_entropy, _ = softmax_confidence_margin_entropy(
        ref_logits
    )
    cand_confidence, cand_margin, cand_entropy, _ = (
        softmax_confidence_margin_entropy(cand_logits)
    )

    # -- Noisy label probability --
    ref_probs = F.softmax(ref_logits, dim=1)
    cand_probs = F.softmax(cand_logits, dim=1)
    ref_noisy_prob = (
        ref_probs.gather(1, val_labels.unsqueeze(1)).squeeze(1)
    )
    cand_noisy_prob = (
        cand_probs.gather(1, val_labels.unsqueeze(1)).squeeze(1)
    )

    # -- CE loss --
    ref_ce = per_sample_cross_entropy(ref_logits, val_labels)
    cand_ce = per_sample_cross_entropy(cand_logits, val_labels)

    # -- GCE (q=0.7) --
    ref_gce = per_sample_gce(ref_logits, val_labels, q=0.7)
    cand_gce = per_sample_gce(cand_logits, val_labels, q=0.7)

    # -- Flip logits --
    logger.info("Computing flip metrics...")
    ref_flip_logits = compute_logits_fast(val_flip_features, ref_ckpt, device)
    cand_flip_logits = compute_logits_fast(val_flip_features, cand_ckpt, device)

    _, _, _, ref_flip_pred = softmax_confidence_margin_entropy(ref_flip_logits)
    _, _, _, cand_flip_pred = softmax_confidence_margin_entropy(cand_flip_logits)

    ref_flip_pred_agree = ref_preds_np == ref_flip_pred.numpy()
    cand_flip_pred_agree = cand_preds_np == cand_flip_pred.numpy()

    # -- Flip JSD --
    ref_flip_jsd = jensen_shannon_divergence(ref_logits, ref_flip_logits)
    cand_flip_jsd = jensen_shannon_divergence(cand_logits, cand_flip_logits)

    # -- TTA (average original + flip logits) --
    ref_tta_logits = (ref_logits + ref_flip_logits) / 2.0
    cand_tta_logits = (cand_logits + cand_flip_logits) / 2.0

    _, _, _, ref_tta_pred = softmax_confidence_margin_entropy(ref_tta_logits)
    _, _, _, cand_tta_pred = softmax_confidence_margin_entropy(cand_tta_logits)

    ref_tta_correct = ref_tta_pred.numpy() == val_labels_np
    cand_tta_correct = cand_tta_pred.numpy() == val_labels_np

    # -- CLIP flip cosine (features are L2-normalized, so dot = cosine) --
    clip_flip_cosine = (val_features * val_flip_features).sum(dim=1).numpy()

    # -- kNN --
    logger.info("Computing kNN (k=%d)...", args.knn_k)
    knn_idx, knn_sim = chunked_topk_cosine(
        val_features,
        train_features,
        k=args.knn_k,
        device=args.device,
    )

    knn_metrics = knn_label_metrics(
        knn_idx, train_labels, val_labels, num_classes
    )
    knn_label_agreement = knn_metrics["knn_label_agreement"].numpy()
    knn_majority_label = knn_metrics["knn_majority_label"].numpy().astype(int)
    knn_majority_fraction = knn_metrics["knn_majority_fraction"].numpy()

    nearest_train_similarity = knn_sim[:, 0].numpy()
    mean_topk_similarity = knn_sim.mean(dim=1).numpy()

    # kNN support for each prediction
    train_labels_np = train_labels.numpy()
    knn_labels = train_labels_np[knn_idx.numpy()]  # (N_val, k)
    knn_support_d3 = (knn_labels == ref_preds_np[:, None]).mean(axis=1)
    knn_support_b2 = (knn_labels == cand_preds_np[:, None]).mean(axis=1)

    # -- Prototypes --
    logger.info(
        "Building robust prototypes (trim=%.2f)...", args.prototype_trim
    )
    prototypes = build_trimmed_class_prototypes(
        train_features,
        train_labels,
        num_classes,
        args.prototype_trim,
    )

    proto_met = prototype_metrics(val_features, val_labels, prototypes)

    # Prototype similarity for each model's predictions
    proto_sims = torch.mm(val_features, prototypes.T)  # (N_val, C)
    ref_proto_sim = (
        proto_sims.gather(
            1, torch.from_numpy(ref_preds_np).unsqueeze(1)
        )
        .squeeze(1)
        .numpy()
    )
    cand_proto_sim = (
        proto_sims.gather(
            1, torch.from_numpy(cand_preds_np).unsqueeze(1)
        )
        .squeeze(1)
        .numpy()
    )

    # ------------------------------------------------------------------
    # 4. Partition into 4 groups
    # ------------------------------------------------------------------
    group = np.full(n_val, "both_wrong", dtype=object)
    both_correct_mask = ref_correct & cand_correct
    ref_only_mask = ref_correct & ~cand_correct
    cand_only_mask = ~ref_correct & cand_correct

    group[both_correct_mask] = "both_correct"
    group[ref_only_mask] = f"{ref_name}_only_correct"
    group[cand_only_mask] = f"{cand_name}_only_correct"
    # "both_wrong" is the default

    diff_count = int(ref_only_mask.sum()) - int(cand_only_mask.sum())

    logger.info("Group counts:")
    logger.info("  both_correct:        %d", both_correct_mask.sum())
    logger.info("  %s_only_correct: %d", ref_name, ref_only_mask.sum())
    logger.info("  %s_only_correct: %d", cand_name, cand_only_mask.sum())
    logger.info("  both_wrong:          %d", (~ref_correct & ~cand_correct).sum())
    logger.info("  %s_only - %s_only = %d", ref_name, cand_name, diff_count)

    # Verify groups are mutually exclusive and exhaustive
    total_grouped = (
        both_correct_mask.sum()
        + ref_only_mask.sum()
        + cand_only_mask.sum()
        + (~ref_correct & ~cand_correct).sum()
    )
    assert total_grouped == n_val, (
        f"Groups not exhaustive: {total_grouped} != {n_val}"
    )
    logger.info("Groups verified: mutually exclusive and exhaustive.")

    # ------------------------------------------------------------------
    # 5. Build DataFrame with ALL columns
    # ------------------------------------------------------------------
    logger.info("Building DataFrame with %d columns...", 50)

    # Resolve names for empty columns
    empty_str_list = [""] * n_val

    df = pd.DataFrame(
        {
            "sample_index": np.arange(n_val),
            "image_path": val_paths,
            "image_sha256": (
                val_sha256 if val_sha256 is not None else empty_str_list
            ),
            "class_name": (
                val_class_names
                if val_class_names is not None
                else empty_str_list
            ),
            "noisy_label": val_labels_np,
            # Predictions
            f"{ref_name}_pred": ref_preds_np,
            f"{cand_name}_pred": cand_preds_np,
            # Correctness
            f"{ref_name}_correct": ref_correct,
            f"{cand_name}_correct": cand_correct,
            # Group
            "group": group,
            # Confidence / margin / entropy
            f"{ref_name}_confidence": ref_confidence.numpy(),
            f"{cand_name}_confidence": cand_confidence.numpy(),
            f"{ref_name}_margin": ref_margin.numpy(),
            f"{cand_name}_margin": cand_margin.numpy(),
            f"{ref_name}_entropy": ref_entropy.numpy(),
            f"{cand_name}_entropy": cand_entropy.numpy(),
            # Noisy label probability
            f"{ref_name}_noisy_label_probability": ref_noisy_prob.numpy(),
            f"{cand_name}_noisy_label_probability": cand_noisy_prob.numpy(),
            # CE loss
            f"{ref_name}_ce_loss": ref_ce.numpy(),
            f"{cand_name}_ce_loss": cand_ce.numpy(),
            # GCE (q=0.7)
            f"{ref_name}_gce_q07": ref_gce.numpy(),
            f"{cand_name}_gce_q07": cand_gce.numpy(),
            # Flip
            f"{ref_name}_flip_pred": ref_flip_pred.numpy(),
            f"{cand_name}_flip_pred": cand_flip_pred.numpy(),
            f"{ref_name}_flip_pred_agree": ref_flip_pred_agree,
            f"{cand_name}_flip_pred_agree": cand_flip_pred_agree,
            f"{ref_name}_flip_jsd": ref_flip_jsd.numpy(),
            f"{cand_name}_flip_jsd": cand_flip_jsd.numpy(),
            # TTA
            f"{ref_name}_tta_pred": ref_tta_pred.numpy(),
            f"{cand_name}_tta_pred": cand_tta_pred.numpy(),
            f"{ref_name}_tta_correct": ref_tta_correct,
            f"{cand_name}_tta_correct": cand_tta_correct,
            # CLIP flip
            "clip_flip_cosine": clip_flip_cosine,
            # kNN
            "knn_label_agreement": knn_label_agreement,
            "knn_majority_label": knn_majority_label,
            "knn_majority_fraction": knn_majority_fraction,
            f"knn_support_{ref_name}_pred": knn_support_d3,
            f"knn_support_{cand_name}_pred": knn_support_b2,
            "nearest_train_similarity": nearest_train_similarity,
            "mean_topk_similarity": mean_topk_similarity,
            # Prototypes
            "prototype_label_similarity": proto_met[
                "prototype_label_similarity"
            ].numpy(),
            "prototype_top1_label": proto_met["prototype_top1_label"]
            .numpy()
            .astype(int),
            "prototype_top1_similarity": proto_met[
                "prototype_top1_similarity"
            ].numpy(),
            "prototype_second_similarity": proto_met[
                "prototype_second_similarity"
            ].numpy(),
            "prototype_margin": proto_met["prototype_margin"].numpy(),
            "prototype_supports_noisy_label": proto_met[
                "prototype_supports_noisy_label"
            ].numpy(),
            f"prototype_similarity_{ref_name}_pred": ref_proto_sim,
            f"prototype_similarity_{cand_name}_pred": cand_proto_sim,
        }
    )

    # -- Cross-class duplicate conflict and dedup metadata --
    dedup_available = (
        val_sha256 is not None and train_sha256 is not None
    )
    df["dedup_metadata_available"] = dedup_available

    if dedup_available:
        logger.info("Computing cross-class duplicate conflicts...")
        sha_to_labels = defaultdict(set)
        for sha, label in zip(train_sha256, train_labels.numpy()):
            sha_to_labels[sha].add(int(label))

        conflict = np.zeros(n_val, dtype=bool)
        for i, sha in enumerate(val_sha256):
            if sha in sha_to_labels:
                train_labels_for_sha = sha_to_labels[sha]
                val_label = int(val_labels_np[i])
                if val_label not in train_labels_for_sha:
                    conflict[i] = True
                elif len(train_labels_for_sha) > 1:
                    conflict[i] = True
        df["cross_class_duplicate_conflict"] = conflict
    else:
        df["cross_class_duplicate_conflict"] = False

    logger.info(
        "DataFrame shape: %d rows x %d columns", df.shape[0], df.shape[1]
    )

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving outputs to %s", output_dir)

    # Full metrics
    df.to_csv(output_dir / "sample_metrics.csv", index=False)
    logger.info("Saved sample_metrics.csv (%d rows)", len(df))

    # Per-group CSVs
    group_csv_map = {
        "both_correct": both_correct_mask,
        f"{ref_name}_only_correct": ref_only_mask,
        f"{cand_name}_only_correct": cand_only_mask,
        "both_wrong": ~ref_correct & ~cand_correct,
    }
    for group_name, mask in group_csv_map.items():
        g_df = df[mask].copy()
        g_df.to_csv(output_dir / f"{group_name}.csv", index=False)
        logger.info("Saved %s.csv (%d rows)", group_name, len(g_df))

    # Group summary JSON
    ref_acc = float(ref_correct.mean())
    cand_acc = float(cand_correct.mean())
    group_summary = {
        "reference_name": ref_name,
        "candidate_name": cand_name,
        "total_samples": int(n_val),
        "both_correct": int(both_correct_mask.sum()),
        f"{ref_name}_only_correct": int(ref_only_mask.sum()),
        f"{cand_name}_only_correct": int(cand_only_mask.sum()),
        "both_wrong": int((~ref_correct & ~cand_correct).sum()),
        f"{ref_name}_correct": int(ref_correct.sum()),
        f"{cand_name}_correct": int(cand_correct.sum()),
        f"{ref_name}_accuracy": ref_acc,
        f"{cand_name}_accuracy": cand_acc,
        f"{ref_name}_only_minus_{cand_name}_only": diff_count,
        "shared_feature_fast_path": shared_feature_fast_path,
    }

    with open(output_dir / "group_summary.json", "w") as f:
        json.dump(group_summary, f, indent=2)
    logger.info("Saved group_summary.json")

    # Class summary CSV
    class_summary_rows = []
    for c in range(num_classes):
        c_mask = val_labels_np == c
        c_count = int(c_mask.sum())
        if c_count > 0:
            ref_c_acc = float(ref_correct[c_mask].mean())
            cand_c_acc = float(cand_correct[c_mask].mean())
        else:
            ref_c_acc = 0.0
            cand_c_acc = 0.0
        class_name_str = (
            idx_to_class.get(str(c), str(c))
            if idx_to_class is not None
            else str(c)
        )
        class_summary_rows.append(
            {
                "class_index": c,
                "class_name": class_name_str,
                "count": c_count,
                f"{ref_name}_accuracy": ref_c_acc,
                f"{cand_name}_accuracy": cand_c_acc,
            }
        )

    class_summary_df = pd.DataFrame(class_summary_rows)
    class_summary_df.to_csv(output_dir / "class_summary.csv", index=False)
    logger.info("Saved class_summary.csv (%d rows)", len(class_summary_df))

    # ------------------------------------------------------------------
    # 7. Print final summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Total validation samples: %d", n_val)
    logger.info(
        "%s accuracy:  %.4f (%d/%d)", ref_name, ref_acc, ref_correct.sum(), n_val
    )
    logger.info(
        "%s accuracy: %.4f (%d/%d)",
        cand_name,
        cand_acc,
        cand_correct.sum(),
        n_val,
    )
    logger.info(
        "both_correct:     %d (%.2f%%)",
        both_correct_mask.sum(),
        100.0 * both_correct_mask.mean(),
    )
    logger.info(
        "%s_only_correct: %d (%.2f%%)",
        ref_name,
        ref_only_mask.sum(),
        100.0 * ref_only_mask.mean(),
    )
    logger.info(
        "%s_only_correct: %d (%.2f%%)",
        cand_name,
        cand_only_mask.sum(),
        100.0 * cand_only_mask.mean(),
    )
    logger.info(
        "both_wrong:       %d (%.2f%%)",
        (~ref_correct & ~cand_correct).sum(),
        100.0 * (~ref_correct & ~cand_correct).mean(),
    )
    logger.info(
        "%s_only - %s_only = %d", ref_name, cand_name, diff_count
    )
    logger.info("Shared feature fast path: %s", shared_feature_fast_path)
    logger.info("=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()
