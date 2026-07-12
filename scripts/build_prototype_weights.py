#!/usr/bin/env python3
"""Build per-sample prototype-confidence weights for the D3 training set.

Uses a frozen CLIP ViT-B/32 encoder to extract image features, builds
per-class 10%-trimmed centroids, and assigns each sample a weight in
[0.2, 1.0] based on within-class similarity and margin percentiles.

Usage:
    PYTHONPATH=. python scripts/build_prototype_weights.py \
        --config configs/d3_strict.yaml \
        --train-csv outputs/d3_strict/seed42/train.csv \
        --output-dir outputs/phase2/prototype_weights \
        --batch-size 256
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.clip_utils import load_openai_clip
from common.dataset import TrainImageDataset, seed_worker
from common.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def _extract_features(
    model: torch.nn.Module,
    visual: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_paths: list,
) -> dict:
    """Extract L2-normalized CLIP features, keyed by image path.

    Returns dict[image_path -> (feature_tensor, label_int)].
    """
    results = {}
    path_idx = 0
    for images, labels, paths in tqdm(loader, desc="Extracting features"):
        images = images.to(device, non_blocking=True)
        # Cast to match CLIP conv1 dtype
        dtype = visual.conv1.weight.dtype
        images = images.to(dtype)

        features = visual(images)
        # Handle spatial or token output
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])
        elif features.dim() == 3:
            features = features[:, 0, :]

        features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        features = features.cpu()

        for i in range(len(paths)):
            results[paths[i]] = {
                "feature": features[i],
                "label": labels[i].item(),
            }

    return results


def _build_centroids(
    features_by_class: dict,
    trim_ratio: float = 0.10,
) -> dict:
    """Build 10%-trimmed centroid per class.

    For each class, removes the trim_ratio fraction of samples farthest
    from the mean, then computes the mean of the remainder and L2-normalizes.
    """
    centroids = {}
    for cls, feats in features_by_class.items():
        feats_tensor = torch.stack(feats)
        mean = feats_tensor.mean(dim=0)
        # Distances to mean
        dists = torch.norm(feats_tensor - mean.unsqueeze(0), dim=1)
        n_keep = max(1, int(len(feats) * (1.0 - trim_ratio)))
        _, keep_idx = dists.topk(n_keep, largest=False)
        trimmed = feats_tensor[keep_idx]
        centroid = torch.nn.functional.normalize(trimmed.mean(dim=0), p=2, dim=-1)
        centroids[cls] = centroid
    return centroids


def _compute_weights(
    data: dict,
    centroids: dict,
    all_classes: list,
    min_weight: float = 0.2,
) -> list:
    """Compute per-sample prototype-confidence weights.

    For each sample:
      sim_i = z_i · p_{y_i}
      margin_i = sim_i - max_{c != y_i} (z_i · p_c)
      r_sim = within-class percentile of sim (higher = more typical)
      r_margin = within-class percentile of margin
      confidence = 0.5 * r_sim + 0.5 * r_margin
      weight = min_weight + (1.0 - min_weight) * confidence
    """
    centroids_tensor = torch.stack([centroids[c] for c in all_classes])  # (C, D)

    # Group data by class
    by_class = {}
    for path, info in data.items():
        cls = info["label"]
        by_class.setdefault(cls, []).append(path)

    all_paths = []
    all_weights = []
    all_sim = []
    all_margin = []
    all_r_sim = []
    all_r_margin = []

    for cls, paths in tqdm(sorted(by_class.items()), desc="Computing weights"):
        feats = torch.stack([data[p]["feature"] for p in paths])  # (N_c, D)
        sims = (feats * centroids[cls].unsqueeze(0)).sum(dim=1)  # (N_c,)

        # Margin: own similarity - max similarity to other classes
        all_sims_to_all = feats @ centroids_tensor.T  # (N_c, C)
        # Zero out own class
        cls_idx = all_classes.index(cls)
        all_sims_to_all[:, cls_idx] = -float("inf")
        max_other = all_sims_to_all.max(dim=1).values
        margins = sims - max_other

        # Within-class percentile ranks (0 = worst, 1 = best)
        r_sim = _within_class_percentile(sims)
        r_margin = _within_class_percentile(margins)

        confidence = 0.5 * r_sim + 0.5 * r_margin
        weights = min_weight + (1.0 - min_weight) * confidence

        for i, p in enumerate(paths):
            all_paths.append(p)
            all_weights.append(float(weights[i]))
            all_sim.append(float(sims[i]))
            all_margin.append(float(margins[i]))
            all_r_sim.append(float(r_sim[i]))
            all_r_margin.append(float(r_margin[i]))

    return all_paths, all_weights, all_sim, all_margin, all_r_sim, all_r_margin


def _within_class_percentile(values: torch.Tensor) -> torch.Tensor:
    """Compute within-class percentile ranks [0, 1].

    Uses PyTorch for consistency; handles the single-sample edge case.
    """
    n = len(values)
    if n <= 1:
        return torch.ones(n)
    # argsort → rank (0 = smallest, n-1 = largest)
    _, idx = values.sort()
    ranks = torch.zeros_like(values)
    for rank, i in enumerate(idx):
        ranks[i] = rank / (n - 1)
    return ranks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build prototype-confidence sample weights."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/phase2/prototype_weights")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--trim-ratio", type=float, default=0.10)
    parser.add_argument("--min-weight", type=float, default=0.2)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    setup_logging(str(output_dir), name="build_weights")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load CLIP ──────────────────────────────────────────────────
    logger.info("Loading CLIP ViT-B/32...")
    clip_model, preprocess = load_openai_clip(device, model_name="ViT-B/32")
    clip_model.eval()
    visual = clip_model.visual

    # ── Load training set ──────────────────────────────────────────
    train_csv = Path(args.train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    logger.info(f"Loading training images from {train_csv}...")
    dataset = TrainImageDataset(
        data_root=config["data"]["train_dir"],
        split_csv=str(train_csv),
        class_to_idx=None,  # auto-build from CSV
        transform=preprocess,
        return_path=True,
    )
    logger.info(f"Training set: {len(dataset)} images, {len(dataset.class_to_idx)} classes")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config["train"].get("num_workers", 8),
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    all_classes = sorted(class_to_idx.values())

    # ── Extract features ───────────────────────────────────────────
    logger.info("Extracting CLIP features...")
    data = _extract_features(clip_model, visual, loader, device, [])
    logger.info(f"Extracted features for {len(data)} images")

    # ── Build centroids ────────────────────────────────────────────
    logger.info(f"Building centroids (trim={args.trim_ratio})...")
    features_by_class = {}
    for path, info in data.items():
        cls = info["label"]
        features_by_class.setdefault(cls, []).append(info["feature"])

    centroids = _build_centroids(features_by_class, trim_ratio=args.trim_ratio)
    logger.info(f"Built centroids for {len(centroids)} classes")

    # ── Compute weights ────────────────────────────────────────────
    logger.info(f"Computing weights (min_weight={args.min_weight})...")
    paths, weights, sims, margins, r_sims, r_margins = _compute_weights(
        data, centroids, all_classes, min_weight=args.min_weight,
    )

    weights_t = torch.tensor(weights)
    logger.info(f"Weight stats: mean={weights_t.mean():.4f}, "
                f"std={weights_t.std():.4f}, "
                f"min={weights_t.min():.4f}, max={weights_t.max():.4f}")

    # ── Per-class statistics ───────────────────────────────────────
    by_class_stats = {}
    for path, w, label in zip(paths, weights, [data[p]["label"] for p in paths]):
        by_class_stats.setdefault(label, []).append(w)

    class_stats_rows = []
    for cls in sorted(by_class_stats):
        ws = torch.tensor(by_class_stats[cls])
        class_stats_rows.append({
            "class_idx": cls,
            "n_samples": len(ws),
            "mean_weight": float(ws.mean()),
            "std_weight": float(ws.std()),
            "p10": float(ws.quantile(0.10)),
            "p50": float(ws.median()),
            "p90": float(ws.quantile(0.90)),
        })

    # ── Weight distribution ────────────────────────────────────────
    dist = {
        "n_samples": len(weights),
        "mean": float(weights_t.mean()),
        "std": float(weights_t.std()),
        "min": float(weights_t.min()),
        "max": float(weights_t.max()),
        "p10": float(weights_t.quantile(0.10)),
        "p25": float(weights_t.quantile(0.25)),
        "p50": float(weights_t.median()),
        "p75": float(weights_t.quantile(0.75)),
        "p90": float(weights_t.quantile(0.90)),
        "n_below_0d3": int((weights_t < 0.3).sum()),
        "n_below_0d5": int((weights_t < 0.5).sum()),
        "n_above_0d9": int((weights_t > 0.9).sum()),
    }

    # ── Save sample_weights.json ───────────────────────────────────
    weights_dict = {}
    for p, w, s, m, rs, rm in zip(paths, weights, sims, margins, r_sims, r_margins):
        weights_dict[p] = {
            "weight": w,
            "own_similarity": s,
            "margin": m,
            "similarity_percentile": rs,
            "margin_percentile": rm,
            "label": data[p]["label"],
        }

    with open(output_dir / "sample_weights.json", "w") as f:
        json.dump(weights_dict, f, indent=2)
    logger.info(f"Saved sample_weights.json ({len(weights_dict)} entries)")

    # ── Save sample_weights.csv ────────────────────────────────────
    with open(output_dir / "sample_weights.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path", "label", "own_similarity", "best_other_similarity",
            "margin", "similarity_percentile", "margin_percentile", "weight",
        ])
        for p, w, s, m, rs, rm, label in zip(
            paths, weights, sims, margins, r_sims, r_margins,
            [data[p]["label"] for p in paths],
        ):
            # best_other = own_sim - margin
            best_other = s - m
            writer.writerow([p, label, s, best_other, m, rs, rm, w])
    logger.info(f"Saved sample_weights.csv")

    # ── Save class_statistics.csv ──────────────────────────────────
    with open(output_dir / "class_statistics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class_idx", "n_samples", "mean_weight", "std_weight",
            "p10", "p50", "p90",
        ])
        writer.writeheader()
        writer.writerows(class_stats_rows)
    logger.info(f"Saved class_statistics.csv ({len(class_stats_rows)} classes)")

    # ── Save weight_distribution.json ──────────────────────────────
    with open(output_dir / "weight_distribution.json", "w") as f:
        json.dump(dist, f, indent=2)
    logger.info(f"Saved weight_distribution.json")

    # ── Save manifest.json ─────────────────────────────────────────
    manifest = {
        "train_csv": str(train_csv.resolve()),
        "train_csv_sha256": _sha256_hex(train_csv),
        "num_samples": len(weights),
        "num_classes": len(class_to_idx),
        "trim_ratio": args.trim_ratio,
        "min_weight": args.min_weight,
        "sample_weights_sha256": _sha256_hex(output_dir / "sample_weights.json"),
        "weight_stats": dist,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest.json")

    logger.info("=" * 50)
    logger.info("Prototype weight generation complete.")
    logger.info(f"  Samples:     {len(weights)}")
    logger.info(f"  Classes:     {len(class_to_idx)}")
    logger.info(f"  Weight range: [{weights_t.min():.3f}, {weights_t.max():.3f}]")
    logger.info(f"  Weight mean:  {weights_t.mean():.3f}")
    logger.info(f"  Output:       {output_dir}")


if __name__ == "__main__":
    main()
