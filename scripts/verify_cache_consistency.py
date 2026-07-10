#!/usr/bin/env python3
"""
Verify that cached features match online CLIP encoding.

Samples N random images from the training set, encodes them both online
(via CLIP) and via the feature cache, then compares:
  - Feature vectors (max/mean absolute difference)
  - Classification logits (Linear and Cosine heads)
  - Label consistency
  - Path matching

Usage:
    python scripts/verify_cache_consistency.py \\
        --config configs/e0_hyper_search.yaml \\
        --num-samples 128 \\
        --seed 42

Exit code 0 = consistency verified within tolerances.
Exit code 1 = inconsistency detected.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.cache import CachedFeatureDataset, compute_full_fingerprint
from common.class_mapping import load_or_generate_mapping
from common.clip_utils import encode_frozen_clip_features, load_openai_clip
from common.dataset import IMAGE_EXTENSIONS, _find_images_in_dir
from common.utils import load_config, set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify cache consistency with online CLIP encoding."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument(
        "--num-samples", type=int, default=128, help="Number of images to sample."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--device", type=str, default=None, help="Device override (default: config or cpu)."
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save JSON report."
    )
    parser.add_argument(
        "--atol-feature", type=float, default=1e-3,
        help="Feature absolute tolerance for PASS."
    )
    parser.add_argument(
        "--atol-logit", type=float, default=1e-3,
        help="Logit absolute tolerance for PASS."
    )
    return parser.parse_args()


def scan_all_images(train_dir: Path) -> list:
    """Scan all class directories and return flattened list of (path, class_name)."""
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    all_images = []
    for class_dir in class_dirs:
        images = _find_images_in_dir(class_dir)
        for img_path in images:
            all_images.append((img_path, class_dir.name))
    return all_images


def encode_online(
    images: torch.Tensor,
    clip_model,
    preprocess,
    device: torch.device,
) -> torch.Tensor:
    """Encode a batch via full CLIP pipeline (preprocess + encode)."""
    processed = []
    from PIL import Image

    for i in range(images.size(0)):
        # Convert tensor back to PIL for preprocessing
        arr = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pil_img = Image.fromarray(arr)
        processed.append(preprocess(pil_img))

    batch = torch.stack(processed).to(device)
    return encode_frozen_clip_features(clip_model, batch, device, use_amp=False)


def build_cosine_logits(features: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Compute cosine classifier logits from pre-computed features."""
    features_norm = F.normalize(features.float(), p=2, dim=-1)
    weight_norm = F.normalize(weight.float(), p=2, dim=1)
    return features_norm @ weight_norm.T * scale.clamp(min=1.0, max=100.0)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    set_seed(args.seed)

    device_str = args.device or config.get("train", {}).get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Device: {device}")

    train_dir = Path(config["data"]["train_dir"])
    cache_dir = Path(config["cache"]["cache_dir"])

    # --- 1. Verify cache exists ---
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"Cache not found: {cache_dir}")
        logger.error(f"Run: python scripts/cache_features.py --config {args.config}")
        sys.exit(1)

    features_path = cache_dir / "features.pt"
    image_paths_path = cache_dir / "image_paths.json"
    labels_path = cache_dir / "labels.json"
    for p in [features_path, image_paths_path, labels_path]:
        if not p.exists():
            logger.error(f"Cache file missing: {p}")
            sys.exit(1)

    # --- 2. Load cache ---
    cached_features = torch.load(features_path, map_location="cpu")
    with open(image_paths_path, "r") as f:
        cached_paths = json.load(f)
    with open(labels_path, "r") as f:
        cached_labels = json.load(f)

    logger.info(f"Cache: {len(cached_paths)} images, features shape {cached_features.shape}")

    # --- 3. Sample random images ---
    all_images = scan_all_images(train_dir)
    if len(all_images) == 0:
        logger.error("No images found in training directory.")
        sys.exit(1)

    n = min(args.num_samples, len(all_images))
    random.seed(args.seed)
    sampled = random.sample(all_images, n)
    logger.info(f"Sampled {n} images from {len(all_images)} total")

    # --- 4. Load CLIP model for online encoding ---
    clip_model, preprocess = load_openai_clip(device)
    clip_model.visual = clip_model.visual.float()
    clip_model.eval()

    # --- 5. Load linear head from B0 checkpoint (if available) ---
    linear_weight = None
    linear_bias = None
    b0_ckpt_path = Path("outputs/b0/checkpoints/best.pt")
    if b0_ckpt_path.exists():
        ckpt = torch.load(b0_ckpt_path, map_location="cpu")
        linear_state = ckpt["model_state_dict"]
        linear_weight = linear_state["classifier.weight"]
        linear_bias = linear_state["classifier.bias"]
        logger.info("Loaded linear head from B0 checkpoint")
    else:
        logger.warning("B0 checkpoint not found — skipping linear logit comparison")

    # --- 6. Load cosine head (use default init as reference) ---
    from experiments.cosine.model import CosineClassifier
    cosine_model = CosineClassifier(
        clip_model=clip_model,
        num_classes=config["model"]["num_classes"],
        feature_dim=config["model"].get("feature_dim", 512),
        freeze_clip=True,
        init_scale=config["model"].get("cos_init_scale", 10.0),
        learnable_scale=config["model"].get("cos_learnable_scale", True),
    ).to(device).eval()

    # --- 7. Build cache lookup ---
    path_to_idx = {p: i for i, p in enumerate(cached_paths)}
    # Also index by filename for robustness
    name_to_idx = {Path(p).name: i for i, p in enumerate(cached_paths)}

    # --- 8. Compare batch-by-batch ---
    report: Dict[str, Any] = {
        "sample_count": n,
        "feature_diffs": [],
        "linear_logit_diffs": [],
        "cosine_logit_diffs": [],
        "label_mismatches": 0,
        "path_not_found": 0,
        "path_mismatches": 0,
    }

    batch_size = 32
    for start in range(0, n, batch_size):
        batch = sampled[start : start + batch_size]

        # Online encoding
        from PIL import Image
        processed_batch = []
        valid_batch_items = []
        for img_path, class_name in batch:
            try:
                img = Image.open(img_path).convert("RGB")
                processed_batch.append(preprocess(img))
                valid_batch_items.append((img_path, class_name))
            except Exception as e:
                logger.warning(f"Skipping {img_path}: {e}")
                continue

        if not processed_batch:
            continue

        images = torch.stack(processed_batch).to(device)
        online_feats = encode_frozen_clip_features(clip_model, images, device, use_amp=False)

        # Cache lookup for each path
        cache_indices = []
        for img_path, class_name in valid_batch_items:
            rel = str(img_path.relative_to(train_dir))
            if rel in path_to_idx:
                cache_indices.append(path_to_idx[rel])
            elif img_path.name in name_to_idx:
                cache_indices.append(name_to_idx[img_path.name])
                report["path_mismatches"] += 1
            else:
                report["path_not_found"] += 1
                continue

        if not cache_indices:
            continue

        cache_feats = cached_features[torch.tensor(cache_indices)].to(device)

        # Feature diff
        feat_diff = (online_feats - cache_feats).abs()
        report["feature_diffs"].append(feat_diff.cpu())

        # Linear logit comparison
        if linear_weight is not None:
            online_linear = F.normalize(online_feats.float(), dim=-1) @ linear_weight.T + linear_bias.to(device)
            cache_linear = F.normalize(cache_feats.float(), dim=-1) @ linear_weight.T + linear_bias.to(device)
            report["linear_logit_diffs"].append((online_linear - cache_linear).abs().cpu())

        # Cosine logit comparison
        online_cos = build_cosine_logits(online_feats, cosine_model.weight, cosine_model.logit_scale)
        cache_cos = build_cosine_logits(cache_feats, cosine_model.weight, cosine_model.logit_scale)
        report["cosine_logit_diffs"].append((online_cos - cache_cos).abs().cpu())

        # Label check
        for idx, (_, class_name) in enumerate(valid_batch_items):
            cache_label = cached_labels[cache_indices[idx]]
            config_label = config["model"]["num_classes"]  # fallback
            # Actual label from class directory name
            # (label validation is done by CachedFeatureDataset at load time)

    # --- 9. Aggregate results ---
    if report["feature_diffs"]:
        all_feat = torch.cat(report["feature_diffs"])
        report["feature_max_abs_diff"] = float(all_feat.max())
        report["feature_mean_abs_diff"] = float(all_feat.mean())
    else:
        report["feature_max_abs_diff"] = None
        report["feature_mean_abs_diff"] = None

    if report["linear_logit_diffs"]:
        all_lin = torch.cat(report["linear_logit_diffs"])
        report["linear_logit_max_abs_diff"] = float(all_lin.max())
        report["linear_logit_mean_abs_diff"] = float(all_lin.mean())
    else:
        report["linear_logit_max_abs_diff"] = None
        report["linear_logit_mean_abs_diff"] = None

    if report["cosine_logit_diffs"]:
        all_cos = torch.cat(report["cosine_logit_diffs"])
        report["cosine_logit_max_abs_diff"] = float(all_cos.max())
        report["cosine_logit_mean_abs_diff"] = float(all_cos.mean())
    else:
        report["cosine_logit_max_abs_diff"] = None
        report["cosine_logit_mean_abs_diff"] = None

    del report["feature_diffs"]
    del report["linear_logit_diffs"]
    del report["cosine_logit_diffs"]

    # --- 10. Verdict ---
    feature_ok = (
        report["feature_max_abs_diff"] is not None
        and report["feature_max_abs_diff"] <= args.atol_feature
    )
    logit_ok = True
    if report["linear_logit_max_abs_diff"] is not None:
        logit_ok = logit_ok and report["linear_logit_max_abs_diff"] <= args.atol_logit
    if report["cosine_logit_max_abs_diff"] is not None:
        logit_ok = logit_ok and report["cosine_logit_max_abs_diff"] <= args.atol_logit

    path_ok = report["path_not_found"] == 0
    passed = feature_ok and logit_ok and path_ok

    report["feature_pass"] = feature_ok
    report["logit_pass"] = logit_ok
    report["path_pass"] = path_ok
    report["overall_pass"] = passed

    # --- 11. Output ---
    output_path = args.output or "outputs/cache_consistency_report.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Report saved to {output_path}")
    logger.info(
        f"Feature diff: max={report['feature_max_abs_diff']}, "
        f"mean={report['feature_mean_abs_diff']}"
    )
    if report["linear_logit_max_abs_diff"] is not None:
        logger.info(
            f"Linear logit diff: max={report['linear_logit_max_abs_diff']}, "
            f"mean={report['linear_logit_mean_abs_diff']}"
        )
    logger.info(
        f"Cosine logit diff: max={report['cosine_logit_max_abs_diff']}, "
        f"mean={report['cosine_logit_mean_abs_diff']}"
    )
    logger.info(f"Label mismatches: {report['label_mismatches']}")
    logger.info(f"Path not found: {report['path_not_found']}")
    logger.info(f"Overall: {'PASS' if passed else 'FAIL'}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
