"""Evaluate an Aegis checkpoint with the clean-proxy selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import CachedFeatureDataset, OnlineImageDataset, TrustBundle
from aegis_clip.evaluation import evaluate
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.runtime import atomic_json_dump, seed_worker
from aegis_clip.tta import TTA_FUSION_MODES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tta", choices=["none", "horizontal_flip"], default="none"
    )
    parser.add_argument(
        "--tta-fusion", choices=sorted(TTA_FUSION_MODES), default="mean_logits"
    )
    parser.add_argument("--tta-temperature", type=float, default=1.0)
    args = parser.parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model, preprocess, checkpoint = build_from_checkpoint(args.checkpoint, device)
    multiprototype_head = checkpoint.get("multiprototype_head")
    if multiprototype_head is not None:
        multiprototype_head = dict(multiprototype_head)
        multiprototype_head["prototypes"] = multiprototype_head["prototypes"].to(
            device=device, dtype=torch.float32
        )
    config = load_config(args.config) if args.config else checkpoint["config"]
    evaluation_config = config.get("evaluation", {})
    measure_flip_consistency = bool(
        evaluation_config.get("measure_flip_consistency", False)
    )
    feature_config = config["features"]
    feature_store = FrozenFeatureStore(
        feature_config["tensor_path"],
        feature_config["paths_path"],
        feature_config.get("manifest_path"),
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    trust_bundle = (
        TrustBundle(config["trust"]["bundle_path"])
        if config.get("trust", {}).get("enabled", False)
        else None
    )
    if (
        model.visual_requires_grad
        or args.tta != "none"
        or measure_flip_consistency
    ):
        dataset = OnlineImageDataset(
            config["data"]["val_csv"],
            config["data"]["train_root"],
            preprocess,
            feature_store,
            trust_bundle,
        )
    else:
        dataset = CachedFeatureDataset(
            config["data"]["val_csv"], feature_store, trust_bundle
        )
    workers = int(config["train"].get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=int(evaluation_config.get("batch_size", 256)),
        shuffle=False,
        num_workers=workers,
        timeout=int(config["train"].get("loader_timeout", 120 if workers else 0)),
        pin_memory=bool(config["train"].get("pin_memory", True)),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )
    metrics = evaluate(
        model,
        loader,
        device,
        num_classes=int(config["model"]["num_classes"]),
        use_amp=bool(config["train"].get("amp", True)),
        drift_budget=float(evaluation_config.get("drift_budget", 0.01)),
        drift_penalty=float(evaluation_config.get("drift_penalty", 0.5)),
        selector_metric=str(
            evaluation_config.get("selector_metric", "proxy_macro")
        ),
        tta_mode=args.tta,
        tta_fusion=args.tta_fusion,
        tta_temperature=args.tta_temperature,
        multiprototype_head=multiprototype_head,
        clean_core_threshold=float(
            evaluation_config.get("clean_core_threshold", 0.70)
        ),
        measure_flip_consistency=measure_flip_consistency,
    )
    if args.output:
        atomic_json_dump(metrics, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
