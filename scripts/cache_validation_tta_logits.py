#!/usr/bin/env python3
"""Cache horizontal-flip TTA validation logits for prior/TTA evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
import sys
if str(AEGIS_ROOT) not in sys.path:
    sys.path.insert(0, str(AEGIS_ROOT))

from aegis_clip.checkpoint import build_from_checkpoint  # noqa: E402
from aegis_clip.config import load_config  # noqa: E402
from aegis_clip.data import OnlineImageDataset, TrustBundle  # noqa: E402
from aegis_clip.features import FrozenFeatureStore  # noqa: E402
from aegis_clip.runtime import seed_worker, sha256_file  # noqa: E402
from aegis_clip.tta import TTA_FUSION_MODES, fuse_paired_logits  # noqa: E402


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tta-fusion", choices=sorted(TTA_FUSION_MODES), default="mean_probabilities")
    parser.add_argument("--tta-temperature", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = load_config(args.config) if args.config else None
    checkpoint_path = Path(args.checkpoint).resolve()
    model, preprocess, checkpoint = build_from_checkpoint(
        checkpoint_path, device, config_override=config
    )
    config = config or checkpoint["config"]
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
    dataset = OnlineImageDataset(
        config["data"]["val_csv"],
        config["data"]["train_root"],
        preprocess,
        feature_store,
        trust_bundle,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        timeout=int(config["train"].get("loader_timeout", 120 if args.num_workers else 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
        worker_init_fn=seed_worker,
    )
    model.eval()
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    logits_parts = []
    labels_parts = []
    clean_parts = []
    pseudo_parts = []
    correction_parts = []
    paths = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            first = model(images=images)
            second = model(images=torch.flip(images, dims=(3,)))
        fused = fuse_paired_logits(
            first.detach().float(),
            second.detach().float(),
            mode=str(args.tta_fusion),
            temperature=float(args.tta_temperature),
        )
        logits_parts.append(fused.cpu())
        labels_parts.append(batch["label"].long().cpu())
        clean_parts.append(batch["clean_probability"].float().cpu())
        pseudo_parts.append(batch["pseudo_label"].long().cpu())
        correction_parts.append(batch["correction_alpha"].float().cpu())
        paths.extend(str(path) for path in batch["path"])
    payload = {
        "format_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_csv": str(config["data"]["val_csv"]),
        "validation_csv_sha256": sha256_file(config["data"]["val_csv"]),
        "tta_fusion": str(args.tta_fusion),
        "tta_temperature": float(args.tta_temperature),
        "paths": paths,
        "labels": torch.cat(labels_parts),
        "clean_probability": torch.cat(clean_parts),
        "pseudo_labels": torch.cat(pseudo_parts),
        "correction_alpha": torch.cat(correction_parts),
        "logits": torch.cat(logits_parts),
    }
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
