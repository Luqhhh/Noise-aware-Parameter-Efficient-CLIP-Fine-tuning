"""Cache deterministic global and attention-local features for adapter training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import OnlineImageDataset, TrustBundle
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.localization import (
    extract_attention_crops,
    forward_features_with_last_block_attention,
)
from aegis_clip.local_adapter import validate_local_view_cache
from aegis_clip.runtime import seed_worker, set_seed, sha256_file
from aegis_clip.trust import atomic_torch_save


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if not 1 <= args.crop_size <= 224:
        raise ValueError("crop-size must be in [1, 224]")
    if not 1 <= args.top_k <= 49:
        raise ValueError("top-k must be in [1, 49]")

    config = load_config(args.config)
    set_seed(int(config["project"].get("seed", 42)), deterministic=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model, preprocess, _ = build_from_checkpoint(args.checkpoint, device)
    model.eval()
    base_checkpoint_sha256 = sha256_file(args.checkpoint)

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
    split_csv = config["data"][f"{args.split}_csv"]
    dataset = OnlineImageDataset(
        split_csv,
        config["data"]["train_root"],
        preprocess,
        feature_store,
        trust_bundle,
    )
    workers = int(config["train"].get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        timeout=int(config["train"].get("loader_timeout", 120 if workers else 0)),
        pin_memory=bool(config["train"].get("pin_memory", True)),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    paths: list[str] = []
    labels: list[torch.Tensor] = []
    clean_probabilities: list[torch.Tensor] = []
    pseudo_labels: list[torch.Tensor] = []
    correction_alphas: list[torch.Tensor] = []
    global_features: list[torch.Tensor] = []
    local_features: list[torch.Tensor] = []

    for batch in tqdm(loader, desc=f"Cache {args.split} local views"):
        images = batch["images"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            _, global_encoded, attention = (
                forward_features_with_last_block_attention(model, images)
            )
            local_images, _ = extract_attention_crops(
                images,
                attention,
                crop_size=args.crop_size,
                top_k=args.top_k,
            )
            _, local_encoded = model(images=local_images, return_features=True)
        paths.extend(str(value) for value in batch["path"])
        labels.append(batch["label"].long())
        clean_probabilities.append(batch["clean_probability"].float())
        pseudo_labels.append(batch["pseudo_label"].long())
        correction_alphas.append(batch["correction_alpha"].float())
        global_features.append(global_encoded.float().cpu())
        local_features.append(local_encoded.float().cpu())

    payload = {
        "format_version": 1,
        "protocol": "last_block_mean_head_topk_attention_crop",
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "split_csv": str(Path(split_csv).resolve()),
        "split_csv_sha256": sha256_file(split_csv),
        "crop_size": int(args.crop_size),
        "top_k": int(args.top_k),
        "paths": paths,
        "labels": torch.cat(labels),
        "clean_probability": torch.cat(clean_probabilities),
        "pseudo_label": torch.cat(pseudo_labels),
        "correction_alpha": torch.cat(correction_alphas),
        "global_features": torch.cat(global_features),
        "local_features": torch.cat(local_features),
    }
    size = validate_local_view_cache(
        payload,
        expected_checkpoint_sha256=base_checkpoint_sha256,
        expected_split=args.split,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, output_path)
    print(
        json.dumps(
            {
                "status": "passed",
                "split": args.split,
                "samples": size,
                "crop_size": args.crop_size,
                "top_k": args.top_k,
                "base_checkpoint_sha256": base_checkpoint_sha256,
                "output": str(output_path.resolve()),
                "output_sha256": sha256_file(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
