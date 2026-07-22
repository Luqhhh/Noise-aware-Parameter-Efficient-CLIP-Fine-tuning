"""Cache F1+M1 logits, CLS features, and fixed pooled local patch tokens."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.data import OnlineImageDataset, TrustBundle
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.local_inference import (
    attention_guided_crop,
    logits_with_last_block_attention,
    native_visual_forward_with_patch_features,
)
from aegis_clip.part_token_adapter import (
    PART_POOL_METHOD,
    pool_cls_aligned_patch_features,
    validate_part_token_cache,
)
from aegis_clip.runtime import seed_worker, sha256_file, sha256_lines


@torch.no_grad()
def cache_part_token_adapter_features(
    checkpoint_path: str | Path,
    split_csv: str | Path,
    output_path: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    crop_size: int = 160,
    top_patches: int = 5,
    part_top_patches: int = 8,
    part_temperature: float = 0.07,
) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    split_csv = Path(split_csv).resolve()
    destination = Path(output_path).resolve()
    if int(batch_size) <= 0 or int(num_workers) < 0:
        raise ValueError("Invalid cache loader settings")
    if int(part_top_patches) <= 0 or float(part_temperature) <= 0.0:
        raise ValueError("Invalid part pooling settings")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, checkpoint = build_from_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
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
        split_csv,
        config["data"]["train_root"],
        preprocess,
        feature_store,
        trust_bundle,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=seed_worker,
    )
    model.eval()
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    parts: dict[str, list[torch.Tensor]] = {
        "labels": [],
        "clean_probability": [],
        "pseudo_labels": [],
        "correction_alpha": [],
        "global_logits": [],
        "local_features": [],
        "local_logits": [],
        "part_features": [],
    }
    paths: list[str] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            global_logits = model(images=images)
            _, _, attention = logits_with_last_block_attention(model, images)
            local_images = attention_guided_crop(
                images,
                attention,
                crop_size=int(crop_size),
                top_patches=int(top_patches),
            )
            local_logits, local_features, patch_features = (
                native_visual_forward_with_patch_features(model, local_images)
            )
            part_features = pool_cls_aligned_patch_features(
                local_features,
                patch_features,
                top_patches=int(part_top_patches),
                temperature=float(part_temperature),
            )
        parts["labels"].append(batch["label"].long().cpu())
        parts["clean_probability"].append(
            batch["clean_probability"].float().cpu()
        )
        parts["pseudo_labels"].append(batch["pseudo_label"].long().cpu())
        parts["correction_alpha"].append(
            batch["correction_alpha"].float().cpu()
        )
        parts["global_logits"].append(global_logits.float().cpu())
        parts["local_features"].append(local_features.float().cpu())
        parts["local_logits"].append(local_logits.float().cpu())
        parts["part_features"].append(part_features.float().cpu())
        paths.extend(str(path) for path in batch["path"])
    payload: dict[str, object] = {
        "format_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_csv": str(split_csv),
        "split_csv_sha256": sha256_file(split_csv),
        "path_order_sha256": sha256_lines(paths),
        "execution": {
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "device_type": device.type,
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "amp_enabled": bool(use_amp),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "crop_size": int(crop_size),
        "top_patches": int(top_patches),
        "part_pool_spec": {
            "method": PART_POOL_METHOD,
            "top_patches": int(part_top_patches),
            "temperature": float(part_temperature),
            "feature_source": "final_local_patch_tokens_after_ln_post_and_visual_proj",
            "query": "same_view_normalized_local_cls",
        },
        "feature_dtype": "float32",
        "paths": paths,
        **{name: torch.cat(values) for name, values in parts.items()},
    }
    validate_part_token_cache(
        payload,
        expected_feature_dim=int(config["model"].get("feature_dim", 512)),
        expected_num_classes=int(config["model"]["num_classes"]),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--top-patches", type=int, default=5)
    parser.add_argument("--part-top-patches", type=int, default=8)
    parser.add_argument("--part-temperature", type=float, default=0.07)
    args = parser.parse_args()
    print(
        cache_part_token_adapter_features(
            args.checkpoint,
            args.split_csv,
            args.output,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            crop_size=args.crop_size,
            top_patches=args.top_patches,
            part_top_patches=args.part_top_patches,
            part_temperature=args.part_temperature,
        )
    )


if __name__ == "__main__":
    main()
