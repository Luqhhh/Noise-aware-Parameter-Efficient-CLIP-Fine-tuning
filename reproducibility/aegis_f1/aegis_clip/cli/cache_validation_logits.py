"""Cache deterministic validation logits for label-free calibration studies."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.data import CachedFeatureDataset, OnlineImageDataset, TrustBundle
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.image_preprocess import select_inference_preprocess
from aegis_clip.local_inference import (
    attention_local_global_logits,
    complementary_flip_local_global_logits,
)
from aegis_clip.multi_region_inference import discriminative_multi_region_logits
from aegis_clip.runtime import seed_worker, sha256_file


@torch.no_grad()
def cache_validation_logits(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    view_mode: str = "center",
    force_online_images: bool = False,
    input_resize_mode: str = "clip_center_crop",
) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    destination = Path(output_path).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if view_mode not in {
        "center",
        "attention_local_global",
        "complementary_flip_local_global",
        "discriminative_multi_region",
    }:
        raise ValueError(
            "view_mode must be center, attention_local_global, "
            "complementary_flip_local_global, or discriminative_multi_region"
        )
    model, preprocess, checkpoint = build_from_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    preprocess = select_inference_preprocess(
        preprocess,
        mode=input_resize_mode,
        input_resolution=int(config["model"].get("input_resolution", 224)),
    )
    data = config["data"]
    features = config["features"]
    feature_store = FrozenFeatureStore(
        tensor_path=features["tensor_path"],
        paths_path=features["paths_path"],
        manifest_path=features.get("manifest_path"),
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    trust_bundle = (
        TrustBundle(config["trust"]["bundle_path"])
        if config.get("trust", {}).get("enabled", False)
        else None
    )
    if (
        model.visual_requires_grad
        or view_mode
        in {
            "attention_local_global",
            "complementary_flip_local_global",
            "discriminative_multi_region",
        }
        or bool(force_online_images)
    ):
        dataset = OnlineImageDataset(
            data["val_csv"],
            data["train_root"],
            preprocess,
            feature_store,
            trust_bundle,
        )
    else:
        dataset = CachedFeatureDataset(
            data["val_csv"], feature_store, trust_bundle
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
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    model.eval()
    values: dict[str, list[torch.Tensor]] = {
        "logits": [],
        "labels": [],
        "clean_probability": [],
        "pseudo_labels": [],
        "correction_alpha": [],
    }
    paths: list[str] = []
    global_logits_parts: list[torch.Tensor] = []
    local_logits_parts: list[torch.Tensor] = []
    attention_local_logits_parts: list[torch.Tensor] = []
    selected_region_indices_parts: list[torch.Tensor] = []
    selected_region_weights_parts: list[torch.Tensor] = []
    m1_logits_parts: list[torch.Tensor] = []
    flip_logits_parts: list[torch.Tensor] = []
    flip_fused_logits_parts: list[torch.Tensor] = []
    for batch in loader:
        argument_name = "images" if "images" in batch else "features"
        inputs = batch[argument_name].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            if view_mode in {
                "attention_local_global",
                "complementary_flip_local_global",
                "discriminative_multi_region",
            }:
                if argument_name != "images":
                    raise ValueError("Local-region inference requires online images")
                if view_mode == "attention_local_global":
                    result = attention_local_global_logits(
                        model, inputs, crop_size=160, top_patches=5
                    )
                elif view_mode == "discriminative_multi_region":
                    result = discriminative_multi_region_logits(model, inputs)
                else:
                    result = complementary_flip_local_global_logits(
                        model, inputs, crop_size=160, top_patches=5
                    )
                logits = result["logits"]
                global_logits_parts.append(result["global_logits"].cpu())
                local_logits_parts.append(result["local_logits"].cpu())
                if "attention_local_logits" in result:
                    attention_local_logits_parts.append(
                        result["attention_local_logits"].cpu()
                    )
                    selected_region_indices_parts.append(
                        result["selected_region_indices"].cpu()
                    )
                    selected_region_weights_parts.append(
                        result["selected_region_weights"].cpu()
                    )
                if "m1_logits" in result:
                    m1_logits_parts.append(result["m1_logits"].cpu())
                    flip_logits_parts.append(result["flip_logits"].cpu())
                    flip_fused_logits_parts.append(
                        result["flip_fused_logits"].cpu()
                    )
            else:
                logits = model(**{argument_name: inputs})
        values["logits"].append(logits.float().cpu())
        values["labels"].append(batch["label"].long().cpu())
        values["clean_probability"].append(
            batch["clean_probability"].float().cpu()
        )
        values["pseudo_labels"].append(batch["pseudo_label"].long().cpu())
        values["correction_alpha"].append(
            batch["correction_alpha"].float().cpu()
        )
        paths.extend(str(path) for path in batch["path"])
    payload = {
        "format_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_csv": str(data["val_csv"]),
        "validation_csv_sha256": sha256_file(data["val_csv"]),
        "view_mode": view_mode,
        "force_online_images": bool(force_online_images),
        "input_resize_mode": input_resize_mode,
        "paths": paths,
        **{name: torch.cat(parts) for name, parts in values.items()},
    }
    if global_logits_parts:
        payload["global_logits"] = torch.cat(global_logits_parts)
        payload["local_logits"] = torch.cat(local_logits_parts)
    if attention_local_logits_parts:
        payload["attention_local_logits"] = torch.cat(
            attention_local_logits_parts
        )
        payload["selected_region_indices"] = torch.cat(
            selected_region_indices_parts
        )
        payload["selected_region_weights"] = torch.cat(
            selected_region_weights_parts
        )
    if m1_logits_parts:
        payload["m1_logits"] = torch.cat(m1_logits_parts)
        payload["flip_logits"] = torch.cat(flip_logits_parts)
        payload["flip_fused_logits"] = torch.cat(flip_fused_logits_parts)
    if payload["logits"].shape[0] != len(dataset):
        raise RuntimeError("Validation logit cache is incomplete")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--view-mode",
        choices=[
            "center",
            "attention_local_global",
            "complementary_flip_local_global",
            "discriminative_multi_region",
        ],
        default="center",
    )
    parser.add_argument("--force-online-images", action="store_true")
    parser.add_argument(
        "--input-resize-mode",
        choices=["clip_center_crop", "clip_letterbox"],
        default="clip_center_crop",
    )
    args = parser.parse_args()
    path = cache_validation_logits(
        args.checkpoint,
        args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        view_mode=args.view_mode,
        force_online_images=args.force_online_images,
        input_resize_mode=args.input_resize_mode,
    )
    print(path)


if __name__ == "__main__":
    main()
