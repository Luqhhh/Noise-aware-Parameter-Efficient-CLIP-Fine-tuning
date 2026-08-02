"""Evaluate attention-guided local-view inference without touching test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import OnlineImageDataset, TrustBundle
from aegis_clip.evaluation import weighted_accuracy, weighted_macro_accuracy
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.image_preprocess import select_inference_preprocess
from aegis_clip.localization import (
    extract_attention_crops,
    forward_with_last_block_attention,
    fuse_global_local_flip_probabilities,
    fuse_global_local_probabilities,
    fuse_global_multilocal_probabilities,
    parse_int_sequence,
)
from aegis_clip.runtime import atomic_json_dump, seed_worker, sha256_file


def _parse_float_sequence(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("At least one float value is required")
    if len(parsed) != len(set(parsed)):
        raise ValueError("Sequence values must be unique")
    return parsed


def _metrics(
    prediction: torch.Tensor,
    *,
    noisy: torch.Tensor,
    proxy: torch.Tensor,
    clean_weight: torch.Tensor,
    proxy_weight: torch.Tensor,
    num_classes: int,
    clean_core_threshold: float,
) -> dict[str, float | int]:
    unit_weight = torch.ones_like(clean_weight)
    clean_core_weight = (clean_weight >= clean_core_threshold).float()
    return {
        "predicted_class_count": int(prediction.unique().numel()),
        "raw_micro": float((prediction == noisy).float().mean()),
        "raw_macro": weighted_macro_accuracy(
            prediction, noisy, unit_weight, num_classes
        ),
        "trusted_micro": weighted_accuracy(prediction, noisy, clean_weight),
        "trusted_macro": weighted_macro_accuracy(
            prediction, noisy, clean_weight, num_classes
        ),
        "proxy_micro": weighted_accuracy(prediction, proxy, proxy_weight),
        "proxy_macro": weighted_macro_accuracy(
            prediction, proxy, proxy_weight, num_classes
        ),
        "clean_core_micro": weighted_accuracy(
            prediction, noisy, clean_core_weight
        ),
        "clean_core_macro": weighted_macro_accuracy(
            prediction, noisy, clean_core_weight, num_classes
        ),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-sizes", default="160")
    parser.add_argument("--top-ks", default="5")
    parser.add_argument("--local-weights", default="0.5")
    parser.add_argument("--include-horizontal-flip", action="store_true")
    parser.add_argument("--flip-weights", default="0.5")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--input-resize-mode",
        choices=["clip_center_crop", "clip_letterbox"],
        default="clip_center_crop",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    crop_sizes = parse_int_sequence(args.crop_sizes)
    top_ks = parse_int_sequence(args.top_ks)
    local_weights = _parse_float_sequence(args.local_weights)
    flip_weights = _parse_float_sequence(args.flip_weights)
    if any(not 0.0 <= value <= 1.0 for value in local_weights):
        raise ValueError("All local weights must be in [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in flip_weights):
        raise ValueError("All flip weights must be in [0, 1]")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model, preprocess, checkpoint = build_from_checkpoint(args.checkpoint, device)
    config = load_config(args.config)
    preprocess = select_inference_preprocess(
        preprocess,
        mode=args.input_resize_mode,
        input_resolution=int(config["model"].get("input_resolution", 224)),
    )
    model.eval()

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
    workers = int(config["train"].get("num_workers", 4))
    batch_size = int(
        args.batch_size or config["evaluation"].get("batch_size", 128)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        timeout=int(config["train"].get("loader_timeout", 120 if workers else 0)),
        pin_memory=bool(config["train"].get("pin_memory", True)),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )

    candidate_keys = [
        (crop_size, top_k)
        for crop_size in crop_sizes
        for top_k in top_ks
    ]
    global_logits_parts: list[torch.Tensor] = []
    local_logits_parts: dict[tuple[int, int], list[torch.Tensor]] = {
        key: [] for key in candidate_keys
    }
    flipped_global_logits_parts: list[torch.Tensor] = []
    flipped_local_logits_parts: dict[
        tuple[int, int], list[torch.Tensor]
    ] = {key: [] for key in candidate_keys}
    labels_parts: list[torch.Tensor] = []
    clean_parts: list[torch.Tensor] = []
    pseudo_parts: list[torch.Tensor] = []
    correction_parts: list[torch.Tensor] = []
    box_edge_counts = {key: 0 for key in candidate_keys}
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"

    for batch in tqdm(loader, desc="Localization sweep"):
        images = batch["images"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            global_logits, attention = forward_with_last_block_attention(
                model, images
            )
        global_logits_parts.append(global_logits.float().cpu())
        labels_parts.append(batch["label"].long())
        clean_parts.append(batch["clean_probability"].float())
        pseudo_parts.append(batch["pseudo_label"].long())
        correction_parts.append(batch["correction_alpha"].float())

        for key in candidate_keys:
            crop_size, top_k = key
            crops, boxes = extract_attention_crops(
                images,
                attention,
                crop_size=crop_size,
                top_k=top_k,
            )
            with torch.autocast(device_type=device.type, enabled=use_amp):
                local_logits = model(images=crops)
            local_logits_parts[key].append(local_logits.float().cpu())
            image_height, image_width = images.shape[-2:]
            box_edge_counts[key] += sum(
                x0 == 0
                or y0 == 0
                or x1 == image_width
                or y1 == image_height
                for x0, y0, x1, y1 in boxes
            )
        if args.include_horizontal_flip:
            flipped_images = torch.flip(images, dims=(3,))
            with torch.autocast(device_type=device.type, enabled=use_amp):
                (
                    flipped_global_logits,
                    flipped_attention,
                ) = forward_with_last_block_attention(
                    model,
                    flipped_images,
                )
            flipped_global_logits_parts.append(
                flipped_global_logits.float().cpu()
            )
            for key in candidate_keys:
                crop_size, top_k = key
                flipped_crops, _ = extract_attention_crops(
                    flipped_images,
                    flipped_attention,
                    crop_size=crop_size,
                    top_k=top_k,
                )
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    flipped_local_logits = model(images=flipped_crops)
                flipped_local_logits_parts[key].append(
                    flipped_local_logits.float().cpu()
                )

    global_logits = torch.cat(global_logits_parts)
    noisy = torch.cat(labels_parts)
    clean_weight = torch.cat(clean_parts)
    pseudo_label = torch.cat(pseudo_parts)
    correction = torch.cat(correction_parts)
    proxy = torch.where(correction > 0.0, pseudo_label, noisy)
    proxy_weight = torch.maximum(clean_weight, correction)
    num_classes = int(config["model"]["num_classes"])
    clean_core_threshold = float(
        config["evaluation"].get("clean_core_threshold", 0.70)
    )
    global_prediction = global_logits.argmax(dim=1)
    local_logits_by_key = {
        key: torch.cat(parts) for key, parts in local_logits_parts.items()
    }
    flipped_global_logits = (
        torch.cat(flipped_global_logits_parts)
        if args.include_horizontal_flip
        else None
    )
    flipped_local_logits_by_key = (
        {
            key: torch.cat(parts)
            for key, parts in flipped_local_logits_parts.items()
        }
        if args.include_horizontal_flip
        else {}
    )
    global_metrics = _metrics(
        global_prediction,
        noisy=noisy,
        proxy=proxy,
        clean_weight=clean_weight,
        proxy_weight=proxy_weight,
        num_classes=num_classes,
        clean_core_threshold=clean_core_threshold,
    )

    candidates: list[dict[str, Any]] = []
    for key in candidate_keys:
        crop_size, top_k = key
        local_logits = local_logits_by_key[key]
        local_prediction = local_logits.argmax(dim=1)
        local_metrics = _metrics(
            local_prediction,
            noisy=noisy,
            proxy=proxy,
            clean_weight=clean_weight,
            proxy_weight=proxy_weight,
            num_classes=num_classes,
            clean_core_threshold=clean_core_threshold,
        )
        common = {
            "crop_size": crop_size,
            "top_k": top_k,
            "global_local_agreement": float(
                (global_prediction == local_prediction).float().mean()
            ),
            "either_view_raw_correct": float(
                ((global_prediction == noisy) | (local_prediction == noisy))
                .float()
                .mean()
            ),
            "global_only_raw_correct": int(
                ((global_prediction == noisy) & (local_prediction != noisy)).sum()
            ),
            "local_only_raw_correct": int(
                ((global_prediction != noisy) & (local_prediction == noisy)).sum()
            ),
            "edge_clamped_fraction": box_edge_counts[key] / len(dataset),
            "local": local_metrics,
            "fusions": [],
            "stacked_flip_fusions": [],
        }
        m1_metrics_by_weight: dict[float, dict[str, float | int]] = {}
        for local_weight in local_weights:
            fused = fuse_global_local_probabilities(
                global_logits,
                local_logits,
                local_weight=local_weight,
                temperature=args.temperature,
            )
            fused_metrics = _metrics(
                fused.argmax(dim=1),
                noisy=noisy,
                proxy=proxy,
                clean_weight=clean_weight,
                proxy_weight=proxy_weight,
                num_classes=num_classes,
                clean_core_threshold=clean_core_threshold,
            )
            common["fusions"].append(
                {
                    "local_weight": local_weight,
                    "temperature": float(args.temperature),
                    **fused_metrics,
                    "raw_micro_delta_vs_global": (
                        float(fused_metrics["raw_micro"])
                        - float(global_metrics["raw_micro"])
                    ),
                    "clean_core_micro_delta_vs_global": (
                        float(fused_metrics["clean_core_micro"])
                        - float(global_metrics["clean_core_micro"])
                    ),
                }
            )
            m1_metrics_by_weight[local_weight] = fused_metrics
        if args.include_horizontal_flip:
            assert flipped_global_logits is not None
            flipped_local_logits = flipped_local_logits_by_key[key]
            flipped_local_prediction = flipped_local_logits.argmax(dim=1)
            common["flipped_local"] = _metrics(
                flipped_local_prediction,
                noisy=noisy,
                proxy=proxy,
                clean_weight=clean_weight,
                proxy_weight=proxy_weight,
                num_classes=num_classes,
                clean_core_threshold=clean_core_threshold,
            )
            for local_weight in local_weights:
                m1_metrics = m1_metrics_by_weight[local_weight]
                for flip_weight in flip_weights:
                    fused = fuse_global_local_flip_probabilities(
                        global_logits,
                        local_logits,
                        flipped_global_logits,
                        flipped_local_logits,
                        local_weight=local_weight,
                        flip_weight=flip_weight,
                        temperature=args.temperature,
                    )
                    fused_metrics = _metrics(
                        fused.argmax(dim=1),
                        noisy=noisy,
                        proxy=proxy,
                        clean_weight=clean_weight,
                        proxy_weight=proxy_weight,
                        num_classes=num_classes,
                        clean_core_threshold=clean_core_threshold,
                    )
                    common["stacked_flip_fusions"].append(
                        {
                            "local_weight": local_weight,
                            "flip_weight": flip_weight,
                            "temperature": float(args.temperature),
                            **fused_metrics,
                            "raw_micro_delta_vs_m1": (
                                float(fused_metrics["raw_micro"])
                                - float(m1_metrics["raw_micro"])
                            ),
                            "trusted_macro_delta_vs_m1": (
                                float(fused_metrics["trusted_macro"])
                                - float(m1_metrics["trusted_macro"])
                            ),
                            "proxy_macro_delta_vs_m1": (
                                float(fused_metrics["proxy_macro"])
                                - float(m1_metrics["proxy_macro"])
                            ),
                            "clean_core_micro_delta_vs_m1": (
                                float(fused_metrics["clean_core_micro"])
                                - float(m1_metrics["clean_core_micro"])
                            ),
                            "clean_core_macro_delta_vs_m1": (
                                float(fused_metrics["clean_core_macro"])
                                - float(m1_metrics["clean_core_macro"])
                            ),
                        }
                    )
        candidates.append(common)

    multiscale_candidates: list[dict[str, Any]] = []
    if len(crop_sizes) > 1:
        for top_k in top_ks:
            keys = [(crop_size, top_k) for crop_size in crop_sizes]
            local_views = [local_logits_by_key[key] for key in keys]
            mean_local_probabilities = torch.stack(
                [value.softmax(dim=1) for value in local_views],
                dim=0,
            ).mean(dim=0)
            mean_local_metrics = _metrics(
                mean_local_probabilities.argmax(dim=1),
                noisy=noisy,
                proxy=proxy,
                clean_weight=clean_weight,
                proxy_weight=proxy_weight,
                num_classes=num_classes,
                clean_core_threshold=clean_core_threshold,
            )
            multiscale: dict[str, Any] = {
                "crop_sizes": list(crop_sizes),
                "top_k": top_k,
                "mean_local": mean_local_metrics,
                "fusions": [],
            }
            for local_weight in local_weights:
                fused = fuse_global_multilocal_probabilities(
                    global_logits,
                    local_views,
                    local_weight=local_weight,
                    temperature=args.temperature,
                )
                fused_metrics = _metrics(
                    fused.argmax(dim=1),
                    noisy=noisy,
                    proxy=proxy,
                    clean_weight=clean_weight,
                    proxy_weight=proxy_weight,
                    num_classes=num_classes,
                    clean_core_threshold=clean_core_threshold,
                )
                multiscale["fusions"].append(
                    {
                        "local_weight": local_weight,
                        "temperature": float(args.temperature),
                        **fused_metrics,
                        "raw_micro_delta_vs_global": (
                            float(fused_metrics["raw_micro"])
                            - float(global_metrics["raw_micro"])
                        ),
                        "clean_core_micro_delta_vs_global": (
                            float(fused_metrics["clean_core_micro"])
                            - float(global_metrics["clean_core_micro"])
                        ),
                    }
                )
            multiscale_candidates.append(multiscale)

    payload = {
        "format_version": 2,
        "protocol": (
            "last_block_mean_head_topk_attention_crop_plus_horizontal_flip"
            if args.include_horizontal_flip
            else "last_block_mean_head_topk_attention_crop"
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(Path(args.config).resolve()),
        "input_resize_mode": args.input_resize_mode,
        "samples": len(dataset),
        "clean_core_threshold": clean_core_threshold,
        "global": global_metrics,
        "flipped_global": (
            _metrics(
                flipped_global_logits.argmax(dim=1),
                noisy=noisy,
                proxy=proxy,
                clean_weight=clean_weight,
                proxy_weight=proxy_weight,
                num_classes=num_classes,
                clean_core_threshold=clean_core_threshold,
            )
            if flipped_global_logits is not None
            else None
        ),
        "flip_weights": (
            list(flip_weights) if args.include_horizontal_flip else []
        ),
        "candidates": candidates,
        "multiscale_candidates": multiscale_candidates,
    }
    atomic_json_dump(payload, output_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
