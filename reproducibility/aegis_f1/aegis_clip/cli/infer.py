"""Bare or explicitly acknowledged same-model TTA with fail-closed output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import TestImageDataset, load_class_mapping
from aegis_clip.image_preprocess import select_inference_preprocess
from aegis_clip.local_feature_adapter import load_local_feature_adapter
from aegis_clip.local_inference import (
    attention_local_adapter_global_logits,
    attention_local_global_logits,
    complementary_flip_local_global_logits,
)
from aegis_clip.multiprototype import blend_multiprototype_logits
from aegis_clip.prior_alignment import align_logits_to_prior
from aegis_clip.runtime import seed_worker, set_seed
from aegis_clip.submission import create_submission
from aegis_clip.tta import TTA_FUSION_MODES, fuse_paired_logits


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tta",
        choices=[
            "none",
            "horizontal_flip",
            "attention_local_global",
            "attention_local_adapter_global",
            "complementary_flip_local_global",
        ],
        default="none",
    )
    parser.add_argument(
        "--tta-fusion", choices=sorted(TTA_FUSION_MODES), default="mean_logits"
    )
    parser.add_argument("--tta-temperature", type=float, default=1.0)
    parser.add_argument("--acknowledge-tta-risk", action="store_true")
    parser.add_argument("--prior-alignment-strength", type=float, default=0.0)
    parser.add_argument("--prior-alignment-iterations", type=int, default=50)
    parser.add_argument("--acknowledge-balanced-test-prior", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--input-resize-mode",
        choices=["clip_center_crop", "clip_letterbox"],
        default="clip_center_crop",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Explicit inference batch size for validation-aligned numerical reproducibility",
    )
    args = parser.parse_args()
    if args.tta != "none" and not args.acknowledge_tta_risk:
        raise ValueError(
            "TTA is a competition gray area; pass --acknowledge-tta-risk explicitly"
        )
    if args.tta in {
        "attention_local_global",
        "attention_local_adapter_global",
        "complementary_flip_local_global",
    } and (
        args.tta_fusion != "mean_logits" or args.tta_temperature != 1.0
    ):
        raise ValueError(
            "Local-global TTA modes use frozen probability averaging; do not "
            "pass fusion or temperature overrides"
        )
    if args.prior_alignment_strength > 0.0 and not args.acknowledge_balanced_test_prior:
        raise ValueError(
            "Balanced-prior calibration uses the declared test-set distribution; "
            "pass --acknowledge-balanced-test-prior explicitly"
        )
    if not 0.0 <= args.prior_alignment_strength <= 1.0:
        raise ValueError("--prior-alignment-strength must be in [0, 1]")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    config = load_config(args.config) if args.config else None
    model, preprocess, checkpoint = build_from_checkpoint(
        args.checkpoint, device, config_override=config
    )
    config = config or checkpoint["config"]
    preprocess = select_inference_preprocess(
        preprocess,
        mode=args.input_resize_mode,
        input_resolution=int(config["model"].get("input_resolution", 224)),
    )
    set_seed(int(config["project"].get("seed", 42)), deterministic=True)
    _, idx_to_class = load_class_mapping(config["data"]["class_mapping"])
    dataset = TestImageDataset(config["data"]["test_root"], preprocess)
    expected_test_samples = int(config["data"]["expected_test_samples"])
    if len(dataset) != expected_test_samples:
        raise ValueError(
            f"Test image count {len(dataset)} does not match the declared "
            f"official count {expected_test_samples}"
        )
    workers = int(config["train"].get("num_workers", 4))
    inference_batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else config["evaluation"].get(
            "inference_batch_size",
            min(int(config["evaluation"].get("batch_size", 256)), 256),
        )
    )
    if inference_batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    loader = DataLoader(
        dataset,
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=workers,
        timeout=int(config["train"].get("loader_timeout", 120 if workers else 0)),
        pin_memory=bool(config["train"].get("pin_memory", True)),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )
    model.eval()
    local_feature_adapter = (
        load_local_feature_adapter(checkpoint, device)
        if args.tta == "attention_local_adapter_global"
        else None
    )
    multiprototype_head = checkpoint.get("multiprototype_head")
    if multiprototype_head is not None:
        multiprototype_head = dict(multiprototype_head)
        multiprototype_head["prototypes"] = multiprototype_head["prototypes"].to(
            device=device, dtype=torch.float32
        )
    if args.tta in {
        "attention_local_global",
        "attention_local_adapter_global",
        "complementary_flip_local_global",
    } and multiprototype_head is not None:
        raise ValueError(
            "Local-global TTA has not been validated with a multiprototype head"
        )
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    logit_batches: list[torch.Tensor] = []
    prediction_names: list[str] = []
    corrupt_count = 0
    for batch in tqdm(loader, desc="Aegis inference"):
        images = batch["images"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            if args.tta == "attention_local_global":
                logits = attention_local_global_logits(
                    model,
                    images,
                    crop_size=160,
                    top_patches=5,
                )["logits"]
            elif args.tta == "attention_local_adapter_global":
                if local_feature_adapter is None:
                    raise RuntimeError("O3 local feature adapter was not loaded")
                logits = attention_local_adapter_global_logits(
                    model,
                    local_feature_adapter,
                    images,
                    crop_size=160,
                    top_patches=5,
                )["logits"]
            elif args.tta == "complementary_flip_local_global":
                logits = complementary_flip_local_global_logits(
                    model,
                    images,
                    crop_size=160,
                    top_patches=5,
                )["logits"]
            else:
                if multiprototype_head is None:
                    logits = model(images=images)
                else:
                    logits, features = model(images=images, return_features=True)
                    logits = blend_multiprototype_logits(
                        logits, features, multiprototype_head
                    )
                if args.tta == "horizontal_flip":
                    if multiprototype_head is None:
                        second_logits = model(
                            images=torch.flip(images, dims=(3,))
                        )
                    else:
                        second_logits, second_features = model(
                            images=torch.flip(images, dims=(3,)),
                            return_features=True,
                        )
                        second_logits = blend_multiprototype_logits(
                            second_logits, second_features, multiprototype_head
                        )
                    logits = fuse_paired_logits(
                        logits,
                        second_logits,
                        mode=args.tta_fusion,
                        temperature=args.tta_temperature,
                    )
        names = list(batch["name"])
        corrupt_count += int(batch["corrupt"].sum())
        logit_batches.append(logits.detach().float().cpu())
        prediction_names.extend(names)
    expected_names = [path.name for path in dataset.paths]
    if corrupt_count:
        raise RuntimeError(
            f"Refusing to publish: Pillow failed to decode {corrupt_count} test images"
        )
    all_logits = torch.cat(logit_batches, dim=0)
    prior_alignment = None
    if args.prior_alignment_strength > 0.0:
        all_logits, prior_alignment = align_logits_to_prior(
            all_logits,
            strength=float(args.prior_alignment_strength),
            max_iterations=int(args.prior_alignment_iterations),
        )
    indices = all_logits.argmax(dim=1).tolist()
    predictions = [
        (name, str(idx_to_class[index]).zfill(4))
        for name, index in zip(prediction_names, indices)
    ]
    if args.tta == "attention_local_global":
        inference_mode = "attention_local_global:crop=160:top5:mean_probabilities"
    elif args.tta == "attention_local_adapter_global":
        inference_mode = (
            "attention_local_adapter_global:crop=160:top5:"
            "mean_probabilities"
        )
    elif args.tta == "complementary_flip_local_global":
        inference_mode = (
            "complementary_flip_local_global:crop=160:top5:"
            "branch_mean_probabilities"
        )
    else:
        inference_mode = (
            args.tta
            if args.tta == "none" or args.tta_fusion == "mean_logits"
            else f"{args.tta}:{args.tta_fusion}:t={args.tta_temperature:g}"
        )
    if args.prior_alignment_strength > 0.0:
        inference_mode += f":balanced_prior={args.prior_alignment_strength:g}"
    if args.input_resize_mode != "clip_center_crop":
        inference_mode += f":resize={args.input_resize_mode}"
    manifest = create_submission(
        predictions,
        expected_names,
        args.output_dir,
        args.checkpoint,
        inference_mode=inference_mode,
        tta_risk_acknowledged=args.acknowledge_tta_risk,
        valid_labels={str(value).zfill(4) for value in idx_to_class.values()},
        extra_manifest={
            "corrupt_images": corrupt_count,
            "tta_fusion": (
                "mean_probabilities"
                if args.tta
                in {"attention_local_global", "attention_local_adapter_global"}
                else "branch_mean_probabilities"
                if args.tta == "complementary_flip_local_global"
                else args.tta_fusion if args.tta != "none" else "none"
            ),
            "tta_temperature": (
                float(args.tta_temperature) if args.tta != "none" else 1.0
            ),
            "prediction_head": (
                "linear_plus_multiprototype"
                if multiprototype_head is not None
                else "linear"
            ),
            "input_resolution": int(
                config["model"].get("input_resolution", 224)
            ),
            "input_resize_mode": args.input_resize_mode,
            "inference_batch_size": inference_batch_size,
            "balanced_test_prior_acknowledged": bool(
                args.acknowledge_balanced_test_prior
            ),
            "prior_alignment": prior_alignment,
            "attention_local_global": (
                {
                    "attention_block": "last",
                    "attention_heads": "mean_all_12",
                    "top_patches": 5,
                    "crop_size": 160,
                    "input_size": 224,
                    "fusion": "1:1_probability_mean",
                }
                if args.tta
                in {"attention_local_global", "attention_local_adapter_global"}
                else None
            ),
            "local_feature_adapter": (
                {
                    **checkpoint["local_feature_adapter"]["spec"],
                    "gate": checkpoint["local_feature_adapter"]["gate"],
                    "view_condition": "attention_local_only",
                    "global_path": "native_parent_checkpoint_unchanged",
                }
                if args.tta == "attention_local_adapter_global"
                else None
            ),
            "complementary_flip_local_global": (
                {
                    "flip_branch": "mean_center_flip_logits",
                    "m1_branch": "1:1_center_attention_local_probabilities",
                    "branch_fusion": "1:1_probability_mean",
                    "attention_block": "last",
                    "attention_heads": "mean_all_12",
                    "top_patches": 5,
                    "crop_size": 160,
                    "input_size": 224,
                }
                if args.tta == "complementary_flip_local_global"
                else None
            ),
            "multiprototype": (
                {
                    key: value
                    for key, value in multiprototype_head.items()
                    if key != "prototypes"
                }
                if multiprototype_head is not None
                else None
            ),
        },
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
