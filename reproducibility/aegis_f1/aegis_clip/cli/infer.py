"""Audited bare, flip, local-view, local-global TTA, or explicitly stacked inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aegis_clip.checkpoint import _atomic_torch_save, build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import TestImageDataset, load_class_mapping
from aegis_clip.image_preprocess import select_inference_preprocess
from aegis_clip.local_feature_adapter import load_local_feature_adapter
from aegis_clip.local_inference import (
    adapted_local_view_logits,
    adapted_part_token_local_view_logits,
    attention_local_adapter_global_logits,
    attention_local_global_logits,
    attention_part_token_adapter_global_logits,
    complementary_flip_local_global_logits,
)
from aegis_clip.localization import (
    extract_attention_crops,
    forward_with_last_block_attention,
    fuse_global_local_flip_probabilities,
    fuse_global_local_probabilities,
    fuse_global_multilocal_flip_probabilities,
    fuse_global_multilocal_probabilities,
    normalized_probability_weights,
    parse_int_sequence,
)
from aegis_clip.multiprototype import blend_multiprototype_logits
from aegis_clip.part_token_adapter import load_part_token_adapter
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
            "attention_part_token_adapter_global",
            "complementary_flip_local_global",
        ],
        default="none",
    )
    parser.add_argument(
        "--tta-fusion", choices=sorted(TTA_FUSION_MODES), default="mean_logits"
    )
    parser.add_argument("--tta-temperature", type=float, default=1.0)
    parser.add_argument("--tta-view-weight", type=float, default=0.5)
    parser.add_argument("--acknowledge-tta-risk", action="store_true")
    parser.add_argument(
        "--local-view",
        choices=["none", "attention_crop", "attention_multiscale"],
        default="none",
    )
    parser.add_argument("--local-crop-size", type=int, default=160)
    parser.add_argument("--local-crop-sizes", default="144,160,176")
    parser.add_argument(
        "--local-scale-weights",
        help="Comma-separated multiscale local probability weights",
    )
    parser.add_argument("--local-top-k", type=int, default=5)
    parser.add_argument("--local-weight", type=float, default=0.5)
    parser.add_argument("--local-temperature", type=float, default=1.0)
    parser.add_argument(
        "--adapt-local-features",
        action="store_true",
        help="Apply the checkpoint-embedded O3 adapter to every local crop",
    )
    parser.add_argument(
        "--adapt-part-token-features",
        action="store_true",
        help="Apply the checkpoint-embedded part-token adapter to every local crop",
    )
    parser.add_argument("--acknowledge-local-view-risk", action="store_true")
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
    parser.add_argument(
        "--dump-logits",
        metavar="PATH",
        help="Save the fused logits (before prior alignment) and image names for offline sweeps",
    )
    args = parser.parse_args()
    local_crop_sizes = (
        parse_int_sequence(args.local_crop_sizes)
        if args.local_view == "attention_multiscale"
        else (int(args.local_crop_size),)
    )
    local_scale_weights = (
        normalized_probability_weights(
            args.local_scale_weights,
            len(local_crop_sizes),
            name="local-scale-weights",
        )
        if args.local_scale_weights is not None
        else None
    )
    if local_scale_weights is not None and args.local_view != "attention_multiscale":
        raise ValueError("--local-scale-weights requires attention_multiscale")
    if args.tta != "none" and not args.acknowledge_tta_risk:
        raise ValueError(
            "TTA is a competition gray area; pass --acknowledge-tta-risk explicitly"
        )
    if (
        args.local_view != "none"
        and not args.acknowledge_local_view_risk
    ):
        raise ValueError(
            "Local-view inference changes the inference protocol; pass "
            "--acknowledge-local-view-risk explicitly"
        )
    if not 0.0 <= args.tta_view_weight <= 1.0:
        raise ValueError("tta-view-weight must be in [0, 1]")
    stacked_local_tta = args.local_view != "none" and args.tta != "none"
    if stacked_local_tta and args.tta_fusion != "mean_probabilities":
        raise ValueError(
            "Stacked local-view TTA requires --tta-fusion mean_probabilities"
        )
    if (
        stacked_local_tta
        and (args.tta_temperature <= 0.0 or args.local_temperature <= 0.0)
    ):
        raise ValueError(
            "Stacked local-view TTA temperatures must be positive"
        )
    if not stacked_local_tta and args.tta_view_weight != 0.5:
        raise ValueError(
            "tta-view-weight is only defined for stacked local-view TTA"
        )
    if args.local_view != "none":
        if any(not 1 <= crop_size <= 224 for crop_size in local_crop_sizes):
            raise ValueError("All local crop sizes must be in [1, 224]")
        if not 1 <= args.local_top_k <= 49:
            raise ValueError("local-top-k must be in [1, 49]")
        if not 0.0 <= args.local_weight <= 1.0:
            raise ValueError("local-weight must be in [0, 1]")
        if args.local_temperature <= 0.0:
            raise ValueError("local-temperature must be positive")
    if args.adapt_local_features and args.local_view == "none":
        raise ValueError("--adapt-local-features requires an attention local view")
    if args.adapt_part_token_features and args.local_view == "none":
        raise ValueError(
            "--adapt-part-token-features requires an attention local view"
        )
    if args.adapt_local_features and args.adapt_part_token_features:
        raise ValueError(
            "--adapt-local-features and --adapt-part-token-features are "
            "mutually exclusive"
        )
    if args.tta in {
        "attention_local_global",
        "attention_local_adapter_global",
        "attention_part_token_adapter_global",
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
        if args.tta == "attention_local_adapter_global" or args.adapt_local_features
        else None
    )
    part_token_adapter = (
        load_part_token_adapter(checkpoint, device)
        if args.tta == "attention_part_token_adapter_global"
        or args.adapt_part_token_features
        else None
    )
    part_pool_spec = (
        checkpoint["part_token_adapter"]["spec"]["part_pool_spec"]
        if part_token_adapter is not None
        else None
    )
    multiprototype_head = checkpoint.get("multiprototype_head")
    if multiprototype_head is not None:
        if args.local_view != "none":
            raise ValueError(
                "Attention-local inference is not defined for multiprototype heads"
            )
        multiprototype_head = dict(multiprototype_head)
        multiprototype_head["prototypes"] = multiprototype_head["prototypes"].to(
            device=device, dtype=torch.float32
        )
    if args.tta in {
        "attention_local_global",
        "attention_local_adapter_global",
        "attention_part_token_adapter_global",
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
            if args.local_view in {"attention_crop", "attention_multiscale"}:
                global_logits, attention = forward_with_last_block_attention(
                    model, images
                )
                local_logits_views = []
                for crop_size in local_crop_sizes:
                    local_images, _ = extract_attention_crops(
                        images,
                        attention,
                        crop_size=crop_size,
                        top_k=args.local_top_k,
                    )
                    if args.adapt_local_features:
                        local_logits = adapted_local_view_logits(
                            model, local_feature_adapter, local_images
                        )
                    elif args.adapt_part_token_features:
                        local_logits = adapted_part_token_local_view_logits(
                            model,
                            part_token_adapter,
                            local_images,
                            part_top_patches=int(part_pool_spec["top_patches"]),
                            part_temperature=float(part_pool_spec["temperature"]),
                        )
                    else:
                        local_logits = model(images=local_images)
                    local_logits_views.append(local_logits)
                if args.tta == "horizontal_flip":
                    flipped_images = torch.flip(images, dims=(3,))
                    (
                        flipped_global_logits,
                        flipped_attention,
                    ) = forward_with_last_block_attention(
                        model,
                        flipped_images,
                    )
                    flipped_local_logits_views = []
                    for crop_size in local_crop_sizes:
                        flipped_local_images, _ = extract_attention_crops(
                            flipped_images,
                            flipped_attention,
                            crop_size=crop_size,
                            top_k=args.local_top_k,
                        )
                        if args.adapt_local_features:
                            flipped_local_logits = adapted_local_view_logits(
                                model,
                                local_feature_adapter,
                                flipped_local_images,
                            )
                        elif args.adapt_part_token_features:
                            flipped_local_logits = (
                                adapted_part_token_local_view_logits(
                                    model,
                                    part_token_adapter,
                                    flipped_local_images,
                                    part_top_patches=int(
                                        part_pool_spec["top_patches"]
                                    ),
                                    part_temperature=float(
                                        part_pool_spec["temperature"]
                                    ),
                                )
                            )
                        else:
                            flipped_local_logits = model(
                                images=flipped_local_images
                            )
                        flipped_local_logits_views.append(flipped_local_logits)
                    if args.local_view == "attention_multiscale":
                        logits = fuse_global_multilocal_flip_probabilities(
                            global_logits,
                            local_logits_views,
                            flipped_global_logits,
                            flipped_local_logits_views,
                            local_weight=args.local_weight,
                            flip_weight=args.tta_view_weight,
                            temperature=args.local_temperature,
                            global_temperature=args.tta_temperature,
                            local_temperature=args.local_temperature,
                            local_scale_weights=local_scale_weights,
                        )
                    else:
                        logits = fuse_global_local_flip_probabilities(
                            global_logits,
                            local_logits_views[0],
                            flipped_global_logits,
                            flipped_local_logits_views[0],
                            local_weight=args.local_weight,
                            flip_weight=args.tta_view_weight,
                            temperature=args.local_temperature,
                            local_scale_weights=local_scale_weights,
                            global_temperature=args.tta_temperature,
                            local_temperature=args.local_temperature,
                        )
                else:
                    if args.local_view == "attention_multiscale":
                        logits = fuse_global_multilocal_probabilities(
                            global_logits,
                            local_logits_views,
                            local_weight=args.local_weight,
                            temperature=args.local_temperature,
                        )
                    else:
                        logits = fuse_global_local_probabilities(
                            global_logits,
                            local_logits_views[0],
                            local_weight=args.local_weight,
                            temperature=args.local_temperature,
                        )
            elif args.tta == "attention_local_global":
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
            elif args.tta == "attention_part_token_adapter_global":
                if part_token_adapter is None:
                    raise RuntimeError("R1 part-token adapter was not loaded")
                logits = attention_part_token_adapter_global_logits(
                    model,
                    part_token_adapter,
                    images,
                    crop_size=160,
                    top_patches=5,
                    part_top_patches=int(part_pool_spec["top_patches"]),
                    part_temperature=float(part_pool_spec["temperature"]),
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
    if args.dump_logits:
        _atomic_torch_save(
            {
                "logits": all_logits.detach().float().cpu(),
                "names": prediction_names,
                "inference_mode": (
                    (
                        "attention_multiscale_flip:topk=%d:crops=%s:local_weight=%g:flip_weight=%g:t=%g"
                        % (
                            args.local_top_k,
                            "-".join(str(value) for value in local_crop_sizes),
                            args.local_weight,
                            args.tta_view_weight,
                            args.local_temperature,
                        )
                        if args.local_view == "attention_multiscale"
                        else "attention_crop_flip:topk=%d:crop=%d:local_weight=%g:flip_weight=%g:t=%g"
                        % (
                            args.local_top_k,
                            args.local_crop_size,
                            args.local_weight,
                            args.tta_view_weight,
                            args.local_temperature,
                        )
                    )
                    + (
                        ":weights="
                        + "-".join(f"{value:g}" for value in local_scale_weights)
                        if local_scale_weights is not None
                        else ""
                    )
                    + (":adapter=o3" if args.adapt_local_features else "")
                    + (
                        ":adapter=part_token"
                        if args.adapt_part_token_features
                        else ""
                    )
                    if args.local_view != "none"
                    else args.tta
                ),
            },
            args.dump_logits,
        )
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
    if args.local_view != "none":
        view_name = (
            "attention_multiscale" if args.local_view == "attention_multiscale"
            else "attention_crop"
        )
        inference_mode = view_name + ("_flip:" if stacked_local_tta else ":")
        inference_mode += f"topk={args.local_top_k}:"
        if args.local_view == "attention_multiscale":
            inference_mode += "crops=" + "-".join(
                str(value) for value in local_crop_sizes
            ) + ":"
            if local_scale_weights is not None:
                inference_mode += "weights=" + "-".join(
                    f"{value:g}" for value in local_scale_weights
                ) + ":"
        else:
            inference_mode += f"crop={args.local_crop_size}:"
        inference_mode += f"local_weight={args.local_weight:g}:"
        if stacked_local_tta:
            inference_mode += f"flip_weight={args.tta_view_weight:g}:"
        inference_mode += f"t={args.local_temperature:g}"
        if args.adapt_local_features:
            inference_mode += ":adapter=o3"
        elif args.adapt_part_token_features:
            inference_mode += ":adapter=part_token"
    elif args.tta == "attention_local_global":
        inference_mode = "attention_local_global:crop=160:top5:mean_probabilities"
    elif args.tta == "attention_local_adapter_global":
        inference_mode = (
            "attention_local_adapter_global:crop=160:top5:"
            "mean_probabilities"
        )
    elif args.tta == "attention_part_token_adapter_global":
        inference_mode = (
            "attention_part_token_adapter_global:crop=160:top5:"
            "part_top8:mean_probabilities"
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
        tta_risk_acknowledged=(
            args.acknowledge_tta_risk or args.acknowledge_local_view_risk
        ),
        valid_labels={str(value).zfill(4) for value in idx_to_class.values()},
        extra_manifest={
            "corrupt_images": corrupt_count,
            "tta_fusion": (
                "weighted_mean_probabilities"
                if stacked_local_tta
                else "mean_probabilities"
                if args.tta
                in {
                    "attention_local_global",
                    "attention_local_adapter_global",
                    "attention_part_token_adapter_global",
                }
                else "branch_mean_probabilities"
                if args.tta == "complementary_flip_local_global"
                else args.tta_fusion if args.tta != "none" else "none"
            ),
            "tta_temperature": (
                float(args.tta_temperature) if args.tta != "none" else 1.0
            ),
            "tta_view_weight": (
                float(args.tta_view_weight) if stacked_local_tta else None
            ),
            "local_view": args.local_view,
            "local_crop_size": (
                int(args.local_crop_size)
                if args.local_view == "attention_crop"
                else None
            ),
            "local_crop_sizes": (
                list(local_crop_sizes)
                if args.local_view == "attention_multiscale"
                else None
            ),
            "local_scale_weights": (
                list(local_scale_weights)
                if local_scale_weights is not None
                else None
            ),
            "local_top_k": (
                int(args.local_top_k) if args.local_view != "none" else None
            ),
            "local_weight": (
                float(args.local_weight)
                if args.local_view != "none"
                else None
            ),
            "local_temperature": (
                float(args.local_temperature)
                if args.local_view != "none"
                else None
            ),
            "local_view_risk_acknowledged": bool(
                args.acknowledge_local_view_risk
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
                in {
                    "attention_local_global",
                    "attention_local_adapter_global",
                    "attention_part_token_adapter_global",
                }
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
                or args.adapt_local_features
                else None
            ),
            "part_token_adapter": (
                {
                    **checkpoint["part_token_adapter"]["spec"],
                    "gate": checkpoint["part_token_adapter"]["gate"],
                    "view_condition": "attention_local_only",
                    "global_path": "native_parent_checkpoint_unchanged",
                    "epoch_zero_baseline": "F1+M1",
                }
                if args.tta == "attention_part_token_adapter_global"
                or args.adapt_part_token_features
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
