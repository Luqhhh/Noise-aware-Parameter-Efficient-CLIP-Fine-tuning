"""Bare or explicitly acknowledged flip-TTA inference with fail-closed output."""

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
from aegis_clip.multiprototype import blend_multiprototype_logits
from aegis_clip.runtime import seed_worker, set_seed
from aegis_clip.submission import create_submission
from aegis_clip.tta import TTA_FUSION_MODES, fuse_paired_logits


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tta", choices=["none", "horizontal_flip"], default="none")
    parser.add_argument(
        "--tta-fusion", choices=sorted(TTA_FUSION_MODES), default="mean_logits"
    )
    parser.add_argument("--tta-temperature", type=float, default=1.0)
    parser.add_argument("--acknowledge-tta-risk", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.tta != "none" and not args.acknowledge_tta_risk:
        raise ValueError(
            "TTA is a competition gray area; pass --acknowledge-tta-risk explicitly"
        )
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model, preprocess, checkpoint = build_from_checkpoint(args.checkpoint, device)
    config = load_config(args.config) if args.config else checkpoint["config"]
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
    loader = DataLoader(
        dataset,
        batch_size=int(
            config["evaluation"].get(
                "inference_batch_size",
                min(int(config["evaluation"].get("batch_size", 256)), 256),
            )
        ),
        shuffle=False,
        num_workers=workers,
        timeout=int(config["train"].get("loader_timeout", 120 if workers else 0)),
        pin_memory=bool(config["train"].get("pin_memory", True)),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )
    model.eval()
    multiprototype_head = checkpoint.get("multiprototype_head")
    if multiprototype_head is not None:
        multiprototype_head = dict(multiprototype_head)
        multiprototype_head["prototypes"] = multiprototype_head["prototypes"].to(
            device=device, dtype=torch.float32
        )
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    predictions: list[tuple[str, str]] = []
    corrupt_count = 0
    for batch in tqdm(loader, desc="Aegis inference"):
        images = batch["images"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            if multiprototype_head is None:
                logits = model(images=images)
            else:
                logits, features = model(images=images, return_features=True)
                logits = blend_multiprototype_logits(
                    logits, features, multiprototype_head
                )
            if args.tta == "horizontal_flip":
                if multiprototype_head is None:
                    second_logits = model(images=torch.flip(images, dims=(3,)))
                else:
                    second_logits, second_features = model(
                        images=torch.flip(images, dims=(3,)), return_features=True
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
        indices = logits.argmax(dim=1).cpu().tolist()
        names = list(batch["name"])
        corrupt_count += int(batch["corrupt"].sum())
        predictions.extend(
            (name, str(idx_to_class[index]).zfill(4))
            for name, index in zip(names, indices)
        )
    expected_names = [path.name for path in dataset.paths]
    if corrupt_count:
        raise RuntimeError(
            f"Refusing to publish: Pillow failed to decode {corrupt_count} test images"
        )
    manifest = create_submission(
        predictions,
        expected_names,
        args.output_dir,
        args.checkpoint,
        inference_mode=(
            args.tta
            if args.tta == "none" or args.tta_fusion == "mean_logits"
            else f"{args.tta}:{args.tta_fusion}:t={args.tta_temperature:g}"
        ),
        tta_risk_acknowledged=args.acknowledge_tta_risk,
        valid_labels={str(value).zfill(4) for value in idx_to_class.values()},
        extra_manifest={
            "corrupt_images": corrupt_count,
            "tta_fusion": args.tta_fusion if args.tta != "none" else "none",
            "tta_temperature": (
                float(args.tta_temperature) if args.tta != "none" else 1.0
            ),
            "prediction_head": (
                "linear_plus_multiprototype"
                if multiprototype_head is not None
                else "linear"
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
