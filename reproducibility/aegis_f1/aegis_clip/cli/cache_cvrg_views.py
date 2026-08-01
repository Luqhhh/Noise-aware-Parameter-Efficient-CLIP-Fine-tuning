"""Generate audited four-view CVRG validation or test caches."""

from __future__ import annotations

import argparse
import hashlib
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.data import TestImageDataset, resolve_image_path
from aegis_clip.image_preprocess import select_inference_preprocess
from aegis_clip.localization import (
    extract_attention_crops,
    forward_features_with_last_block_attention,
)
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.view_reliability import (
    CVRGProtocol,
    atomic_torch_save,
    validate_cvrg_cache,
)


class ValidationImages(Dataset):
    def __init__(self, csv_path, image_root, transform):
        self.frame = pd.read_csv(csv_path)
        self.root = Path(image_root)
        self.transform = transform
        self.paths = self.frame["image_path"].astype(str).tolist()
        self.labels = self.frame["label"].astype(int).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        relative = self.paths[index]
        with Image.open(resolve_image_path(self.root, relative)) as image:
            return {
                "images": self.transform(image.convert("RGB")),
                "path": relative,
                "label": torch.tensor(self.labels[index], dtype=torch.long),
                "clean_probability": torch.tensor(1.0),
                "pseudo_label": torch.tensor(self.labels[index], dtype=torch.long),
                "correction_alpha": torch.tensor(0.0),
            }


@torch.no_grad()
def build_cache(checkpoint, output, *, split, config_path=None, batch_size=64, num_workers=0, device_name="cuda", limit=None):
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    model, preprocess, checkpoint_payload = build_from_checkpoint(checkpoint, device)
    config = checkpoint_payload["config"]
    if config_path:
        from aegis_clip.config import load_config
        config = load_config(config_path)
    preprocess = select_inference_preprocess(preprocess, mode="clip_center_crop", input_resolution=int(config["model"].get("input_resolution", 224)))
    if split == "validation":
        dataset = ValidationImages(config["data"]["val_csv"], config["data"]["train_root"], preprocess)
    else:
        dataset = TestImageDataset(config["data"]["test_root"], preprocess)
    if limit is not None:
        dataset = Subset(dataset, range(min(int(limit), len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits_parts, feature_parts, att_parts, box_parts = [], [], [], []
    paths, labels, clean, pseudo, correction = [], [], [], [], []
    corrupt = 0
    model.eval()
    for batch in loader:
        images = batch["images"].to(device)
        original_logits, original_features, original_attention = forward_features_with_last_block_attention(model, images)
        local_images, original_boxes = extract_attention_crops(images, original_attention, crop_size=160, top_k=5)
        local_logits, local_features = model(images=local_images, return_features=True)
        flipped_images = torch.flip(images, dims=(3,))
        flipped_logits, flipped_features, flipped_attention = forward_features_with_last_block_attention(model, flipped_images)
        flipped_local_images, flipped_boxes = extract_attention_crops(flipped_images, flipped_attention, crop_size=160, top_k=5)
        flipped_local_logits, flipped_local_features = model(images=flipped_local_images, return_features=True)
        boxes = torch.tensor([original_boxes, flipped_boxes], dtype=torch.int64).permute(1, 0, 2)
        logits_parts.append(torch.stack((original_logits, local_logits, flipped_logits, flipped_local_logits), 1).float().cpu())
        feature_parts.append(torch.stack((original_features, local_features, flipped_features, flipped_local_features), 1).float().cpu())
        att_parts.append(torch.stack((original_attention, flipped_attention), 1).float().cpu())
        box_parts.append(boxes)
        if split == "validation":
            paths.extend(str(x) for x in batch["path"])
            labels.append(batch["label"].cpu())
            clean.append(batch["clean_probability"].cpu())
            pseudo.append(batch["pseudo_label"].cpu())
            correction.append(batch["correction_alpha"].cpu())
        else:
            paths.extend(str(x) for x in batch["name"])
            corrupt += int(batch["corrupt"].sum())
    if corrupt:
        raise RuntimeError(f"refusing to publish cache: {corrupt} corrupt test images")
    checkpoint_sha = sha256_file(checkpoint)
    payload = {
        "format_version": 1,
        "split": split,
        "view_order": ["original_global", "original_local", "flipped_global", "flipped_local"],
        "checkpoint_sha256": checkpoint_sha,
        "split_sha256": sha256_file(config["data"]["val_csv"]) if split == "validation" else hashlib.sha256("\n".join(paths).encode()).hexdigest(),
        "protocol": CVRGProtocol().__dict__,
        "view_logits": torch.cat(logits_parts),
        "view_features": torch.nn.functional.normalize(torch.cat(feature_parts), dim=-1),
        "orientation_attention": torch.cat(att_parts),
        "crop_boxes": torch.cat(box_parts),
        "paths": paths,
    }
    if split == "validation":
        payload.update({
            "labels": torch.cat(labels),
            "clean_probability": torch.cat(clean),
            "pseudo_label": torch.cat(pseudo),
            "correction_alpha": torch.cat(correction),
        })
    validate_cvrg_cache(payload, require_labels=split == "validation")
    output = Path(output)
    atomic_torch_save(payload, output)
    cache_sha = sha256_file(output)
    atomic_json_dump({
        "format_version": 1, "cache": str(output.resolve()), "cache_sha256": cache_sha,
        "checkpoint_sha256": checkpoint_sha, "split": split, "sample_count": len(paths),
        "class_count": 500, "feature_schema_version": 1, "contains_labels": split == "validation",
        "protocol": CVRGProtocol().__dict__,
    }, output.with_name("view_cache_manifest.json"))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--config")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    build_cache(args.checkpoint, args.output, split=args.split, config_path=args.config, batch_size=args.batch_size, num_workers=args.num_workers, device_name=args.device, limit=args.limit)


if __name__ == "__main__":
    main()
