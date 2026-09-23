#!/usr/bin/env python3
"""Extract frozen V3 visual features for the H-group head-only trials.

The V3 checkpoint is a same-split full-finetune model.  This script loads its
visual tower, runs a deterministic 224px center-crop pass over the full current
training pool in ``full_train.csv`` order, and writes a feature cache whose
manifest is bound to the requested V4 H config.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aegis_clip.checkpoint import load_initial_weights
from aegis_clip.config import load_config
from aegis_clip.data import resolve_image_path
from aegis_clip.device import resolve_device
from aegis_clip.model import build_model
from aegis_clip.runtime import (
    atomic_json_dump,
    seed_worker,
    sha256_file,
    sha256_lines,
)
from aegis_clip.rematch_search_v4 import expected_feature_binding, validate_dataset
from aegis_clip.trust import atomic_torch_save


class PoolDataset(Dataset):
    """Read full_train.csv in-order, returning preprocessed image and path."""

    def __init__(self, train_root: str | Path, split_csv: str | Path, transform) -> None:
        self.root = Path(train_root)
        self.transform = transform
        with Path(split_csv).open(encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError("full_train.csv is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        relative = str(row["image_path"]).removeprefix("train/")
        path = resolve_image_path(self.root, relative)
        raw = path.read_bytes()
        expected_hash = row.get("file_sha256", "")
        if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"Training image changed after split audit: {path}")
        with Image.open(io.BytesIO(raw)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, relative


@torch.no_grad()
def extract(config_path: str | Path, checkpoint: str | Path, output_dir: str | Path, *,
            batch_size: int, workers: int, device_name: str) -> dict:
    config = load_config(config_path)
    if config["model"].get("backbone") != "ViT-B/32":
        raise ValueError("V3 feature extraction only supports ViT-B/32")
    train_config = config["train"]
    train_config["device"] = device_name
    device = resolve_device(device_name)
    model_config = config["model"]
    model_config["peft_mode"] = "full_finetune"
    model_config["use_cached_training"] = False
    model_config["input_resolution"] = 224
    model, preprocess = build_model(config, device)
    state = load_initial_weights(model, checkpoint, device)
    del state
    model.float().eval()
    source_hashes = {}
    dataset = PoolDataset(
        config["data"]["train_root"], config["data"]["train_csv"], preprocess
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        timeout=120 if int(workers) else 0,
        pin_memory=device.type in {"cuda", "npu"},
        persistent_workers=int(workers) > 0,
        worker_init_fn=seed_worker,
    )
    chunks = []
    paths: list[str] = []
    for images, batch_paths in tqdm(loader, desc="V3 frozen features"):
        images = images.to(device, non_blocking=True)
        encoded = model.encode_image(images)
        chunks.append(F.normalize(encoded.float(), dim=-1).cpu())
        paths.extend(str(value) for value in batch_paths)
    features = torch.cat(chunks, dim=0).float()
    if features.shape[1] != int(model_config.get("feature_dim", 512)):
        raise ValueError("Unexpected V3 feature dimension")
    if len(paths) != features.shape[0]:
        raise ValueError("V3 path/feature length mismatch")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    tensor_path = destination / "features.pt"
    paths_path = destination / "image_paths.json"
    manifest_path = destination / "manifest.json"
    atomic_torch_save(features, tensor_path)
    atomic_json_dump(paths, paths_path)
    atomic_json_dump([int(row["label"]) for row in dataset.rows], destination / "labels.json")

    manifest = validate_dataset(config)
    binding = expected_feature_binding(config, manifest)
    runtime_manifest = {
        "format_version": 1,
        "stage": "repechage",
        "artifact_scope": "v3_frozen_feature_cache",
        "backbone": "ViT-B/32",
        "pretrained": "openai",
        "normalized": True,
        "augmentation": "none",
        "dataset_size": len(paths),
        "feature_dim": int(features.shape[1]),
        "source_root": str(Path(config["data"]["train_root"]).resolve()),
        "external_data": False,
        "test_data_used": False,
        "path_index_sha256": sha256_lines(paths),
        "rematch_binding": binding,
        "tensor_sha256": sha256_file(tensor_path),
        "paths_file_sha256": sha256_file(paths_path),
        "preprocessing_repr": repr(preprocess),
        "source_checkpoint": str(Path(checkpoint).resolve()),
        "source_checkpoint_sha256": sha256_file(checkpoint),
    }
    atomic_json_dump(runtime_manifest, manifest_path)
    return runtime_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    manifest = extract(
        args.config,
        args.checkpoint,
        args.output_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        device_name=args.device,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
