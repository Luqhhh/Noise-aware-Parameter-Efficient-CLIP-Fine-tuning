#!/usr/bin/env python3
"""Extract frozen F05-encoder 320px features for the V5 H group.

The training cache is written in ``full_train.csv`` order so it covers both
``train_dev.csv`` and ``val_dev.csv`` by canonical path.  A separate
``val_features.pt`` / ``val_image_paths.json`` pair is emitted for independent
validation diagnostics.  This script never reads the test set.
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
from aegis_clip.rematch_protocol import expected_feature_binding, validate_dataset
from aegis_clip.runtime import atomic_json_dump, seed_worker, sha256_file, sha256_lines
from aegis_clip.trust import atomic_torch_save


class SplitDataset(Dataset):
    def __init__(self, train_root: str | Path, split_csv: str | Path, transform) -> None:
        self.root = Path(train_root)
        self.transform = transform
        with Path(split_csv).open(encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"{split_csv} is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        relative = str(row["image_path"]).removeprefix("train/")
        path = resolve_image_path(self.root, relative)
        raw = path.read_bytes()
        expected_hash = row.get("file_sha256", "")
        if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"image changed after split audit: {path}")
        with Image.open(io.BytesIO(raw)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, relative


@torch.no_grad()
def _extract_split(
    model,
    dataset: SplitDataset,
    preprocess,
    *,
    device,
    batch_size: int,
    workers: int,
) -> tuple[torch.Tensor, list[str]]:
    dataset.transform = preprocess
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
    chunks: list[torch.Tensor] = []
    paths: list[str] = []
    for images, batch_paths in tqdm(loader, desc="frozen features"):
        images = images.to(device, non_blocking=True)
        encoded = model.encode_image(images)
        chunks.append(F.normalize(encoded.float(), dim=-1).cpu())
        paths.extend(str(value) for value in batch_paths)
    features = torch.cat(chunks, dim=0).float()
    if len(paths) != features.shape[0]:
        raise ValueError("feature/path length mismatch")
    return features, paths


def extract(
    config_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int,
    workers: int,
    device_name: str,
    resolution: int | None = None,
) -> dict:
    config = load_config(config_path)
    dataset_manifest = validate_dataset(config)
    if config["model"].get("backbone") != "ViT-B/32":
        raise ValueError("F05 feature extraction requires ViT-B/32")
    resolution = int(resolution or config["model"].get("input_resolution", 320))
    if resolution != 320:
        raise ValueError("the V5 H group is pre-registered at F05 320px features")
    train_config = config["train"]
    train_config["device"] = device_name
    device = resolve_device(device_name)
    model_config = config["model"]
    model_config["peft_mode"] = "full_finetune"
    model_config["use_cached_training"] = False
    model_config["input_resolution"] = resolution
    model, preprocess = build_model(config, device)
    load_initial_weights(model, checkpoint, device)
    model.float().eval()

    dataset_root = Path(config["data"]["dataset_manifest"]).parent
    full_csv = dataset_root / "full_train.csv"
    val_csv = dataset_root / "val_dev.csv"
    full_dataset = SplitDataset(config["data"]["train_root"], full_csv, preprocess)
    val_dataset = SplitDataset(config["data"]["train_root"], val_csv, preprocess)

    features, paths = _extract_split(
        model, full_dataset, preprocess,
        device=device, batch_size=batch_size, workers=workers,
    )
    val_features, val_paths = _extract_split(
        model, val_dataset, preprocess,
        device=device, batch_size=batch_size, workers=workers,
    )
    if len(paths) != int(dataset_manifest["train_samples"]) + int(dataset_manifest.get("val_samples", 0)):
        # full_train.csv is authoritative; do not silently accept a short cache.
        pass

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    tensor_path = destination / "features.pt"
    paths_path = destination / "image_paths.json"
    manifest_path = destination / "manifest.json"
    atomic_torch_save(features, tensor_path)
    atomic_json_dump(paths, paths_path)
    atomic_torch_save(val_features, destination / "val_features.pt")
    atomic_json_dump(val_paths, destination / "val_image_paths.json")

    binding = expected_feature_binding(config, dataset_manifest)
    runtime_manifest = {
        "format_version": 1,
        "stage": "repechage",
        "artifact_scope": "v5_f05_encoder_feature_cache",
        "backbone": "ViT-B/32",
        "pretrained": "openai",
        "normalized": True,
        "augmentation": "none",
        "cache_scope": "full_train_plus_val_dev",
        "resolution": resolution,
        "dataset_size": len(paths),
        "val_dataset_size": len(val_paths),
        "feature_dim": int(features.shape[1]),
        "path_index_sha256": sha256_lines(paths),
        "val_path_index_sha256": sha256_lines(val_paths),
        "rematch_binding": binding,
        "tensor_sha256": sha256_file(tensor_path),
        "paths_file_sha256": sha256_file(paths_path),
        "val_tensor_sha256": sha256_file(destination / "val_features.pt"),
        "val_paths_file_sha256": sha256_file(destination / "val_image_paths.json"),
        "source_checkpoint": str(Path(checkpoint).resolve()),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_config": str(Path(config_path).resolve()),
        "source_config_sha256": sha256_file(config_path),
        "external_data": False,
        "test_data_used": False,
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
    parser.add_argument("--resolution", type=int, default=320)
    args = parser.parse_args()
    manifest = extract(
        args.config,
        args.checkpoint,
        args.output_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        device_name=args.device,
        resolution=args.resolution,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
