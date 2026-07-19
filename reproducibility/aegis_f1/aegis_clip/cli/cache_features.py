"""Rebuild frozen official OpenAI CLIP ViT-B/32 features from stage training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aegis_clip.config import load_config
from aegis_clip.data import IMAGE_EXTENSIONS
from aegis_clip.runtime import (
    atomic_json_dump,
    environment_manifest,
    seed_worker,
    sha256_lines,
)
from aegis_clip.trust import atomic_torch_save


def apply_feature_augmentation(
    images: torch.Tensor, augmentation: str
) -> torch.Tensor:
    if augmentation == "none":
        return images
    if augmentation == "horizontal_flip":
        return torch.flip(images, dims=(3,))
    raise ValueError(f"Unsupported feature augmentation: {augmentation}")


class StageImageDataset(Dataset):
    def __init__(self, root: str | Path, transform) -> None:
        self.root = Path(root).resolve()
        self.transform = transform
        class_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
        self.class_to_idx = {
            directory.name: index for index, directory in enumerate(class_dirs)
        }
        self.records: list[tuple[Path, str, int]] = []
        for directory in class_dirs:
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    canonical = f"{directory.name}/{path.name}"
                    self.records.append(
                        (path, canonical, self.class_to_idx[directory.name])
                    )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path, canonical, label = self.records[index]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Pillow failed to decode official image: {path}") from exc
        return tensor, canonical, label


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--augmentation", choices=["none", "horizontal_flip"], default="none"
    )
    parser.add_argument(
        "--output-dir",
        help="Required for augmented caches so the canonical cache is never overwritten",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.augmentation != "none" and not args.output_dir:
        raise ValueError("augmented feature caches require --output-dir")
    config = load_config(args.config)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    try:
        import clip
    except ImportError as exc:
        raise ImportError("Install the pinned official OpenAI CLIP package") from exc
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    dataset = StageImageDataset(config["data"]["train_root"], preprocess)
    expected_samples = int(config["data"]["expected_official_train_samples"])
    expected_classes = int(config["model"]["num_classes"])
    if len(dataset) != expected_samples:
        raise ValueError(f"Feature source has {len(dataset)} images, expected {expected_samples}")
    if len(dataset.class_to_idx) != expected_classes:
        raise ValueError(
            f"Feature source has {len(dataset.class_to_idx)} classes, "
            f"expected {expected_classes}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        timeout=120 if args.workers else 0,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
    )
    chunks = []
    paths: list[str] = []
    labels: list[int] = []
    for images, batch_paths, batch_labels in tqdm(loader, desc="Official CLIP cache"):
        images = apply_feature_augmentation(
            images.to(device, non_blocking=True), args.augmentation
        )
        encoded = clip_model.encode_image(images)
        chunks.append(F.normalize(encoded.float(), dim=1).cpu())
        paths.extend(list(batch_paths))
        labels.extend(torch.as_tensor(batch_labels).tolist())
    features = torch.cat(chunks, dim=0)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = output_dir / "features.pt"
        paths_path = output_dir / "image_paths.json"
        manifest_path = output_dir / "manifest.json"
    else:
        feature_config = config["features"]
        tensor_path = Path(feature_config["tensor_path"])
        paths_path = Path(feature_config["paths_path"])
        manifest_path = Path(feature_config["manifest_path"])
    atomic_torch_save(features, tensor_path)
    atomic_json_dump(paths, paths_path)
    atomic_json_dump(labels, paths_path.with_name("labels.json"))
    runtime = environment_manifest()
    manifest = {
        "format_version": 1,
        "stage": config["project"]["stage"],
        "backbone": "ViT-B/32",
        "pretrained": "openai",
        "normalized": True,
        "augmentation": args.augmentation,
        "dataset_size": len(dataset),
        "feature_dim": int(features.shape[1]),
        "source_root": str(Path(config["data"]["train_root"]).resolve()),
        "external_data": False,
        "test_data_used": False,
        "path_index_sha256": sha256_lines(paths),
        "runtime": runtime,
    }
    atomic_json_dump(manifest, manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
