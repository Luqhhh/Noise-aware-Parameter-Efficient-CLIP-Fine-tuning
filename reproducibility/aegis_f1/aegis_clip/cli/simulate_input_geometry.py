"""Ask whether the test set's input geometry explains the local->platform gap.

The rematch train/val images carry 33,631 distinct shapes with long sides of
800-1390px+. The test images carry 807 distinct shapes and *every* dimension is
<= 500px (375x500 alone is 37.6%). That is a systematic geometry shift, and the
obvious hypothesis is that it explains why the best checkpoint scores ~72.9%
on val but only 60.97% on the platform.

This tests the hypothesis on the *labelled* validation split: each image is
downscaled so its long side is L, then handed to the ordinary CLIP
preprocessing. Test images contribute their size statistics only -- no test
image, label or prediction is read. ``L=None`` is the control and must
reproduce the checkpoint's recorded accuracy.

Answer as of 2026-09-22: the hypothesis is refuted. See
``docs/rematch750_inference_side_probe_20260922.md``.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import _load_split, _trust_values, resolve_image_path
from aegis_clip.features import FrozenFeatureStore, canonical_sample_path
from aegis_clip.runtime import seed_worker


def target_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Scale (width, height) so the longer side becomes long_side."""
    scale = long_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


class RescaledValidation(Dataset):
    """OnlineImageDataset.getitem with an optional pre-resize of the long side."""

    def __init__(self, split_csv, image_root, transform, feature_store, long_side):
        self.frame = _load_split(split_csv)
        self.paths = self.frame["image_path"].astype(str).tolist()
        self.labels = self.frame["label"].astype(int).tolist()
        self.image_root = Path(image_root)
        self.transform = transform
        self.feature_store = feature_store
        self.long_side = long_side

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        relative_path = self.paths[index]
        label = self.labels[index]
        absolute_path = resolve_image_path(self.image_root, relative_path)
        with Image.open(io.BytesIO(absolute_path.read_bytes())) as image:
            image = image.convert("RGB")
            if self.long_side is not None:
                image = image.resize(
                    target_size(*image.size, self.long_side), Image.BICUBIC
                )
            tensor = self.transform(image)
        item = {
            "images": tensor,
            "reference_features": self.feature_store.get(relative_path),
            "index": torch.tensor(index, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "path": canonical_sample_path(relative_path),
        }
        item.update(_trust_values(None, relative_path, label))
        return item


@torch.no_grad()
def accuracy(model, loader, device, use_amp):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["label"].to(device)
        with torch.autocast(
            device_type=device.type, enabled=use_amp and device.type == "cuda"
        ):
            logits, _ = model(images=images, return_features=True)
        correct += int((logits.float().argmax(dim=1) == labels).sum())
        total += labels.numel()
    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--long-sides", type=int, nargs="*",
                        default=[800, 640, 500, 400])
    parser.add_argument("--anchor-micro", type=float, default=None)
    parser.add_argument(
        "--anchor-tolerance",
        type=float,
        default=1e-6,
        help="the recorded metric is float32-accumulated, so it is not exactly "
             "k/N; 1e-6 is far below one image and far above that rounding",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    config = load_config(args.config)
    model, preprocess, _ = build_from_checkpoint(
        args.checkpoint, device, config_override=config
    )
    feature_config = config["features"]
    feature_store = FrozenFeatureStore(
        feature_config["tensor_path"],
        feature_config["paths_path"],
        feature_config.get("manifest_path"),
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    workers = int(config["train"].get("num_workers", 4))
    use_amp = bool(config["train"].get("amp", True))
    batch_size = int(config.get("evaluation", {}).get("batch_size", 128))

    results = []
    for long_side in [None] + args.long_sides:
        dataset = RescaledValidation(
            config["data"]["val_csv"], config["data"]["train_root"],
            preprocess, feature_store, long_side,
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
        correct, total = accuracy(model, loader, device, use_amp)
        name = "original" if long_side is None else f"long_side_{long_side}"
        results.append({"variant": name, "long_side": long_side,
                        "correct": correct, "total": total,
                        "accuracy": correct / total})
        print(f"  {name:18s} {correct:6d}/{total} = "
              f"{correct / total * 100:.4f}%", flush=True)

    control = results[0]
    if args.anchor_micro is not None:
        if abs(control["accuracy"] - args.anchor_micro) > args.anchor_tolerance:
            raise SystemExit(
                f"ANCHOR FAILED: control {control['accuracy']!r} != recorded "
                f"{args.anchor_micro!r}; the simulated pipeline is not the real one"
            )
        print(f"anchor OK: the no-resize control matches the recorded accuracy "
              f"(is {control['accuracy']!r}, recorded {args.anchor_micro!r})")

    print()
    for row in results:
        print(f"{row['variant']:18s} {row['accuracy'] * 100:9.4f}% "
              f"{(row['accuracy'] - control['accuracy']) * 100:+12.4f}pp")

    Path(args.output).write_text(
        json.dumps(
            {"checkpoint": str(args.checkpoint), "config": str(args.config),
             "results": results},
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
