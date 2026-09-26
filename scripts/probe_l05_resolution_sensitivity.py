#!/usr/bin/env python3
"""Measure L05 validation sensitivity to a fixed 500px bicubic pre-resize."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility" / "aegis_f1"))

from aegis_clip.checkpoint import build_from_checkpoint  # noqa: E402
from aegis_clip.runtime import sha256_file  # noqa: E402
from aegis_clip.tta import fuse_paired_logits  # noqa: E402


class PairedValidation(Dataset):
    def __init__(self, csv_path: Path, image_root: Path, paths: list[str], transform, limit: int):
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_path = {row["image_path"].removeprefix("train/"): row for row in rows}
        if len(by_path) != len(rows) or len(paths) != len(rows) or set(paths) != set(by_path):
            raise ValueError("validation CSV and frozen cache have different sample sets")
        self.paths = paths[:limit]
        self.by_path = by_path
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        row = self.by_path[path]
        absolute = self.image_root / path
        raw = absolute.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["file_sha256"]:
            raise ValueError(f"validation image hash mismatch: {path}")
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        from io import BytesIO
        with Image.open(BytesIO(raw)) as source:
            original = source.convert("RGB")
            width, height = original.size
            target = original
            changed = max(width, height) > 500
            if changed:
                factor = 500.0 / max(width, height)
                target = original.resize(
                    (max(1, round(width * factor)), max(1, round(height * factor))),
                    Image.Resampling.BICUBIC,
                )
            transformed = self.transform(target)
            if index < 32:
                original_tensor = self.transform(original)
            else:
                original_tensor = torch.empty(0)
        return transformed, original_tensor, int(row["label"]), changed, path


def per_class_macro(prediction: torch.Tensor, labels: torch.Tensor, classes: int,
                    *, require_all: bool) -> float:
    counts = torch.bincount(labels, minlength=classes)
    correct = torch.bincount(labels[prediction == labels], minlength=classes)
    if require_all and bool((counts == 0).any()):
        raise ValueError("validation split lacks a class")
    present = counts > 0
    return float((correct[present].double() / counts[present].double()).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=14880)
    args = parser.parse_args()
    if args.batch_size != 32:
        raise ValueError("fixed 32-sample native replay requires --batch-size 32")
    expected_checkpoint = "35d17c0c3b9f281e13353959d098c2713def6b7d508d334de5d3a7e769cd2397"
    expected_csv = "d91106df365b62f9bf56eff51474c222925301546642fa427f6778258cb833ab"
    if sha256_file(args.checkpoint) != expected_checkpoint or sha256_file(args.val_csv) != expected_csv:
        raise ValueError("frozen L05 checkpoint or validation CSV hash mismatch")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache["checkpoint_sha256"] != expected_checkpoint or cache["validation_csv_sha256"] != expected_csv:
        raise ValueError("frozen validation cache has a different source")
    paths = list(cache["paths"])
    if args.limit <= 0 or args.limit > len(paths):
        raise ValueError("limit outside frozen validation cache")
    device = torch.device("cuda:0")
    model, preprocess, _ = build_from_checkpoint(args.checkpoint, device)
    model.eval()
    dataset = PairedValidation(args.val_csv, args.image_root, paths, preprocess, args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    downsized_first, downsized_flip, labels, resized = [], [], [], []
    baseline_max_diff = 0.0
    baseline_agreement = True
    offset = 0
    with torch.inference_mode():
        for images, originals, batch_labels, batch_changed, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=True):
                first = model(images=images).float().cpu()
                flip = model(images=torch.flip(images, dims=(3,))).float().cpu()
            n = len(batch_labels)
            if offset < 32:
                original_batch = originals.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=True):
                    base_first = model(images=original_batch).float().cpu()
                    base_flip = model(images=torch.flip(original_batch, dims=(3,))).float().cpu()
                ref_first = cache["original_logits"][offset:offset+n]
                ref_flip = cache["flip_logits"][offset:offset+n]
                baseline_max_diff = max(baseline_max_diff,
                    float((base_first - ref_first).abs().max()),
                    float((base_flip - ref_flip).abs().max()))
                baseline_agreement &= bool(torch.equal(base_first.argmax(1), ref_first.argmax(1)))
                baseline_agreement &= bool(torch.equal(base_flip.argmax(1), ref_flip.argmax(1)))
                if not baseline_agreement or baseline_max_diff > 0.02:
                    raise ValueError(f"native L05 replay mismatch: {baseline_max_diff=}, {baseline_agreement=}")
            downsized_first.append(first)
            downsized_flip.append(flip)
            labels.append(batch_labels)
            resized.append(batch_changed)
            offset += n
            if offset % 3200 < n:
                print(f"processed {offset}/{args.limit}", flush=True)
    labels = torch.cat(labels).long()
    if not torch.equal(labels, cache["labels"][:args.limit].long()):
        raise ValueError("validation labels disagree with frozen cache")
    reduced = fuse_paired_logits(torch.cat(downsized_first), torch.cat(downsized_flip),
                                  mode="mean_probabilities", temperature=1.4).argmax(1)
    original = fuse_paired_logits(cache["original_logits"][:args.limit],
                                   cache["flip_logits"][:args.limit],
                                   mode="mean_probabilities", temperature=1.4).argmax(1)
    changed = torch.cat(resized).bool()
    num_classes = int(cache["original_logits"].shape[1])
    original_correct = original == labels
    reduced_correct = reduced == labels
    result = {
        "checkpoint_sha256": expected_checkpoint,
        "cache_sha256": sha256_file(args.cache),
        "validation_csv_sha256": expected_csv,
        "limit": args.limit,
        "pre_resize": "bicubic, longest side at most 500px",
        "model_input_resolution": 384,
        "fusion": "mean_probabilities, temperature 1.4, horizontal flip",
        "native_replay_first_32": {"max_abs_logit_diff": baseline_max_diff,
                                    "all_branch_top1_agree": baseline_agreement},
        "actually_resized": int(changed.sum()),
        "original": {"micro": float(original_correct.double().mean()),
                     "macro": per_class_macro(original, labels, num_classes,
                                              require_all=args.limit == len(paths))},
        "downscaled": {"micro": float(reduced_correct.double().mean()),
                       "macro": per_class_macro(reduced, labels, num_classes,
                                                require_all=args.limit == len(paths))},
        "paired": {"prediction_changed": int((original != reduced).sum()),
                   "corrections": int(((~original_correct) & reduced_correct).sum()),
                   "regressions": int((original_correct & (~reduced_correct)).sum())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
