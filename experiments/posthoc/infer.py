"""Horizontal-flip TTA inference for a checkpoint-embedded prototype head.

Usage:
    python -m experiments.posthoc.infer --config configs/e20_posthoc.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.dataset import TestImageDataset
from common.posthoc import blend_multiprototype_logits, fuse_paired_logits
from common.submission import validate_submission_coverage
from common.utils import load_config, set_seed
from experiments.baseline.model import build_model


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    recipe = config["posthoc"]
    set_seed(int(config["data"].get("train_seed", config["data"]["seed"])))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint or recipe["output_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    head = checkpoint.get("multiprototype_head")
    if head is None:
        raise ValueError("checkpoint does not contain multiprototype_head")

    model, preprocess = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    idx_to_class = checkpoint.get("idx_to_class")
    if idx_to_class is None:
        mapping_path = Path(config["data"]["class_mapping_path"]) / "idx_to_class.json"
        with open(mapping_path, "r", encoding="utf-8") as handle:
            idx_to_class = json.load(handle)

    dataset = TestImageDataset(config["data"]["test_dir"], transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=int(config["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 8)),
        pin_memory=device.type == "cuda",
    )
    fusion = str(recipe.get("tta_fusion", "mean_probabilities"))
    temperature = float(recipe.get("tta_temperature", 1.0))
    predictions = []
    corrupt_images = 0
    for images, names, _paths in tqdm(loader, desc="Post-hoc TTA inference"):
        corrupt_images += int((images.abs().sum(dim=(1, 2, 3)) == 0).sum())
        images = images.to(device, non_blocking=True)
        first_features = model.encode_image(images)
        second_features = model.encode_image(torch.flip(images, dims=[3]))
        first_logits = blend_multiprototype_logits(
            model.classifier(first_features), first_features, head
        )
        second_logits = blend_multiprototype_logits(
            model.classifier(second_features), second_features, head
        )
        scores = fuse_paired_logits(
            first_logits,
            second_logits,
            mode=fusion,
            temperature=temperature,
        )
        for name, prediction in zip(names, scores.argmax(1).cpu().tolist()):
            label = str(idx_to_class[str(int(prediction))]).zfill(4)
            predictions.append((name, int(prediction), label))

    output_dir = Path(args.output_dir or config["output"]["submission_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "pred_raw.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "pred_idx", "pred_label"])
        writer.writerows(predictions)
    result_path = output_dir / "pred_results.csv"
    with open(result_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for name, _prediction, label in predictions:
            writer.writerow([name, f" {label}"])
    validate_submission_coverage(config["data"]["test_dir"], str(result_path))
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(result_path, arcname="pred_results.csv")
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "prediction_head": "linear_plus_multiprototype",
        "inference_mode": f"horizontal_flip:{fusion}:t={temperature:g}",
        "tta_fusion": fusion,
        "tta_temperature": temperature,
        "prediction_count": len(predictions),
        "corrupt_images": corrupt_images,
        "prediction_csv_sha256": _sha256(result_path),
        "submission_zip_sha256": _sha256(zip_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
