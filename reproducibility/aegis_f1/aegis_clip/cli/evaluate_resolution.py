"""Evaluate an existing checkpoint at multiple input resolutions.

Uses the built-in ViT 2-D position-embedding interpolation to run a 224px
checkpoint at higher resolutions (must be divisible by the patch size 32).
Reports global raw / clean-core metrics on the validation split at each
resolution, without retraining.  Decision input for whether to invest in
high-resolution training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import OnlineImageDataset, TrustBundle
from aegis_clip.evaluation import weighted_accuracy, weighted_macro_accuracy
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.runtime import atomic_json_dump, seed_worker


def _parse_int_sequence(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolutions", default="224,256,288,320,384,448")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    resolutions = _parse_int_sequence(args.resolutions)
    for resolution in resolutions:
        if resolution % 32:
            raise ValueError(
                f"Resolution {resolution} must be divisible by the patch size 32"
            )

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    base_config = load_config(args.config)
    num_classes = int(base_config["model"]["num_classes"])
    clean_core_threshold = float(
        base_config["evaluation"].get("clean_core_threshold", 0.70)
    )
    feature_config = base_config["features"]
    feature_store = FrozenFeatureStore(
        feature_config["tensor_path"],
        feature_config["paths_path"],
        feature_config.get("manifest_path"),
        expected_dim=int(base_config["model"].get("feature_dim", 512)),
    )
    trust_bundle = (
        TrustBundle(base_config["trust"]["bundle_path"])
        if base_config.get("trust", {}).get("enabled", False)
        else None
    )
    workers = int(base_config["train"].get("num_workers", 4))

    results: dict[str, dict[str, float | int]] = {}
    for resolution in resolutions:
        config = load_config(args.config)
        config["model"]["input_resolution"] = resolution
        model, preprocess, _ = build_from_checkpoint(
            args.checkpoint, device, config_override=config
        )
        model.eval()
        dataset = OnlineImageDataset(
            config["data"]["val_csv"],
            config["data"]["train_root"],
            preprocess,
            feature_store,
            trust_bundle,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(
                args.batch_size or config["evaluation"].get("batch_size", 128)
            ),
            shuffle=False,
            num_workers=workers,
            timeout=int(config["train"].get("loader_timeout", 120 if workers else 0)),
            pin_memory=bool(config["train"].get("pin_memory", True)),
            persistent_workers=workers > 0,
            worker_init_fn=seed_worker,
        )
        prediction_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        clean_parts: list[torch.Tensor] = []
        use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
        for batch in tqdm(loader, desc=f"Resolution {resolution}"):
            images = batch["images"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images=images)
            prediction_parts.append(logits.float().argmax(dim=1).cpu())
            label_parts.append(batch["label"].long())
            clean_parts.append(batch["clean_probability"].float())
        prediction = torch.cat(prediction_parts)
        label = torch.cat(label_parts)
        clean = torch.cat(clean_parts)
        num_classes_actual = len(torch.unique(prediction))
        clean_core_mask = clean >= clean_core_threshold
        results[str(resolution)] = {
            "raw_micro": float(weighted_accuracy(prediction, label, torch.ones_like(label))),
            "raw_macro": float(
                weighted_macro_accuracy(prediction, label, torch.ones_like(label), num_classes)
            ),
            "clean_core_micro": float(
                weighted_accuracy(
                    prediction,
                    label,
                    clean_core_mask.float(),
                )
            ),
            "clean_core_macro": float(
                weighted_macro_accuracy(prediction, label, clean_core_mask.float(), num_classes)
            ),
            "clean_core_samples": int(clean_core_mask.sum()),
            "predicted_class_count": num_classes_actual,
        }

    atomic_json_dump({"resolutions": list(resolutions), "results": results}, args.output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
