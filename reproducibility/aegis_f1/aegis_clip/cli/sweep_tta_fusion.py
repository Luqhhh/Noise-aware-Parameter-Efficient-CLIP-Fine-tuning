"""Sweep deterministic paired-view fusion without reading the test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import TrustBundle
from aegis_clip.evaluation import weighted_accuracy, weighted_macro_accuracy
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.multiprototype import blend_multiprototype_logits
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.tta import TTA_FUSION_MODES, fuse_paired_logits


def _parse_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated temperatures")
    return values


@torch.no_grad()
def _adapt(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    values = []
    model.eval()
    for start in range(0, features.shape[0], batch_size):
        values.append(
            model.adapt_features(features[start : start + batch_size].to(device))
            .float()
            .cpu()
        )
    return torch.cat(values)


def _metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    clean: torch.Tensor,
    proxy: torch.Tensor,
    proxy_weight: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    prediction = scores.argmax(1).cpu()
    return {
        "raw_micro": float((prediction == labels).float().mean()),
        "raw_macro": weighted_macro_accuracy(
            prediction, labels, torch.ones_like(clean), num_classes
        ),
        "trusted_micro": weighted_accuracy(prediction, labels, clean),
        "trusted_macro": weighted_macro_accuracy(
            prediction, labels, clean, num_classes
        ),
        "proxy_micro": weighted_accuracy(prediction, proxy, proxy_weight),
        "proxy_macro": weighted_macro_accuracy(
            prediction, proxy, proxy_weight, num_classes
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--augmented-feature-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperatures", type=_parse_floats, default=[0.5, 1.0, 2.0])
    parser.add_argument("--selection-metric", default="proxy_macro")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    config = load_config(args.config)
    model, _, checkpoint = build_from_checkpoint(args.checkpoint, device)
    original = FrozenFeatureStore(
        config["features"]["tensor_path"],
        config["features"]["paths_path"],
        config["features"].get("manifest_path"),
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    augmented_dir = Path(args.augmented_feature_dir)
    augmented = FrozenFeatureStore(
        augmented_dir / "features.pt",
        augmented_dir / "image_paths.json",
        augmented_dir / "manifest.json",
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    if original.paths != augmented.paths:
        raise ValueError("paired feature caches have different path indices")
    if augmented.manifest.get("augmentation") == "none":
        raise ValueError("second cache does not declare an augmentation")

    frame = pd.read_csv(config["data"]["val_csv"])
    paths = frame["image_path"].astype(str).tolist()
    labels = torch.as_tensor(frame["label"].astype(int).tolist(), dtype=torch.long)
    trust_bundle = TrustBundle(config["trust"]["bundle_path"])
    trust = [
        trust_bundle.values_for(path, int(label))
        for path, label in zip(paths, labels.tolist())
    ]
    clean = torch.stack([item["clean_probability"] for item in trust]).float()
    pseudo = torch.stack([item["pseudo_label"] for item in trust]).long()
    correction = torch.stack([item["correction_alpha"] for item in trust]).float()
    proxy = torch.where(correction > 0.0, pseudo, labels)
    proxy_weight = torch.maximum(clean, correction)
    first_features = _adapt(model, original.get_many(paths), device, args.batch_size)
    second_features = _adapt(model, augmented.get_many(paths), device, args.batch_size)
    weight = model.classifier.weight.detach().float().cpu()
    bias = model.classifier.bias.detach().float().cpu()
    first_logits = torch.nn.functional.linear(first_features, weight, bias)
    second_logits = torch.nn.functional.linear(second_features, weight, bias)
    multiprototype_head = checkpoint.get("multiprototype_head")
    if multiprototype_head is not None:
        first_logits = blend_multiprototype_logits(
            first_logits, first_features, multiprototype_head
        )
        second_logits = blend_multiprototype_logits(
            second_logits, second_features, multiprototype_head
        )
    num_classes = int(config["model"]["num_classes"])
    rows = []
    for mode in sorted(TTA_FUSION_MODES):
        temperatures = (
            args.temperatures
            if mode in {"mean_probabilities", "entropy_weighted_probabilities"}
            else [1.0]
        )
        for temperature in temperatures:
            scores = fuse_paired_logits(
                first_logits,
                second_logits,
                mode=mode,
                temperature=temperature,
            )
            rows.append(
                {
                    "mode": mode,
                    "temperature": float(temperature),
                    **_metrics(
                        scores,
                        labels,
                        clean,
                        proxy,
                        proxy_weight,
                        num_classes,
                    ),
                }
            )
    if args.selection_metric not in rows[0]:
        raise ValueError(f"unsupported selection metric: {args.selection_metric}")
    rows.sort(key=lambda row: float(row[args.selection_metric]), reverse=True)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "augmented_manifest_sha256": sha256_file(augmented_dir / "manifest.json"),
        "selection_metric": args.selection_metric,
        "prediction_head": (
            "linear_plus_multiprototype"
            if multiprototype_head is not None
            else "linear"
        ),
        "winner": rows[0],
        "rows": rows,
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
