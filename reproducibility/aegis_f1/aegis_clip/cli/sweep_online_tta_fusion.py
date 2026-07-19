"""Sweep paired-view fusion for visual PEFT models on validation images."""

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
from aegis_clip.multiprototype import blend_multiprototype_logits
from aegis_clip.runtime import atomic_json_dump, seed_worker, set_seed, sha256_file
from aegis_clip.tta import TTA_FUSION_MODES, fuse_paired_logits


def _parse_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated temperatures")
    return values


def _metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    clean: torch.Tensor,
    proxy: torch.Tensor,
    proxy_weight: torch.Tensor,
    *,
    clean_core_threshold: float,
    num_classes: int,
) -> dict[str, float]:
    prediction = scores.argmax(1)
    unit_weight = torch.ones_like(clean)
    clean_core_weight = (clean >= clean_core_threshold).float()
    return {
        "raw_micro": float((prediction == labels).float().mean()),
        "raw_macro": weighted_macro_accuracy(
            prediction, labels, unit_weight, num_classes
        ),
        "trusted_micro": weighted_accuracy(prediction, labels, clean),
        "trusted_macro": weighted_macro_accuracy(
            prediction, labels, clean, num_classes
        ),
        "proxy_micro": weighted_accuracy(prediction, proxy, proxy_weight),
        "proxy_macro": weighted_macro_accuracy(
            prediction, proxy, proxy_weight, num_classes
        ),
        "clean_core_micro": weighted_accuracy(
            prediction, labels, clean_core_weight
        ),
        "clean_core_macro": weighted_macro_accuracy(
            prediction, labels, clean_core_weight, num_classes
        ),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperatures", type=_parse_floats, default=[0.5, 1.0, 2.0])
    parser.add_argument("--selection-metric", default="clean_core_micro")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["project"].get("seed", 42)), deterministic=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model, preprocess, checkpoint = build_from_checkpoint(args.checkpoint, device)
    feature_config = config["features"]
    feature_store = FrozenFeatureStore(
        feature_config["tensor_path"],
        feature_config["paths_path"],
        feature_config.get("manifest_path"),
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    trust_bundle = TrustBundle(config["trust"]["bundle_path"])
    dataset = OnlineImageDataset(
        config["data"]["val_csv"],
        config["data"]["train_root"],
        preprocess,
        feature_store,
        trust_bundle,
    )
    workers = int(config["train"].get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"].get("batch_size", 256)),
        shuffle=False,
        num_workers=workers,
        timeout=int(config["train"].get("loader_timeout", 120 if workers else 0)),
        pin_memory=bool(config["train"].get("pin_memory", True)),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )
    multiprototype_head = checkpoint.get("multiprototype_head")
    if multiprototype_head is not None:
        multiprototype_head = dict(multiprototype_head)
        multiprototype_head["prototypes"] = multiprototype_head["prototypes"].to(
            device=device, dtype=torch.float32
        )
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    first_values: list[torch.Tensor] = []
    second_values: list[torch.Tensor] = []
    label_values: list[torch.Tensor] = []
    clean_values: list[torch.Tensor] = []
    proxy_values: list[torch.Tensor] = []
    proxy_weight_values: list[torch.Tensor] = []
    model.eval()
    for batch in tqdm(loader, desc="Paired validation inference"):
        images = batch["images"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            first, first_features = model(images=images, return_features=True)
            second, second_features = model(
                images=torch.flip(images, dims=(3,)), return_features=True
            )
            if multiprototype_head is not None:
                first = blend_multiprototype_logits(
                    first, first_features, multiprototype_head
                )
                second = blend_multiprototype_logits(
                    second, second_features, multiprototype_head
                )
        labels = batch["label"].long()
        clean = batch["clean_probability"].float()
        pseudo = batch["pseudo_label"].long()
        correction = batch["correction_alpha"].float()
        first_values.append(first.float().cpu())
        second_values.append(second.float().cpu())
        label_values.append(labels)
        clean_values.append(clean)
        proxy_values.append(torch.where(correction > 0.0, pseudo, labels))
        proxy_weight_values.append(torch.maximum(clean, correction))

    first = torch.cat(first_values)
    second = torch.cat(second_values)
    labels = torch.cat(label_values)
    clean = torch.cat(clean_values)
    proxy = torch.cat(proxy_values)
    proxy_weight = torch.cat(proxy_weight_values)
    clean_core_threshold = float(
        config["evaluation"].get("clean_core_threshold", 0.70)
    )
    metric_arguments = {
        "labels": labels,
        "clean": clean,
        "proxy": proxy,
        "proxy_weight": proxy_weight,
        "clean_core_threshold": clean_core_threshold,
        "num_classes": int(config["model"]["num_classes"]),
    }
    rows = [
        {
            "mode": "none",
            "temperature": 1.0,
            **_metrics(first, **metric_arguments),
        }
    ]
    for mode in sorted(TTA_FUSION_MODES):
        temperatures = (
            args.temperatures
            if mode in {"mean_probabilities", "entropy_weighted_probabilities"}
            else [1.0]
        )
        for temperature in temperatures:
            scores = fuse_paired_logits(
                first,
                second,
                mode=mode,
                temperature=temperature,
            )
            rows.append(
                {
                    "mode": mode,
                    "temperature": float(temperature),
                    **_metrics(scores, **metric_arguments),
                }
            )
    if args.selection_metric not in rows[0]:
        raise ValueError(f"unsupported selection metric: {args.selection_metric}")
    tie_breakers = [
        metric
        for metric in (
            args.selection_metric,
            "clean_core_macro",
            "trusted_macro",
            "raw_macro",
        )
        if metric in rows[0]
    ]
    tie_breakers = list(dict.fromkeys(tie_breakers))
    rows.sort(
        key=lambda row: tuple(float(row[metric]) for metric in tie_breakers),
        reverse=True,
    )
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "selection_metric": args.selection_metric,
        "tie_breakers": tie_breakers,
        "clean_core_threshold": clean_core_threshold,
        "clean_core_samples": int((clean >= clean_core_threshold).sum()),
        "flip_prediction_agreement": float(
            (first.argmax(1) == second.argmax(1)).float().mean()
        ),
        "winner": rows[0],
        "rows": rows,
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
