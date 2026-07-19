"""Screen multi-modal visual prototypes on the untouched development split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import TrustBundle
from aegis_clip.evaluation import weighted_accuracy, weighted_macro_accuracy
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.multiprototype import (
    fit_weighted_multiprototypes,
    match_score_scale,
    multiprototype_logits,
    paired_top1_changes,
)
from aegis_clip.runtime import atomic_json_dump, sha256_file


def _parse_numbers(value: str, cast: type) -> list:
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return values


@torch.no_grad()
def _adapt(model: torch.nn.Module, features: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    values = []
    for start in range(0, features.shape[0], batch_size):
        values.append(model.adapt_features(features[start : start + batch_size]).float())
    return torch.cat(values).cpu()


def _split(frame: pd.DataFrame, trust_bundle: TrustBundle) -> dict[str, torch.Tensor | list[str]]:
    paths = frame["image_path"].astype(str).tolist()
    labels = torch.as_tensor(frame["label"].astype(int).tolist(), dtype=torch.long)
    trust = [
        trust_bundle.values_for(path, int(label))
        for path, label in zip(paths, labels.tolist())
    ]
    clean = torch.stack([item["clean_probability"] for item in trust]).float()
    pseudo = torch.stack([item["pseudo_label"] for item in trust]).long()
    correction = torch.stack([item["correction_alpha"] for item in trust]).float()
    return {
        "paths": paths,
        "labels": labels,
        "clean": clean,
        "proxy": torch.where(correction > 0.0, pseudo, labels),
        "proxy_weight": torch.maximum(clean, correction),
    }


def _metrics(scores: torch.Tensor, split: dict, num_classes: int) -> dict[str, float]:
    prediction = scores.argmax(1).cpu()
    labels = split["labels"]
    clean = split["clean"]
    proxy = split["proxy"]
    proxy_weight = split["proxy_weight"]
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
    parser.add_argument(
        "--prototypes-per-class",
        type=lambda value: _parse_numbers(value, int),
        default=[1, 2, 4],
    )
    parser.add_argument(
        "--trust-powers",
        type=lambda value: _parse_numbers(value, float),
        default=[0.0, 1.0, 2.0],
    )
    parser.add_argument(
        "--alphas",
        type=lambda value: _parse_numbers(value, float),
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--selection-metric", default="proxy_macro")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    model, _, _ = build_from_checkpoint(args.checkpoint, torch.device("cpu"))
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

    trust_bundle = TrustBundle(config["trust"]["bundle_path"])
    train_split = _split(pd.read_csv(config["data"]["train_csv"]), trust_bundle)
    eval_split = _split(pd.read_csv(config["data"]["val_csv"]), trust_bundle)
    train_first = _adapt(
        model, original.get_many(train_split["paths"]), args.batch_size
    )
    train_second = _adapt(
        model, augmented.get_many(train_split["paths"]), args.batch_size
    )
    eval_first = _adapt(model, original.get_many(eval_split["paths"]), args.batch_size)
    eval_second = _adapt(
        model, augmented.get_many(eval_split["paths"]), args.batch_size
    )
    weight = model.classifier.weight.detach().float().cpu()
    bias = model.classifier.bias.detach().float().cpu()
    base_scores = 0.5 * (
        F.linear(eval_first, weight, bias) + F.linear(eval_second, weight, bias)
    )
    num_classes = int(config["model"]["num_classes"])
    rows = [
        {
            "kind": "base",
            "prototypes_per_class": None,
            "trust_power": None,
            "aggregation": None,
            "alpha": 0.0,
            "candidate_scale": 0.0,
            "changed_predictions": 0,
            "raw_fixed": 0,
            "raw_broken": 0,
            "raw_net_fixed": 0,
            **_metrics(base_scores, eval_split, num_classes),
        }
    ]
    fit_features = F.normalize(0.5 * (train_first + train_second), dim=1)
    for trust_power in args.trust_powers:
        sample_weights = train_split["clean"].pow(float(trust_power))
        for prototypes_per_class in args.prototypes_per_class:
            prototypes = fit_weighted_multiprototypes(
                fit_features,
                train_split["labels"],
                sample_weights,
                num_classes=num_classes,
                prototypes_per_class=int(prototypes_per_class),
                random_state=args.random_state,
            )
            for aggregation in ("max", "logmeanexp"):
                candidate_scores = 0.5 * (
                    multiprototype_logits(
                        eval_first, prototypes, aggregation=aggregation
                    )
                    + multiprototype_logits(
                        eval_second, prototypes, aggregation=aggregation
                    )
                )
                scale = match_score_scale(base_scores, candidate_scores)
                for alpha in args.alphas:
                    scores = base_scores + float(alpha) * scale * candidate_scores
                    rows.append(
                        {
                            "kind": "multiprototype_blend",
                            "prototypes_per_class": int(prototypes_per_class),
                            "trust_power": float(trust_power),
                            "aggregation": aggregation,
                            "alpha": float(alpha),
                            "candidate_scale": scale,
                            **paired_top1_changes(
                                base_scores, scores, eval_split["labels"]
                            ),
                            **_metrics(scores, eval_split, num_classes),
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
        "base": next(row for row in rows if row["kind"] == "base"),
        "winner": rows[0],
        "rows": rows,
    }
    atomic_json_dump(report, args.output)
    print(json.dumps({key: report[key] for key in ("base", "winner")}, indent=2))


if __name__ == "__main__":
    main()
