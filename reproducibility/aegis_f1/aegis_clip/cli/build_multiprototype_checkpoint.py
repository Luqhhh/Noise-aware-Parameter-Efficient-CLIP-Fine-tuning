"""Embed a fixed, development-selected multi-prototype residual in one checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import TrustBundle
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.multiprototype import fit_weighted_multiprototypes
from aegis_clip.runtime import sha256_file
from aegis_clip.trust import atomic_torch_save


@torch.no_grad()
def _adapt(model: torch.nn.Module, features: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    values = []
    for start in range(0, features.shape[0], batch_size):
        values.append(model.adapt_features(features[start : start + batch_size]).float())
    return torch.cat(values).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--augmented-feature-dir", required=True)
    parser.add_argument("--fit-csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prototypes-per-class", type=int, required=True)
    parser.add_argument("--trust-power", type=float, required=True)
    parser.add_argument("--aggregation", choices=["max", "logmeanexp"], required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--candidate-scale", type=float, required=True)
    parser.add_argument("--softmax-temperature", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if args.alpha < 0.0 or args.candidate_scale <= 0.0:
        raise ValueError("alpha must be non-negative and candidate-scale must be positive")

    config = load_config(args.config)
    model, _, checkpoint = build_from_checkpoint(args.checkpoint, torch.device("cpu"))
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
    fit_csv = args.fit_csv or config["data"]["train_csv"]
    frame = pd.read_csv(fit_csv)
    paths = frame["image_path"].astype(str).tolist()
    labels = torch.as_tensor(frame["label"].astype(int).tolist(), dtype=torch.long)
    original.verify_coverage(paths)
    trust_bundle = TrustBundle(config["trust"]["bundle_path"])
    trust_bundle.verify_coverage(paths)
    clean = torch.stack(
        [
            trust_bundle.values_for(path, int(label))["clean_probability"]
            for path, label in zip(paths, labels.tolist())
        ]
    ).float()
    first = _adapt(model, original.get_many(paths), args.batch_size)
    second = _adapt(model, augmented.get_many(paths), args.batch_size)
    prototypes = fit_weighted_multiprototypes(
        F.normalize(0.5 * (first + second), dim=1),
        labels,
        clean.pow(args.trust_power),
        num_classes=int(config["model"]["num_classes"]),
        prototypes_per_class=args.prototypes_per_class,
        random_state=args.random_state,
    )
    winner = copy.deepcopy(checkpoint)
    winner["multiprototype_head"] = {
        "prototypes": prototypes,
        "prototypes_per_class": args.prototypes_per_class,
        "aggregation": args.aggregation,
        "softmax_temperature": args.softmax_temperature,
        "alpha": args.alpha,
        "candidate_scale": args.candidate_scale,
        "trust_power": args.trust_power,
        "fit_representation": "mean_original_horizontal_flip",
        "fit_csv": str(Path(fit_csv).resolve()),
        "fit_csv_sha256": sha256_file(fit_csv),
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "original_feature_manifest_sha256": sha256_file(
            config["features"]["manifest_path"]
        ),
        "augmented_feature_manifest_sha256": sha256_file(
            augmented_dir / "manifest.json"
        ),
        "single_checkpoint": True,
    }
    atomic_torch_save(winner, args.output)
    report = {
        key: value
        for key, value in winner["multiprototype_head"].items()
        if key != "prototypes"
    }
    report["output_checkpoint"] = str(Path(args.output).resolve())
    report["output_checkpoint_sha256"] = sha256_file(args.output)
    report["prototype_shape"] = list(prototypes.shape)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
