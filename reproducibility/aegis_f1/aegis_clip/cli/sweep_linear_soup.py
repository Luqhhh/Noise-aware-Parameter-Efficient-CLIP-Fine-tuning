"""Sweep a strictly single-backbone interpolation of two linear heads."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import TrustBundle
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.multiprototype import paired_top1_changes
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.soup import assert_non_classifier_state_equal, interpolate_linear_heads
from aegis_clip.trust import atomic_torch_save
from aegis_clip.cli.sweep_structural_head import (
    _adapt_features,
    _metrics,
    _parse_floats,
    _split_tensors,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--first-checkpoint", required=True)
    parser.add_argument("--second-checkpoint", required=True)
    parser.add_argument("--augmented-feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--alphas", type=_parse_floats, default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--selection-metric", default="proxy_macro")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if any(not 0.0 <= alpha <= 1.0 for alpha in args.alphas):
        raise ValueError("all alphas must be in [0,1]")

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    config = load_config(args.config)
    model, _, first_checkpoint = build_from_checkpoint(
        args.first_checkpoint, device
    )
    second_checkpoint = torch.load(
        args.second_checkpoint, map_location="cpu", weights_only=False
    )
    if first_checkpoint.get("effective_model_spec") != second_checkpoint.get(
        "effective_model_spec"
    ):
        raise ValueError("checkpoints have different effective model specifications")
    first_state = first_checkpoint["model_state_dict"]
    second_state = second_checkpoint["model_state_dict"]
    assert_non_classifier_state_equal(first_state, second_state)

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
    first_split = _split_tensors(config["data"]["val_csv"], original, trust_bundle)
    second_split = _split_tensors(config["data"]["val_csv"], augmented, trust_bundle)
    first_features = _adapt_features(
        model, first_split["features"], device=device, batch_size=args.batch_size
    )
    second_features = _adapt_features(
        model, second_split["features"], device=device, batch_size=args.batch_size
    )
    num_classes = int(config["model"]["num_classes"])
    second_reference_logits = 0.5 * (
        F.linear(
            first_features,
            second_state["classifier.weight"].float().cpu(),
            second_state["classifier.bias"].float().cpu(),
        )
        + F.linear(
            second_features,
            second_state["classifier.weight"].float().cpu(),
            second_state["classifier.bias"].float().cpu(),
        )
    )
    rows = []
    candidates = {}
    for alpha in args.alphas:
        weight, bias = interpolate_linear_heads(
            first_state["classifier.weight"].float().cpu(),
            first_state["classifier.bias"].float().cpu(),
            second_state["classifier.weight"].float().cpu(),
            second_state["classifier.bias"].float().cpu(),
            alpha=float(alpha),
        )
        logits = 0.5 * (
            F.linear(first_features, weight, bias)
            + F.linear(second_features, weight, bias)
        )
        rows.append(
            {
                "alpha": float(alpha),
                **paired_top1_changes(
                    second_reference_logits, logits, first_split["labels"]
                ),
                **_metrics(logits, first_split, num_classes),
            }
        )
        candidates[float(alpha)] = (weight, bias)
    if args.selection_metric not in rows[0]:
        raise ValueError(f"unsupported selection metric: {args.selection_metric}")
    rows.sort(key=lambda row: float(row[args.selection_metric]), reverse=True)
    best = rows[0]
    best_weight, best_bias = candidates[float(best["alpha"])]
    winner = copy.deepcopy(first_checkpoint)
    winner["model_state_dict"]["classifier.weight"] = best_weight
    winner["model_state_dict"]["classifier.bias"] = best_bias
    winner["linear_soup"] = {
        "first_checkpoint": str(Path(args.first_checkpoint).resolve()),
        "first_checkpoint_sha256": sha256_file(args.first_checkpoint),
        "second_checkpoint": str(Path(args.second_checkpoint).resolve()),
        "second_checkpoint_sha256": sha256_file(args.second_checkpoint),
        "alpha": float(best["alpha"]),
        "selection_metric": args.selection_metric,
        "single_backbone_verified": True,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    atomic_torch_save(winner, checkpoint_path)
    report = {
        **winner["linear_soup"],
        "winner": best,
        "rows": rows,
        "output_checkpoint": str(checkpoint_path.resolve()),
        "output_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    atomic_json_dump(report, output_dir / "sweep.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
