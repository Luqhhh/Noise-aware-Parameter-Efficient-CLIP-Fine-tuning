"""Sweep robust structural heads and emit a single-checkpoint winner."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import TrustBundle
from aegis_clip.evaluation import weighted_accuracy, weighted_macro_accuracy
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.structural import (
    blend_linear_heads,
    discriminant_from_statistics,
    match_linear_logit_scale,
    ridge_head_from_statistics,
    weighted_class_statistics,
    weighted_ridge_statistics,
)
from aegis_clip.trust import atomic_torch_save


@torch.no_grad()
def _adapt_features(
    model: torch.nn.Module,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    output = []
    model.eval()
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size].to(device)
        output.append(model.adapt_features(batch).float().cpu())
    return torch.cat(output)


def _split_tensors(
    split_csv: str,
    feature_store: FrozenFeatureStore,
    trust_bundle: TrustBundle,
) -> dict[str, torch.Tensor]:
    frame = pd.read_csv(split_csv)
    paths = frame["image_path"].astype(str).tolist()
    labels = torch.as_tensor(frame["label"].astype(int).tolist(), dtype=torch.long)
    feature_store.verify_coverage(paths)
    trust_bundle.verify_coverage(paths)
    trust = [
        trust_bundle.values_for(path, int(label))
        for path, label in zip(paths, labels.tolist())
    ]
    clean = torch.stack([item["clean_probability"] for item in trust]).float()
    pseudo = torch.stack([item["pseudo_label"] for item in trust]).long()
    correction = torch.stack([item["correction_alpha"] for item in trust]).float()
    proxy = torch.where(correction > 0.0, pseudo, labels)
    return {
        "features": feature_store.get_many(paths).float(),
        "labels": labels,
        "clean": clean,
        "pseudo": pseudo,
        "correction": correction,
        "proxy": proxy,
        "proxy_weight": torch.maximum(clean, correction),
    }


def _metrics(
    logits: torch.Tensor,
    split: dict[str, torch.Tensor],
    num_classes: int,
) -> dict[str, float]:
    prediction = logits.argmax(dim=1).cpu()
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


def _parse_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trust-powers", type=_parse_floats, default=[0.0, 1.0, 2.0])
    parser.add_argument("--method", choices=["lda", "ridge"], default="lda")
    parser.add_argument(
        "--shrinkages", type=_parse_floats, default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--alphas", type=_parse_floats, default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--ridge-strengths",
        type=_parse_floats,
        default=[0.01, 0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument(
        "--ridge-targets", choices=["noisy", "corrected"], default="noisy"
    )
    parser.add_argument("--selection-metric", default="proxy_macro")
    parser.add_argument(
        "--augmented-feature-dir",
        help=(
            "Optional audited feature cache; when set, heads are fitted and "
            "evaluated on the mean of original and augmented adapted features"
        ),
    )
    parser.add_argument(
        "--fixed-candidate",
        action="store_true",
        help=(
            "Require exactly one trust power, method parameter, and alpha and emit that "
            "pre-registered candidate without selecting on the evaluation split"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    valid_metrics = {
        "raw_micro",
        "raw_macro",
        "trusted_micro",
        "trusted_macro",
        "proxy_micro",
        "proxy_macro",
    }
    if args.selection_metric not in valid_metrics:
        raise ValueError(f"selection metric must be one of {sorted(valid_metrics)}")
    method_parameters = (
        args.shrinkages if args.method == "lda" else args.ridge_strengths
    )
    if args.fixed_candidate and not (
        len(args.trust_powers) == len(method_parameters) == len(args.alphas) == 1
    ):
        raise ValueError(
            "--fixed-candidate requires one trust power, one method parameter, and one alpha"
        )
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    config = load_config(args.config)
    model, _, checkpoint = build_from_checkpoint(args.checkpoint, device)
    if getattr(model, "classifier_mode", "linear") != "linear":
        raise ValueError("structural sweep currently requires a linear classifier")
    feature_store = FrozenFeatureStore(
        config["features"]["tensor_path"],
        config["features"]["paths_path"],
        config["features"].get("manifest_path"),
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    augmented_store = None
    if args.augmented_feature_dir:
        augmented_dir = Path(args.augmented_feature_dir)
        augmented_store = FrozenFeatureStore(
            augmented_dir / "features.pt",
            augmented_dir / "image_paths.json",
            augmented_dir / "manifest.json",
            expected_dim=int(config["model"].get("feature_dim", 512)),
        )
        if augmented_store.paths != feature_store.paths:
            raise ValueError("original and augmented feature caches have different paths")
        if augmented_store.manifest.get("augmentation", "none") == "none":
            raise ValueError("augmented feature cache does not declare an augmentation")
    trust_bundle = TrustBundle(config["trust"]["bundle_path"])
    fit_split = _split_tensors(config["data"]["train_csv"], feature_store, trust_bundle)
    eval_split = _split_tensors(config["data"]["val_csv"], feature_store, trust_bundle)
    if augmented_store is not None:
        fit_split["augmented_features"] = augmented_store.get_many(
            pd.read_csv(config["data"]["train_csv"])["image_path"].astype(str).tolist()
        ).float()
        eval_split["augmented_features"] = augmented_store.get_many(
            pd.read_csv(config["data"]["val_csv"])["image_path"].astype(str).tolist()
        ).float()
    fit_features = _adapt_features(
        model, fit_split["features"], device=device, batch_size=args.batch_size
    )
    eval_features = _adapt_features(
        model, eval_split["features"], device=device, batch_size=args.batch_size
    )
    representation = "original"
    if augmented_store is not None:
        fit_augmented = _adapt_features(
            model,
            fit_split["augmented_features"],
            device=device,
            batch_size=args.batch_size,
        )
        eval_augmented = _adapt_features(
            model,
            eval_split["augmented_features"],
            device=device,
            batch_size=args.batch_size,
        )
        fit_features = (fit_features + fit_augmented) / 2.0
        eval_features = (eval_features + eval_augmented) / 2.0
        representation = "mean_original_augmented"
    base_weight = model.classifier.weight.detach().float().cpu()
    base_bias = model.classifier.bias.detach().float().cpu()
    num_classes = int(config["model"]["num_classes"])
    rows: list[dict[str, Any]] = []
    base_metrics = _metrics(
        F.linear(eval_features, base_weight, base_bias), eval_split, num_classes
    )
    rows.append(
        {
            "kind": "base",
            "method": None,
            "trust_power": None,
            "shrinkage": None,
            "ridge_strength": None,
            "ridge_targets": None,
            "alpha": 0.0,
            "candidate_scale": 0.0,
            **base_metrics,
        }
    )
    candidate_parameters: dict[tuple[float, float], tuple[torch.Tensor, torch.Tensor, float]] = {}
    fit_device = device
    fit_features_device = fit_features.to(fit_device)
    fit_labels_device = fit_split["labels"].to(fit_device)
    for trust_power in args.trust_powers:
        sample_weights = fit_split["clean"].pow(float(trust_power)).to(fit_device)
        if args.method == "lda":
            statistics = weighted_class_statistics(
                fit_features_device,
                fit_labels_device,
                sample_weights,
                num_classes=num_classes,
                covariance_batch_size=args.batch_size,
            )
            parameter_values = args.shrinkages
        else:
            statistics = weighted_ridge_statistics(
                fit_features_device,
                fit_labels_device,
                sample_weights,
                num_classes=num_classes,
                pseudo_labels=(
                    fit_split["pseudo"].to(fit_device)
                    if args.ridge_targets == "corrected"
                    else None
                ),
                correction_alpha=(
                    fit_split["correction"].to(fit_device)
                    if args.ridge_targets == "corrected"
                    else None
                ),
            )
            parameter_values = args.ridge_strengths
        for method_parameter in parameter_values:
            if args.method == "lda":
                structural_weight, structural_bias = discriminant_from_statistics(
                    *statistics, shrinkage=float(method_parameter)
                )
            else:
                structural_weight, structural_bias = ridge_head_from_statistics(
                    *statistics, ridge_strength=float(method_parameter)
                )
            structural_weight = structural_weight.cpu()
            structural_bias = structural_bias.cpu()
            scale = match_linear_logit_scale(
                fit_features,
                base_weight,
                base_bias,
                structural_weight,
                structural_bias,
            )
            candidate_parameters[(float(trust_power), float(method_parameter))] = (
                structural_weight,
                structural_bias,
                scale,
            )
            for alpha in args.alphas:
                weight, bias = blend_linear_heads(
                    base_weight,
                    base_bias,
                    structural_weight,
                    structural_bias,
                    alpha=float(alpha),
                    candidate_scale=scale,
                )
                values = _metrics(
                    F.linear(eval_features, weight, bias), eval_split, num_classes
                )
                rows.append(
                    {
                        "kind": "structural_blend",
                        "method": args.method,
                        "trust_power": float(trust_power),
                        "shrinkage": (
                            float(method_parameter) if args.method == "lda" else None
                        ),
                        "ridge_strength": (
                            float(method_parameter) if args.method == "ridge" else None
                        ),
                        "ridge_targets": (
                            args.ridge_targets if args.method == "ridge" else None
                        ),
                        "alpha": float(alpha),
                        "candidate_scale": scale,
                        **values,
                    }
                )

    if args.fixed_candidate:
        best = next(row for row in rows if row["kind"] == "structural_blend")
        selection_mode = "fixed_from_development"
    else:
        best = max(rows, key=lambda row: float(row[args.selection_metric]))
        selection_mode = "evaluation_metric_sweep"
    if best["kind"] == "base":
        best_weight, best_bias = base_weight, base_bias
    else:
        method_parameter = (
            float(best["shrinkage"])
            if args.method == "lda"
            else float(best["ridge_strength"])
        )
        structural_weight, structural_bias, scale = candidate_parameters[
            (float(best["trust_power"]), method_parameter)
        ]
        best_weight, best_bias = blend_linear_heads(
            base_weight,
            base_bias,
            structural_weight,
            structural_bias,
            alpha=float(best["alpha"]),
            candidate_scale=scale,
        )
    winner = copy.deepcopy(checkpoint)
    winner["model_state_dict"]["classifier.weight"] = best_weight
    winner["model_state_dict"]["classifier.bias"] = best_bias
    winner["structural_head_fit"] = {
        "method": args.method,
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_csv": config["data"]["train_csv"],
        "fit_csv_sha256": sha256_file(config["data"]["train_csv"]),
        "eval_csv": config["data"]["val_csv"],
        "eval_csv_sha256": sha256_file(config["data"]["val_csv"]),
        "trust_bundle": config["trust"]["bundle_path"],
        "trust_bundle_sha256": sha256_file(config["trust"]["bundle_path"]),
        "selection_metric": args.selection_metric,
        "selection_mode": selection_mode,
        "representation": representation,
        "augmented_feature_manifest_sha256": (
            sha256_file(Path(args.augmented_feature_dir) / "manifest.json")
            if args.augmented_feature_dir
            else None
        ),
        "winner": best,
        "single_linear_head": True,
    }
    checkpoint_path = output_dir / "best.pt"
    atomic_torch_save(winner, checkpoint_path)
    report = {
        "method": args.method,
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "output_checkpoint": str(checkpoint_path.resolve()),
        "output_checkpoint_sha256": sha256_file(checkpoint_path),
        "selection_metric": args.selection_metric,
        "selection_mode": selection_mode,
        "representation": representation,
        "base": rows[0],
        "winner": best,
        "rows": sorted(
            rows, key=lambda row: float(row[args.selection_metric]), reverse=True
        ),
    }
    atomic_json_dump(report, output_dir / "sweep.json")
    print(json.dumps({key: report[key] for key in ("base", "winner", "output_checkpoint", "output_checkpoint_sha256")}, indent=2))


if __name__ == "__main__":
    main()
