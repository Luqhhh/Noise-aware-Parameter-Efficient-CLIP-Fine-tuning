"""Diagnose whether a train-only visual memory beats a frozen linear head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from aegis_clip.cache_diagnostic import (
    complementarity_metrics,
    prediction_metrics,
    topk_cache_predictions,
)
from aegis_clip.features import FrozenFeatureStore, canonical_sample_path
from aegis_clip.runtime import atomic_json_dump, sha256_file


def _frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)[["image_path", "label"]].copy()
    frame["canonical_path"] = frame["image_path"].map(canonical_sample_path)
    if frame["canonical_path"].duplicated().any():
        raise ValueError(f"Duplicate paths in {path}")
    return frame.set_index("canonical_path", drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--evaluation-csv", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature-paths", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trust-bundle", required=True)
    parser.add_argument("--rejected-paths", required=True)
    parser.add_argument("--content-groups", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--clean-threshold", type=float, default=0.70)
    parser.add_argument("--query-batch-size", type=int, default=256)
    args = parser.parse_args()

    train = _frame(args.train_csv)
    evaluation = _frame(args.evaluation_csv)
    rejected = {
        canonical_sample_path(line)
        for line in Path(args.rejected_paths).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not rejected <= set(train.index):
        raise ValueError("Rejected paths are not a subset of train")
    train = train.drop(index=sorted(rejected))

    with Path(args.content_groups).open("r", encoding="utf-8") as handle:
        raw_groups = json.load(handle)
    groups = {canonical_sample_path(path): str(value) for path, value in raw_groups.items()}
    missing_groups = (set(train.index) | set(evaluation.index)) - set(groups)
    if missing_groups:
        raise ValueError(f"Content groups miss {len(missing_groups)} samples")
    train_groups = {groups[path] for path in train.index}
    conflicted_evaluation = {
        path for path in evaluation.index if groups[path] in train_groups
    }
    evaluation = evaluation.drop(index=sorted(conflicted_evaluation))
    if set(train.index) & set(evaluation.index):
        raise ValueError("Train and evaluation paths overlap")
    if {groups[path] for path in train.index} & {groups[path] for path in evaluation.index}:
        raise ValueError("Train and evaluation content groups overlap after filtering")

    store = FrozenFeatureStore(
        args.features, args.feature_paths, args.feature_manifest
    )
    train_features = store.get_many(train.index)
    evaluation_features = store.get_many(evaluation.index)
    train_labels = torch.tensor(train["label"].to_numpy(), dtype=torch.long)
    evaluation_labels = torch.tensor(evaluation["label"].to_numpy(), dtype=torch.long)

    trust = torch.load(args.trust_bundle, map_location="cpu", weights_only=False)
    trust_index = {
        canonical_sample_path(path): index for index, path in enumerate(trust["paths"])
    }
    missing_trust = (set(train.index) | set(evaluation.index)) - set(trust_index)
    if missing_trust:
        raise ValueError(f"Trust bundle misses {len(missing_trust)} samples")
    clean = torch.as_tensor(trust["clean_probability"], dtype=torch.float32)
    train_clean = clean[
        torch.tensor([trust_index[path] for path in train.index], dtype=torch.long)
    ]
    evaluation_clean = clean[
        torch.tensor([trust_index[path] for path in evaluation.index], dtype=torch.long)
    ]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    weight = state["classifier.weight"].float().cpu()
    bias = state["classifier.bias"].float().cpu()
    baseline_logits = F.normalize(evaluation_features.float(), dim=1) @ weight.T + bias
    baseline = baseline_logits.argmax(dim=1)

    variants: dict[str, torch.Tensor] = {
        "a2_kept": torch.ones(len(train), dtype=torch.bool),
        "oof_clean": train_clean >= float(args.clean_threshold),
    }
    results: dict[str, object] = {
        "protocol": {
            "k": int(args.k),
            "beta": float(args.beta),
            "clean_threshold": float(args.clean_threshold),
            "device": str(args.device),
            "test_data_used": False,
            "external_data_used": False,
            "parameter_sweep": False,
        },
        "counts": {
            "a2_train_original": len(train) + len(rejected),
            "a2_rejected": len(rejected),
            "a2_kept": len(train),
            "evaluation_after_content_filter": len(evaluation),
            "evaluation_content_conflicts_removed": len(conflicted_evaluation),
        },
        "baseline": prediction_metrics(
            baseline,
            evaluation_labels,
            evaluation_clean,
            num_classes=500,
            clean_threshold=args.clean_threshold,
        ),
        "variants": {},
        "lineage": {
            "train_csv_sha256": sha256_file(args.train_csv),
            "evaluation_csv_sha256": sha256_file(args.evaluation_csv),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "trust_bundle_sha256": sha256_file(args.trust_bundle),
            "rejected_paths_sha256": sha256_file(args.rejected_paths),
            "content_groups_sha256": sha256_file(args.content_groups),
            "features_sha256": sha256_file(args.features),
        },
    }
    saved = {"paths": list(evaluation.index), "target": evaluation_labels, "clean": evaluation_clean, "baseline": baseline}
    for name, mask in variants.items():
        candidate, margin = topk_cache_predictions(
            evaluation_features,
            train_features[mask],
            train_labels[mask],
            num_classes=500,
            k=args.k,
            beta=args.beta,
            query_batch_size=args.query_batch_size,
            device=args.device,
        )
        results["variants"][name] = {
            "bank_samples": int(mask.sum()),
            "metrics": prediction_metrics(
                candidate,
                evaluation_labels,
                evaluation_clean,
                num_classes=500,
                clean_threshold=args.clean_threshold,
            ),
            "complementarity": complementarity_metrics(
                baseline,
                candidate,
                evaluation_labels,
                evaluation_clean,
                clean_threshold=args.clean_threshold,
            ),
            "mean_margin": float(margin.mean()),
        }
        saved[f"{name}_prediction"] = candidate
        saved[f"{name}_margin"] = margin

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(results, output / "diagnostic.json")
    torch.save(saved, output / "predictions.pt")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
