"""Measure every flip-TTA fusion rule on a labelled split, in one GPU pass.

Purpose: put a *number* on the inference-side ceiling. The rematch strategy
record already decided qualitatively that "推理侧不是答案" (the inference side
is not the answer); this tool quantifies how large the best available
inference-side move actually is on this checkpoint.

Why one pass instead of calling ``cli/evaluate.py`` once per rule: each call
re-decodes the whole split and re-runs both forward passes. Here the paired
(original, flipped) logits are produced once and cached in memory, then every
rule -- and every temperature -- is fused offline.

The metric functions are imported from ``aegis_clip.evaluation`` rather than
reimplemented, so the numbers stay on the same definition as the run's own
``best_evaluation.json``.  ``fusion="none"`` must reproduce that file; the
comparison is asserted before any TTA number is trusted.

This deliberately does NOT write a submission. ``cli/infer.py`` refuses TTA
under a rematch config ("Rematch first-round inference is global only"), and
that guard is a stage policy, not an oversight -- see
``docs/rematch750_inference_side_probe_20260922.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import OnlineImageDataset, TrustBundle
from aegis_clip.evaluation import (
    longtail_segment_metrics,
    support_metrics,
    weighted_accuracy,
    weighted_macro_accuracy,
)
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.multiprototype import blend_multiprototype_logits
from aegis_clip.runtime import seed_worker
from aegis_clip.tta import TTA_FUSION_MODES, fuse_paired_logits


def collect_paired_logits(model, loader, device, use_amp, multiprototype_head):
    """One pass over the split, keeping both views' logits."""
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {
        key: [] for key in
        ("original", "flipped", "labels", "clean", "pseudo", "correction")
    }
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, enabled=use_amp and device.type == "cuda"
            ):
                logits, encoded = model(images=images, return_features=True)
                if multiprototype_head is not None:
                    logits = blend_multiprototype_logits(
                        logits, encoded, multiprototype_head
                    )
                flipped_logits, flipped_encoded = model(
                    images=torch.flip(images, dims=(3,)), return_features=True
                )
                if multiprototype_head is not None:
                    flipped_logits = blend_multiprototype_logits(
                        flipped_logits, flipped_encoded, multiprototype_head
                    )
            collected["original"].append(logits.float().cpu())
            collected["flipped"].append(flipped_logits.float().cpu())
            collected["labels"].append(batch["label"].cpu())
            collected["clean"].append(batch["clean_probability"].float().cpu())
            collected["pseudo"].append(batch["pseudo_label"].long().cpu())
            collected["correction"].append(batch["correction_alpha"].float().cpu())
    return {key: torch.cat(value) for key, value in collected.items()}


def fuse(first: torch.Tensor, second: torch.Tensor, mode: str, temperature: float,
         chunk: int = 2048) -> torch.Tensor:
    if mode == "none":
        return first
    return torch.cat(
        [
            fuse_paired_logits(
                first[start : start + chunk],
                second[start : start + chunk],
                mode=mode,
                temperature=temperature,
            )
            for start in range(0, first.shape[0], chunk)
        ]
    )


def score(fused, cached, class_counts, num_classes, clean_core_threshold):
    prediction = fused.argmax(dim=1)
    noisy = cached["labels"]
    clean_weight = cached["clean"]
    proxy = torch.where(cached["correction"] > 0.0, cached["pseudo"], noisy)
    proxy_weight = torch.maximum(clean_weight, cached["correction"])
    ones = torch.ones_like(clean_weight)

    metrics = {
        "raw_micro": float((prediction == noisy).float().mean()),
        "raw_macro": weighted_macro_accuracy(prediction, noisy, ones, num_classes),
        "trusted_micro": weighted_accuracy(prediction, noisy, clean_weight),
        "trusted_macro": weighted_macro_accuracy(
            prediction, noisy, clean_weight, num_classes
        ),
        "proxy_micro": weighted_accuracy(prediction, proxy, proxy_weight),
        "proxy_macro": weighted_macro_accuracy(
            prediction, proxy, proxy_weight, num_classes
        ),
        "predicted_class_count": int(prediction.unique().numel()),
    }
    core_weight = (clean_weight >= float(clean_core_threshold)).float()
    metrics["clean_core_micro"] = weighted_accuracy(prediction, noisy, core_weight)
    metrics["clean_core_macro"] = weighted_macro_accuracy(
        prediction, noisy, core_weight, num_classes
    )
    metrics.update(support_metrics(prediction, noisy, class_counts))
    metrics.update(
        longtail_segment_metrics(prediction, noisy, class_counts, weight=core_weight)
    )
    return metrics


def train_class_counts(train_csv: str | Path, num_classes: int) -> torch.Tensor:
    """support_metrics() reports these as train_samples, so they come from train."""
    with Path(train_csv).open(encoding="utf-8") as handle:
        labels = [int(row["label"]) for row in csv.DictReader(handle)]
    return torch.bincount(torch.as_tensor(labels, dtype=torch.long),
                          minlength=num_classes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperatures", type=float, nargs="*",
                        default=[0.5, 1.0, 2.0])
    parser.add_argument("--anchor-macro", type=float, default=None)
    parser.add_argument("--anchor-micro", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    config = load_config(args.config)
    model, preprocess, checkpoint = build_from_checkpoint(
        args.checkpoint, device, config_override=config
    )
    multiprototype_head = checkpoint.get("multiprototype_head")
    if multiprototype_head is not None:
        multiprototype_head = dict(multiprototype_head)
        multiprototype_head["prototypes"] = multiprototype_head["prototypes"].to(
            device=device, dtype=torch.float32
        )

    evaluation_config = config.get("evaluation", {})
    feature_config = config["features"]
    feature_store = FrozenFeatureStore(
        feature_config["tensor_path"],
        feature_config["paths_path"],
        feature_config.get("manifest_path"),
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    trust_bundle = (
        TrustBundle(config["trust"]["bundle_path"])
        if config.get("trust", {}).get("enabled", False)
        else None
    )
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
        batch_size=int(evaluation_config.get("batch_size", 128)),
        shuffle=False,
        num_workers=workers,
        timeout=int(config["train"].get("loader_timeout", 120 if workers else 0)),
        pin_memory=bool(config["train"].get("pin_memory", True)),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )

    num_classes = int(config["model"]["num_classes"])
    class_counts = train_class_counts(config["data"]["train_csv"], num_classes)
    use_amp = bool(config["train"].get("amp", True))
    clean_core_threshold = float(evaluation_config.get("clean_core_threshold", 0.70))

    cached = collect_paired_logits(model, loader, device, use_amp,
                                   multiprototype_head)
    agreement = float(
        (cached["original"].argmax(1) == cached["flipped"].argmax(1)).float().mean()
    )
    print(f"flip top-1 agreement = {agreement:.6f}", flush=True)

    variants = [("none", 1.0)]
    for mode in sorted(TTA_FUSION_MODES):
        if mode in {"mean_probabilities", "entropy_weighted_probabilities"}:
            variants.extend((mode, value) for value in args.temperatures)
        else:
            variants.append((mode, 1.0))

    results = []
    for mode, temperature in variants:
        metrics = score(
            fuse(cached["original"], cached["flipped"], mode, temperature),
            cached, class_counts, num_classes, clean_core_threshold,
        )
        metrics["tta_fusion"] = mode
        metrics["tta_temperature"] = temperature
        metrics["flip_prediction_agreement"] = agreement
        results.append(metrics)
        print(
            f"  {mode:34s} T={temperature:<4} "
            f"raw_macro={metrics['raw_macro'] * 100:.4f}%  "
            f"raw_micro={metrics['raw_micro'] * 100:.4f}%  "
            f"eqthirds_tail_macro={metrics['tail_macro'] * 100:.4f}%",
            flush=True,
        )

    control = results[0]
    for key, expected in (("raw_macro", args.anchor_macro),
                          ("raw_micro", args.anchor_micro)):
        if expected is None:
            continue
        if abs(control[key] - expected) > 1e-6:
            raise SystemExit(
                f"ANCHOR FAILED: {key} recomputed {control[key]!r} != recorded "
                f"{expected!r}; this path does not reproduce the run, so no TTA "
                "number is trustworthy"
            )
    print("anchor OK: fusion=none reproduces the recorded checkpoint metrics")

    Path(args.output).write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "config": str(args.config),
                "samples": int(cached["labels"].numel()),
                "flip_top1_agreement": agreement,
                "num_classes": num_classes,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
