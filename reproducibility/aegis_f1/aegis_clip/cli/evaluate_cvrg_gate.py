"""Evaluate and, when promoted, freeze a CVRG reliability gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aegis_clip.cvrg_crossfit import (
    CVRGFitConfig,
    GateFitMetadata,
    cross_fit_reliability,
    fit_frozen_gate,
    select_regularization_c,
)
from aegis_clip.localization import fuse_global_local_flip_probabilities
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.view_reliability import (
    atomic_torch_save,
    extract_reliability_features,
    frozen_gate_to_payload,
    validate_cvrg_cache,
)


def _accuracy(log_scores, labels):
    return float((log_scores.argmax(-1) == labels).float().mean())


def evaluate(validation_cache, output_dir, *, checkpoint_sha256=None, cache_sha256=None):
    payload = torch.load(validation_cache, map_location="cpu", weights_only=False)
    validate_cvrg_cache(payload, require_labels=True)
    features, names = extract_reliability_features(
        payload["view_logits"], payload["view_features"],
        payload["orientation_attention"], payload["crop_boxes"],
    )
    labels = payload["labels"].long()
    logits = payload["view_logits"].float()
    baseline = fuse_global_local_flip_probabilities(
        logits[:,0], logits[:,1], logits[:,2], logits[:,3],
        local_weight=0.4, flip_weight=0.5, temperature=1.0,
    )
    config = CVRGFitConfig()
    result = cross_fit_reliability(features, logits, labels, payload["paths"], config=config)
    weights = torch.softmax(torch.log(torch.tensor([.30,.20,.30,.20]))[None] + torch.logit(result.oof_reliability.clamp(1e-4, 1-1e-4)), 1)
    dynamic = (torch.softmax(logits, -1) * weights[...,None]).sum(1).clamp_min(1e-12).log()
    baseline_acc = _accuracy(baseline, labels)
    dynamic_acc = _accuracy(dynamic, labels)
    selected_c, inner_scores = select_regularization_c(
        features, logits, labels, payload["paths"], torch.arange(labels.numel()), config=config, seed_offset=9000,
    )
    metadata = GateFitMetadata(
        checkpoint_sha256=checkpoint_sha256 or str(payload["checkpoint_sha256"]),
        validation_cache_sha256=cache_sha256 or sha256_file(validation_cache),
    )
    gate = fit_frozen_gate(
        features, logits, labels, torch.arange(labels.numel()), c=selected_c,
        seed=config.seed, metadata=metadata, maximum_iterations=config.maximum_iterations,
    )
    promoted = dynamic_acc >= baseline_acc
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "promoted" if promoted else "closed_no_test_inference",
        "baseline_raw_micro": baseline_acc,
        "dynamic_oof_raw_micro": dynamic_acc,
        "delta": dynamic_acc - baseline_acc,
        "selected_c": selected_c,
        "inner_brier": inner_scores,
        "outer_folds": config.outer_folds,
        "test_data_used": False,
        "model_parameters_updated": False,
    }
    atomic_json_dump(report, output_dir / "promotion_gate.json")
    if promoted:
        atomic_torch_save(frozen_gate_to_payload(gate), output_dir / "final_gate.pt")
        atomic_json_dump({
            "format_version": 1, "promoted": True, "gate_sha256": sha256_file(output_dir / "final_gate.pt"),
            "checkpoint_sha256": gate.checkpoint_sha256, "validation_cache_sha256": gate.validation_cache_sha256,
            "feature_schema_sha256": gate.feature_schema_sha256, "protocol": gate.protocol.__dict__,
        }, output_dir / "final_gate_manifest.json")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--cache-sha256")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.validation_cache, args.output_dir, checkpoint_sha256=args.checkpoint_sha256, cache_sha256=args.cache_sha256), indent=2))


if __name__ == "__main__":
    main()
