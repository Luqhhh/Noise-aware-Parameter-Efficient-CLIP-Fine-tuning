"""Create a validated submission from a promoted CVRG gate and test cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aegis_clip.config import load_config
from aegis_clip.runtime import sha256_file
from aegis_clip.submission import create_submission
from aegis_clip.view_reliability import (
    extract_reliability_features,
    fuse_dynamic_view_probabilities,
    load_frozen_gate,
    validate_cvrg_cache,
)


def infer(test_cache, gate_path, output_dir, checkpoint, config_path):
    payload = torch.load(test_cache, map_location="cpu", weights_only=False)
    validate_cvrg_cache(payload, require_labels=False)
    gate = load_frozen_gate(gate_path)
    features, _ = extract_reliability_features(
        payload["view_logits"], payload["view_features"],
        payload["orientation_attention"], payload["crop_boxes"],
    )
    scores, weights, reliability = fuse_dynamic_view_probabilities(payload["view_logits"], gate, features)
    config = load_config(config_path)
    mapping = config["data"]["class_mapping"]
    with Path(mapping).open("r", encoding="utf-8") as handle:
        class_to_idx = json.load(handle)
    idx_to_class = {int(index): name for name, index in class_to_idx.items()}
    predictions = [(name, str(idx_to_class[int(index)]).zfill(4)) for name, index in zip(payload["paths"], scores.argmax(-1).tolist())]
    return create_submission(
        predictions, payload["paths"], output_dir, checkpoint,
        inference_mode="cvrg_dynamic_four_view",
        tta_risk_acknowledged=True,
        valid_labels={str(name).zfill(4) for name in idx_to_class.values()},
        extra_manifest={
            "gate_sha256": sha256_file(gate_path), "test_cache_sha256": sha256_file(test_cache),
            "mean_view_weights": weights.mean(0).tolist(), "mean_reliability": reliability.mean(0).tolist(),
        }, overwrite=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(infer(args.test_cache, args.gate, args.output_dir, args.checkpoint, args.config), indent=2))


if __name__ == "__main__":
    main()
