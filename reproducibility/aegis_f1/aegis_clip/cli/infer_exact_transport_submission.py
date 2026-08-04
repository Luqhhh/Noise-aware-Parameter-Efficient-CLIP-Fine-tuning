"""Build an exact near-uniform submission from an audited fused-logits dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aegis_clip.balanced_transport import (
    exact_near_uniform_transport_prediction,
)
from aegis_clip.config import load_config
from aegis_clip.data import IMAGE_EXTENSIONS, load_class_mapping
from aegis_clip.runtime import sha256_file
from aegis_clip.submission import create_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--acknowledge-balanced-test-prior", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_balanced_test_prior:
        raise ValueError(
            "Exact near-uniform allocation uses the declared test-set prior; "
            "pass --acknowledge-balanced-test-prior explicitly"
        )

    dump_path = Path(args.dump).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not dump_path.is_file():
        raise FileNotFoundError(f"Fused-logits dump is missing: {dump_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint_path}")
    payload = torch.load(dump_path, map_location="cpu", weights_only=False)
    if not {"logits", "names", "inference_mode"}.issubset(payload):
        raise ValueError("Fused-logits dump misses required fields")
    logits = torch.as_tensor(payload["logits"]).float().cpu()
    names = [str(value) for value in payload["names"]]
    if logits.ndim != 2 or logits.shape[0] != len(names):
        raise ValueError("Dumped logits must be [N,C] matching the name count")
    if not torch.isfinite(logits).all():
        raise ValueError("Dumped logits contain NaN or Inf")

    config = load_config(args.config)
    num_classes = int(config["model"]["num_classes"])
    expected_samples = int(config["data"]["expected_test_samples"])
    if logits.shape != (expected_samples, num_classes):
        raise ValueError(
            f"Dumped logits shape {tuple(logits.shape)} differs from "
            f"({expected_samples}, {num_classes})"
        )
    test_root = Path(config["data"]["test_root"])
    expected_names = [
        path.name
        for path in sorted(
            path
            for path in test_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]
    if names != expected_names:
        raise ValueError("Dumped image order differs from the official test set")

    _, idx_to_class = load_class_mapping(config["data"]["class_mapping"])
    prediction, transport = exact_near_uniform_transport_prediction(
        logits,
        temperature=float(args.temperature),
        iterations=int(args.iterations),
    )
    predictions = [
        (name, str(idx_to_class[int(index)]).zfill(4))
        for name, index in zip(names, prediction.tolist())
    ]
    source_mode = str(payload["inference_mode"])
    inference_mode = (
        f"{source_mode}:exact_near_uniform_transport:"
        f"temperature={args.temperature:g}:iterations={args.iterations}"
    )
    manifest = create_submission(
        predictions,
        expected_names,
        args.output_dir,
        checkpoint_path,
        inference_mode=inference_mode,
        tta_risk_acknowledged=True,
        valid_labels={str(value).zfill(4) for value in idx_to_class.values()},
        extra_manifest={
            "corrupt_images": 0,
            "balanced_test_prior_acknowledged": True,
            "exact_integer_quota_risk_acknowledged": True,
            "source_logits_dump": str(dump_path),
            "source_logits_sha256": sha256_file(dump_path),
            "source_inference_mode": source_mode,
            "transport": transport,
        },
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
