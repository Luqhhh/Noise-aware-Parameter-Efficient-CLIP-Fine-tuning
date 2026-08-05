"""Build an audited submission by reweighting nested multiscale logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from aegis_clip.config import load_config
from aegis_clip.data import IMAGE_EXTENSIONS, load_class_mapping
from aegis_clip.prior_alignment import align_logits_to_prior
from aegis_clip.runtime import sha256_file
from aegis_clip.scale_reweighting import (
    parse_scale_weights,
    reconstruct_nested_scale_probabilities,
    weighted_scale_probabilities,
)
from aegis_clip.submission import create_submission


def _load_dump(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Fused-logits dump is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"logits", "names", "inference_mode"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        missing = required - set(payload if isinstance(payload, dict) else ())
        raise ValueError(f"Fused-logits dump misses fields: {sorted(missing)}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triple-dump", required=True)
    parser.add_argument("--pair-dump", required=True)
    parser.add_argument("--single-dump", required=True)
    parser.add_argument("--scales", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--acknowledge-balanced-test-prior", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.acknowledge_balanced_test_prior:
        raise ValueError(
            "Prior alignment uses the declared balanced test-set prior; pass "
            "--acknowledge-balanced-test-prior explicitly"
        )
    if not 0.0 <= float(args.strength) <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if int(args.iterations) <= 0:
        raise ValueError("iterations must be positive")
    scales, weights = parse_scale_weights(args.scales, args.weights)

    paths = {
        "triple": Path(args.triple_dump).resolve(),
        "pair": Path(args.pair_dump).resolve(),
        "single": Path(args.single_dump).resolve(),
    }
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint_path}")
    payloads = {name: _load_dump(path) for name, path in paths.items()}
    names = [str(value) for value in payloads["triple"]["names"]]
    if any(
        [str(value) for value in payload["names"]] != names
        for payload in payloads.values()
    ):
        raise ValueError("Nested fused-logits dumps have different image orders")

    expected_protocols = {
        "triple": f"crops={'-'.join(str(value) for value in scales)}",
        "pair": f"crops={scales[1]}-{scales[2]}",
        "single": f"crop={scales[2]}",
    }
    for name, fragment in expected_protocols.items():
        mode = str(payloads[name]["inference_mode"])
        if fragment not in mode:
            raise ValueError(f"{name} dump mode {mode!r} does not contain {fragment!r}")

    config = load_config(args.config)
    num_classes = int(config["model"]["num_classes"])
    expected_samples = int(config["data"]["expected_test_samples"])
    expected_shape = (expected_samples, num_classes)
    logits = {
        name: torch.as_tensor(payload["logits"]).float().cpu()
        for name, payload in payloads.items()
    }
    if any(tuple(value.shape) != expected_shape for value in logits.values()):
        raise ValueError(f"Every logits dump must have shape {expected_shape}")

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

    recovered, reconstruction = reconstruct_nested_scale_probabilities(
        logits["triple"], logits["pair"], logits["single"]
    )
    fused = weighted_scale_probabilities(recovered, weights)
    aligned, alignment = align_logits_to_prior(
        fused.log(),
        strength=float(args.strength),
        max_iterations=int(args.iterations),
    )
    _, idx_to_class = load_class_mapping(config["data"]["class_mapping"])
    predictions = [
        (name, str(idx_to_class[index]).zfill(4))
        for name, index in zip(names, aligned.argmax(dim=1).tolist())
    ]
    scale_text = "-".join(str(value) for value in scales)
    weight_text = "-".join(f"{value:g}" for value in weights)
    inference_mode = (
        "attention_reweighted_multiscale_flip:"
        f"crops={scale_text}:weights={weight_text}:balanced_prior={args.strength:g}"
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
            "source_logits": {
                name: {
                    "path": str(paths[name]),
                    "sha256": sha256_file(paths[name]),
                    "inference_mode": str(payloads[name]["inference_mode"]),
                }
                for name in paths
            },
            "scales": list(scales),
            "scale_weights": list(weights),
            "scale_reconstruction": reconstruction,
            "prior_alignment": alignment,
        },
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
