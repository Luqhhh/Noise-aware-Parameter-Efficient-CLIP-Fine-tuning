"""Rebuild cross-fitted wrong-event and top-k training trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aegis_clip.oof_rebuild import load_oof_inputs
from aegis_clip.runtime import sha256_file
from aegis_clip.trajectory import rebuild_oof_trajectory


def _verified_hash(path: str | Path, expected: str | None, name: str) -> str:
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected={expected}, actual={actual}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--feature-tensor", required=True)
    parser.add_argument("--feature-paths", required=True)
    parser.add_argument("--feature-labels", required=True)
    parser.add_argument("--reference-oof", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-assignments-sha256")
    parser.add_argument("--expected-feature-tensor-sha256")
    parser.add_argument("--expected-feature-paths-sha256")
    parser.add_argument("--expected-feature-labels-sha256")
    parser.add_argument("--expected-reference-oof-sha256")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--infer-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    hashes = {
        "assignments_sha256": _verified_hash(
            args.assignments, args.expected_assignments_sha256, "assignments"
        ),
        "feature_tensor_sha256": _verified_hash(
            args.feature_tensor,
            args.expected_feature_tensor_sha256,
            "feature tensor",
        ),
        "feature_paths_sha256": _verified_hash(
            args.feature_paths, args.expected_feature_paths_sha256, "feature paths"
        ),
        "feature_labels_sha256": _verified_hash(
            args.feature_labels,
            args.expected_feature_labels_sha256,
            "feature labels",
        ),
        "reference_oof_sha256": _verified_hash(
            args.reference_oof,
            args.expected_reference_oof_sha256,
            "reference OOF artifact",
        ),
    }
    inputs = load_oof_inputs(
        args.assignments,
        args.feature_tensor,
        args.feature_paths,
        args.feature_labels,
    )
    result = rebuild_oof_trajectory(
        inputs,
        args.output_dir,
        reference_oof_path=args.reference_oof,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        infer_batch_size=args.infer_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        q=args.q,
        seed=args.seed,
        top_k=args.top_k,
        device=device,
        input_hashes=hashes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
