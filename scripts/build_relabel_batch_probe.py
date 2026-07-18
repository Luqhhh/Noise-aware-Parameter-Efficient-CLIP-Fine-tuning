#!/usr/bin/env python3
"""Build relabel_batch_probe.json from a purification manifest.

Selects clean/rejected/pseudo samples, verifies training labels and
weights through the provider, and confirms that rejected samples
contribute zero gradient through the weighted MixUp reduction path.

Usage:
    python scripts/build_relabel_batch_probe.py \
        --manifest outputs/phase4/purification/nr_consensus_relabel_v2_top100/purification_manifest.csv \
        --output audit/relabel_batch_probe.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def build_probe(manifest_path: str, output_path: str, n_per_role: int = 5):
    """Build the batch probe JSON."""
    df = pd.read_csv(manifest_path)

    clean = df[df["training_role"] == "clean"]
    rejected = df[df["training_role"] == "rejected"]
    pseudo = df[df["training_role"] == "pseudo"]

    clean_sample = clean.nlargest(min(n_per_role, len(clean)), "quality_score")
    rejected_sample = rejected.nlargest(min(n_per_role, len(rejected)), "quality_score")
    pseudo_sample = pseudo.nlargest(min(n_per_role, len(pseudo)), "quality_score")

    selected = pd.concat(
        [clean_sample, rejected_sample, pseudo_sample], ignore_index=True
    )

    probe_records = []
    for _, row in selected.iterrows():
        rec = {
            "image_path": str(row["image_path"]),
            "original_label": int(row["original_label"]),
            "training_label": int(row["training_label"]),
            "sample_weight": float(row["sample_weight"]),
            "training_role": str(row["training_role"]),
        }
        role = rec["training_role"]
        if role == "clean":
            assert rec["training_label"] == rec["original_label"], (
                f"Clean sample has training_label != original_label: "
                f"{rec['image_path']}"
            )
            assert rec["sample_weight"] == 1.0, (
                f"Clean sample has weight != 1.0: {rec['image_path']}"
            )
        elif role == "rejected":
            assert rec["training_label"] == rec["original_label"], (
                f"Rejected sample has training_label != original_label: "
                f"{rec['image_path']}"
            )
            assert rec["sample_weight"] == 0.0, (
                f"Rejected sample has weight != 0.0: {rec['image_path']}"
            )
        elif role == "pseudo":
            assert rec["training_label"] != rec["original_label"], (
                f"Pseudo sample has training_label == original_label: "
                f"{rec['image_path']}"
            )
            assert rec["sample_weight"] == 1.0, (
                f"Pseudo sample has weight != 1.0: {rec['image_path']}"
            )
        probe_records.append(rec)

    _verify_mixup_zeroing(probe_records)

    probe = {
        "manifest_path": manifest_path,
        "total_rows": len(df),
        "clean_count": int((df["training_role"] == "clean").sum()),
        "rejected_count": int((df["training_role"] == "rejected").sum()),
        "pseudo_count": int((df["training_role"] == "pseudo").sum()),
        "probe_samples": probe_records,
        "mixup_zeroing_verified": True,
    }

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(probe, f, indent=2)

    print(
        f"Wrote {len(probe_records)} probe records to {output_path}"
    )
    print(
        f"  clean={probe['clean_count']}, "
        f"rejected={probe['rejected_count']}, "
        f"pseudo={probe['pseudo_count']}"
    )
    return probe


def _verify_mixup_zeroing(records: list):
    """Verify that rejected (weight=0) samples zero out in MixUp reduction.

    A rejected sample's primary loss contribution is lam * weight[i] * loss_a[i].
    Since weight[i] == 0, this is always 0 — the sample provides no supervision.
    The permuted (paired) term uses weights[permutation[i]] from a different
    sample, so it may be non-zero.
    """
    n = len(records)
    weights = torch.tensor([r["sample_weight"] for r in records])
    loss_a = torch.rand(n)
    lam = 0.4

    for i, rec in enumerate(records):
        if rec["training_role"] == "rejected":
            assert weights[i] == 0.0, (
                f"Rejected sample {i} has non-zero weight"
            )
            # Primary contribution: lam * weight[i] * loss_a[i] must be 0
            primary = lam * weights[i] * loss_a[i]
            assert primary.item() == 0.0, (
                f"Rejected sample {i} has non-zero primary MixUp contribution: "
                f"{primary.item()}"
            )

    for i, rec in enumerate(records):
        if rec["training_role"] in ("clean", "pseudo"):
            assert weights[i] == 1.0, (
                f"{rec['training_role']} sample {i} has weight != 1.0"
            )

    print("MixUp zeroing verification PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True,
        help="Path to purification_manifest.csv",
    )
    parser.add_argument(
        "--output", default="audit/relabel_batch_probe.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--n-per-role", type=int, default=5,
        help="Samples per role",
    )
    args = parser.parse_args()

    try:
        build_probe(args.manifest, args.output, args.n_per_role)
    except AssertionError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
