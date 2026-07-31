"""Offline balanced-prior strength sweep over dumped fused logits.

Takes a logits dump produced by ``infer.py --dump-logits`` (raw fused logits
before any prior alignment) and evaluates ``align_logits_to_prior`` at several
strengths without re-running model inference.  Writes one ``pred_results.csv``
per strength and a JSON summary of the class-balance diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from aegis_clip.config import load_config
from aegis_clip.data import load_class_mapping
from aegis_clip.prior_alignment import align_logits_to_prior
from aegis_clip.runtime import atomic_json_dump


def _parse_float_sequence(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, help="Path to infer.py --dump-logits output")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strengths", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    dump_path = Path(args.dump)
    if not dump_path.exists():
        raise FileNotFoundError(f"Logits dump not found: {dump_path}")
    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    logits = dump["logits"]
    names = dump["names"]
    if logits.ndim != 2 or logits.shape[0] != len(names):
        raise ValueError("Dumped logits must be [N, C] matching the name count")

    strengths = _parse_float_sequence(args.strengths)
    if any(not 0.0 <= value <= 1.0 for value in strengths):
        raise ValueError("All strengths must be in [0, 1]")

    config = load_config(args.config)
    _, idx_to_class = load_class_mapping(config["data"]["class_mapping"])
    num_classes = int(config["model"]["num_classes"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float | int | list[int]]] = {}
    for strength in strengths:
        aligned, report = align_logits_to_prior(
            logits,
            strength=float(strength),
            max_iterations=args.iterations,
        )
        indices = aligned.argmax(dim=1).tolist()
        counts = [
            indices.count(index)
            for index in range(num_classes)
        ]
        changed = 0
        if strength == 0.0:
            baseline_indices = list(indices)
        else:
            changed = sum(
                1 for index, base in zip(indices, baseline_indices) if index != base
            )
        csv_path = output_dir / f"pred_strength_{strength:g}.csv"
        with csv_path.open("w", encoding="utf-8") as handle:
            for name, index in zip(names, indices):
                handle.write(f"{name}, {str(idx_to_class[index]).zfill(4)}\n")
        summary[f"strength_{strength:g}"] = {
            "prediction_count": len(indices),
            "class_count": len({index for index in indices}),
            "count_min": min(counts),
            "count_max": max(counts),
            "count_mean": float(sum(counts) / num_classes),
            "changed_vs_strength_0": changed,
            "marginal_l1": float(report["final_marginal_l1"]),
            "iterations": int(report["iterations"]),
        }

    report_path = output_dir / "prior_alignment_sweep.json"
    atomic_json_dump(
        {
            "strengths": list(strengths),
            "inference_mode": dump.get("inference_mode"),
            "results": summary,
        },
        report_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
