#!/usr/bin/env python3
"""Grid-search L05 flip-TTA temperature and prior strength on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
import sys
if str(AEGIS_ROOT) not in sys.path:
    sys.path.insert(0, str(AEGIS_ROOT))

from aegis_clip.balanced_inference import prediction_metrics  # noqa: E402
from aegis_clip.prior_alignment import apply_prior_bias, fit_prior_bias  # noqa: E402
from aegis_clip.tta import fuse_paired_logits  # noqa: E402


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in str(value).split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperatures", default="1.0,1.1,1.2,1.3,1.4,1.5")
    parser.add_argument(
        "--strengths", default="0.25,0.50,0.60,0.70,0.75,0.80,0.90,1.00"
    )
    parser.add_argument("--selector-metric", default="raw_macro")
    args = parser.parse_args()

    payload = torch.load(args.branch_cache, map_location="cpu", weights_only=False)
    required = {
        "original_logits",
        "flip_logits",
        "labels",
        "clean_probability",
        "pseudo_labels",
        "correction_alpha",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"branch cache missing keys: {sorted(missing)}")
    original = torch.as_tensor(payload["original_logits"]).float()
    flipped = torch.as_tensor(payload["flip_logits"]).float()
    labels = payload["labels"]
    clean = payload["clean_probability"]
    pseudo = payload["pseudo_labels"]
    correction = payload["correction_alpha"]
    num_classes = int(original.shape[1])
    metric_args = {
        "labels": labels,
        "clean_probability": clean,
        "pseudo_labels": pseudo,
        "correction_alpha": correction,
        "num_classes": num_classes,
        "clean_core_threshold": 0.70,
    }
    temperatures = _parse_floats(args.temperatures)
    strengths = _parse_floats(args.strengths)
    rows = []
    for temperature in temperatures:
        fused = fuse_paired_logits(
            original,
            flipped,
            mode="mean_probabilities",
            temperature=float(temperature),
        )
        bias, fit_report = fit_prior_bias(fused, max_iterations=50)
        for strength in strengths:
            corrected = apply_prior_bias(fused, bias, strength=float(strength))
            metrics = prediction_metrics(corrected.argmax(dim=1), **metric_args)
            rows.append(
                {
                    "temperature": float(temperature),
                    "strength": float(strength),
                    "raw_macro": float(metrics["raw_macro"]),
                    "raw_micro": float(metrics["raw_micro"]),
                    "changed_predictions_vs_raw": int(
                        (fused.argmax(dim=1) != corrected.argmax(dim=1)).sum()
                    ),
                    "bias_fit_iterations": int(fit_report["iterations"]),
                }
            )
    rows.sort(key=lambda item: item[str(args.selector_metric)], reverse=True)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "selector_metric": str(args.selector_metric),
                "temperatures": list(temperatures),
                "strengths": list(strengths),
                "grid": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for row in rows[:10]:
        print(
            f"T={row['temperature']:.2f} s={row['strength']:.2f} "
            f"macro={row['raw_macro']:.6f} micro={row['raw_micro']:.6f}"
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
