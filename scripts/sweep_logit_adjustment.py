#!/usr/bin/env python3
"""Sweep tau values for logit adjustment using pre-computed val logits.

Usage:
    python3 scripts/sweep_logit_adjustment.py \
        --val-logits outputs/experiment/val_logits.pt \
        --val-labels outputs/experiment/val_labels.pt \
        --train-csv outputs/experiment/split/train.csv \
        --taus 0 0.25 0.5 0.75 1.0 \
        --output-dir outputs/experiment/sweep
"""

import argparse
import json
import os
import sys

import torch

# Add repo root to path for common imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.logit_adjustment import (
    _compute_metrics_from_logits,
    adjust_logits,
    compute_class_priors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep tau values for logit adjustment"
    )
    parser.add_argument(
        "--val-logits",
        required=True,
        type=str,
        help="Path to val_logits.pt",
    )
    parser.add_argument(
        "--val-labels",
        required=True,
        type=str,
        help="Path to val_labels.pt",
    )
    parser.add_argument(
        "--train-csv",
        required=True,
        type=str,
        help="Path to train.csv for computing class priors",
    )
    parser.add_argument(
        "--taus",
        required=True,
        nargs="+",
        type=float,
        help="Tau values to sweep (e.g. 0 0.25 0.5 0.75 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to write sweep_results.json",
    )
    return parser.parse_args()


def select_best_tau(
    sweep_results: dict,
) -> tuple[float, dict]:
    """Select best tau from sweep results.

    Priority: macro_accuracy > micro_accuracy > bottom_10_percent_accuracy.
    Returns (best_tau, best_metrics).
    """
    keys = ["macro_accuracy", "micro_accuracy", "bottom_10_percent_accuracy"]
    best_tau = None
    best_metrics = None

    for tau, metrics in sweep_results.items():
        if best_tau is None:
            best_tau = tau
            best_metrics = metrics
            continue

        for key in keys:
            if metrics[key] > best_metrics[key]:
                best_tau = tau
                best_metrics = metrics
                break
            elif metrics[key] < best_metrics[key]:
                break
            # else equal, continue to next tiebreaker

    return best_tau, best_metrics


def main() -> None:
    args = parse_args()

    # Load validation data
    print(f"Loading val logits from {args.val_logits}")
    val_logits = torch.load(args.val_logits, map_location="cpu", weights_only=True)
    print(f"Loading val labels from {args.val_labels}")
    val_labels = torch.load(args.val_labels, map_location="cpu", weights_only=True)

    if val_logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape {val_logits.shape}")
    if val_labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {val_labels.shape}")
    if val_logits.size(0) != val_labels.size(0):
        raise ValueError(
            f"Number of logits ({val_logits.size(0)}) does not match "
            f"number of labels ({val_labels.size(0)})"
        )

    num_classes = val_logits.size(1)
    print(f"  Logits shape: {val_logits.shape}")
    print(f"  Labels shape: {val_labels.shape}")
    print(f"  Num classes:  {num_classes}")

    # Compute class priors
    print(f"Computing class priors from {args.train_csv}")
    priors = compute_class_priors(args.train_csv, num_classes=num_classes)
    priors_list = priors.tolist()
    print(f"  Priors computed (min={priors.min().item():.6e}, "
          f"max={priors.max().item():.6f})")

    # Sweep taus
    print(f"Sweeping tau values: {args.taus}")
    sweep = {}
    for tau in args.taus:
        adjusted = adjust_logits(val_logits, priors, tau)
        metrics = _compute_metrics_from_logits(adjusted, val_labels)
        sweep[tau] = metrics
        print(
            f"  tau={tau:6.3f}  micro={metrics['micro_accuracy']:.4f}  "
            f"macro={metrics['macro_accuracy']:.4f}  "
            f"median={metrics['median_per_class_accuracy']:.4f}  "
            f"bottom10={metrics['bottom_10_percent_accuracy']:.4f}  "
            f"gap={metrics['micro_macro_gap']:.4f}"
        )

    # Select best tau
    best_tau, best_metrics = select_best_tau(sweep)
    print(f"\nBest tau: {best_tau}")
    print(f"  micro_accuracy={best_metrics['micro_accuracy']:.4f}")
    print(f"  macro_accuracy={best_metrics['macro_accuracy']:.4f}")
    print(f"  median_per_class_accuracy={best_metrics['median_per_class_accuracy']:.4f}")
    print(f"  bottom_10_percent_accuracy={best_metrics['bottom_10_percent_accuracy']:.4f}")
    print(f"  micro_macro_gap={best_metrics['micro_macro_gap']:.4f}")

    # Get baseline (tau=0) metrics
    baseline_metrics = sweep.get(0.0, None)

    # Build output dict
    # Convert sweep dict keys (float) to string for JSON
    sweep_list = []
    for tau in args.taus:
        sweep_list.append({
            "tau": tau,
            "micro_accuracy": sweep[tau]["micro_accuracy"],
            "macro_accuracy": sweep[tau]["macro_accuracy"],
            "median_per_class_accuracy": sweep[tau]["median_per_class_accuracy"],
            "bottom_10_percent_accuracy": sweep[tau]["bottom_10_percent_accuracy"],
            "micro_macro_gap": sweep[tau]["micro_macro_gap"],
        })

    output = {
        "priors": priors_list,
        "sweep": sweep_list,
        "best_tau": best_tau,
        "best_metrics": {
            "micro_accuracy": best_metrics["micro_accuracy"],
            "macro_accuracy": best_metrics["macro_accuracy"],
            "median_per_class_accuracy": best_metrics["median_per_class_accuracy"],
            "bottom_10_percent_accuracy": best_metrics["bottom_10_percent_accuracy"],
            "micro_macro_gap": best_metrics["micro_macro_gap"],
        },
        "baseline": (
            {
                "micro_accuracy": baseline_metrics["micro_accuracy"],
                "macro_accuracy": baseline_metrics["macro_accuracy"],
                "median_per_class_accuracy": baseline_metrics["median_per_class_accuracy"],
                "bottom_10_percent_accuracy": baseline_metrics["bottom_10_percent_accuracy"],
                "micro_macro_gap": baseline_metrics["micro_macro_gap"],
            }
            if baseline_metrics is not None
            else None
        ),
    }

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "sweep_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
