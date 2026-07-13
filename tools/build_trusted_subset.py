#!/usr/bin/env python3
"""
Build trusted validation subset using model-agnostic V1 rules.

Reads a sample_metrics CSV, applies configurable thresholds, and outputs
a manifest CSV and summary JSON for the trusted subset.

Usage:
    python tools/build_trusted_subset.py \
        --sample-metrics outputs/experiment/sample_metrics.csv \
        --output-manifest outputs/experiment/trusted_manifest.csv \
        --output-summary outputs/experiment/trusted_subset_summary.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.trusted_subset import TrustedSubsetConfig, build_trusted_subset

logger = logging.getLogger(__name__)

# Sensitivity tiers for --sensitivity-tiers flag
SENSITIVITY_TIERS = {
    "T_strict": TrustedSubsetConfig(
        knn_label_agreement_min=0.80,
        prototype_margin_min=0.05,
        clip_flip_cosine_min=0.95,
    ),
    "T_main": TrustedSubsetConfig(
        knn_label_agreement_min=0.60,
        prototype_margin_min=0.02,
        clip_flip_cosine_min=0.90,
    ),
    "T_loose": TrustedSubsetConfig(
        knn_label_agreement_min=0.40,
        prototype_margin_min=0.01,
        clip_flip_cosine_min=0.80,
    ),
}

MANIFEST_COLUMNS = [
    "sample_index",
    "image_path",
    "image_sha256",
    "noisy_label",
    "class_name",
    "trusted_v1",
    "rejection_reasons",
    "knn_label_agreement",
    "prototype_top1_label",
    "prototype_margin",
    "clip_flip_cosine",
    "cross_class_duplicate_conflict",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Build trusted validation subset using model-agnostic V1 rules"
    )
    p.add_argument(
        "--sample-metrics",
        required=True,
        help="Path to sample_metrics CSV with per-sample features",
    )
    p.add_argument(
        "--output-manifest",
        default=None,
        help="Path to output manifest CSV (subset of columns)",
    )
    p.add_argument(
        "--output-summary",
        default=None,
        help="Path to output summary JSON",
    )
    p.add_argument(
        "--knn-label-agreement-min",
        type=float,
        default=0.60,
        help="Minimum kNN label agreement threshold (default: 0.60)",
    )
    p.add_argument(
        "--prototype-margin-min",
        type=float,
        default=0.02,
        help="Minimum prototype margin threshold (default: 0.02)",
    )
    p.add_argument(
        "--clip-flip-cosine-min",
        type=float,
        default=0.90,
        help="Minimum CLIP flip cosine threshold (default: 0.90)",
    )
    p.add_argument(
        "--sensitivity-tiers",
        action="store_true",
        help="Generate T_strict/T_main/T_loose sensitivity tier manifests and summaries",
    )
    return p.parse_args()


def build_output_paths(args):
    """Resolve default output paths if not specified."""
    metrics_path = Path(args.sample_metrics)
    metrics_dir = metrics_path.parent

    output_manifest = (
        Path(args.output_manifest)
        if args.output_manifest
        else metrics_dir / "trusted_manifest.csv"
    )
    output_summary = (
        Path(args.output_summary)
        if args.output_summary
        else metrics_dir / "trusted_subset_summary.json"
    )
    return output_manifest, output_summary


def build_single_trusted_subset(df, config, label):
    """Run build_trusted_subset and return (manifest, summary)."""
    manifest, summary = build_trusted_subset(df, config)
    summary["tier"] = label
    # Use label in the threshold keys
    summary["config"]["tier"] = label
    return manifest, summary


def save_tier_manifest(manifest, base_manifest_path, tier_label):
    """Save manifest CSV for a specific tier alongside the main manifest."""
    tier_path = base_manifest_path.parent / f"trusted_manifest_{tier_label}.csv"
    _save_manifest(manifest, tier_path)
    logger.info("Saved %s tier manifest: %s", tier_label, tier_path)
    return tier_path


def save_tier_summary(summary, base_summary_path, tier_label):
    """Save summary JSON for a specific tier."""
    tier_path = base_summary_path.parent / f"trusted_subset_summary_{tier_label}.json"
    tier_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved %s tier summary: %s", tier_label, tier_path)
    return tier_path


def _save_manifest(manifest, path):
    """Save manifest CSV with only the desired subset of columns."""
    available_cols = [c for c in MANIFEST_COLUMNS if c in manifest.columns]
    missing_cols = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing_cols:
        logger.warning(
            "Columns missing from manifest, will be omitted: %s", missing_cols
        )
    manifest[available_cols].to_csv(path, index=False)


def print_summary_report(summary):
    """Print coverage, class representation, and rejection reason counts."""
    print(f"Trusted subset summary:")
    print(f"  Total samples:        {summary['total_samples']}")
    print(f"  Trusted count:        {summary['trusted_count']}")
    print(f"  Rejected count:       {summary['rejected_count']}")
    print(f"  Coverage:             {summary['coverage']:.4f} ({summary['coverage']*100:.2f}%)")
    print(f"  Represented classes:  {summary['represented_classes']} / {summary['total_classes']}")
    print(f"  Missing classes:      {summary['missing_classes']}")
    print(f"  Rejection reasons:")
    for reason_key, count in summary["rejection_reason_counts"].items():
        print(f"    - {reason_key}: {count}")
    print()


def check_warnings(summary):
    """Warn if coverage < 25% or represented_classes < 475."""
    warnings_issued = False
    if summary["coverage"] < 0.25:
        logger.warning(
            "Low coverage: %.2f%% (< 25%%) — trusted subset may be too small",
            summary["coverage"] * 100,
        )
        warnings_issued = True
    if summary["represented_classes"] < 475:
        logger.warning(
            "Few represented classes: %d / %d (< 475) — some classes have no trusted samples",
            summary["represented_classes"],
            summary["total_classes"],
        )
        warnings_issued = True
    return warnings_issued


def process_sensitivity_tiers(df, base_manifest_path, base_summary_path):
    """Process all three sensitivity tiers and return list of tier summaries."""
    tier_summaries = []
    for tier_label, tier_config in SENSITIVITY_TIERS.items():
        logger.info("Processing sensitivity tier: %s", tier_label)
        tier_manifest, tier_summary = build_single_trusted_subset(
            df, tier_config, tier_label
        )
        save_tier_manifest(tier_manifest, base_manifest_path, tier_label)
        save_tier_summary(tier_summary, base_summary_path, tier_label)
        tier_summaries.append(tier_summary)

    # Print coverage comparison across tiers
    print("=== Sensitivity Tier Comparison ===")
    for s in tier_summaries:
        print(
            f"  {s['tier']:10s}: coverage={s['coverage']:.4f}, "
            f"trusted={s['trusted_count']:5d}, "
            f"represented_classes={s['represented_classes']:3d}"
        )
    print()
    return tier_summaries


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    # Validate input file exists
    metrics_path = Path(args.sample_metrics)
    if not metrics_path.exists():
        logger.error("Sample metrics file not found: %s", metrics_path)
        sys.exit(4)

    # Resolve output paths
    output_manifest, output_summary = build_output_paths(args)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    # Read sample metrics
    logger.info("Reading sample metrics from: %s", metrics_path)
    try:
        df = pd.read_csv(metrics_path)
    except Exception as e:
        logger.error("Failed to read sample metrics CSV: %s", e)
        sys.exit(4)

    logger.info("Loaded %d samples with %d columns", len(df), len(df.columns))

    # Check required columns
    required_cols = [
        "knn_label_agreement",
        "prototype_supports_noisy_label",
        "prototype_margin",
        "clip_flip_cosine",
    ]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        logger.error(
            "Missing required columns in sample metrics: %s", missing_required
        )
        sys.exit(4)

    # Build config from CLI args
    config = TrustedSubsetConfig(
        knn_label_agreement_min=args.knn_label_agreement_min,
        prototype_margin_min=args.prototype_margin_min,
        clip_flip_cosine_min=args.clip_flip_cosine_min,
    )

    # Build trusted subset
    logger.info(
        "Building trusted subset with config: knn_agreement>=%.2f, "
        "prototype_margin>=%.2f, clip_flip_cosine>=%.2f",
        config.knn_label_agreement_min,
        config.prototype_margin_min,
        config.clip_flip_cosine_min,
    )
    try:
        manifest, summary = build_trusted_subset(df, config)
        summary["tier"] = "main"
    except Exception as e:
        logger.error("Failed to build trusted subset: %s", e)
        sys.exit(4)

    # Save manifest
    _save_manifest(manifest, output_manifest)
    logger.info("Saved manifest: %s (%d rows)", output_manifest, len(manifest))

    # Save summary
    output_summary.write_text(json.dumps(summary, indent=2))
    logger.info("Saved summary: %s", output_summary)

    # Print summary report
    print_summary_report(summary)

    # Process sensitivity tiers if requested
    if args.sensitivity_tiers:
        process_sensitivity_tiers(df, output_manifest, output_summary)

    # Check warnings
    check_warnings(summary)

    logger.info("Done.")


if __name__ == "__main__":
    main()
