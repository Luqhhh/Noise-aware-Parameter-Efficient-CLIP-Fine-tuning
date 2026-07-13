#!/usr/bin/env python3
"""
Aggregate dual-validation results across multiple random seeds into a
cross-seed summary report.

Reads one ``dual_validation.json`` per seed for both reference (e.g. D3) and
candidate (e.g. B2) experiments, computes per-seed and cross-seed statistics,
and writes a JSON report and a Markdown findings file.

Usage:
    # After running evaluate_dual_validation.py for each seed:
    python tools/summarize_cross_seed.py \\
        --reference-name ref \\
        --candidate-name gce_q07 \\
        --seeds 42,2026,3407 \\
        --ref-results outputs/analysis/d3_vs_b2_seed{seed}/dual_validation_d3.json \\
        --cand-results outputs/analysis/d3_vs_b2_seed{seed}/dual_validation_b2.json \\
        --output-dir outputs/analysis/cross_seed
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-seed aggregation of dual validation results"
    )
    p.add_argument("--reference-name", required=True, help="e.g. ref")
    p.add_argument("--candidate-name", required=True, help="e.g. gce_q07")
    p.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated list of seeds, e.g. 42,2026,3407",
    )
    p.add_argument(
        "--ref-results",
        required=True,
        help="Path template for reference dual_validation.json. "
        "Use {seed} placeholder, e.g. outputs/analysis/d3_vs_b2_seed{seed}/dual_validation_d3.json",
    )
    p.add_argument(
        "--cand-results",
        required=True,
        help="Path template for candidate dual_validation.json.",
    )
    p.add_argument("--output-dir", required=True, help="Output directory for reports")
    return p.parse_args()


def _load_one(path_template: str, seed: str, name: str) -> dict:
    """Load a dual_validation.json for one seed."""
    path = Path(path_template.replace("{seed}", seed))
    if not path.exists():
        raise FileNotFoundError(f"[{name} seed={seed}] Missing: {path}")
    with open(path) as f:
        return json.load(f)


def _safe_get(d: dict, *keys, default=None):
    """Navigate nested dicts safely."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def _stats(values: List[float]) -> dict:
    """Compute mean, std, min, max for a list of floats."""
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": [float(v) for v in arr],
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    seeds = [s.strip() for s in args.seeds.split(",")]
    ref_name = args.reference_name
    cand_name = args.candidate_name

    logger.info("=== Cross-Seed Summary ===")
    logger.info("Reference:  %s", ref_name)
    logger.info("Candidate:  %s", cand_name)
    logger.info("Seeds:      %s", seeds)

    # ── 1. Load all data ──
    ref_data: Dict[str, dict] = {}
    cand_data: Dict[str, dict] = {}
    for seed in seeds:
        ref_data[seed] = _load_one(args.ref_results, seed, ref_name)
        cand_data[seed] = _load_one(args.cand_results, seed, cand_name)

    # ── 2. Extract per-seed metrics ──
    metrics_to_track = {
        "raw_micro": (None, "raw_noisy_validation", "micro_accuracy"),
        "raw_macro": (None, "raw_noisy_validation", "macro_accuracy_present_classes"),
        "raw_median": (None, "raw_noisy_validation", "median_class_accuracy"),
        "trusted_v1_micro": (None, "trusted_validation", "micro_accuracy"),
        "trusted_v1_coverage": (None, "trusted_validation", "coverage"),
        "trusted_v1_classes": (None, "trusted_validation", "represented_classes"),
        "v2_trust_weighted": ("trusted_validation_v2", "trust_weighted", "accuracy"),
        "v2_class_balanced": ("trusted_validation_v2", "class_balanced", "macro_accuracy"),
        "v2_cb_classes": ("trusted_validation_v2", "class_balanced", "num_classes_with_k"),
    }

    per_seed = {}
    for seed in seeds:
        seed_metrics = {"ref": {}, "cand": {}, "delta": {}}
        for metric_key, path in metrics_to_track.items():
            ref_val = _safe_get(ref_data[seed], *filter(None, path))
            cand_val = _safe_get(cand_data[seed], *filter(None, path))

            if ref_val is not None and cand_val is not None:
                seed_metrics["ref"][metric_key] = ref_val
                seed_metrics["cand"][metric_key] = cand_val
                seed_metrics["delta"][metric_key] = cand_val - ref_val
        per_seed[seed] = seed_metrics

    # ── 3. Aggregate across seeds ──
    cross_seed = {}
    for metric_key in metrics_to_track:
        ref_vals = [per_seed[s]["ref"].get(metric_key) for s in seeds
                     if metric_key in per_seed[s]["ref"]]
        cand_vals = [per_seed[s]["cand"].get(metric_key) for s in seeds
                      if metric_key in per_seed[s]["cand"]]
        delta_vals = [per_seed[s]["delta"].get(metric_key) for s in seeds
                       if metric_key in per_seed[s]["delta"]]

        cross_seed[metric_key] = {
            "ref": _stats(ref_vals) if ref_vals else None,
            "cand": _stats(cand_vals) if cand_vals else None,
            "delta": _stats(delta_vals) if delta_vals else None,
        }

    # ── 4. Build report ──
    report = {
        "schema_version": 1,
        "reference_name": ref_name,
        "candidate_name": cand_name,
        "seeds": seeds,
        "per_seed": {
            seed: {
                "ref": {k: v for k, v in per_seed[seed]["ref"].items()},
                "cand": {k: v for k, v in per_seed[seed]["cand"].items()},
                "delta": {k: v for k, v in per_seed[seed]["delta"].items()},
            }
            for seed in seeds
        },
        "cross_seed": {k: v for k, v in cross_seed.items()},
    }

    # ── 5. Save JSON ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cross_seed_summary.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("JSON report saved to: %s", json_path)

    # ── 6. Generate Markdown ──
    md_lines = [
        f"# Cross-Seed Validation Report",
        f"",
        f"**Reference:** {ref_name}  ",
        f"**Candidate:** {cand_name}  ",
        f"**Seeds:** {', '.join(seeds)}  ",
        f"",
        f"## 1. Per-Seed Results",
        f"",
    ]

    # Table header
    header = "| Seed | Raw Micro | Raw Macro | V1 Trusted | V1 Cov% | V2 TW Acc | V2 CB Acc |"
    sep =    "|------|-----------|-----------|------------|---------|-----------|-----------|"
    md_lines.append(header)
    md_lines.append(sep)

    for seed in seeds:
        m = per_seed[seed]
        row = (
            f"| {seed} "
            f"| {m['ref'].get('raw_micro', 0):.4f} → {m['cand'].get('raw_micro', 0):.4f} "
            f"| {m['ref'].get('raw_macro', 0):.4f} → {m['cand'].get('raw_macro', 0):.4f} "
            f"| {m['ref'].get('trusted_v1_micro', 0):.4f} → {m['cand'].get('trusted_v1_micro', 0):.4f} "
            f"| {m['ref'].get('trusted_v1_coverage', 0)*100:.1f}% "
            f"| {m['ref'].get('v2_trust_weighted', 0):.4f} → {m['cand'].get('v2_trust_weighted', 0):.4f} "
            f"| {m['ref'].get('v2_class_balanced', 0):.4f} → {m['cand'].get('v2_class_balanced', 0):.4f} |"
        )
        md_lines.append(row)

    md_lines.append("")

    # ── 7. Cross-seed deltas ──
    md_lines.extend([
        f"## 2. Cross-Seed Deltas ({cand_name} − {ref_name})",
        f"",
        f"| Metric | Mean ± Std | Min | Max |",
        f"|--------|-----------|-----|-----|",
    ])

    metric_labels = {
        "raw_micro": "Raw Micro Acc",
        "raw_macro": "Raw Macro Acc",
        "trusted_v1_micro": "V1 Trusted Acc",
        "v2_trust_weighted": "V2 Trust-Weighted Acc",
        "v2_class_balanced": "V2 Class-Balanced Acc",
    }

    for metric_key, label in metric_labels.items():
        delta_stats = cross_seed.get(metric_key, {}).get("delta")
        if delta_stats and delta_stats["values"]:
            md_lines.append(
                f"| {label} "
                f"| {delta_stats['mean']:+.4f} ± {delta_stats['std']:.4f} "
                f"| {delta_stats['min']:+.4f} "
                f"| {delta_stats['max']:+.4f} |"
            )

    md_lines.extend([
        f"",
        f"## 3. Stability Assessment",
        f"",
    ])

    # Check if all deltas have same sign
    raw_delta_signs = [np.sign(per_seed[s]["delta"].get("raw_micro", 0)) for s in seeds]
    v2_delta_signs = [np.sign(per_seed[s]["delta"].get("v2_trust_weighted", 0)) for s in seeds]

    if all(s == raw_delta_signs[0] for s in raw_delta_signs):
        md_lines.append(f"- Raw noisy-label delta: **consistent sign** across all seeds ({raw_delta_signs[0]:+d})")
    else:
        md_lines.append(f"- ⚠️ Raw noisy-label delta: **sign flips** across seeds")

    if all(s == v2_delta_signs[0] for s in v2_delta_signs):
        md_lines.append(f"- V2 trust-weighted delta: **consistent sign** across all seeds ({v2_delta_signs[0]:+d})")
    else:
        md_lines.append(f"- ⚠️ V2 trust-weighted delta: **sign flips** across seeds")

    # Platform-local correlation
    md_lines.extend([
        f"",
        f"### B2 Stability Conclusion",
        f"",
        f"If the per-seed deltas show consistent sign and magnitude, gce_q07 is a "
        f"stable improvement over ref. If signs flip or variance is large, "
        f"B2 should be treated as high-variance and not the primary candidate.",
        f"",
        f"---",
        f"",
        f"*Generated by tools/summarize_cross_seed.py*",
    ])

    md_path = output_dir / "cross_seed_findings.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info("Markdown report saved to: %s", md_path)

    # ── 8. Print key deltas ──
    logger.info("=" * 60)
    logger.info("CROSS-SEED DELTA SUMMARY (%s − %s)", cand_name, ref_name)
    logger.info("=" * 60)
    for metric_key, label in metric_labels.items():
        delta_stats = cross_seed.get(metric_key, {}).get("delta")
        if delta_stats and delta_stats["values"]:
            logger.info(
                "%25s: %+.4f ± %.4f  [%+.4f, %+.4f]",
                label,
                delta_stats["mean"],
                delta_stats["std"],
                delta_stats["min"],
                delta_stats["max"],
            )
    logger.info("=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()
