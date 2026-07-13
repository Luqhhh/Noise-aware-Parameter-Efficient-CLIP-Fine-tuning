#!/usr/bin/env python3
"""
Generate findings.md from all analysis outputs.

Consumes audit JSON, sample_metrics CSV, group_summary JSON, trusted_summary JSON,
and both dual validation JSONs (reference, candidate) to produce a structured
markdown report covering protocol audit, raw / trusted accuracy, four-group
composition, metric comparisons, and a conclusion.

Usage:
    python tools/summarize_disagreement.py \\
        --audit outputs/d3_vs_b2/audit.json \\
        --sample-metrics outputs/d3_vs_b2_disagreement/sample_metrics.csv \\
        --group-summary outputs/d3_vs_b2_disagreement/group_summary.json \\
        --trusted-summary outputs/d3_vs_b2_disagreement/trusted_subset_summary.json \\
        --reference-dual outputs/d3_strict/seed42/checkpoints/dual_validation.json \\
        --candidate-dual outputs/b2_gce07/seed42/checkpoints/dual_validation.json \\
        --output outputs/d3_vs_b2_disagreement/findings.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────


def load_json(path: str) -> dict:
    """Load a JSON file and return the parsed dict."""
    path = Path(path)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(4)
    with open(path) as f:
        return json.load(f)


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    path = Path(path)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(4)
    return pd.read_csv(path)


# ──────────────────────────────────────────────────────────────────────
# Report helpers
# ──────────────────────────────────────────────────────────────────────


def micro_correct(metrics: dict) -> int:
    """Compute number of correct samples from micro_accuracy and num_samples."""
    return int(round(metrics["micro_accuracy"] * metrics["num_samples"]))


# ──────────────────────────────────────────────────────────────────────
# Section builders
# ──────────────────────────────────────────────────────────────────────


def section_protocol_audit(audit: dict) -> str:
    """Build ## 1. Protocol Audit."""
    lines = ["## 1. Protocol Audit\n"]
    lines.append(f"- **paired_valid**: {audit.get('paired_valid', 'UNKNOWN')}")
    lines.append(
        f"- **causal_claim_allowed**: {audit.get('causal_claim_allowed', 'UNKNOWN')}"
    )
    lines.append(
        f"- **sample_classification**: {audit.get('sample_classification', 'not_checked')}"
    )
    lines.append(
        f"- **max_visual_abs_diff**: {audit.get('max_visual_abs_diff', -1.0)}"
    )
    unexpected_count = len(audit.get("unexpected_differences", []))
    lines.append(f"- **unexpected_differences count**: {unexpected_count}")
    warnings_count = len(audit.get("warnings", []))
    lines.append(f"- **warnings count**: {warnings_count}")

    if audit.get("causal_claim_allowed"):
        lines.append(
            "\n*Causal claim is allowed: experiments are properly paired and "
            "visual encoders are byte-identical.*"
        )
    elif audit.get("paired_valid"):
        lines.append(
            "\n*Paired comparison is valid, but visual encoders differ. "
            "Causal claim is NOT allowed (correlational only).*"
        )
    else:
        lines.append(
            "\n*Experiments are not properly paired. Results are not "
            "directly comparable.*"
        )

    return "\n".join(lines) + "\n"


def section_raw_accuracy(ref_dual: dict, cand_dual: dict) -> str:
    """Build ## 2. Raw Noisy-Label Validation Accuracy."""
    ref_raw = ref_dual["raw_noisy_validation"]
    cand_raw = cand_dual["raw_noisy_validation"]

    ref_correct = micro_correct(ref_raw)
    cand_correct = micro_correct(cand_raw)
    total = ref_raw["num_samples"]
    raw_delta = cand_raw["micro_accuracy"] - ref_raw["micro_accuracy"]

    lines = ["## 2. Raw Noisy-Label Validation Accuracy\n"]
    lines.append(
        "| Model | Correct | Total | Micro Accuracy | Macro (present) | "
        "Macro (all 500) | Median Class | Bottom 10% |"
    )
    lines.append(
        "|-------|---------|-------|---------------|-----------------|"
        "----------------|--------------|------------|"
    )

    def _row(label, metrics):
        return (
            f"| {label} | {micro_correct(metrics)} | {metrics['num_samples']} | "
            f"{metrics['micro_accuracy']:.4f} | "
            f"{metrics['macro_accuracy_present_classes']:.4f} | "
            f"{metrics['macro_accuracy_all']:.4f} | "
            f"{metrics['median_class_accuracy']:.4f} | "
            f"{metrics['bottom_10pct_class_accuracy']:.4f} |"
        )

    lines.append(_row(f"**{ref_dual['experiment_name']}**", ref_raw))
    lines.append(_row(f"**{cand_dual['experiment_name']}**", cand_raw))

    delta_row = (
        f"| **Delta ({cand_dual['experiment_name']} - "
        f"{ref_dual['experiment_name']})** | "
        f"{cand_correct - ref_correct:+d} | — | "
        f"{raw_delta:+.4f} | "
        f"{cand_raw['macro_accuracy_present_classes'] - ref_raw['macro_accuracy_present_classes']:+.4f} | "
        f"{cand_raw['macro_accuracy_all'] - ref_raw['macro_accuracy_all']:+.4f} | "
        f"{cand_raw['median_class_accuracy'] - ref_raw['median_class_accuracy']:+.4f} | "
        f"{cand_raw['bottom_10pct_class_accuracy'] - ref_raw['bottom_10pct_class_accuracy']:+.4f} |"
    )
    lines.append(delta_row)
    lines.append("")

    return "\n".join(lines)


def section_trusted_accuracy(ref_dual: dict, cand_dual: dict) -> str:
    """Build ## 3. Trusted Validation Accuracy."""
    ref_trust = ref_dual["trusted_validation"]
    cand_trust = cand_dual["trusted_validation"]

    ref_correct = micro_correct(ref_trust)
    cand_correct = micro_correct(cand_trust)
    trusted_delta = cand_trust["micro_accuracy"] - ref_trust["micro_accuracy"]

    lines = ["## 3. Trusted Validation Accuracy\n"]
    lines.append(
        "| Model | Correct | Total (Trusted) | Coverage | Micro Accuracy | "
        "Macro (present) | Macro (all 500) |"
    )
    lines.append(
        "|-------|---------|-----------------|----------|---------------|"
        "-----------------|----------------|"
    )

    def _row(label, metrics):
        return (
            f"| {label} | {micro_correct(metrics)} | "
            f"{metrics['num_trusted_samples']} | "
            f"{metrics['coverage']:.4f} | "
            f"{metrics['micro_accuracy']:.4f} | "
            f"{metrics['macro_accuracy_present_classes']:.4f} | "
            f"{metrics['macro_accuracy_all']:.4f} |"
        )

    lines.append(_row(f"**{ref_dual['experiment_name']}**", ref_trust))
    lines.append(_row(f"**{cand_dual['experiment_name']}**", cand_trust))

    delta_row = (
        f"| **Delta ({cand_dual['experiment_name']} - "
        f"{ref_dual['experiment_name']})** | "
        f"{cand_correct - ref_correct:+d} | — | "
        f"{cand_trust['coverage'] - ref_trust['coverage']:.4f} | "
        f"{trusted_delta:+.4f} | "
        f"{cand_trust['macro_accuracy_present_classes'] - ref_trust['macro_accuracy_present_classes']:+.4f} | "
        f"{cand_trust['macro_accuracy_all'] - ref_trust['macro_accuracy_all']:+.4f} |"
    )
    lines.append(delta_row)
    lines.append("")

    return "\n".join(lines)


def section_four_group(summary: dict) -> str:
    """Build ## 4. Four-Group Composition."""
    ref_name = summary["reference_name"]
    cand_name = summary["candidate_name"]
    total = summary["total_samples"]

    both = summary["both_correct"]
    ref_only = summary[f"{ref_name}_only_correct"]
    cand_only = summary[f"{cand_name}_only_correct"]
    both_wrong = summary["both_wrong"]

    ref_minus_cand = summary.get(
        f"{ref_name}_only_minus_{cand_name}_only",
        ref_only - cand_only,
    )

    lines = ["## 4. Four-Group Composition\n"]
    lines.append(
        "| Group | Count | Percentage |"
    )
    lines.append(
        "|-------|-------|------------|"
    )

    groups = [
        ("both_correct", both),
        (f"{ref_name}_only_correct", ref_only),
        (f"{cand_name}_only_correct", cand_only),
        ("both_wrong", both_wrong),
    ]
    for label, count in groups:
        pct = 100.0 * count / total if total > 0 else 0.0
        lines.append(f"| {label} | {count} | {pct:.2f}% |")

    lines.append("")
    lines.append(
        f"**{ref_name}_only - {cand_name}_only = {ref_only} - {cand_only} = "
        f"{ref_minus_cand}**"
    )
    lines.append("")

    return "\n".join(lines)


def section_metric_comparison(
    sample_metrics: pd.DataFrame,
    summary: dict,
) -> str:
    """Build ## 5. Key Metric Comparison: d3_only_correct vs both_correct."""
    ref_name = summary["reference_name"]

    ref_only_group = f"{ref_name}_only_correct"
    both_group = "both_correct"

    # Filter groups
    ref_only_df = sample_metrics[sample_metrics["group"] == ref_only_group]
    both_df = sample_metrics[sample_metrics["group"] == both_group]

    metric_cols = [
        "knn_label_agreement",
        "prototype_margin",
        "clip_flip_cosine",
        "prototype_supports_noisy_label",
    ]

    lines = [
        "## 5. Key Metric Comparison: "
        f"`{ref_name}_only_correct` vs `both_correct`\n"
    ]
    lines.append(
        "| Metric | " + ref_only_group + " (mean) | both_correct (mean) | "
        "Delta | Interpretation |"
    )
    lines.append(
        "|--------|-------------------------|----------------------|"
        "-------|----------------|"
    )

    for col in metric_cols:
        if col not in sample_metrics.columns:
            logger.warning("Column '%s' not found in sample_metrics. Skipping.", col)
            continue

        ref_only_mean = ref_only_df[col].mean() if len(ref_only_df) > 0 else float("nan")
        both_mean = both_df[col].mean() if len(both_df) > 0 else float("nan")
        delta = ref_only_mean - both_mean

        if col == "prototype_supports_noisy_label":
            # Boolean column: mean = proportion True
            interp = (
                "Lower in d3_only → noisy-label samples concentrated "
                "in d3_only region"
                if delta < 0
                else (
                    "Higher or equal → d3_only samples not more noise-like "
                    "by this metric"
                )
            )
        elif col == "prototype_margin":
            interp = (
                "Lower margin in d3_only → less confident prototype "
                "assignment"
                if delta < 0
                else "Higher or equal margin"
            )
        elif col == "knn_label_agreement":
            interp = (
                "Lower kNN agreement in d3_only → more label noise"
                if delta < 0
                else "Higher or equal kNN agreement"
            )
        elif col == "clip_flip_cosine":
            interp = (
                "Lower flip agreement in d3_only → less visually stable"
                if delta < 0
                else "Higher or equal flip agreement"
            )
        else:
            interp = ""

        lines.append(
            f"| {col} | {ref_only_mean:.4f} | {both_mean:.4f} | "
            f"{delta:+.4f} | {interp} |"
        )

    lines.append("")

    # Add sample counts
    lines.append(
        f"*Note: {ref_only_group} has {len(ref_only_df)} samples, "
        f"{both_group} has {len(both_df)} samples.*\n"
    )

    return "\n".join(lines)


def section_b2_in_d3_only(
    sample_metrics: pd.DataFrame,
    summary: dict,
) -> str:
    """Build ## 6. B2 Predictions in d3_only_correct Region."""
    ref_name = summary["reference_name"]
    cand_name = summary["candidate_name"]

    ref_only_group = f"{ref_name}_only_correct"

    ref_only_df = sample_metrics[sample_metrics["group"] == ref_only_group]

    lines = [
        "## 6. " +
        f"{cand_name.title()} Predictions in `{ref_name}_only_correct` Region\n"
    ]

    # Comparison 1: knn_support_{cand_name}_pred vs knn_label_agreement
    knn_support_col = f"knn_support_{cand_name}_pred"
    if knn_support_col in sample_metrics.columns and "knn_label_agreement" in sample_metrics.columns:
        support_mean = ref_only_df[knn_support_col].mean()
        agreement_mean = ref_only_df["knn_label_agreement"].mean()
        lines.append(
            f"**kNN support for {cand_name} prediction vs kNN label agreement:**\n"
        )
        lines.append(f"- Mean `knn_support_{cand_name}_pred`: {support_mean:.4f}")
        lines.append(f"- Mean `knn_label_agreement`: {agreement_mean:.4f}")
        lines.append(
            f"- Difference: {support_mean - agreement_mean:+.4f}\n"
        )
        gap = support_mean - agreement_mean
        if gap > 0.05:
            lines.append(
                f"*{cand_name}'s new prediction in the {ref_name}_only region "
                f"receives substantially higher kNN support than the noisy label, "
                f"suggesting the new prediction is more consistent with the "
                f"training neighborhood.*\n"
            )
        elif gap > 0.0:
            lines.append(
                f"*Mildly higher kNN support for {cand_name}'s prediction vs the "
                f"noisy label.*\n"
            )
        else:
            lines.append(
                f"*kNN support for {cand_name}'s prediction does not exceed noisy-label "
                f"agreement in this region.*\n"
            )
    else:
        lines.append(f"*Column {knn_support_col} not available. Skipping.*\n")

    # Comparison 2: prototype_similarity_{cand_name}_pred vs prototype_label_similarity
    proto_sim_col = f"prototype_similarity_{cand_name}_pred"
    if proto_sim_col in sample_metrics.columns and "prototype_label_similarity" in sample_metrics.columns:
        proto_sim_mean = ref_only_df[proto_sim_col].mean()
        proto_label_mean = ref_only_df["prototype_label_similarity"].mean()
        lines.append(
            f"**Prototype similarity for {cand_name} prediction vs "
            f"prototype-label similarity:**\n"
        )
        lines.append(f"- Mean `{proto_sim_col}`: {proto_sim_mean:.4f}")
        lines.append(f"- Mean `prototype_label_similarity`: {proto_label_mean:.4f}")
        lines.append(
            f"- Difference: {proto_sim_mean - proto_label_mean:+.4f}\n"
        )
        gap = proto_sim_mean - proto_label_mean
        if gap > 0.05:
            lines.append(
                f"*{cand_name}'s new prediction in the {ref_name}_only region "
                f"has substantially higher prototype similarity than the noisy "
                f"label, consistent with noise-correcting behavior.*\n"
            )
        elif gap > 0.0:
            lines.append(
                f"*Mildly higher prototype similarity for {cand_name}'s "
                f"prediction.*\n"
            )
        else:
            lines.append(
                f"*Prototype similarity for {cand_name}'s prediction does not "
                f"exceed noisy-label similarity in this region.*\n"
            )
    else:
        lines.append(
            f"*Column {proto_sim_col} not available. Skipping.*\n"
        )

    return "\n".join(lines)


def section_conclusion(
    audit: dict,
    ref_dual: dict,
    cand_dual: dict,
) -> str:
    """Build ## 7. Conclusion."""
    lines = ["## 7. Conclusion\n"]

    causal_allowed = audit.get("causal_claim_allowed", False)
    if not causal_allowed:
        lines.append(
            "**Causal claim NOT allowed.** Results are correlational only.\n"
        )

    ref_raw = ref_dual["raw_noisy_validation"]
    cand_raw = cand_dual["raw_noisy_validation"]
    ref_trust = ref_dual["trusted_validation"]
    cand_trust = cand_dual["trusted_validation"]

    raw_delta = cand_raw["micro_accuracy"] - ref_raw["micro_accuracy"]
    trusted_delta = cand_trust["micro_accuracy"] - ref_trust["micro_accuracy"]

    lines.append(
        f"- Raw delta ({cand_dual['experiment_name']} - "
        f"{ref_dual['experiment_name']}): **{raw_delta:+.4f}**"
    )
    lines.append(
        f"- Trusted delta ({cand_dual['experiment_name']} - "
        f"{ref_dual['experiment_name']}): **{trusted_delta:+.4f}**\n"
    )

    if trusted_delta > 0 and raw_delta < 0:
        lines.append(
            "**Strong evidence: "
            f"{cand_dual['experiment_name']} gains on trusted subset "
            "despite raw loss.** "
            f"(trusted_delta={trusted_delta:+.4f} > 0, raw_delta={raw_delta:+.4f} < 0)\n"
        )
    elif trusted_delta > raw_delta:
        lines.append(
            "**Evidence supports: local noisy validation underestimates "
            f"{cand_dual['experiment_name']}.** "
            f"(trusted_delta={trusted_delta:+.4f} > raw_delta={raw_delta:+.4f})\n"
        )
    else:
        lines.append(
            "**Inconclusive:** trusted delta does not exceed raw delta. "
            f"(trusted_delta={trusted_delta:+.4f}, raw_delta={raw_delta:+.4f})\n"
        )

    if not causal_allowed:
        lines.append(
            "\n---\n"
            "*Note on correlation vs causation:* "
            "Because the two experiments do not share a byte-identical visual "
            "encoder, the accuracy differences could be driven by changes in "
            "the visual representation rather than the classifier head / loss "
            "function alone. A paired experiment with frozen visual encoder "
            "is required for causal attribution."
        )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Generate findings.md from all analysis outputs"
    )
    p.add_argument("--audit", required=True, help="Path to audit JSON")
    p.add_argument(
        "--sample-metrics",
        required=True,
        help="Path to sample_metrics CSV",
    )
    p.add_argument(
        "--group-summary",
        required=True,
        help="Path to group_summary JSON",
    )
    p.add_argument(
        "--trusted-summary",
        required=True,
        help="Path to trusted_subset_summary JSON",
    )
    p.add_argument(
        "--reference-dual",
        required=True,
        help="Path to reference experiment dual validation JSON",
    )
    p.add_argument(
        "--candidate-dual",
        required=True,
        help="Path to candidate experiment dual validation JSON",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to output findings.md",
    )
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    # ── Load all inputs ──
    logger.info("Loading audit: %s", args.audit)
    audit = load_json(args.audit)

    logger.info("Loading sample_metrics: %s", args.sample_metrics)
    sample_metrics = load_csv(args.sample_metrics)

    logger.info("Loading group_summary: %s", args.group_summary)
    group_summary = load_json(args.group_summary)

    logger.info("Loading trusted_summary: %s", args.trusted_summary)
    trusted_summary = load_json(args.trusted_summary)

    logger.info("Loading reference dual: %s", args.reference_dual)
    ref_dual = load_json(args.reference_dual)

    logger.info("Loading candidate dual: %s", args.candidate_dual)
    cand_dual = load_json(args.candidate_dual)

    # Verify dual validation schema versions
    for name, dv in [("reference", ref_dual), ("candidate", cand_dual)]:
        if dv.get("schema_version") != 1:
            logger.warning(
                "%s dual validation has unexpected schema_version=%s",
                name,
                dv.get("schema_version"),
            )

    # ── Build sections ──
    logger.info("Building report sections...")
    sections = []

    sections.append("# Disagreement Analysis Findings\n")
    sections.append(
        f"*Generated from: reference={ref_dual['experiment_name']}, "
        f"candidate={cand_dual['experiment_name']}*\n"
    )

    sections.append(section_protocol_audit(audit))
    sections.append(section_raw_accuracy(ref_dual, cand_dual))
    sections.append(section_trusted_accuracy(ref_dual, cand_dual))
    sections.append(section_four_group(group_summary))
    sections.append(section_metric_comparison(sample_metrics, group_summary))
    sections.append(
        section_b2_in_d3_only(sample_metrics, group_summary)
    )
    sections.append(section_conclusion(audit, ref_dual, cand_dual))

    # ── Write output ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(sections)
    output_path.write_text(report_text)
    logger.info("Findings report saved to: %s", output_path)

    # ── Print summary ──
    ref_name = group_summary["reference_name"]
    cand_name = group_summary["candidate_name"]
    logger.info("=" * 60)
    logger.info("FINDINGS SUMMARY")
    logger.info("=" * 60)
    logger.info("Reference: %s", ref_name)
    logger.info("Candidate: %s", cand_name)
    logger.info("Audit paired_valid: %s", audit.get("paired_valid"))
    logger.info("Audit causal_claim_allowed: %s", audit.get("causal_claim_allowed"))
    ref_raw = ref_dual["raw_noisy_validation"]
    cand_raw = cand_dual["raw_noisy_validation"]
    raw_delta = cand_raw["micro_accuracy"] - ref_raw["micro_accuracy"]
    ref_trust = ref_dual["trusted_validation"]
    cand_trust = cand_dual["trusted_validation"]
    trusted_delta = cand_trust["micro_accuracy"] - ref_trust["micro_accuracy"]
    logger.info("Raw delta: %+.4f", raw_delta)
    logger.info("Trusted delta: %+.4f", trusted_delta)
    logger.info("=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()
