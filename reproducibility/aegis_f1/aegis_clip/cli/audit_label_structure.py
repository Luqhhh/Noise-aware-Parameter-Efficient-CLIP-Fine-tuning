"""Audit OOF disagreement structure before choosing a denoising strategy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from aegis_clip.features import canonical_sample_path
from aegis_clip.runtime import atomic_json_dump, sha256_file


REQUIRED_QUALITY_COLUMNS = {
    "image_path",
    "original_label",
    "oof_top1",
    "p_top1",
    "top1_margin",
    "prototype_top1",
    "knn_top1",
    "knn_top1_agreement",
    "flip_consistency",
    "duplicate_conflict_flag",
}


def audit_label_structure(
    *,
    quality_path: str | Path,
    issues_path: str | Path,
    a2_manifest_path: str | Path,
    output_path: str | Path,
    num_classes: int = 500,
) -> dict:
    quality_file = Path(quality_path)
    issues_file = Path(issues_path)
    a2_file = Path(a2_manifest_path)
    quality = pd.read_csv(quality_file)
    missing = REQUIRED_QUALITY_COLUMNS - set(quality.columns)
    if missing:
        raise ValueError(f"Quality asset is missing columns: {sorted(missing)}")
    if len(quality) == 0 or quality["image_path"].duplicated().any():
        raise ValueError("Quality asset must contain unique, non-empty paths")
    labels = quality["original_label"].to_numpy(dtype=np.int64)
    oof = quality["oof_top1"].to_numpy(dtype=np.int64)
    knn = quality["knn_top1"].to_numpy(dtype=np.int64)
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError("Observed labels fall outside the declared class range")

    issues = pd.read_csv(issues_file)
    required_issues = {"index", "selected"}
    if required_issues - set(issues.columns):
        raise ValueError("Issues asset must contain index and selected")
    selected_issue_indices = issues.loc[
        issues["selected"].astype(bool), "index"
    ].to_numpy(dtype=np.int64)
    if len(selected_issue_indices) and (
        selected_issue_indices.min() < 0
        or selected_issue_indices.max() >= len(quality)
    ):
        raise ValueError("Issue indices do not align with the quality asset")
    issue_mask = np.zeros(len(quality), dtype=bool)
    issue_mask[selected_issue_indices] = True

    duplicate_free = ~quality["duplicate_conflict_flag"].astype(bool).to_numpy()
    flip_stable = quality["flip_consistency"].to_numpy(dtype=float) == 1.0
    prototype = quality["prototype_top1"].to_numpy(dtype=np.int64)
    kta = quality["knn_top1_agreement"].to_numpy(dtype=float)
    p_top1 = quality["p_top1"].to_numpy(dtype=float)
    margin = quality["top1_margin"].to_numpy(dtype=float)
    disagreement = oof != labels
    consensus_other = disagreement & (knn == oof) & duplicate_free
    kta_consensus = consensus_other & (kta >= 0.60)
    moderate_relabel = issue_mask & kta_consensus
    strict_relabel = (
        moderate_relabel
        & (prototype == oof)
        & flip_stable
        & (p_top1 >= 0.90)
        & (margin >= 0.70)
    )
    trusted_original = (
        (oof == labels)
        & (knn == labels)
        & (prototype == labels)
        & flip_stable
    )

    a2_manifest = pd.read_csv(a2_file)
    required_a2 = {"image_path", "training_role"}
    if required_a2 - set(a2_manifest.columns):
        raise ValueError("A2 manifest must contain image_path and training_role")
    rejected_paths = {
        canonical_sample_path(path)
        for path in a2_manifest.loc[
            a2_manifest["training_role"] == "rejected", "image_path"
        ]
    }
    quality_paths = np.array(
        [canonical_sample_path(path) for path in quality["image_path"]], dtype=object
    )
    a2_rejected = np.fromiter(
        (path in rejected_paths for path in quality_paths),
        dtype=bool,
        count=len(quality_paths),
    )
    if int(a2_rejected.sum()) != len(rejected_paths):
        raise ValueError("A2 rejected paths do not have one-to-one quality coverage")

    class_counts = np.bincount(labels, minlength=num_classes)
    disagree_counts = np.bincount(labels[disagreement], minlength=num_classes)
    dominant_shares = []
    normalized_entropies = []
    edge_counts: dict[tuple[int, int], int] = {}
    for source in range(num_classes):
        targets = oof[(labels == source) & disagreement]
        if len(targets) == 0:
            dominant_shares.append(0.0)
            normalized_entropies.append(0.0)
            continue
        counts = np.bincount(targets, minlength=num_classes)
        counts[source] = 0
        positive = counts[counts > 0]
        dominant_shares.append(float(positive.max() / positive.sum()))
        probabilities = positive / positive.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        normalized_entropies.append(
            entropy / math.log(max(len(positive), 2))
        )
        for target in np.flatnonzero(counts):
            edge_counts[(source, int(target))] = int(counts[target])

    sorted_edges = sorted(edge_counts.items(), key=lambda item: item[1], reverse=True)
    strict_labels = labels.copy()
    strict_labels[strict_relabel] = oof[strict_relabel]
    original_counts = np.bincount(labels, minlength=num_classes)
    corrected_counts = np.bincount(strict_labels, minlength=num_classes)
    flow = corrected_counts - original_counts

    def mask_summary(mask: np.ndarray) -> dict:
        per_class = np.bincount(labels[mask], minlength=num_classes)
        return {
            "count": int(mask.sum()),
            "fraction": float(mask.mean()),
            "source_classes": int((per_class > 0).sum()),
            "median_per_source_class": float(np.median(per_class)),
            "max_per_source_class": int(per_class.max()),
        }

    result = {
        "assets": {
            "quality_path": str(quality_file.resolve()),
            "quality_sha256": sha256_file(quality_file),
            "issues_path": str(issues_file.resolve()),
            "issues_sha256": sha256_file(issues_file),
            "a2_manifest_path": str(a2_file.resolve()),
            "a2_manifest_sha256": sha256_file(a2_file),
        },
        "sample_count": int(len(quality)),
        "num_classes": int(num_classes),
        "class_count_range": [int(class_counts.min()), int(class_counts.max())],
        "oof_disagreement": mask_summary(disagreement),
        "confident_joint_issue": mask_summary(issue_mask),
        "trusted_original": mask_summary(trusted_original),
        "oof_knn_consensus_other": mask_summary(consensus_other),
        "kta_consensus_other": mask_summary(kta_consensus),
        "moderate_relabel": mask_summary(moderate_relabel),
        "strict_relabel": mask_summary(strict_relabel),
        "a2_rejected": mask_summary(a2_rejected),
        "a2_rejected_overlap": {
            "with_moderate_relabel": int((a2_rejected & moderate_relabel).sum()),
            "with_strict_relabel": int((a2_rejected & strict_relabel).sum()),
            "with_oof_disagreement": int((a2_rejected & disagreement).sum()),
        },
        "disagreement_structure": {
            "median_class_disagreement_rate": float(
                np.median(disagree_counts / np.maximum(class_counts, 1))
            ),
            "median_dominant_alternative_share": float(np.median(dominant_shares)),
            "p90_dominant_alternative_share": float(
                np.quantile(dominant_shares, 0.90)
            ),
            "median_normalized_alternative_entropy": float(
                np.median(normalized_entropies)
            ),
            "unique_directed_edges": int(len(edge_counts)),
            "top_edges": [
                {"source": source, "target": target, "count": count}
                for (source, target), count in sorted_edges[:20]
            ],
        },
        "strict_relabel_flow": {
            "l1_count_shift": int(np.abs(flow).sum()),
            "max_absolute_class_shift": int(np.abs(flow).max()),
            "classes_changed": int((flow != 0).sum()),
            "net_shift": int(flow.sum()),
        },
    }
    atomic_json_dump(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--issues", required=True)
    parser.add_argument("--a2-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-classes", type=int, default=500)
    args = parser.parse_args()
    result = audit_label_structure(
        quality_path=args.quality,
        issues_path=args.issues,
        a2_manifest_path=args.a2_manifest,
        output_path=args.output,
        num_classes=args.num_classes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
