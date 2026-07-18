"""Build purification manifests from OOF predictions.

Modes:
  cl_classwise_drop  — confident-joint per-class capped drop
  cl_knn_drop        — confident-joint ∩ OOF/kNN consensus drop
  consensus_relabel  — three-signal consensus high-confidence relabel

Output: outputs/phase4/purification/<experiment_id>/
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from analysis.noisy_labels.confident_joint import (
    build_confident_joint,
    estimate_class_thresholds,
    rank_label_issues,
)
from analysis.noisy_labels.consensus import (
    select_consensus_relabel_v2,
    select_consensus_drop,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_quality_data(sample_quality_csv: str, oof_logits_pt: str) -> tuple:
    """Load quality CSV and OOF logits, returning probabilities."""
    quality = pd.read_csv(sample_quality_csv)
    logits_data = torch.load(oof_logits_pt, map_location="cpu")
    logits = logits_data["logits"].float()
    probs = F.softmax(logits, dim=1)
    return quality, probs


def _build_cl_classwise_drop(
    quality: pd.DataFrame,
    probs: torch.Tensor,
    output_dir: Path,
    max_class_reject_rate: float,
    max_global_reject_rate: float,
) -> pd.DataFrame:
    """Build purification manifest via class-conditional confident joint drop."""
    labels = torch.tensor(quality["original_label"].to_numpy(copy=True))
    num_classes = int(labels.max()) + 1

    thresholds = estimate_class_thresholds(probs, labels, num_classes)
    cj = build_confident_joint(probs, labels, thresholds, num_classes)

    # Build signals from quality DataFrame
    knn = quality["knn_agreement"].to_numpy(copy=True) if "knn_agreement" in quality.columns else None
    flip = quality["flip_consistency"].to_numpy(copy=True) if "flip_consistency" in quality.columns else None
    margin = quality["top1_margin"].to_numpy(copy=True) if "top1_margin" in quality.columns else None

    issues = rank_label_issues(
        probs, labels, thresholds, cj,
        max_class_reject_rate=max_class_reject_rate,
        max_global_reject_rate=max_global_reject_rate,
        knn_agreement=knn,
        flip_consistency=flip,
        top1_margin=margin,
    )
    return _build_manifest_from_issues(quality, issues)


def _build_manifest_from_issues(
    quality: pd.DataFrame,
    issues: pd.DataFrame,
) -> pd.DataFrame:
    """Convert issue selections to a full purification manifest."""
    rows = []
    selected_set = set(issues[issues["selected"]]["index"].values)

    for i, (_, r) in enumerate(quality.iterrows()):
        role = "rejected" if i in selected_set else "clean"
        weight = 0.0 if role == "rejected" else 1.0
        rows.append({
            "sample_id": r["sample_id"],
            "image_path": r["image_path"],
            "original_label": int(r["original_label"]),
            "training_label": int(r["original_label"]),
            "sample_weight": weight,
            "quality_score": float(r.get("quality", r.get("p_original_label", 0.5))),
            "training_role": role,
            "selection_reason": _get_selection_reason(i, issues),
            "suggested_label": int(r.get("suggested_label", r["original_label"])),
            "oof_top1": int(r.get("oof_top1", r["original_label"])),
            "p_original_label": float(r.get("p_original_label", 0.5)),
            "p_top1": float(r.get("p_top1", 0.5)),
            "top1_margin": float(r.get("top1_margin", 0.0)),
            "prototype_top1": int(r.get("prototype_top1", r["original_label"])),
            "prototype_margin": float(r.get("prototype_margin", 0.0)),
            "knn_top1": int(r.get("knn_top1", r["original_label"])),
            "knn_agreement": float(r.get("knn_agreement", 0.5)),
            "flip_consistency": float(r.get("flip_consistency", 1.0)),
        })

    df = pd.DataFrame(rows)
    return df


def _get_selection_reason(idx: int, issues: pd.DataFrame) -> str:
    sel = issues[issues["index"] == idx]
    if len(sel) > 0 and sel.iloc[0]["selected"]:
        return f"confident_joint_issue_score={sel.iloc[0]['score']:.4f}"
    return "clean"


def _build_partition_metrics(df: pd.DataFrame) -> dict:
    """Compute partition_metrics.json from manifest."""
    role_counts = df["training_role"].value_counts()
    n = len(df)
    num_classes = df["original_label"].nunique()
    clean_counts = df[df["training_role"] == "clean"].groupby("original_label").size()
    zero_clean = sorted(
        [int(c) for c in range(num_classes) if c not in clean_counts.index]
    )

    class_reject_rates = []
    for c in range(num_classes):
        cls = df[df["original_label"] == c]
        if len(cls) > 0:
            class_reject_rates.append(
                float((cls["training_role"] == "rejected").mean())
            )

    return {
        "clean_count": int(role_counts.get("clean", 0)),
        "rejected_count": int(role_counts.get("rejected", 0)),
        "pseudo_count": int(role_counts.get("pseudo", 0)),
        "global_reject_rate": float((df["training_role"] == "rejected").mean()),
        "global_relabel_rate": float((df["training_role"] == "pseudo").mean()),
        "max_class_reject_rate": float(max(class_reject_rates)) if class_reject_rates else 0.0,
        "max_class_relabel_rate": float(
            max(
                (df[df["original_label"] == c]["training_role"] == "pseudo").mean()
                for c in range(df["original_label"].nunique())
                if (df["original_label"] == c).sum() > 0
            )
        ) if (df["training_role"] == "pseudo").any() else 0.0,
        "classes_with_zero_clean_samples": zero_clean,
        "manifest_coverage": 1.0,
    }


def _write_outputs(df: pd.DataFrame, output_dir: Path, mode: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    # purification_manifest.csv
    manifest_path = output_dir / "purification_manifest.csv"
    df.to_csv(manifest_path, index=False)

    # partition_metrics.json
    metrics = _build_partition_metrics(df)
    (output_dir / "partition_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # protocol_audit.json
    training_allowed = (
        len(metrics["classes_with_zero_clean_samples"]) == 0
        and metrics["global_reject_rate"] <= 0.10
        and metrics["manifest_coverage"] == 1.0
    )
    audit = {
        "mode": mode,
        "training_allowed": training_allowed,
        "manifest_sha256": _sha256(manifest_path),
        **metrics,
    }

    # Save confident joint
    cj_path = output_dir / "confident_joint.npy"
    if "cj" in dir():
        np.save(cj_path, cj)
        audit["confident_joint_path"] = str(cj_path)
    else:
        audit["confident_joint_path"] = None

    # Save artifact manifest
    import hashlib, json as _json
    manifest_path = output_dir / "purification_manifest.csv"
    artifact = {
        "schema_version": 1,
        "manifest_sha256": _sha256(manifest_path),
        "created_at": _json.dumps(str(pd.Timestamp.now())),
    }
    (output_dir / "artifact_manifest.json").write_text(_json.dumps(artifact, indent=2))

    (output_dir / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print(f"Wrote {len(df)} rows to {output_dir}")
    print(json.dumps(audit, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=["cl_classwise_drop", "cl_knn_drop", "consensus_relabel_v2"])
    parser.add_argument("--sample-quality",
                        default="outputs/phase/phase3/oof/sample_quality_with_kta.csv")
    parser.add_argument("--oof-logits",
                        default="outputs/phase/phase3/oof/oof_logits.pt")
    parser.add_argument("--strict-train",
                        default="outputs/data/d3_strict/seed42/train.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-class-reject-rate", type=float, default=0.10)
    parser.add_argument("--max-global-reject-rate", type=float, default=0.10)
    parser.add_argument("--target-relabel-fraction", type=float, default=0.01)
    args = parser.parse_args()

    quality, probs = _load_quality_data(args.sample_quality, args.oof_logits)
    output_dir = Path(args.output_dir)

    if args.mode in ("cl_classwise_drop", "cl_knn_drop"):
        # Build issues via confident joint
        labels = torch.tensor(quality["original_label"].to_numpy(copy=True))
        num_classes = int(labels.max()) + 1
        thresholds = estimate_class_thresholds(probs, labels, num_classes)
        cj = build_confident_joint(probs, labels, thresholds, num_classes)

        knn = quality["knn_agreement"].to_numpy(copy=True) if "knn_agreement" in quality.columns else None
        flip = quality["flip_consistency"].to_numpy(copy=True) if "flip_consistency" in quality.columns else None
        margin = quality["top1_margin"].to_numpy(copy=True) if "top1_margin" in quality.columns else None

        issues = rank_label_issues(
            probs, labels, thresholds, cj,
            max_class_reject_rate=args.max_class_reject_rate,
            max_global_reject_rate=args.max_global_reject_rate,
            knn_agreement=knn,
            flip_consistency=flip,
            top1_margin=margin,
        )

        if args.mode == "cl_knn_drop":
            selected = select_consensus_drop(quality, issues)
            issue_rows = []
            for i, (_, r) in enumerate(quality.iterrows()):
                role = "rejected" if i in selected else "clean"
                weight = 0.0 if role == "rejected" else 1.0
                issue_rows.append({
                    "sample_id": r["sample_id"],
                    "image_path": r["image_path"],
                    "original_label": int(r["original_label"]),
                    "training_label": int(r["original_label"]),
                    "sample_weight": weight,
                    "quality_score": float(r.get("p_original_label", 0.5)),
                    "training_role": role,
                    "selection_reason": "knn_consensus_drop" if role == "rejected" else "clean",
                    "suggested_label": int(r.get("oof_top1", r["original_label"])),
                    "oof_top1": int(r.get("oof_top1", r["original_label"])),
                    "p_original_label": float(r.get("p_original_label", 0.5)),
                    "p_top1": float(r.get("p_top1", 0.5)),
                    "top1_margin": float(r.get("top1_margin", 0.0)),
                    "prototype_top1": int(r.get("prototype_top1", r["original_label"])),
                    "prototype_margin": float(r.get("prototype_margin", 0.0)),
                    "knn_top1": int(r.get("knn_top1", r["original_label"])),
                    "knn_agreement": float(r.get("knn_agreement", 0.5)),
                    "flip_consistency": float(r.get("flip_consistency", 1.0)),
                })
            df = pd.DataFrame(issue_rows)
        else:
            df = _build_manifest_from_issues(quality, issues)


    elif args.mode == "consensus_relabel_v2":
        labels = torch.tensor(quality["original_label"].to_numpy(copy=True))
        num_classes = int(labels.max()) + 1
        thresholds = estimate_class_thresholds(probs, labels, num_classes)
        cj = build_confident_joint(probs, labels, thresholds, num_classes)

        knn = quality["knn_agreement"].to_numpy(copy=True) if "knn_agreement" in quality.columns else None
        flip = quality["flip_consistency"].to_numpy(copy=True) if "flip_consistency" in quality.columns else None
        margin = quality["top1_margin"].to_numpy(copy=True) if "top1_margin" in quality.columns else None

        issues = rank_label_issues(
            probs, labels, thresholds, cj,
            max_class_reject_rate=0.10,
            max_global_reject_rate=0.10,
            knn_agreement=knn,
            flip_consistency=flip,
            top1_margin=margin,
        )

        # For v2: if fraction > 1, treat as absolute count; otherwise as fraction
        if args.target_relabel_fraction >= 1:
            top_k = int(args.target_relabel_fraction)
        else:
            top_k = max(1, int(args.target_relabel_fraction * len(quality)))
        pseudo_set = select_consensus_relabel_v2(
            quality, issues, top_k=top_k,
        )

        rows = []
        for i, (_, r) in enumerate(quality.iterrows()):
            if i in pseudo_set:
                rows.append({
                    "sample_id": r["sample_id"],
                    "image_path": r["image_path"],
                    "original_label": int(r["original_label"]),
                    "training_label": int(r["oof_top1"]),
                    "sample_weight": 1.0,
                    "quality_score": float(r.get("p_original_label", 0.5)),
                    "training_role": "pseudo",
                    "selection_reason": f"consensus_relabel_v2",
                    "suggested_label": int(r["oof_top1"]),
                    "oof_top1": int(r.get("oof_top1", r["original_label"])),
                    "p_original_label": float(r.get("p_original_label", 0.5)),
                    "p_top1": float(r.get("p_top1", 0.5)),
                    "top1_margin": float(r.get("top1_margin", 0.0)),
                    "prototype_top1": int(r.get("prototype_top1", r["original_label"])),
                    "prototype_margin": float(r.get("prototype_margin", 0.0)),
                    "knn_top1": int(r.get("knn_top1", r["original_label"])),
                    "knn_agreement": float(r.get("knn_agreement", 0.5)),
                    "flip_consistency": float(r.get("flip_consistency", 1.0)),
                })
            else:
                rows.append({
                    "sample_id": r["sample_id"],
                    "image_path": r["image_path"],
                    "original_label": int(r["original_label"]),
                    "training_label": int(r["original_label"]),
                    "sample_weight": 1.0,
                    "quality_score": float(r.get("p_original_label", 0.5)),
                    "training_role": "clean",
                    "selection_reason": "clean",
                    "suggested_label": int(r.get("oof_top1", r["original_label"])),
                    "oof_top1": int(r.get("oof_top1", r["original_label"])),
                    "p_original_label": float(r.get("p_original_label", 0.5)),
                    "p_top1": float(r.get("p_top1", 0.5)),
                    "top1_margin": float(r.get("top1_margin", 0.0)),
                    "prototype_top1": int(r.get("prototype_top1", r["original_label"])),
                    "prototype_margin": float(r.get("prototype_margin", 0.0)),
                    "knn_top1": int(r.get("knn_top1", r["original_label"])),
                    "knn_agreement": float(r.get("knn_agreement", 0.5)),
                    "flip_consistency": float(r.get("flip_consistency", 1.0)),
                })
        df = pd.DataFrame(rows)

    _write_outputs(df, output_dir, args.mode)


if __name__ == "__main__":
    main()
