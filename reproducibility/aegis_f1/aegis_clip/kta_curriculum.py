"""Build a high-precision KTA anchor bundle for cyclic noisy-label training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from aegis_clip.features import canonical_sample_path
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.trust import atomic_torch_save


REQUIRED_COLUMNS = {
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


def build_kta_curriculum_bundle(
    *,
    base_bundle_path: str | Path,
    quality_path: str | Path,
    issues_path: str | Path,
    output_bundle_path: str | Path,
    output_manifest_path: str | Path,
) -> dict[str, Any]:
    """Replace only strict cross-fitted consensus labels and protect anchors.

    The base clean probabilities remain unchanged except for two high-precision
    groups: three-view original-label agreement is promoted to a protected
    anchor, while strict OOF/kNN/prototype consensus is hard-corrected and also
    promoted. All pre-existing soft corrections are cleared so the resulting
    bundle has exactly one auditable correction source.
    """
    base_file = Path(base_bundle_path)
    quality_file = Path(quality_path)
    issues_file = Path(issues_path)
    output_file = Path(output_bundle_path)
    manifest_file = Path(output_manifest_path)

    base = torch.load(base_file, map_location="cpu", weights_only=False)
    required_bundle = {
        "paths",
        "clean_probability",
        "pseudo_label",
        "pseudo_confidence",
        "correction_alpha",
    }
    if required_bundle - set(base):
        raise ValueError("Base trust bundle is incomplete")
    paths = [canonical_sample_path(path) for path in base["paths"]]
    path_to_index = {path: index for index, path in enumerate(paths)}
    if len(path_to_index) != len(paths):
        raise ValueError("Base trust bundle contains duplicate canonical paths")

    quality = pd.read_csv(quality_file)
    missing = REQUIRED_COLUMNS - set(quality.columns)
    if missing:
        raise ValueError(f"Quality asset is missing columns: {sorted(missing)}")
    quality_paths = [canonical_sample_path(path) for path in quality["image_path"]]
    if len(set(quality_paths)) != len(quality_paths):
        raise ValueError("Quality asset contains duplicate canonical paths")
    missing_paths = sorted(set(quality_paths) - set(path_to_index))
    if missing_paths:
        raise ValueError(f"Base bundle misses quality path: {missing_paths[0]}")

    issues = pd.read_csv(issues_file)
    if {"index", "selected"} - set(issues.columns):
        raise ValueError("Issues asset must contain index and selected")
    selected = set(
        issues.loc[issues["selected"].astype(bool), "index"].astype(int).tolist()
    )

    clean_probability = torch.as_tensor(
        base["clean_probability"], dtype=torch.float32
    ).clone()
    pseudo_label = torch.full_like(
        torch.as_tensor(base["pseudo_label"], dtype=torch.long), -1
    )
    pseudo_confidence = torch.zeros_like(
        torch.as_tensor(base["pseudo_confidence"], dtype=torch.float32)
    )
    correction_alpha = torch.zeros_like(
        torch.as_tensor(base["correction_alpha"], dtype=torch.float32)
    )

    trusted_original_paths: list[str] = []
    strict_rows: list[dict[str, Any]] = []
    rescued_rejects = 0
    for row_index, row in quality.iterrows():
        path = quality_paths[row_index]
        bundle_index = path_to_index[path]
        observed = int(row["original_label"])
        oof = int(row["oof_top1"])
        knn = int(row["knn_top1"])
        prototype = int(row["prototype_top1"])
        duplicate_free = not bool(row["duplicate_conflict_flag"])
        flip_stable = float(row["flip_consistency"]) == 1.0

        trusted_original = (
            oof == observed
            and knn == observed
            and prototype == observed
            and flip_stable
        )
        if trusted_original:
            clean_probability[bundle_index] = 1.0
            trusted_original_paths.append(path)

        strict = (
            row_index in selected
            and oof != observed
            and knn == oof
            and prototype == oof
            and duplicate_free
            and flip_stable
            and float(row["knn_top1_agreement"]) >= 0.60
            and float(row["p_top1"]) >= 0.90
            and float(row["top1_margin"]) >= 0.70
        )
        if not strict:
            continue
        if float(clean_probability[bundle_index]) == 0.0:
            rescued_rejects += 1
        confidence = float(
            (float(row["p_top1"]) * float(row["knn_top1_agreement"])) ** 0.5
        )
        clean_probability[bundle_index] = 1.0
        pseudo_label[bundle_index] = oof
        pseudo_confidence[bundle_index] = confidence
        correction_alpha[bundle_index] = 1.0
        strict_rows.append(
            {
                "image_path": path,
                "original_label": observed,
                "corrected_label": oof,
                "p_top1": float(row["p_top1"]),
                "top1_margin": float(row["top1_margin"]),
                "knn_top1_agreement": float(row["knn_top1_agreement"]),
                "pseudo_confidence": confidence,
            }
        )

    output = dict(base)
    output["paths"] = paths
    output["clean_probability"] = clean_probability
    output["pseudo_label"] = pseudo_label
    output["pseudo_confidence"] = pseudo_confidence
    output["correction_alpha"] = correction_alpha
    output["metadata"] = {
        **dict(base.get("metadata", {})),
        "kta_cyclic_anchor": {
            "method": "strict_oof_knn_prototype_flip_anchor_v1",
            "strict_corrected": len(strict_rows),
            "rescued_a2_rejects": rescued_rejects,
            "trusted_original": len(trusted_original_paths),
            "quality_sha256": sha256_file(quality_file),
            "issues_sha256": sha256_file(issues_file),
            "base_bundle_sha256": sha256_file(base_file),
        },
    }
    atomic_torch_save(output, output_file)
    correction_csv = output_file.with_suffix(".corrections.csv")
    pd.DataFrame(strict_rows).to_csv(correction_csv, index=False)
    manifest = {
        "method": "strict_oof_knn_prototype_flip_anchor_v1",
        "samples": len(paths),
        "quality_samples": len(quality),
        "strict_corrected": len(strict_rows),
        "rescued_a2_rejects": rescued_rejects,
        "trusted_original": len(trusted_original_paths),
        "corrected_source_classes": len(
            {row["original_label"] for row in strict_rows}
        ),
        "corrected_target_classes": len(
            {row["corrected_label"] for row in strict_rows}
        ),
        "output_bundle": str(output_file.resolve()),
        "output_bundle_sha256": sha256_file(output_file),
        "correction_csv": str(correction_csv.resolve()),
        "correction_csv_sha256": sha256_file(correction_csv),
        "base_bundle_sha256": sha256_file(base_file),
        "quality_sha256": sha256_file(quality_file),
        "issues_sha256": sha256_file(issues_file),
    }
    atomic_json_dump(manifest, manifest_file)
    return manifest
