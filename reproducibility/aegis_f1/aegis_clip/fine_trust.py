"""Cross-fitted FINE geometry scores and conflict-aware trust capping."""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from aegis_clip.features import FrozenFeatureStore, canonical_sample_path
from aegis_clip.runtime import atomic_json_dump, sha256_file


def cross_fitted_fine_scores(
    features: torch.Tensor,
    labels: torch.Tensor,
    fold_ids: torch.Tensor,
    *,
    num_classes: int,
    power_iterations: int = 20,
) -> torch.Tensor:
    """Score each sample against a class direction that never saw its fold."""
    features = F.normalize(torch.as_tensor(features).float(), dim=1)
    labels = torch.as_tensor(labels).long().flatten()
    fold_ids = torch.as_tensor(fold_ids).long().flatten()
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features and labels must have matching first dimensions")
    if len(fold_ids) != len(labels):
        raise ValueError("fold_ids and labels must have matching lengths")
    if power_iterations < 1:
        raise ValueError("power_iterations must be positive")
    if labels.numel() == 0 or labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError("labels must cover valid class indices")
    folds = torch.unique(fold_ids, sorted=True)
    if len(folds) < 2:
        raise ValueError("cross-fitted FINE requires at least two folds")

    scores = torch.full((len(labels),), float("nan"), dtype=torch.float32)
    for class_index in range(num_classes):
        class_mask = labels == class_index
        if not class_mask.any():
            raise ValueError(f"class {class_index} has no samples")
        for fold in folds.tolist():
            held_out = class_mask & (fold_ids == fold)
            if not held_out.any():
                continue
            fitted = class_mask & (fold_ids != fold)
            if not fitted.any():
                raise ValueError(
                    f"class {class_index} fold {fold} has no fit samples"
                )
            fit_features = features[fitted]
            direction = fit_features.mean(dim=0)
            direction = direction / direction.norm().clamp_min(1.0e-12)
            for _ in range(power_iterations):
                direction = fit_features.T.mv(fit_features.mv(direction))
                direction = direction / direction.norm().clamp_min(1.0e-12)
            scores[held_out] = features[held_out].mv(direction).square()
    if not torch.isfinite(scores).all():
        raise ValueError("cross-fitted FINE left non-finite scores")
    return scores.clamp(0.0, 1.0)


def conflict_geometry_cap(
    clean_probability: torch.Tensor,
    labels: torch.Tensor,
    prototype_top1: torch.Tensor,
    probe_top1: torch.Tensor,
    fine_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap trust only when two OOF views agree on the same alternative class."""
    clean = torch.as_tensor(clean_probability).float().flatten()
    labels = torch.as_tensor(labels).long().flatten()
    prototype = torch.as_tensor(prototype_top1).long().flatten()
    probe = torch.as_tensor(probe_top1).long().flatten()
    scores = torch.as_tensor(fine_scores).float().flatten()
    lengths = {len(clean), len(labels), len(prototype), len(probe), len(scores)}
    if len(lengths) != 1:
        raise ValueError("conflict cap inputs must have equal lengths")
    if not torch.isfinite(clean).all() or not torch.isfinite(scores).all():
        raise ValueError("conflict cap inputs must be finite")
    conflict = (prototype == probe) & (prototype != labels)
    capped = torch.where(conflict, torch.minimum(clean, scores), clean)
    return capped.clamp(0.0, 1.0), conflict


def build_fine_conflict_cap_bundle(
    *,
    source_csv: str | Path,
    base_bundle_path: str | Path,
    feature_tensor_path: str | Path,
    feature_paths_path: str | Path,
    feature_manifest_path: str | Path,
    output_bundle_path: str | Path,
    output_audit_path: str | Path,
    num_classes: int = 500,
    power_iterations: int = 20,
    expected_samples: int | None = None,
    expected_conflicts: int | None = None,
    expected_changed: int | None = None,
) -> dict[str, Any]:
    """Build an auditable bundle while leaving non-source trust entries untouched."""
    source_csv = Path(source_csv).resolve()
    base_bundle_path = Path(base_bundle_path).resolve()
    frame = pd.read_csv(source_csv)
    if {"image_path", "label"} - set(frame):
        raise ValueError("source CSV must contain image_path and label")
    if frame["image_path"].duplicated().any():
        raise ValueError("source CSV contains duplicate image paths")
    if expected_samples is not None and len(frame) != int(expected_samples):
        raise ValueError(f"source has {len(frame)} samples, expected {expected_samples}")

    store = FrozenFeatureStore(
        feature_tensor_path,
        feature_paths_path,
        feature_manifest_path,
    )
    source_paths = [canonical_sample_path(p) for p in frame["image_path"].astype(str)]
    store.verify_coverage(source_paths)
    source_features = store.get_many(source_paths)
    labels = torch.tensor(frame["label"].astype(int).tolist(), dtype=torch.long)

    payload = torch.load(base_bundle_path, map_location="cpu", weights_only=False)
    required = {"paths", "clean_probability", "diagnostics"}
    if required - set(payload):
        raise ValueError("base bundle lacks trust or diagnostic fields")
    diagnostics = payload["diagnostics"]
    diagnostic_required = {"fold_id", "prototype_top1", "probe_top1"}
    if diagnostic_required - set(diagnostics):
        raise ValueError("base bundle lacks OOF conflict diagnostics")
    bundle_paths = [canonical_sample_path(p) for p in payload["paths"]]
    if len(set(bundle_paths)) != len(bundle_paths):
        raise ValueError("base bundle contains duplicate canonical paths")
    bundle_index = {path: index for index, path in enumerate(bundle_paths)}
    missing = [path for path in source_paths if path not in bundle_index]
    if missing:
        raise ValueError(f"base bundle misses source path: {missing[0]}")
    source_indices = torch.tensor([bundle_index[p] for p in source_paths])

    fine_scores = cross_fitted_fine_scores(
        source_features,
        labels,
        diagnostics["fold_id"][source_indices],
        num_classes=num_classes,
        power_iterations=power_iterations,
    )
    old_clean = payload["clean_probability"][source_indices].float()
    new_clean, conflict = conflict_geometry_cap(
        old_clean,
        labels,
        diagnostics["prototype_top1"][source_indices],
        diagnostics["probe_top1"][source_indices],
        fine_scores,
    )
    changed = new_clean < old_clean
    if expected_conflicts is not None and int(conflict.sum()) != expected_conflicts:
        raise ValueError(
            f"found {int(conflict.sum())} conflicts, expected {expected_conflicts}"
        )
    if expected_changed is not None and int(changed.sum()) != expected_changed:
        raise ValueError(
            f"changed {int(changed.sum())} samples, expected {expected_changed}"
        )

    output = copy.deepcopy(payload)
    output["clean_probability"] = output["clean_probability"].clone()
    output["clean_probability"][source_indices] = new_clean
    output.setdefault("metadata", {})["fine_conflict_cap"] = {
        "method": "cross_fitted_fine_conflict_cap_v1",
        "source_csv": str(source_csv),
        "source_samples": len(frame),
        "num_classes": int(num_classes),
        "power_iterations": int(power_iterations),
        "conflicts": int(conflict.sum()),
        "changed": int(changed.sum()),
        "base_bundle_sha256": sha256_file(base_bundle_path),
    }
    full_scores = torch.full_like(output["clean_probability"], float("nan"))
    full_conflict = torch.zeros_like(output["clean_probability"], dtype=torch.bool)
    full_scores[source_indices] = fine_scores
    full_conflict[source_indices] = conflict
    output.setdefault("diagnostics", {})["fine_alignment_score"] = full_scores
    output["diagnostics"]["fine_consensus_conflict"] = full_conflict

    destination = Path(output_bundle_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(output, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    audit = {
        "status": "passed",
        "method": "cross_fitted_fine_conflict_cap_v1",
        "source_samples": len(frame),
        "classes": int(labels.unique().numel()),
        "folds": int(diagnostics["fold_id"][source_indices].unique().numel()),
        "power_iterations": int(power_iterations),
        "conflicts": int(conflict.sum()),
        "changed": int(changed.sum()),
        "changed_fraction": float(changed.float().mean()),
        "mean_old_clean": float(old_clean.mean()),
        "mean_new_clean": float(new_clean.mean()),
        "mean_fine_score": float(fine_scores.mean()),
        "mean_conflict_fine_score": float(fine_scores[conflict].mean()),
        "source_csv_sha256": sha256_file(source_csv),
        "base_bundle_sha256": sha256_file(base_bundle_path),
        "feature_tensor_sha256": sha256_file(feature_tensor_path),
        "feature_paths_sha256": sha256_file(feature_paths_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "output_bundle_sha256": sha256_file(destination),
    }
    atomic_json_dump(audit, output_audit_path)
    return audit
