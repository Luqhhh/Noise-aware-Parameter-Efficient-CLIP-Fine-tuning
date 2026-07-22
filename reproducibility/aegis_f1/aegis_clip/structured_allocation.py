"""Cross-fitted, class-marginal-constrained training-label allocation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from aegis_clip.features import canonical_sample_path
from aegis_clip.runtime import atomic_json_dump, sha256_file


REQUIRED_QUALITY_COLUMNS = {
    "sample_id",
    "original_label",
    "oof_top1",
    "prototype_top1",
    "knn_top1",
    "flip_consistency",
    "knn_top1_agreement",
}


def log_sinkhorn_allocation(
    logits: torch.Tensor,
    target_counts: torch.Tensor,
    *,
    temperature: float = 1.0,
    iterations: int = 100,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Allocate unit row mass to exact positive class marginals in log space."""
    if logits.ndim != 2:
        raise ValueError("logits must be rank two")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if int(iterations) < 1:
        raise ValueError("iterations must be positive")
    counts = torch.as_tensor(
        target_counts, device=logits.device, dtype=torch.float32
    ).flatten()
    if counts.numel() != logits.shape[1]:
        raise ValueError("target_counts must contain one value per class")
    if (counts <= 0).any():
        raise ValueError("every residual target count must be positive")
    if not math.isclose(
        float(counts.sum()), float(logits.shape[0]), rel_tol=0.0, abs_tol=1.0e-4
    ):
        raise ValueError("target_counts must sum to the number of rows")

    log_kernel = logits.detach().float() / float(temperature)
    log_target = counts.log()
    log_v = torch.zeros_like(log_target)
    log_u = torch.zeros(logits.shape[0], device=logits.device)
    for _ in range(int(iterations)):
        log_u = -torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_target - torch.logsumexp(
            log_kernel + log_u.unsqueeze(1), dim=0
        )
    allocation = torch.exp(
        log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    )
    row_error = (allocation.sum(dim=1) - 1.0).abs()
    column_error = (allocation.sum(dim=0) - counts).abs()
    diagnostics: dict[str, float | int] = {
        "iterations": int(iterations),
        "temperature": float(temperature),
        "maximum_row_absolute_error": float(row_error.max()),
        "mean_row_absolute_error": float(row_error.mean()),
        "maximum_column_absolute_error": float(column_error.max()),
        "mean_column_absolute_error": float(column_error.mean()),
    }
    return allocation, diagnostics


def classwise_curriculum_selection(
    assigned_labels: torch.Tensor,
    reliability: torch.Tensor,
    eligible: torch.Tensor,
    quotas: torch.Tensor,
) -> torch.Tensor:
    """Take the most reliable eligible rows independently per assigned class."""
    assigned = torch.as_tensor(assigned_labels, dtype=torch.long).flatten().cpu()
    scores = torch.as_tensor(reliability, dtype=torch.float32).flatten().cpu()
    allowed = torch.as_tensor(eligible, dtype=torch.bool).flatten().cpu()
    quota = torch.as_tensor(quotas, dtype=torch.long).flatten().cpu()
    if not (assigned.numel() == scores.numel() == allowed.numel()):
        raise ValueError("assigned_labels, reliability, and eligible must align")
    if (assigned < 0).any() or (assigned >= quota.numel()).any():
        raise ValueError("assigned label lies outside quota range")
    if (quota < 0).any():
        raise ValueError("quotas must be non-negative")
    selected = torch.zeros_like(allowed)
    for class_index in range(quota.numel()):
        members = torch.nonzero(
            allowed & (assigned == class_index), as_tuple=False
        ).flatten()
        take = min(int(quota[class_index]), members.numel())
        if take == 0:
            continue
        order = torch.argsort(scores[members], descending=True, stable=True)
        selected[members[order[:take]]] = True
    return selected


def backfill_minimum_class_support(
    selected: torch.Tensor,
    labels: torch.Tensor,
    reliability: torch.Tensor,
    minimum_per_class: int,
) -> torch.Tensor:
    """Add reliable original-label rows to prevent class collapse, without relabeling."""
    chosen = torch.as_tensor(selected, dtype=torch.bool).flatten().cpu().clone()
    y = torch.as_tensor(labels, dtype=torch.long).flatten().cpu()
    scores = torch.as_tensor(reliability, dtype=torch.float32).flatten().cpu()
    if not (chosen.numel() == y.numel() == scores.numel()):
        raise ValueError("selected, labels, and reliability must align")
    if int(minimum_per_class) < 1:
        raise ValueError("minimum_per_class must be positive")
    num_classes = int(y.max()) + 1
    for class_index in range(num_classes):
        current = int((chosen & (y == class_index)).sum())
        deficit = int(minimum_per_class) - current
        if deficit <= 0:
            continue
        candidates = torch.nonzero(
            ~chosen & (y == class_index), as_tuple=False
        ).flatten()
        if candidates.numel() < deficit:
            raise ValueError(
                f"Class {class_index} cannot reach minimum support {minimum_per_class}"
            )
        order = torch.argsort(scores[candidates], descending=True, stable=True)
        chosen[candidates[order[:deficit]]] = True
    return chosen


def _load_aligned_inputs(
    assignments_path: str | Path,
    logits_path: str | Path,
    quality_path: str | Path,
) -> tuple[pd.DataFrame, torch.Tensor, pd.DataFrame]:
    assignments = pd.read_csv(
        assignments_path,
        dtype={"sample_id": str, "image_path": str, "label": int, "fold": int},
    ).sort_values("image_path").reset_index(drop=True)
    required_assignments = {"sample_id", "image_path", "label", "fold"}
    missing_assignments = required_assignments - set(assignments.columns)
    if missing_assignments:
        raise ValueError(
            f"Assignments miss columns: {sorted(missing_assignments)}"
        )
    if assignments["sample_id"].duplicated().any():
        raise ValueError("Assignments contain duplicate sample IDs")

    payload = torch.load(logits_path, map_location="cpu", weights_only=False)
    required_payload = {"sample_ids", "logits", "folds", "labels"}
    missing_payload = required_payload - set(payload)
    if missing_payload:
        raise ValueError(f"OOF logits payload misses keys: {sorted(missing_payload)}")
    if list(payload["sample_ids"]) != assignments["sample_id"].tolist():
        raise ValueError("OOF logits sample order differs from assignments")
    logits = torch.as_tensor(payload["logits"], dtype=torch.float32)
    if logits.shape[0] != len(assignments):
        raise ValueError("OOF logits row count differs from assignments")
    if not torch.equal(
        torch.as_tensor(payload["labels"]).long(),
        torch.tensor(assignments["label"].to_numpy(copy=True)).long(),
    ):
        raise ValueError("OOF logits labels differ from assignments")
    if not torch.equal(
        torch.as_tensor(payload["folds"]).long(),
        torch.tensor(assignments["fold"].to_numpy(copy=True)).long(),
    ):
        raise ValueError("OOF logits folds differ from assignments")

    quality = pd.read_csv(quality_path, dtype={"sample_id": str})
    missing_quality = REQUIRED_QUALITY_COLUMNS - set(quality.columns)
    if missing_quality:
        raise ValueError(f"Quality table misses columns: {sorted(missing_quality)}")
    if quality["sample_id"].duplicated().any():
        raise ValueError("Quality table contains duplicate sample IDs")
    if set(quality["sample_id"]) != set(assignments["sample_id"]):
        raise ValueError("Quality sample IDs differ from assignments")
    quality = quality.set_index("sample_id").loc[assignments["sample_id"]].reset_index()
    if not np.array_equal(
        quality["original_label"].to_numpy(dtype=np.int64),
        assignments["label"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Quality labels differ from assignments")
    return assignments, logits, quality


def _load_base_trust(
    trust_path: str | Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    bundle = torch.load(trust_path, map_location="cpu", weights_only=False)
    required = {
        "paths",
        "clean_probability",
        "pseudo_label",
        "pseudo_confidence",
        "correction_alpha",
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Base trust bundle misses keys: {sorted(missing)}")
    canonical_paths = [canonical_sample_path(path) for path in bundle["paths"]]
    if len(canonical_paths) != len(set(canonical_paths)):
        raise ValueError("Base trust bundle has duplicate canonical paths")
    bundle = dict(bundle)
    bundle["paths"] = canonical_paths
    return bundle, {path: index for index, path in enumerate(canonical_paths)}


def _bundle_copy(bundle: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in bundle.items():
        result[key] = value.clone() if isinstance(value, torch.Tensor) else value
    result["metadata"] = dict(bundle.get("metadata", {}))
    return result


def build_structured_allocation(
    assignments_path: str | Path,
    logits_path: str | Path,
    quality_path: str | Path,
    base_trust_path: str | Path,
    output_dir: str | Path,
    *,
    curriculum_budget: float = 0.30,
    temperature: float = 1.0,
    sinkhorn_iterations: int = 100,
    minimum_class_support: int = 10,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Build paired selected-only control and structured-correction bundles."""
    if not 0.0 < float(curriculum_budget) <= 1.0:
        raise ValueError("curriculum_budget must be in (0,1]")
    assignments, logits, quality = _load_aligned_inputs(
        assignments_path, logits_path, quality_path
    )
    base, path_to_trust = _load_base_trust(base_trust_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = torch.tensor(assignments["label"].to_numpy(copy=True), dtype=torch.long)
    num_classes = logits.shape[1]
    if int(labels.max()) >= num_classes:
        raise ValueError("OOF logits do not cover every label")
    assignment_trust_indices = torch.tensor(
        [
            path_to_trust[canonical_sample_path(path)]
            for path in assignments["image_path"]
        ],
        dtype=torch.long,
    )
    base_clean = torch.as_tensor(base["clean_probability"]).float()
    base_pseudo = torch.as_tensor(base["pseudo_label"]).long()
    base_alpha = torch.as_tensor(base["correction_alpha"]).float()
    clean = base_clean[assignment_trust_indices]
    pseudo = base_pseudo[assignment_trust_indices]
    alpha = base_alpha[assignment_trust_indices]
    historical_oof = torch.tensor(
        quality["oof_top1"].to_numpy(copy=True), dtype=torch.long
    )
    prototype_top1 = torch.tensor(
        quality["prototype_top1"].to_numpy(copy=True), dtype=torch.long
    )
    knn_top1 = torch.tensor(
        quality["knn_top1"].to_numpy(copy=True), dtype=torch.long
    )
    flip_stable = torch.tensor(
        quality["flip_consistency"].to_numpy(copy=True), dtype=torch.float32
    ) >= 0.5
    kta = torch.tensor(
        quality["knn_top1_agreement"].to_numpy(copy=True), dtype=torch.float32
    ).clamp(0.0, 1.0)
    explicit_original_anchor = (
        (historical_oof == labels)
        & (prototype_top1 == labels)
        & (knn_top1 == labels)
        & flip_stable
    )
    explicit_correction_anchor = (
        (alpha > 0.0) & (pseudo >= 0) & (pseudo < num_classes)
    )
    fixed = explicit_original_anchor | explicit_correction_anchor
    fixed_labels = torch.where(
        explicit_correction_anchor,
        pseudo,
        labels,
    )
    original_counts = torch.bincount(labels, minlength=num_classes)
    fixed_counts = torch.bincount(fixed_labels[fixed], minlength=num_classes)
    residual_counts = original_counts - fixed_counts
    if (residual_counts <= 0).any():
        classes = torch.nonzero(residual_counts <= 0).flatten().tolist()
        raise ValueError(
            f"Fixed anchors exhaust a class marginal; first classes={classes[:10]}"
        )
    uncertain = ~fixed
    if int(residual_counts.sum()) != int(uncertain.sum()):
        raise RuntimeError("Residual class marginals do not match uncertain rows")

    sinkhorn_device = torch.device(device)
    allocation, sinkhorn = log_sinkhorn_allocation(
        logits[uncertain].to(sinkhorn_device),
        residual_counts.to(sinkhorn_device),
        temperature=temperature,
        iterations=sinkhorn_iterations,
    )
    top2_probability, top2_label = allocation.topk(2, dim=1)
    uncertain_assigned = top2_label[:, 0].cpu()
    uncertain_confidence = top2_probability[:, 0].cpu()
    uncertain_margin = (top2_probability[:, 0] - top2_probability[:, 1]).cpu()
    del allocation
    if sinkhorn_device.type == "cuda":
        torch.cuda.empty_cache()

    assigned = fixed_labels.clone()
    assignment_confidence = torch.ones(len(assignments), dtype=torch.float32)
    assignment_margin = torch.ones(len(assignments), dtype=torch.float32)
    assigned[uncertain] = uncertain_assigned
    assignment_confidence[uncertain] = uncertain_confidence
    assignment_margin[uncertain] = uncertain_margin

    probabilities = F.softmax(logits, dim=1)
    row_indices = torch.arange(len(assignments))
    assigned_oof_support = probabilities[row_indices, assigned]
    # Historical top-1 was computed from the pre-serialization float32 logits.
    # Rebuilt logits are stored in float16, where a small number of near-ties
    # collapse exactly. ``topk`` reproduces the historical tie behavior much
    # better than ``argmax``; the verified historical scalar remains the local
    # OOF vote, while the full rebuilt distribution supplies the OT cost.
    reconstructed_oof = probabilities.topk(2, dim=1).indices[:, 0]
    oof_top1 = historical_oof
    vote_count = (
        (assigned == oof_top1).long()
        + (assigned == prototype_top1).long()
        + (assigned == knn_top1).long()
    )
    local_coherent = (vote_count >= 2) & flip_stable
    neighborhood_factor = torch.where(assigned == knn_top1, kta.sqrt(), 1.0)
    reliability = (
        (assignment_confidence * assigned_oof_support).clamp_min(0.0).sqrt()
        * (vote_count.float() / 3.0)
        * neighborhood_factor
    ).clamp(0.0, 1.0)

    uncertain_quota = torch.floor(
        residual_counts.float() * float(curriculum_budget)
    ).long()
    selected_uncertain_local = classwise_curriculum_selection(
        assigned[uncertain],
        reliability[uncertain],
        local_coherent[uncertain],
        uncertain_quota,
    )
    curriculum_selected = fixed.clone()
    curriculum_selected[uncertain] = selected_uncertain_local
    selected = backfill_minimum_class_support(
        curriculum_selected,
        labels,
        reliability,
        minimum_per_class=minimum_class_support,
    )
    corrected = (
        explicit_correction_anchor
        | (curriculum_selected & ~fixed & local_coherent & (assigned != labels))
    )
    effective_structured_labels = torch.where(corrected, assigned, labels)
    selected = backfill_minimum_class_support(
        selected,
        effective_structured_labels,
        reliability,
        minimum_per_class=minimum_class_support,
    )
    effective_structured_labels = torch.where(corrected, assigned, labels)
    support_backfill = selected & ~curriculum_selected

    selected_frame = assignments.loc[
        selected.numpy(), ["image_path", "label", "sample_id", "fold"]
    ].copy()
    selected_frame.to_csv(output / "selected_train.csv", index=False)
    allocation_frame = assignments[
        ["sample_id", "image_path", "label", "fold"]
    ].rename(columns={"label": "original_label"}).copy()
    allocation_frame["fixed_anchor"] = fixed.numpy()
    allocation_frame["assigned_label"] = assigned.numpy()
    allocation_frame["assignment_confidence"] = assignment_confidence.numpy()
    allocation_frame["assignment_margin"] = assignment_margin.numpy()
    allocation_frame["assigned_oof_support"] = assigned_oof_support.numpy()
    allocation_frame["vote_count"] = vote_count.numpy()
    allocation_frame["local_coherent"] = local_coherent.numpy()
    allocation_frame["reliability"] = reliability.numpy()
    allocation_frame["selected"] = selected.numpy()
    allocation_frame["corrected"] = corrected.numpy()
    allocation_frame.to_csv(output / "allocation.csv", index=False)

    control = _bundle_copy(base)
    structured = _bundle_copy(base)
    for bundle in (control, structured):
        bundle["clean_probability"] = torch.as_tensor(
            bundle["clean_probability"]
        ).float()
        bundle["pseudo_label"] = torch.as_tensor(bundle["pseudo_label"]).long()
        bundle["pseudo_confidence"] = torch.as_tensor(
            bundle["pseudo_confidence"]
        ).float()
        bundle["correction_alpha"] = torch.as_tensor(
            bundle["correction_alpha"]
        ).float()
        bundle["clean_probability"][assignment_trust_indices] = selected.float()
        bundle["pseudo_label"][assignment_trust_indices] = -1
        bundle["pseudo_confidence"][assignment_trust_indices] = 0.0
        bundle["correction_alpha"][assignment_trust_indices] = 0.0

    corrected_trust_indices = assignment_trust_indices[corrected]
    structured["pseudo_label"][corrected_trust_indices] = assigned[corrected]
    structured["pseudo_confidence"][corrected_trust_indices] = reliability[
        corrected
    ]
    structured["correction_alpha"][corrected_trust_indices] = 1.0
    common_metadata = {
        "method": "cross_fitted_marginal_structured_curriculum_v1",
        "curriculum_budget": float(curriculum_budget),
        "temperature": float(temperature),
        "sinkhorn_iterations": int(sinkhorn_iterations),
        "sample_count": len(assignments),
        "selected_count": int(selected.sum()),
        "corrected_count": int(corrected.sum()),
        "test_data_used": False,
        "external_data": False,
    }
    control["metadata"] = {
        **dict(control.get("metadata", {})),
        **common_metadata,
        "variant": "selected_original_label_control",
    }
    structured["metadata"] = {
        **dict(structured.get("metadata", {})),
        **common_metadata,
        "variant": "selected_structured_correction",
    }
    torch.save(control, output / "control_trust.pt")
    torch.save(structured, output / "structured_trust.pt")

    control_selected_counts = torch.bincount(
        labels[selected], minlength=num_classes
    )
    structured_selected_counts = torch.bincount(
        effective_structured_labels[selected], minlength=num_classes
    )
    hard_assigned_counts = torch.bincount(assigned, minlength=num_classes)
    correction_source_counts = torch.bincount(
        labels[corrected], minlength=num_classes
    )
    correction_target_counts = torch.bincount(
        assigned[corrected], minlength=num_classes
    )
    audit: dict[str, Any] = {
        "sample_count": len(assignments),
        "num_classes": num_classes,
        "fixed_anchor_count": int(fixed.sum()),
        "explicit_original_anchor_count": int(explicit_original_anchor.sum()),
        "fixed_correction_count": int(explicit_correction_anchor.sum()),
        "base_clean_ge_0_999_not_fixed_count": int(
            ((clean >= 0.999) & ~fixed).sum()
        ),
        "uncertain_count": int(uncertain.sum()),
        "local_coherent_uncertain_count": int((local_coherent & uncertain).sum()),
        "selected_uncertain_count": int((selected & uncertain).sum()),
        "support_backfill_count": int(support_backfill.sum()),
        "selected_count": int(selected.sum()),
        "selected_fraction": float(selected.float().mean()),
        "corrected_count": int(corrected.sum()),
        "corrected_fraction_of_selected": float(
            corrected.sum() / selected.sum().clamp_min(1)
        ),
        "corrected_source_classes": int((correction_source_counts > 0).sum()),
        "corrected_target_classes": int((correction_target_counts > 0).sum()),
        "all_corrections_local_coherent": bool(local_coherent[corrected].all()),
        "control_selected_class_minimum": int(control_selected_counts.min()),
        "control_selected_class_maximum": int(control_selected_counts.max()),
        "structured_selected_class_minimum": int(
            structured_selected_counts.min()
        ),
        "structured_selected_class_maximum": int(
            structured_selected_counts.max()
        ),
        "hard_assignment_class_count_l1": int(
            (hard_assigned_counts - original_counts).abs().sum()
        ),
        "hard_assignment_maximum_class_shift": int(
            (hard_assigned_counts - original_counts).abs().max()
        ),
        "reconstructed_vs_historical_oof_top1_agreement": float(
            (reconstructed_oof == historical_oof).float().mean()
        ),
        "selected_mean_reliability": float(reliability[selected].mean()),
        "corrected_mean_reliability": float(
            reliability[corrected].mean() if corrected.any() else 0.0
        ),
        "sinkhorn": sinkhorn,
    }
    atomic_json_dump(audit, output / "allocation_audit.json")
    manifest = {
        "protocol": "I1 paired cross-fitted marginal structured curriculum",
        "input_hashes": {
            "assignments_sha256": sha256_file(assignments_path),
            "oof_logits_sha256": sha256_file(logits_path),
            "quality_sha256": sha256_file(quality_path),
            "base_trust_sha256": sha256_file(base_trust_path),
        },
        "parameters": {
            "curriculum_budget": float(curriculum_budget),
            "temperature": float(temperature),
            "sinkhorn_iterations": int(sinkhorn_iterations),
            "minimum_class_support": int(minimum_class_support),
        },
        "outputs": {
            "selected_train_sha256": sha256_file(output / "selected_train.csv"),
            "allocation_sha256": sha256_file(output / "allocation.csv"),
            "control_trust_sha256": sha256_file(output / "control_trust.pt"),
            "structured_trust_sha256": sha256_file(
                output / "structured_trust.pt"
            ),
            "audit_sha256": sha256_file(output / "allocation_audit.json"),
        },
        "external_data": False,
        "test_data_used": False,
    }
    atomic_json_dump(manifest, output / "artifact_manifest.json")
    return {"audit": audit, "manifest": manifest}
