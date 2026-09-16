"""Bounded PRELIM75 v3 recovery cohort, continuation, and delivery.

The V1 teacher is frozen and used only on the official training list.  The
original supervision tensors are never rewritten.  S0 and S1 differ only by
the soft-target loss on the fixed, content-group-deduplicated recovery cohort.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from aegis_clip.balanced_inference import prediction_metrics
from aegis_clip.checkpoint import _atomic_torch_save
from aegis_clip.config import validate_config
from aegis_clip.features import canonical_sample_path
from aegis_clip.lineage import run_lineage_audit
from aegis_clip.local_inference import adapted_dual_local_view_logits
from aegis_clip.localization import extract_attention_crops, forward_features_with_last_block_attention
from aegis_clip.prelim75 import (
    OfficialImages,
    fused_logits,
    gpu_setup,
    load_composite,
    loader,
    repository_root,
    saved_candidate,
    source_manifest,
    supervision,
)
from aegis_clip.prelim75_joint import crop_with_bound_boxes, fixed_choices, official_anchor, weighted_micro_loss
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.trainer import _warmup_cosine


PLAN_ID = "PRELIM75_V3_20260915"
SCALES = (112, 128, 144, 160)
SCALE_WEIGHTS = (0.2, 0.3, 0.4, 0.1)
EXPECTED_PROTOCOL = {
    "temperature": 1.5,
    "confidence_threshold": 0.45,
    "margin_threshold": 0.15,
    "global_weight": 0.6,
    "local_weight": 0.4,
    "flip_weight": 0.5,
    "local_scales": [112, 128, 144, 160],
    "local_scale_weights": [0.2, 0.3, 0.4, 0.1],
    "attention_top_k": 5,
    "class_cap": 32,
    "class_support_fraction": 0.2,
    "minimum_groups": 512,
    "minimum_classes": 100,
    "epochs": 3,
    "schedule_epochs": 18,
    "effective_batch_size": 32,
    "kd_weight": 0.25,
    "epoch_numbers": [4, 5, 6],
    "gpu_budget_seconds": 28800,
}


def _paths_sha256(paths):
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


def load_recovery_plan(path):
    """Load only the single registered v3 plan and verify every frozen asset."""
    source = Path(path).resolve()
    plan = yaml.safe_load(source.read_text())
    path_keys = {
        "parent", "parent_submission", "parent_diagnostic", "trust", "train_csv",
        "val_csv", "class_mapping", "groups", "train_root", "test_root", "output",
    }
    hash_keys = {
        "parent", "parent_submission", "parent_diagnostic", "trust", "train_csv",
        "val_csv", "class_mapping", "groups",
    }
    allowed = {
        "plan_id", "seed", "cache_batch_size", "num_workers", "protocol",
        *path_keys, *(key + "_sha256" for key in hash_keys),
    }
    if set(plan) != allowed or plan.get("plan_id") != PLAN_ID or plan.get("seed") != 42:
        raise ValueError("Unknown or incomplete fixed recovery plan")
    if plan.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("Recovery protocol differs from the registered fixed plan")
    if plan.get("cache_batch_size") != 64 or plan.get("num_workers") != 4:
        raise ValueError("Unregistered cache/loader settings")
    for key in path_keys:
        plan[key] = str((source.parent / plan[key]).resolve())
    for key in hash_keys:
        if sha256_file(plan[key]) != plan[key + "_sha256"]:
            raise ValueError(f"Frozen recovery asset hash mismatch: {key}")
    train_root = Path(plan["train_root"])
    test_root = Path(plan["test_root"])
    if train_root == test_root or train_root.is_relative_to(test_root) or test_root.is_relative_to(train_root):
        raise ValueError("Official train and test roots must be disjoint")
    if Path(plan["output"]).resolve() == Path(plan["parent"]).resolve().parent.parent:
        raise ValueError("v3 output must not overlap the v2 parent output")
    plan["_config_path"] = str(source)
    plan["_config_sha256"] = sha256_file(source)
    return plan


def parent_metadata(plan):
    """Read and validate the actual V1 metadata instead of inheriting an old parent id."""
    checkpoint = torch.load(plan["parent"], map_location="cpu", weights_only=False)
    training = checkpoint.get("prelim75_training", {})
    config = checkpoint.get("config", {})
    experiment_id = config.get("project", {}).get("experiment_id")
    if training.get("name") != "V1" or training.get("plan_id") != "PRELIM75_V2_20260912":
        raise ValueError("Registered parent is not the completed v2 V1 candidate")
    if experiment_id != "PRELIM75_V2_V1":
        raise ValueError("Unexpected actual V1 experiment id")
    if "local_feature_adapter" not in checkpoint or "part_token_adapter" not in checkpoint:
        raise ValueError("V1 checkpoint is missing one or both local adapters")
    diagnostic = json.loads(Path(plan["parent_diagnostic"]).read_text())
    if (diagnostic.get("status") != "complete" or
            diagnostic.get("checkpoint_sha256") != plan["parent_sha256"] or
            diagnostic.get("val_csv_sha256") != plan["val_csv_sha256"] or
            diagnostic.get("prior_in_inference") is not False):
        raise ValueError("Registered V1 diagnostic is incomplete or mismatched")
    return {
        "experiment_id": experiment_id,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "training_metadata": training,
        "diagnostic": diagnostic,
    }


def preflight(plan):
    """Read-only asset/provenance check used before an output directory is created."""
    metadata = parent_metadata(plan)
    data = supervision(plan)
    if len(data["paths"]) != 103218:
        raise ValueError("Official training row count changed")
    groups = json.loads(Path(plan["groups"]).read_text())
    if set(groups) != set(data["paths"]):
        raise ValueError("Content-group map does not exactly cover the official train list")
    train_root = Path(plan["train_root"]).resolve()
    test_root = Path(plan["test_root"]).resolve()
    for path in data["paths"]:
        resolved = (train_root / path).resolve()
        if resolved.is_relative_to(test_root):
            raise ValueError("Test path entered the recovery training list")
    return {
        "status": "passed",
        "plan_id": plan["plan_id"],
        "config_sha256": plan["_config_sha256"],
        "parent_sha256": plan["parent_sha256"],
        "parent_submission_sha256": plan["parent_submission_sha256"],
        "trust_sha256": plan["trust_sha256"],
        "train_csv_sha256": plan["train_csv_sha256"],
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "groups_sha256": plan["groups_sha256"],
        "official_train_rows": len(data["paths"]),
        "actual_parent_experiment_id": metadata["experiment_id"],
        "parent_checkpoint_epoch": metadata["checkpoint_epoch"],
        "parent_checkpoint_global_step": metadata["checkpoint_global_step"],
        "test_used_for_cache_or_training": False,
        "platform_upload_authorized": False,
        "protocol": copy.deepcopy(plan["protocol"]),
    }


def recovery_binding(plan, paths, parent_experiment_id="PRELIM75_V2_V1"):
    return {
        "plan_id": PLAN_ID,
        "parent_sha256": plan["parent_sha256"],
        "parent_experiment_id": parent_experiment_id,
        "train_csv_sha256": plan["train_csv_sha256"],
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "trust_sha256": plan["trust_sha256"],
        "groups_sha256": plan["groups_sha256"],
        "paths_sha256": _paths_sha256(paths),
        "rows": 103218,
        "probability_shape": [103218, 500],
        "boxes_shape": [103218, 2, 4, 4],
        "protocol": copy.deepcopy(plan["protocol"]),
        "probability_semantics": "branch_softmax_at_T_then_0.6_global_0.4_local_then_0.5_flip",
        "teacher_probability_retemperature_applied": False,
        "prior_in_inference": False,
        "dtype": "float32",
    }


@torch.no_grad()
def _teacher_views(model, o3, pta, images):
    global_logits = []
    local_logits = []
    all_boxes = []
    for view in (images, images.flip(3)):
        logits, _, attention = forward_features_with_last_block_attention(model, view)
        global_logits.append(logits.float())
        orientation_logits = []
        orientation_boxes = []
        for scale in SCALES:
            crop, boxes = extract_attention_crops(view, attention, crop_size=scale, top_k=5)
            orientation_logits.append(adapted_dual_local_view_logits(model, o3, pta, crop).float())
            orientation_boxes.append(np.asarray(boxes, dtype=np.int16))
        local_logits.append(torch.stack(orientation_logits, dim=1))
        all_boxes.append(np.stack(orientation_boxes, axis=1))
    return torch.stack(global_logits, dim=1), torch.stack(local_logits, dim=1), np.stack(all_boxes, axis=1)


def fuse_teacher_probabilities(global_logits, local_logits, temperature=1.5):
    """Apply temperature once per branch and return the fixed v3 fusion metadata."""
    if global_logits.ndim != 3 or global_logits.shape[1] != 2:
        raise ValueError("Expected two global orientations")
    if local_logits.ndim != 4 or local_logits.shape[1:3] != (2, 4):
        raise ValueError("Expected two orientations and four local scales")
    g = global_logits.float().div(temperature).softmax(-1)
    local_branch = local_logits.float().div(temperature).softmax(-1)
    scale_weights = torch.tensor(SCALE_WEIGHTS, device=local_logits.device, dtype=torch.float32)
    local = (local_branch * scale_weights.view(1, 1, 4, 1)).sum(2)
    orientations = 0.6 * g + 0.4 * local
    teacher = 0.5 * orientations.sum(1)
    global_pair = 0.5 * g.sum(1)
    local_pair = 0.5 * local.sum(1)
    top2 = teacher.topk(2, dim=1).values
    branches = torch.stack((
        orientations[:, 0].argmax(1), orientations[:, 1].argmax(1),
        global_pair.argmax(1), local_pair.argmax(1),
    ), dim=1)
    return {
        "probabilities": teacher,
        "teacher_prediction": teacher.argmax(1),
        "branch_predictions": branches,
        "confidence": top2[:, 0],
        "margin": top2[:, 0] - top2[:, 1],
    }


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {key: None for key in ("min", "p10", "p25", "p50", "p75", "p90", "max")}
    points = np.quantile(values, [0, .1, .25, .5, .75, .9, 1.])
    return dict(zip(("min", "p10", "p25", "p50", "p75", "p90", "max"), map(float, points)))


def select_recovery_cohort(
    paths, group_ids, weights, targets, teacher_prediction, branch_predictions,
    confidence, margin, *, confidence_threshold=.45, margin_threshold=.15,
    class_cap=32, support_fraction=.20, minimum_groups=512, minimum_classes=100,
):
    """Deterministically select one representative from eligible zero-only groups."""
    n = len(paths)
    arrays = (group_ids, weights, targets, teacher_prediction, branch_predictions, confidence, margin)
    if any(len(value) != n for value in arrays):
        raise ValueError("Recovery selection row mismatch")
    weights = np.asarray(weights)
    targets = np.asarray(targets)
    teacher_prediction = np.asarray(teacher_prediction)
    branch_predictions = np.asarray(branch_predictions)
    confidence = np.asarray(confidence)
    margin = np.asarray(margin)
    if targets.shape != (n, 500) or branch_predictions.shape != (n, 4):
        raise ValueError("Recovery selection tensor shape mismatch")
    if not (np.isfinite(confidence).all() and np.isfinite(margin).all()):
        raise ValueError("Nonfinite recovery metadata")

    members = {}
    for index, group in enumerate(group_ids):
        members.setdefault(str(group), []).append(index)
    positive_groups = {str(group_ids[i]) for i in np.flatnonzero(weights > 0)}
    zero_indices = np.flatnonzero(weights == 0)
    zero_groups = {str(group_ids[i]) for i in zero_indices}
    zero_only_groups = zero_groups - positive_groups
    representatives = []
    for group in zero_only_groups:
        indices = members[group]
        representative = min(indices, key=lambda i: paths[i])
        representatives.append(representative)
    representatives.sort(key=lambda i: paths[i])
    rep = np.asarray(representatives, dtype=np.int64)

    agreement = np.zeros(n, dtype=bool)
    if len(rep):
        agreement[rep] = (branch_predictions[rep] == teacher_prediction[rep, None]).all(1)
    after_agreement = rep[agreement[rep]]
    after_confidence = after_agreement[confidence[after_agreement] >= confidence_threshold]
    eligible = after_confidence[margin[after_confidence] >= margin_threshold]

    support = [set() for _ in range(500)]
    positive_indices = np.flatnonzero(weights > 0)
    positive_class = targets[positive_indices].argmax(1)
    for index, predicted_class in zip(positive_indices, positive_class):
        support[int(predicted_class)].add(str(group_ids[index]))
    positive_group_support = np.asarray([len(groups) for groups in support], dtype=np.int64)
    caps = np.minimum(class_cap, np.floor(support_fraction * positive_group_support).astype(np.int64))

    eligible_by_class = np.bincount(teacher_prediction[eligible], minlength=500)
    ordered = sorted(eligible.tolist(), key=lambda i: (-float(confidence[i]), -float(margin[i]), paths[i]))
    selected = []
    selected_by_class = np.zeros(500, dtype=np.int64)
    for index in ordered:
        predicted_class = int(teacher_prediction[index])
        if selected_by_class[predicted_class] < caps[predicted_class]:
            selected.append(index)
            selected_by_class[predicted_class] += 1
    selected = np.asarray(selected, dtype=np.int64)
    mask = np.zeros(n, dtype=bool)
    recovery_weights = np.zeros(n, dtype=np.float32)
    mask[selected] = True
    recovery_weights[selected] = confidence[selected].astype(np.float32)
    covered_classes = int((selected_by_class > 0).sum())
    gate_passed = len(selected) >= minimum_groups and covered_classes >= minimum_classes

    report = {
        "official_total_rows": n,
        "old_zero_weight_rows": int(len(zero_indices)),
        "old_zero_weight_unique_groups": int(len(zero_groups)),
        "positive_supervision_unique_groups": int(len(positive_groups)),
        "excluded_by_positive_supervision_group_rows": int(sum(str(group_ids[i]) in positive_groups for i in zero_indices)),
        "excluded_by_positive_supervision_unique_groups": int(len(zero_groups & positive_groups)),
        "zero_only_group_representatives": int(len(rep)),
        "representatives_failing_orientation_agreement": int(
            (branch_predictions[rep, 0] != branch_predictions[rep, 1]).sum()
        ),
        "representatives_failing_global_local_agreement": int(
            (branch_predictions[rep, 2] != branch_predictions[rep, 3]).sum()
        ),
        "eliminated_by_view_or_branch_disagreement": int(len(rep) - len(after_agreement)),
        "eliminated_by_confidence": int(len(after_agreement) - len(after_confidence)),
        "eliminated_by_margin": int(len(after_confidence) - len(eligible)),
        "eligible_before_class_cap": int(len(eligible)),
        "selected_after_class_cap": int(len(selected)),
        "eliminated_by_class_cap": int(len(eligible) - len(selected)),
        "covered_prediction_classes": covered_classes,
        "minimum_groups": minimum_groups,
        "minimum_classes": minimum_classes,
        "gate_passed": bool(gate_passed),
        "status": "ready_for_fixed_S0_S1" if gate_passed else "closed_insufficient_recovery_cohort",
        "confidence_quantiles_before_cap": _quantiles(confidence[eligible]),
        "margin_quantiles_before_cap": _quantiles(margin[eligible]),
        "confidence_quantiles_after_cap": _quantiles(confidence[selected]),
        "margin_quantiles_after_cap": _quantiles(margin[selected]),
        "known_limitations": [
            "V1 has already seen the complete official training list.",
            "Multi-view agreement is neither independent teacher voting nor proof of label truth.",
            "Classes with zero positive-supervision group support have cap zero and are not recovered.",
            "The fixed thresholds, group cap, and feasibility gate are engineering choices, not statistical guarantees.",
        ],
    }
    class_rows = [{
        "class_index": class_index,
        "positive_supervision_groups": int(positive_group_support[class_index]),
        "class_cap": int(caps[class_index]),
        "eligible_before_cap": int(eligible_by_class[class_index]),
        "selected_after_cap": int(selected_by_class[class_index]),
    } for class_index in range(500)]
    return {
        "mask": mask,
        "recovery_weights": recovery_weights,
        "selected_indices": selected,
        "report": report,
        "class_rows": class_rows,
    }


@torch.no_grad()
def cache_recovery_teacher(plan):
    """Run D0 once, then freeze the deterministic cohort and feasibility result."""
    output = Path(plan["output"])
    teacher_dir = output / "teacher"
    cohort_dir = output / "cohort"
    teacher_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    device = gpu_setup()
    data = supervision(plan)
    model, preprocess, checkpoint, o3, pta = load_composite(plan["parent"], device)
    model.float().eval().requires_grad_(False)
    o3.float().eval().requires_grad_(False)
    pta.float().eval().requires_grad_(False)
    actual_parent = checkpoint.get("config", {}).get("project", {}).get("experiment_id")
    if actual_parent != "PRELIM75_V2_V1":
        raise ValueError("Loaded teacher metadata is not the registered V1")
    atomic_json_dump(source_manifest(), teacher_dir / "source_manifest.json")
    atomic_json_dump(data["paths"], teacher_dir / "paths.json")
    _atomic_torch_save({key: value for key, value in data.items() if key != "trust"}, teacher_dir / "supervision.pt")
    probabilities = np.lib.format.open_memmap(
        teacher_dir / "probabilities.npy", mode="w+", dtype=np.float32, shape=(len(data["paths"]), 500)
    )
    boxes_array = np.lib.format.open_memmap(
        teacher_dir / "boxes.npy", mode="w+", dtype=np.int16, shape=(len(data["paths"]), 2, 4, 4)
    )
    branch_array = np.lib.format.open_memmap(
        teacher_dir / "branch_predictions.npy", mode="w+", dtype=np.int16, shape=(len(data["paths"]), 4)
    )
    teacher_prediction = np.lib.format.open_memmap(
        teacher_dir / "teacher_prediction.npy", mode="w+", dtype=np.int16, shape=(len(data["paths"]),)
    )
    confidence = np.lib.format.open_memmap(
        teacher_dir / "confidence.npy", mode="w+", dtype=np.float32, shape=(len(data["paths"]),)
    )
    margin = np.lib.format.open_memmap(
        teacher_dir / "margin.npy", mode="w+", dtype=np.float32, shape=(len(data["paths"]),)
    )
    dataset = OfficialImages(data["paths"], plan["train_root"], preprocess)
    stream = loader(dataset, plan["cache_batch_size"], plan)
    epoch0_check = None
    completed = 0
    for batch_index, batch in enumerate(stream):
        images = batch["images"].to(device, non_blocking=True)
        indices = batch["index"].numpy()
        global_logits, local_logits, boxes = _teacher_views(model, o3, pta, images)
        fused = fuse_teacher_probabilities(global_logits, local_logits)
        if (not torch.isfinite(global_logits).all() or not torch.isfinite(local_logits).all() or
                not torch.isfinite(fused["probabilities"]).all()):
            raise RuntimeError("Nonfinite V1 teacher cache value")
        torch.testing.assert_close(
            fused["probabilities"].sum(1), torch.ones(len(images), device=device),
            atol=1e-5, rtol=1e-5,
        )
        if batch_index == 0:
            repeat_global, repeat_local, repeat_boxes = _teacher_views(model, o3, pta, images)
            torch.testing.assert_close(global_logits, repeat_global, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(local_logits, repeat_local, atol=1e-5, rtol=1e-5)
            if not np.array_equal(boxes, repeat_boxes):
                raise RuntimeError("Epoch-0 V1 box reconstruction mismatch")
            views = torch.cat((global_logits, local_logits.reshape(len(images), 8, 500)), dim=1)
            existing_fusion = fused_logits(views)
            torch.testing.assert_close(fused["probabilities"], existing_fusion, atol=1e-5, rtol=1e-5)
            if not torch.equal(fused["teacher_prediction"], existing_fusion.argmax(1)):
                raise RuntimeError("Epoch-0 final fusion argmax mismatch")
            epoch0_check = {
                "rows": len(images),
                "atol": 1e-5,
                "rtol": 1e-5,
                "global_logits_max_abs": float((global_logits - repeat_global).abs().max()),
                "local_logits_max_abs": float((local_logits - repeat_local).abs().max()),
                "fused_probability_max_abs": float((fused["probabilities"] - existing_fusion).abs().max()),
                "argmax_disagreements": int((fused["teacher_prediction"] != existing_fusion.argmax(1)).sum()),
                "global_local_flip_and_final_fusion_checked": True,
            }
        probabilities[indices] = fused["probabilities"].cpu().numpy()
        boxes_array[indices] = boxes
        branch_array[indices] = fused["branch_predictions"].to(torch.int16).cpu().numpy()
        teacher_prediction[indices] = fused["teacher_prediction"].to(torch.int16).cpu().numpy()
        confidence[indices] = fused["confidence"].cpu().numpy()
        margin[indices] = fused["margin"].cpu().numpy()
        completed += len(indices)
        if batch_index % 20 == 0 or completed == len(dataset):
            status = {
                "status": "running", "completed_samples": completed,
                "total_samples": len(dataset), "elapsed_seconds": time.monotonic() - start,
            }
            atomic_json_dump(status, teacher_dir / "status.json")
            print(json.dumps(status), flush=True)
    for array in (probabilities, boxes_array, branch_array, teacher_prediction, confidence, margin):
        array.flush()
    np.savez(
        teacher_dir / "sample_metadata.npz",
        teacher_prediction=np.asarray(teacher_prediction),
        branch_predictions=np.asarray(branch_array),
        confidence=np.asarray(confidence),
        margin=np.asarray(margin),
    )
    files = [
        "probabilities.npy", "boxes.npy", "branch_predictions.npy", "teacher_prediction.npy",
        "confidence.npy", "margin.npy", "sample_metadata.npz", "paths.json", "supervision.pt",
    ]
    manifest = recovery_binding(plan, data["paths"], actual_parent)
    manifest.update({
        "status": "complete",
        "epoch0_reconstruction_check": epoch0_check,
        "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
        "files_sha256": {name: sha256_file(teacher_dir / name) for name in files},
        "test_rows_processed": 0,
    })
    atomic_json_dump(manifest, teacher_dir / "manifest.json")
    atomic_json_dump(manifest, teacher_dir / "status.json")

    cohort_dir.mkdir(exist_ok=False)
    groups = json.loads(Path(plan["groups"]).read_text())
    group_ids = [groups[path] for path in data["paths"]]
    selection = select_recovery_cohort(
        data["paths"], group_ids, data["weights"].numpy(), data["targets"].numpy(),
        np.asarray(teacher_prediction), np.asarray(branch_array), np.asarray(confidence), np.asarray(margin),
    )
    np.save(cohort_dir / "recovery_mask.npy", selection["mask"])
    np.save(cohort_dir / "recovery_weights.npy", selection["recovery_weights"])
    np.save(cohort_dir / "selected_indices.npy", selection["selected_indices"])
    with (cohort_dir / "class_support.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection["class_rows"][0]))
        writer.writeheader()
        writer.writerows(selection["class_rows"])
    report = selection["report"]
    report.update({
        "plan_id": PLAN_ID,
        "parent_sha256": plan["parent_sha256"],
        "teacher_manifest_sha256": sha256_file(teacher_dir / "manifest.json"),
        "threshold_probability_scale": manifest["probability_semantics"],
        "teacher_probability_retemperature_applied": False,
        "representative_rule": "lexicographically_first_canonical_path_before_teacher_selection",
        "noisy_label_used_for_recovery_acceptance": False,
        "original_supervision_modified": False,
    })
    report["files_sha256"] = {
        name: sha256_file(cohort_dir / name)
        for name in ("recovery_mask.npy", "recovery_weights.npy", "selected_indices.npy", "class_support.csv")
    }
    atomic_json_dump(report, cohort_dir / "report.json")
    return report


def read_recovery_assets(plan, verify=True):
    """Fail closed on any cache/cohort identity, order, hash, or supervision drift."""
    output = Path(plan["output"])
    teacher_dir = output / "teacher"
    cohort_dir = output / "cohort"
    manifest = json.loads((teacher_dir / "manifest.json").read_text())
    paths = json.loads((teacher_dir / "paths.json").read_text())
    expected = recovery_binding(plan, paths)
    if manifest.get("status") != "complete":
        raise ValueError("Incomplete V1 recovery teacher cache")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Recovery teacher cache identity mismatch: {key}")
    if verify:
        for name, expected_hash in manifest["files_sha256"].items():
            if sha256_file(teacher_dir / name) != expected_hash:
                raise ValueError(f"Recovery teacher cache hash mismatch: {name}")
    cached = torch.load(teacher_dir / "supervision.pt", map_location="cpu", weights_only=False)
    regenerated = supervision(plan)
    if cached["paths"] != paths or regenerated["paths"] != paths:
        raise ValueError("Recovery training path order mismatch")
    for key in ("labels", "weights", "targets", "clean_probability", "pseudo_label", "correction_alpha"):
        if not torch.equal(cached[key], regenerated[key]):
            raise ValueError(f"Original supervision changed: {key}")
    report = json.loads((cohort_dir / "report.json").read_text())
    if report.get("teacher_manifest_sha256") != sha256_file(teacher_dir / "manifest.json"):
        raise ValueError("Recovery cohort is not bound to this teacher cache")
    for name, expected_hash in report["files_sha256"].items():
        if verify and sha256_file(cohort_dir / name) != expected_hash:
            raise ValueError(f"Recovery cohort hash mismatch: {name}")
    arrays = {
        "probabilities": np.load(teacher_dir / "probabilities.npy", mmap_mode="r"),
        "boxes": np.load(teacher_dir / "boxes.npy", mmap_mode="r"),
        "recovery_mask": np.load(cohort_dir / "recovery_mask.npy", mmap_mode="r"),
        "recovery_weights": np.load(cohort_dir / "recovery_weights.npy", mmap_mode="r"),
    }
    if arrays["probabilities"].shape != (103218, 500) or arrays["boxes"].shape != (103218, 2, 4, 4):
        raise ValueError("Recovery cache array shape mismatch")
    if not np.array_equal(np.asarray(arrays["recovery_mask"]), np.asarray(arrays["recovery_weights"]) > 0):
        raise ValueError("Recovery mask and weights disagree")
    return cached, arrays, manifest, report


def recovery_kd_loss(logits, teacher_probabilities, recovery_weights, denominator, temperature=1.5):
    """Teacher-to-student KL using the full effective-batch denominator."""
    if teacher_probabilities.requires_grad or recovery_weights.requires_grad:
        raise ValueError("Frozen teacher targets/weights must not require gradients")
    if logits.shape != teacher_probabilities.shape or recovery_weights.shape != logits.shape[:1]:
        raise ValueError("KD tensor shape mismatch")
    if float(denominator.detach()) <= 0:
        return logits.sum() * 0.0
    log_student = F.log_softmax(logits.float() / temperature, dim=1)
    per_sample = F.kl_div(log_student, teacher_probabilities.float(), reduction="none").sum(1)
    return temperature ** 2 * (per_sample * recovery_weights).sum() / denominator


def recovery_lambda(step, steps_per_epoch):
    if step < 0 or steps_per_epoch <= 0:
        raise ValueError("Invalid KD schedule position")
    return 0.25 * min(1.0, (step + 1) / steps_per_epoch)


def recovery_candidate_config(checkpoint, plan, name):
    config = copy.deepcopy(checkpoint["config"])
    actual_parent = checkpoint.get("config", {}).get("project", {}).get("experiment_id")
    if actual_parent != "PRELIM75_V2_V1":
        raise ValueError("Recovery parent experiment id must come from actual V1 metadata")
    config["project"]["experiment_id"] = f"PRELIM75_V3_{name}"
    for key in ("train_csv", "val_csv", "class_mapping", "train_root", "test_root"):
        config["data"][key] = plan[key]
    config["data"]["train_augmentation"] = "clip_center_crop"
    config["train"].update({
        "init_checkpoint": plan["parent"], "require_lineage_for_init_checkpoint": True,
        "epochs": 3, "schedule_epochs": 18, "batch_size": 32,
        "backbone_lr": 1e-6, "backbone_weight_decay": 0.0,
        "head_lr": 1.5e-7, "head_weight_decay": 1e-4, "amp": False,
    })
    config["lineage"] = {
        "enabled": True,
        "parent_experiment_id": actual_parent,
        "parent_train_csv": plan["train_csv"],
        "parent_val_csv": plan["val_csv"],
        "require_same_train": True,
        "require_same_val": True,
        "allow_parent_val_in_child_train": True,
        "allow_parent_train_in_child_val": True,
    }
    config["evaluation"]["selection_policy"] = "last_epoch"
    config["output"]["root"] = plan["output"]
    validate_config(config)
    return config


def _grad_norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().float().pow(2).sum()) for p in parameters if p.grad is not None))


def train_recovery(plan, name):
    """Train fixed S0 or S1 independently from V1 with identical geometry/order."""
    if name not in ("S0", "S1"):
        raise ValueError("Only the registered S0/S1 recovery candidates are allowed")
    output = Path(plan["output"]) / name
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    device = gpu_setup()
    data, arrays, cache_manifest, cohort_report = read_recovery_assets(plan)
    if not cohort_report.get("gate_passed"):
        raise ValueError("D0 feasibility gate is closed")
    model, preprocess, checkpoint, o3, pta = load_composite(plan["parent"], device)
    model.float(); o3.float(); pta.float()
    licensed_visual = {parameter_name for parameter_name, parameter in model.visual.named_parameters() if parameter.requires_grad}
    if "conv1.weight" in licensed_visual or "positional_embedding" in licensed_visual:
        raise ValueError("V1 visual permission mask changed")
    model.classifier.requires_grad_(True)
    o3.requires_grad_(True)
    pta.requires_grad_(True)
    model.train(); o3.train(); pta.train()
    anchor_teacher = official_anchor(device)
    config = recovery_candidate_config(checkpoint, plan, name)
    run_lineage_audit(
        config, child_train_csv=plan["train_csv"], child_val_csv=plan["val_csv"],
        checkpoint_path=plan["parent"], output_path=output / "split_lineage_audit.json",
    )
    atomic_json_dump(config, output / "resolved_config.json")
    atomic_json_dump(source_manifest(), output / "source_manifest.json")
    frozen = {parameter_name: parameter.detach().cpu().clone()
              for parameter_name, parameter in model.named_parameters() if not parameter.requires_grad}
    visual_parameters = [parameter for parameter in model.visual.parameters() if parameter.requires_grad]
    head_parameters = list(model.classifier.parameters())
    adapter_parameters = list(o3.parameters()) + list(pta.parameters())
    optimizer = torch.optim.AdamW([
        {"name": "backbone", "params": visual_parameters, "lr": 1e-6, "weight_decay": 0.0},
        {"name": "head", "params": head_parameters, "lr": 1.5e-7, "weight_decay": 1e-4},
        {"name": "local_adapters", "params": adapter_parameters, "lr": 3e-6, "weight_decay": 0.0},
    ])
    dataset = OfficialImages(data["paths"], plan["train_root"], preprocess)
    generator = torch.Generator().manual_seed(42)
    stream = loader(dataset, 32, plan, shuffle=True, generator=generator)
    steps_per_epoch = len(stream)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _warmup_cosine(step, 0, 18 * steps_per_epoch)
    )
    all_parameters = visual_parameters + head_parameters + adapter_parameters
    recipe = {
        "candidate": name,
        "independent_parent_load": True,
        "parent_sha256": plan["parent_sha256"],
        "parent_experiment_id": checkpoint["config"]["project"]["experiment_id"],
        "epochs": 3, "epoch_numbers": [4, 5, 6], "schedule_epochs": 18,
        "batch_size": 32, "effective_batch_size": 32, "microbatch_size": 32, "accumulation": 1,
        "global_weight": .6, "local_weight": .4, "anchor_weight": 2.0,
        "kd_weight": .25 if name == "S1" else 0.0,
        "kd_temperature": 1.5,
        "kd_direction": "teacher_to_student",
        "teacher_targets": "complete_500_probability_vector_without_retemperature",
        "supervision": "bitwise_verified_original_v2_w_q",
        "geometry": "native_clip_center_crop_224_plus_stateless_flip",
        "boxes": "frozen_V1_native_224_orientation_and_scale_bound",
        "boxes_file_sha256": cache_manifest["files_sha256"]["boxes.npy"],
        "teacher_probabilities_file_sha256": cache_manifest["files_sha256"]["probabilities.npy"],
        "cohort_report_sha256": sha256_file(Path(plan["output"]) / "cohort/report.json"),
        "anchor": "frozen_official_CLIP_same_global_pixel_tensor_online",
        "licensed_visual_parameters": sorted(licensed_visual),
        "selection": "fixed_last_epoch",
    }
    atomic_json_dump(recipe, output / "training_recipe.json")
    microbatch = 32
    history = []
    optimizer_step = 0
    first_step_audit = None
    kd_only_gradient_audit = None
    for local_epoch, sample_epoch in enumerate((4, 5, 6), start=1):
        epoch_start = time.monotonic()
        flips, scales = fixed_choices(data["paths"], sample_epoch)
        totals = {
            "supervised_global": 0.0, "supervised_local": 0.0, "anchor": 0.0,
            "kd_global": 0.0, "kd_local": 0.0, "weighted_kd": 0.0,
            "loss": 0.0, "samples": 0, "recovery_hits": 0,
            "recovery_weight_mass": 0.0, "gradient_norm_sum": 0.0, "gradient_norm_max": 0.0,
        }
        for step, batch in enumerate(stream):
            indices = batch["index"].numpy()
            full_supervision_weights = data["weights"][indices]
            supervision_denominator = full_supervision_weights.sum().to(device).clamp_min(1e-8)
            full_recovery_weights = torch.from_numpy(np.asarray(arrays["recovery_weights"][indices]).copy()).to(device)
            recovery_denominator = full_recovery_weights.sum()
            lambda_value = recovery_lambda(optimizer_step, steps_per_epoch) if name == "S1" else 0.0
            while True:
                optimizer.zero_grad(set_to_none=True)
                parts = {key: 0.0 for key in (
                    "supervised_global", "supervised_local", "anchor", "kd_global",
                    "kd_local", "weighted_kd", "loss",
                )}
                try:
                    for offset in range(0, len(indices), microbatch):
                        sub_idx = indices[offset:offset + microbatch]
                        images = batch["images"][offset:offset + microbatch].to(device, non_blocking=True)
                        selected_flips = flips[sub_idx]
                        if selected_flips.any():
                            flip_mask = torch.tensor(selected_flips, device=device)
                            images[flip_mask] = images[flip_mask].flip(3)
                        targets = data["targets"][sub_idx].to(device)
                        weights = data["weights"][sub_idx].to(device)
                        with torch.no_grad():
                            anchor_reference = F.normalize(anchor_teacher(images).float(), dim=1)
                        global_logits, global_features = model(images=images, return_features=True)
                        supervised_global = weighted_micro_loss(
                            global_logits, targets, weights, supervision_denominator,
                            checkpoint["config"]["loss"], sample_epoch,
                        )
                        anchor = (1.0 - F.cosine_similarity(
                            global_features.float(), anchor_reference, dim=1
                        )).sum() / len(indices)
                        bound_boxes = arrays["boxes"][sub_idx, selected_flips.astype(int), scales[sub_idx]]
                        local_images = crop_with_bound_boxes(images, bound_boxes)
                        local_logits = adapted_dual_local_view_logits(model, o3, pta, local_images)
                        supervised_local = weighted_micro_loss(
                            local_logits, targets, weights, supervision_denominator,
                            checkpoint["config"]["loss"], sample_epoch,
                        )
                        loss = .6 * supervised_global + .4 * supervised_local + 2.0 * anchor
                        kd_global = global_logits.sum() * 0.0
                        kd_local = local_logits.sum() * 0.0
                        weighted_kd = global_logits.sum() * 0.0
                        if name == "S1":
                            teacher_targets = torch.from_numpy(
                                np.asarray(arrays["probabilities"][sub_idx]).copy()
                            ).to(device)
                            recovery_weights = full_recovery_weights[offset:offset + len(sub_idx)]
                            kd_global = recovery_kd_loss(
                                global_logits, teacher_targets, recovery_weights, recovery_denominator
                            )
                            kd_local = recovery_kd_loss(
                                local_logits, teacher_targets, recovery_weights, recovery_denominator
                            )
                            if kd_only_gradient_audit is None and bool((recovery_weights > 0).any()):
                                probes = torch.autograd.grad(
                                    .6 * kd_global + .4 * kd_local,
                                    [model.visual.proj, o3.up.weight, pta.up.weight],
                                    retain_graph=True,
                                )
                                kd_only_gradient_audit = {
                                    key: float(gradient.detach().norm()) for key, gradient in zip(
                                        ("visual_proj", "o3_up", "pta_up"), probes
                                    )
                                }
                                if any(value <= 0 or not math.isfinite(value)
                                       for value in kd_only_gradient_audit.values()):
                                    raise RuntimeError("KD-only loss does not reach visual/O3/PTA")
                                atomic_json_dump(
                                    kd_only_gradient_audit, output / "kd_only_gradient_audit.json"
                                )
                            weighted_kd = lambda_value * (.6 * kd_global + .4 * kd_local)
                            loss = loss + weighted_kd
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite recovery continuation loss")
                        for key, value in (
                            ("supervised_global", supervised_global), ("supervised_local", supervised_local),
                            ("anchor", anchor), ("kd_global", kd_global), ("kd_local", kd_local),
                            ("weighted_kd", weighted_kd), ("loss", loss),
                        ):
                            parts[key] += float(value.detach())
                        loss.backward()
                    break
                except torch.cuda.OutOfMemoryError:
                    if optimizer_step or microbatch != 32:
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    import gc
                    gc.collect(); torch.cuda.empty_cache()
                    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
                    microbatch = 16
                    recipe.update({
                        "microbatch_size": 16, "accumulation": 2,
                        "oom_downgrade": "first_effective_batch_before_any_optimizer_step",
                    })
                    atomic_json_dump(recipe, output / "training_recipe.json")
                    print("OOM engineering downgrade: microbatch16 accumulation2, effective32", flush=True)
            gradient_groups = {
                "visual": _grad_norm(visual_parameters),
                "head": _grad_norm(head_parameters),
                "o3": _grad_norm(o3.parameters()),
                "pta": _grad_norm(pta.parameters()),
            }
            frozen_leaks = [parameter_name for parameter_name, parameter in model.named_parameters()
                            if not parameter.requires_grad and parameter.grad is not None]
            if optimizer_step == 0:
                first_step_audit = {
                    **{key + "_gradient_norm": value for key, value in gradient_groups.items()},
                    "frozen_gradient_leaks": frozen_leaks,
                    "kd_lambda": lambda_value,
                    "recovery_hits": int((full_recovery_weights > 0).sum()),
                }
                if any(value <= 0 or not math.isfinite(value) for value in gradient_groups.values()) or frozen_leaks:
                    raise RuntimeError("Recovery first-step trainable/frozen gradient audit failed")
                atomic_json_dump(first_step_audit, output / "first_step_gradient_audit.json")
            total_gradient_norm = float(torch.nn.utils.clip_grad_norm_(all_parameters, 1.0, error_if_nonfinite=True))
            optimizer.step(); scheduler.step(); optimizer_step += 1
            for key in parts:
                totals[key] += parts[key] * len(indices)
            totals["samples"] += len(indices)
            totals["recovery_hits"] += int((full_recovery_weights > 0).sum())
            totals["recovery_weight_mass"] += float(recovery_denominator)
            totals["gradient_norm_sum"] += total_gradient_norm
            totals["gradient_norm_max"] = max(totals["gradient_norm_max"], total_gradient_norm)
            if step % 100 == 0:
                status = {
                    "status": "training", "candidate": name, "epoch": sample_epoch,
                    "completed_samples": totals["samples"], "optimizer_steps": optimizer_step,
                    "kd_lambda": lambda_value, "elapsed_seconds": time.monotonic() - start,
                }
                atomic_json_dump(status, output / "status.json")
                print(json.dumps(status), flush=True)
        row = {
            "epoch": sample_epoch,
            **{key: value / totals["samples"] for key, value in totals.items()
               if key not in ("samples", "recovery_hits", "recovery_weight_mass", "gradient_norm_sum", "gradient_norm_max")},
            "samples": totals["samples"],
            "recovery_hits": totals["recovery_hits"],
            "recovery_weight_mass": totals["recovery_weight_mass"],
            "optimizer_steps": optimizer_step,
            "updates_this_epoch": steps_per_epoch,
            "gradient_norm_mean": totals["gradient_norm_sum"] / steps_per_epoch,
            "gradient_norm_max": totals["gradient_norm_max"],
            "epoch_elapsed_seconds": time.monotonic() - epoch_start,
            "elapsed_seconds": time.monotonic() - start,
            "peak_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
        }
        history.append(row)
        atomic_json_dump(history, output / "training_history.json")
        print(json.dumps(row), flush=True)
    for parameter_name, parameter in model.named_parameters():
        if parameter_name in frozen and not torch.equal(parameter.detach().cpu(), frozen[parameter_name]):
            raise RuntimeError(f"Frozen V1 tensor changed: {parameter_name}")
    if name == "S1" and kd_only_gradient_audit is None:
        raise RuntimeError("S1 never encountered a recovery target")
    payload = saved_candidate(checkpoint, model, config, name, plan, optimizer, scheduler, 6, o3, pta)
    payload["global_step"] = int(checkpoint.get("global_step", 0)) + optimizer_step
    payload["prelim75_training"] = {
        "name": name,
        "plan_id": PLAN_ID,
        "parent_experiment_id": checkpoint["config"]["project"]["experiment_id"],
        "parent_sha256": plan["parent_sha256"],
        "selection": "fixed_epoch_6_last_epoch",
        "supervision": "unchanged_V1_w_q",
        "recovery_kd": name == "S1",
        "teacher_cache_required_for_inference": False,
        "prior_in_inference": False,
        "upstream_provenance_complete": False,
    }
    _atomic_torch_save(payload, output / "candidate.pt")
    checkpoint_hash = sha256_file(output / "candidate.pt")
    del payload, anchor_teacher
    model.cpu(); o3.cpu(); pta.cpu()
    del model, o3, pta
    torch.cuda.empty_cache()
    reloaded_model, _, reloaded_checkpoint, reloaded_o3, reloaded_pta = load_composite(output / "candidate.pt", torch.device("cpu"))
    if reloaded_checkpoint.get("prelim75_training", {}).get("parent_sha256") != plan["parent_sha256"]:
        raise RuntimeError("Saved recovery candidate parent binding changed on reload")
    reload_check = {
        "complete_model_strictly_reloaded": True,
        "shared_head_present": hasattr(reloaded_model, "classifier"),
        "local_feature_adapter_reloaded": reloaded_o3 is not None,
        "part_token_adapter_reloaded": reloaded_pta is not None,
        "teacher_cache_accessed_by_reload": False,
    }
    result = {
        "status": "trained_pending_real_diagnostic",
        "candidate": name,
        "history": history,
        "first_step_gradient_audit": first_step_audit,
        "kd_only_gradient_audit": kd_only_gradient_audit,
        "reload_check": reload_check,
        "checkpoint_sha256": checkpoint_hash,
        "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
        "online_accuracy": None,
        "validation_scope": "overlap_diagnostic",
    }
    atomic_json_dump(result, output / "training_result.json")
    atomic_json_dump(result, output / "status.json")
    return output / "candidate.pt"


@torch.no_grad()
def evaluate_recovery(plan, name):
    if name not in ("S0", "S1"):
        raise ValueError("Unknown recovery candidate")
    root = Path(plan["output"]) / name
    checkpoint_path = root / "candidate.pt"
    output = root / "diagnostic"
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    device = gpu_setup()
    data = supervision(plan)
    model, preprocess, checkpoint, o3, pta = load_composite(checkpoint_path, device)
    model.requires_grad_(False); o3.requires_grad_(False); pta.requires_grad_(False)
    metadata = checkpoint.get("prelim75_training", {})
    if metadata.get("parent_sha256") != plan["parent_sha256"] or metadata.get("plan_id") != PLAN_ID:
        raise ValueError("Recovery candidate lineage mismatch")
    if metadata.get("prior_in_inference") is not False:
        raise ValueError("A prior entered recovery inference")
    frame = pd.read_csv(plan["val_csv"])
    paths = [canonical_sample_path(path) for path in frame.image_path.astype(str)]
    if len(paths) != 10316 or len(set(paths)) != len(paths):
        raise ValueError("Diagnostic row identity mismatch")
    training_index = {path: index for index, path in enumerate(data["paths"])}
    indices = torch.tensor([training_index[path] for path in paths])
    labels = torch.tensor(frame.label.to_numpy(), dtype=torch.long)
    if not torch.equal(labels, data["labels"][indices]):
        raise ValueError("Diagnostic labels differ from the frozen fitting list")
    dataset = OfficialImages(paths, plan["train_root"], preprocess)
    stream = loader(dataset, 64, plan)
    predictions = []
    for step, batch in enumerate(stream):
        images = batch["images"].to(device, non_blocking=True)
        global_logits, local_logits, _ = _teacher_views(model, o3, pta, images)
        scores = fuse_teacher_probabilities(global_logits, local_logits)["probabilities"]
        if scores.shape[1] != 500 or not torch.isfinite(scores).all():
            raise RuntimeError("Invalid recovery diagnostic scores")
        predictions.append(scores.argmax(1).cpu())
        if step % 20 == 0:
            status = {
                "status": "running", "candidate": name,
                "completed_samples": sum(len(prediction) for prediction in predictions),
                "elapsed_seconds": time.monotonic() - start,
            }
            atomic_json_dump(status, output / "status.json")
            print(json.dumps(status), flush=True)
    prediction = torch.cat(predictions)
    fields = {
        "labels": labels,
        "clean_probability": data["clean_probability"][indices],
        "pseudo_labels": data["pseudo_label"][indices],
        "correction_alpha": data["correction_alpha"][indices],
    }
    metrics = prediction_metrics(prediction, **fields, num_classes=500, clean_core_threshold=.7)
    parent = json.loads(Path(plan["parent_diagnostic"]).read_text())
    deltas = {key: 100 * (metrics[key] - parent["metrics"][key]) for key in ("raw_micro", "clean_core_micro")}
    result = {
        "status": "complete", "candidate": name, "validation_scope": "overlap_diagnostic",
        "independent_generalization_claim": False, "metrics": metrics,
        "delta_pp_vs_V1": deltas,
        "engineering_stop": any(value < -2.0 for value in deltas.values()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parent_checkpoint_sha256": plan["parent_sha256"],
        "parent_diagnostic_sha256": plan["parent_diagnostic_sha256"],
        "val_csv_sha256": plan["val_csv_sha256"],
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "prior_in_inference": False,
        "real_candidate_attention_recomputed": True,
        "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
        "online_accuracy": None,
    }
    _atomic_torch_save({"paths": paths, "prediction": prediction, **fields}, output / "predictions.pt")
    atomic_json_dump(result, output / "result.json")
    atomic_json_dump(result, output / "status.json")
    print(json.dumps(result, indent=2), flush=True)
    return result


def deliver_recovery(plan, name):
    """Produce one fixed, uncalibrated, single-checkpoint package and validate it."""
    if name not in ("S0", "S1"):
        raise ValueError("Unknown recovery candidate")
    root = Path(plan["output"]) / name
    diagnostic = json.loads((root / "diagnostic/result.json").read_text())
    if diagnostic.get("status") != "complete" or diagnostic.get("engineering_stop", True):
        raise ValueError("Recovery candidate stopped by the fixed engineering budget")
    checkpoint = root / "candidate.pt"
    if sha256_file(checkpoint) != diagnostic["checkpoint_sha256"]:
        raise ValueError("Candidate changed after the real diagnostic")
    output = root / "submission"
    if output.exists():
        raise FileExistsError("Do not overwrite an existing recovery submission")
    command = [
        sys.executable, "-u", "-m", "aegis_clip.cli.infer",
        "--checkpoint", str(checkpoint), "--output-dir", str(output),
        "--local-view", "attention_multiscale", "--local-crop-sizes", "112,128,144,160",
        "--local-scale-weights", "0.20,0.30,0.40,0.10", "--local-top-k", "5",
        "--local-weight", "0.40", "--local-temperature", "1.5",
        "--adapt-local-features", "--adapt-part-token-features",
        "--tta", "horizontal_flip", "--tta-fusion", "mean_probabilities",
        "--tta-temperature", "1.5", "--tta-view-weight", "0.50",
        "--acknowledge-local-view-risk", "--acknowledge-tta-risk", "--batch-size", "64",
    ]
    with (root / "inference.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    # The competition specification shows comma+space.  Preserve the validated
    # generic predictions while normalizing the final byte representation.
    rows = []
    with (output / "pred_results.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 2:
                raise ValueError("Generic inference emitted a malformed CSV row")
            rows.append((row[0].strip(), row[1].strip()))
    (output / "pred_results.csv").write_text(
        "".join(f"{image_name}, {label}\n" for image_name, label in rows), encoding="utf-8"
    )
    with ZipFile(output / "submission.zip", "w", ZIP_DEFLATED) as archive:
        archive.write(output / "pred_results.csv", arcname="pred_results.csv")
    inference_manifest = json.loads((output / "manifest.json").read_text())
    inference_manifest["prediction_csv_sha256"] = sha256_file(output / "pred_results.csv")
    inference_manifest["submission_zip_sha256"] = sha256_file(output / "submission.zip")
    inference_manifest["csv_delimiter"] = "comma_space"
    atomic_json_dump(inference_manifest, output / "manifest.json")
    repo = repository_root()
    check_command = [
        sys.executable, str(repo / "scripts/check_submission.py"),
        "--test_dir", plan["test_root"], "--num-classes", "500",
        "--csv", str(output / "pred_results.csv"), "--zip", str(output / "submission.zip"),
    ]
    with (root / "submission_validation.log").open("w") as log:
        subprocess.run(check_command, check=True, stdout=log, stderr=subprocess.STDOUT)
    raw = (output / "pred_results.csv").read_bytes()
    with ZipFile(output / "submission.zip") as archive:
        if archive.namelist() != ["pred_results.csv"] or archive.read("pred_results.csv") != raw:
            raise ValueError("Recovery ZIP/CSV byte identity failed")
    if len(rows) != 24967 or len({row[0] for row in rows}) != 24967:
        raise ValueError("Official recovery test coverage mismatch")
    labels = {row[1] for row in rows}
    report = {
        "candidate": name,
        "status": "submission_ready_pending_platform",
        "checkpoint_sha256": sha256_file(checkpoint),
        "csv_sha256": sha256_file(output / "pred_results.csv"),
        "zip_sha256": sha256_file(output / "submission.zip"),
        "inference_manifest_sha256": sha256_file(output / "manifest.json"),
        "resolved_config_sha256": sha256_file(root / "resolved_config.json"),
        "source_manifest_sha256": sha256_file(root / "source_manifest.json"),
        "inference_command": command,
        "validation_command": check_command,
        "validation_exit_code": 0,
        "rows": 24967,
        "unique_prediction_classes": len(labels),
        "prediction_coverage_forced_to_500": False,
        "zip_internal_csv_byte_equal": True,
        "csv_delimiter": "comma_space",
        "diagnostic": diagnostic,
        "single_checkpoint": True,
        "teacher_checkpoint_required_for_final_inference": False,
        "teacher_cache_required_for_final_inference": False,
        "prior_in_inference": False,
        "online_accuracy": None,
        "online_exact_correct": None,
        "actual_platform_upload_time": None,
    }
    atomic_json_dump(report, root / "candidate_report.json")
    atomic_json_dump(report, root / "status.json")
    print(json.dumps(report, indent=2), flush=True)
    return report
