"""Fixed PRELIM75 v4 continuation with a bounded trusted-global CE mix.

Both candidates warm-start weights from v3 S0 and create fresh optimizer and
scheduler state.  C0 preserves the v3 supervised objective.  C1 changes only
the per-sample global loss for a frozen cohort derived from the original
training supervision; local loss and the same-pixel official-CLIP anchor are
unchanged.  V3 class-probability and recovery-cohort assets are never read.
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
from aegis_clip.prelim75 import (
    OfficialImages,
    gpu_setup,
    load_composite,
    loader,
    repository_root,
    saved_candidate,
    source_manifest,
    supervision,
)
from aegis_clip.prelim75_joint import crop_with_bound_boxes, fixed_choices, official_anchor
from aegis_clip.prelim75_recovery import _teacher_views, fuse_teacher_probabilities
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.trainer import _per_sample_loss, _warmup_cosine


PLAN_ID = "PRELIM75_V4_20260916"
EXPECTED_PROTOCOL = {
    "trusted_threshold": 0.90,
    "minimum_groups": 512,
    "minimum_classes": 100,
    "ce_beta": 0.25,
    "epochs": 3,
    "schedule_epochs": 18,
    "effective_batch_size": 32,
    "epoch_numbers": [7, 8, 9],
    "global_weight": 0.6,
    "local_weight": 0.4,
    "anchor_weight": 2.0,
    "local_scales": [112, 128, 144, 160],
    "local_scale_weights": [0.2, 0.3, 0.4, 0.1],
    "temperature": 1.5,
    "gpu_budget_seconds": 28800,
}


def _paths_sha256(paths):
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


def _tensor_digest_update(digest, name, tensor):
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode())
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())


def original_supervision_sha256(data):
    digest = hashlib.sha256()
    digest.update("\n".join(data["paths"]).encode())
    for name in ("labels", "weights", "targets", "clean_probability", "pseudo_label", "correction_alpha"):
        _tensor_digest_update(digest, name, data[name])
    return digest.hexdigest()


def load_trusted_ce_plan(path):
    """Load only the registered v4 plan and verify each immutable asset."""
    source = Path(path).resolve()
    plan = yaml.safe_load(source.read_text())
    path_keys = {
        "weight_parent", "fallback_submission", "parent_diagnostic",
        "geometry_manifest", "geometry_paths", "geometry_boxes", "trust",
        "train_csv", "val_csv", "class_mapping", "groups", "train_root",
        "test_root", "output",
    }
    hash_keys = {
        "weight_parent", "fallback_submission", "parent_diagnostic",
        "geometry_manifest", "geometry_paths", "geometry_boxes", "trust",
        "train_csv", "val_csv", "class_mapping", "groups",
    }
    allowed = {
        "plan_id", "seed", "num_workers", "protocol", "geometry_origin_sha256",
        "original_supervision_sha256", "trusted_mask_semantic_sha256",
        *path_keys, *(key + "_sha256" for key in hash_keys),
    }
    if set(plan) != allowed or plan.get("plan_id") != PLAN_ID or plan.get("seed") != 42:
        raise ValueError("Unknown or incomplete fixed PRELIM75 v4 plan")
    if plan.get("protocol") != EXPECTED_PROTOCOL or plan.get("num_workers") != 4:
        raise ValueError("PRELIM75 v4 protocol differs from the registered fixed plan")
    for key in path_keys:
        plan[key] = str((source.parent / plan[key]).resolve())
    for key in hash_keys:
        if sha256_file(plan[key]) != plan[key + "_sha256"]:
            raise ValueError(f"Frozen PRELIM75 v4 asset hash mismatch: {key}")
    train_root = Path(plan["train_root"]).resolve()
    test_root = Path(plan["test_root"]).resolve()
    if train_root == test_root or train_root.is_relative_to(test_root) or test_root.is_relative_to(train_root):
        raise ValueError("Official train and test roots must be disjoint")
    plan["parent"] = plan["weight_parent"]
    plan["parent_sha256"] = plan["weight_parent_sha256"]
    plan["_config_path"] = str(source)
    plan["_config_sha256"] = sha256_file(source)
    return plan


def _validate_parent(plan):
    checkpoint = torch.load(plan["weight_parent"], map_location="cpu", weights_only=False)
    metadata = checkpoint.get("prelim75_training", {})
    experiment_id = checkpoint.get("config", {}).get("project", {}).get("experiment_id")
    loss = checkpoint.get("config", {}).get("loss", {})
    if experiment_id != "PRELIM75_V3_S0":
        raise ValueError("Registered v4 weight parent is not v3 S0")
    if metadata.get("name") != "S0" or metadata.get("plan_id") != "PRELIM75_V3_20260915":
        raise ValueError("Registered parent metadata is not the completed v3 S0 candidate")
    if metadata.get("recovery_kd") is not False or metadata.get("prior_in_inference") is not False:
        raise ValueError("v4 parent unexpectedly contains recovery KD or an inference prior")
    if loss.get("name") != "gce" or loss.get("gce_q") != 0.5 or loss.get("epsilon") != 1e-7:
        raise ValueError("S0 GCE definition changed")
    if "local_feature_adapter" not in checkpoint or "part_token_adapter" not in checkpoint:
        raise ValueError("S0 checkpoint is missing one or both local adapters")
    diagnostic = json.loads(Path(plan["parent_diagnostic"]).read_text())
    if (diagnostic.get("status") != "complete" or diagnostic.get("engineering_stop") is not False
            or diagnostic.get("checkpoint_sha256") != plan["weight_parent_sha256"]
            or diagnostic.get("prior_in_inference") is not False):
        raise ValueError("Registered S0 diagnostic is incomplete or mismatched")
    return checkpoint, diagnostic


def load_v1_geometry(plan, paths, verify=True):
    """Load only v3's V1 box geometry, never probabilities or recovery state."""
    manifest = json.loads(Path(plan["geometry_manifest"]).read_text())
    cached_paths = json.loads(Path(plan["geometry_paths"]).read_text())
    required = {
        "status": "complete",
        "plan_id": "PRELIM75_V3_20260915",
        "parent_sha256": plan["geometry_origin_sha256"],
        "train_csv_sha256": plan["train_csv_sha256"],
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "trust_sha256": plan["trust_sha256"],
        "groups_sha256": plan["groups_sha256"],
        "paths_sha256": _paths_sha256(paths),
        "rows": 103218,
        "boxes_shape": [103218, 2, 4, 4],
        "test_rows_processed": 0,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"V1 geometry manifest identity mismatch: {key}")
    if cached_paths != paths:
        raise ValueError("V1 geometry path order differs from the official training list")
    if manifest.get("files_sha256", {}).get("paths.json") != plan["geometry_paths_sha256"]:
        raise ValueError("V1 geometry paths are not bound by the manifest")
    if manifest.get("files_sha256", {}).get("boxes.npy") != plan["geometry_boxes_sha256"]:
        raise ValueError("V1 geometry boxes are not bound by the manifest")
    geometry_protocol = manifest.get("protocol", {})
    if (geometry_protocol.get("local_scales") != EXPECTED_PROTOCOL["local_scales"]
            or geometry_protocol.get("local_scale_weights") != EXPECTED_PROTOCOL["local_scale_weights"]
            or geometry_protocol.get("attention_top_k") != 5):
        raise ValueError("V1 geometry protocol changed")
    if verify:
        if sha256_file(plan["geometry_manifest"]) != plan["geometry_manifest_sha256"]:
            raise ValueError("V1 geometry manifest changed")
        if sha256_file(plan["geometry_paths"]) != plan["geometry_paths_sha256"]:
            raise ValueError("V1 geometry paths changed")
        if sha256_file(plan["geometry_boxes"]) != plan["geometry_boxes_sha256"]:
            raise ValueError("V1 geometry boxes changed")
    boxes = np.load(plan["geometry_boxes"], mmap_mode="r")
    if boxes.shape != (103218, 2, 4, 4) or boxes.dtype != np.int16:
        raise ValueError("V1 geometry box tensor shape or dtype changed")
    flat = np.asarray(boxes).reshape(-1, 4)
    valid = ((flat[:, 0] >= 0) & (flat[:, 0] < flat[:, 2]) & (flat[:, 2] <= 224)
             & (flat[:, 1] >= 0) & (flat[:, 1] < flat[:, 3]) & (flat[:, 3] <= 224))
    if not bool(valid.all()):
        raise ValueError("V1 geometry contains an out-of-bounds box")
    return boxes, manifest


def build_trusted_cohort(data, threshold=0.90, minimum_groups=512, minimum_classes=100):
    """Build the fixed C1 mask only from original supervision and labels."""
    paths = data["paths"]
    labels = data["labels"].long()
    weights = data["weights"]
    targets = data["targets"]
    clean = data["clean_probability"]
    groups = [str(group) for group in data["group_ids"]]
    n = len(paths)
    if (targets.shape != (n, 500) or labels.shape != (n,) or weights.shape != (n,)
            or clean.shape != (n,) or len(groups) != n):
        raise ValueError("Unexpected original supervision shape")
    positive = weights > 0
    high_trust = clean >= threshold
    row = torch.arange(len(labels))
    exact_original_one_hot = (targets[row, labels] == 1) & (torch.count_nonzero(targets, dim=1) == 1)
    group_label = {}
    conflicting_groups = set()
    for group, label in zip(groups, labels.tolist()):
        previous = group_label.setdefault(group, label)
        if previous != label:
            conflicting_groups.add(group)
    conflict_free = torch.tensor([group not in conflicting_groups for group in groups], dtype=torch.bool)
    after_positive = positive
    after_trust = after_positive & high_trust
    after_target = after_trust & exact_original_one_hot
    mask = after_target & conflict_free
    indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    eligible_groups = {groups[index] for index in indices}
    eligible_classes = set(labels[mask].tolist())
    gate = len(eligible_groups) >= minimum_groups and len(eligible_classes) >= minimum_classes
    total_weight = float(weights.double().sum())
    eligible_weight = float(weights[mask].double().sum())
    class_groups = [set() for _ in range(500)]
    class_row_counts = np.zeros(500, dtype=np.int64)
    class_weight_mass = np.zeros(500, dtype=np.float64)
    for index in indices:
        class_index = int(labels[index])
        class_groups[class_index].add(groups[index])
        class_row_counts[class_index] += 1
        class_weight_mass[class_index] += float(weights[index])
    class_rows = []
    for class_index in range(500):
        class_rows.append({
            "class_index": class_index,
            "eligible_rows": int(class_row_counts[class_index]),
            "eligible_groups": len(class_groups[class_index]),
            "eligible_weight_mass": float(class_weight_mass[class_index]),
        })
    report = {
        "status": "ready_for_fixed_C0_C1" if gate else "skipped_insufficient_trusted_support",
        "gate_passed": bool(gate),
        "official_total_rows": len(paths),
        "positive_weight_rows": int(after_positive.sum()),
        "rows_after_trusted_threshold": int(after_trust.sum()),
        "rows_after_exact_original_target": int(after_target.sum()),
        "eligible_rows": int(mask.sum()),
        "eligible_unique_groups": len(eligible_groups),
        "eligible_classes": len(eligible_classes),
        "conflicting_groups": len(conflicting_groups),
        "rows_excluded_by_group_conflict_after_other_conditions": int((after_target & ~conflict_free).sum()),
        "eliminated_by_zero_weight": int((~positive).sum()),
        "eliminated_by_trust_after_positive": int(after_positive.sum() - after_trust.sum()),
        "eliminated_by_target_after_trust": int(after_trust.sum() - after_target.sum()),
        "minimum_groups": minimum_groups,
        "minimum_classes": minimum_classes,
        "trusted_threshold": threshold,
        "original_weight_mass": total_weight,
        "eligible_weight_mass": eligible_weight,
        "eligible_weight_mass_fraction": eligible_weight / total_weight,
        "mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
        "known_limitations": [
            "clean_probability is an old model trust score, not an independently calibrated correctness probability.",
            "The support gate controls training expenditure and does not certify label quality.",
            "Content conflicts use only original official labels and are not manually relabeled.",
        ],
    }
    return mask, report, class_rows


def preflight(plan):
    checkpoint, diagnostic = _validate_parent(plan)
    data = supervision(plan)
    boxes, geometry = load_v1_geometry(plan, data["paths"])
    mask, support, _ = build_trusted_cohort(
        data,
        threshold=plan["protocol"]["trusted_threshold"],
        minimum_groups=plan["protocol"]["minimum_groups"],
        minimum_classes=plan["protocol"]["minimum_classes"],
    )
    supervision_hash = original_supervision_sha256(data)
    if supervision_hash != plan["original_supervision_sha256"]:
        raise ValueError("Original v4 supervision differs from the registered tensor hash")
    if support["mask_sha256"] != plan["trusted_mask_semantic_sha256"]:
        raise ValueError("Trusted CE mask differs from the registered semantic hash")
    train_root = Path(plan["train_root"]).resolve()
    test_root = Path(plan["test_root"]).resolve()
    if any((train_root / path).resolve().is_relative_to(test_root) for path in data["paths"]):
        raise ValueError("Test path entered the v4 training list")
    return {
        "status": "passed",
        "plan_id": PLAN_ID,
        "config_sha256": plan["_config_sha256"],
        "weight_parent_experiment_id": checkpoint["config"]["project"]["experiment_id"],
        "weight_parent_sha256": plan["weight_parent_sha256"],
        "fallback_submission_sha256": plan["fallback_submission_sha256"],
        "parent_diagnostic_sha256": plan["parent_diagnostic_sha256"],
        "parent_raw_micro": diagnostic["metrics"]["raw_micro"],
        "parent_clean_core_micro": diagnostic["metrics"]["clean_core_micro"],
        "geometry_origin_sha256": geometry["parent_sha256"],
        "geometry_manifest_sha256": plan["geometry_manifest_sha256"],
        "geometry_boxes_sha256": plan["geometry_boxes_sha256"],
        "geometry_shape": list(boxes.shape),
        "train_csv_sha256": plan["train_csv_sha256"],
        "trust_sha256": plan["trust_sha256"],
        "groups_sha256": plan["groups_sha256"],
        "official_train_rows": len(data["paths"]),
        "original_supervision_sha256": supervision_hash,
        "trusted_support": support,
        "trusted_mask_rows": int(mask.sum()),
        "teacher_probabilities_read": False,
        "recovery_cohort_read": False,
        "test_used_for_training": False,
        "platform_upload_authorized": False,
        "protocol": copy.deepcopy(plan["protocol"]),
    }


def prepare_trusted_assets(plan):
    output = Path(plan["output"]) / "trusted_support"
    output.mkdir(parents=True, exist_ok=False)
    data = supervision(plan)
    mask, report, class_rows = build_trusted_cohort(
        data,
        threshold=plan["protocol"]["trusted_threshold"],
        minimum_groups=plan["protocol"]["minimum_groups"],
        minimum_classes=plan["protocol"]["minimum_classes"],
    )
    supervision_hash = original_supervision_sha256(data)
    if supervision_hash != plan["original_supervision_sha256"]:
        raise ValueError("Original v4 supervision changed before mask preparation")
    if report["mask_sha256"] != plan["trusted_mask_semantic_sha256"]:
        raise ValueError("Trusted CE mask changed before mask preparation")
    mask_path = output / "trusted_ce_mask.npy"
    np.save(mask_path, mask.numpy())
    with (output / "class_support.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader(); writer.writerows(class_rows)
    report.update({
        "plan_id": PLAN_ID,
        "original_supervision_sha256": supervision_hash,
        "train_csv_sha256": plan["train_csv_sha256"],
        "trust_sha256": plan["trust_sha256"],
        "groups_sha256": plan["groups_sha256"],
        "paths_sha256": _paths_sha256(data["paths"]),
        "files_sha256": {
            "trusted_ce_mask.npy": sha256_file(mask_path),
            "class_support.csv": sha256_file(output / "class_support.csv"),
        },
        "teacher_probabilities_read": False,
        "recovery_weights_read": False,
        "recovery_mask_read": False,
    })
    atomic_json_dump(report, output / "report.json")
    return report


def read_v4_assets(plan, verify=True):
    data = supervision(plan)
    boxes, geometry = load_v1_geometry(plan, data["paths"], verify=verify)
    root = Path(plan["output"]) / "trusted_support"
    report = json.loads((root / "report.json").read_text())
    mask_path = root / "trusted_ce_mask.npy"
    class_path = root / "class_support.csv"
    if verify:
        if sha256_file(mask_path) != report["files_sha256"]["trusted_ce_mask.npy"]:
            raise ValueError("Trusted CE mask file changed")
        if sha256_file(class_path) != report["files_sha256"]["class_support.csv"]:
            raise ValueError("Trusted CE class report changed")
    regenerated, regenerated_report, _ = build_trusted_cohort(
        data,
        threshold=plan["protocol"]["trusted_threshold"],
        minimum_groups=plan["protocol"]["minimum_groups"],
        minimum_classes=plan["protocol"]["minimum_classes"],
    )
    mask = np.load(mask_path, mmap_mode="r")
    if mask.shape != (103218,) or mask.dtype != np.bool_ or not np.array_equal(mask, regenerated.numpy()):
        raise ValueError("Trusted CE mask differs from regenerated original supervision")
    for key in ("gate_passed", "eligible_rows", "eligible_unique_groups", "eligible_classes", "mask_sha256"):
        if report.get(key) != regenerated_report.get(key):
            raise ValueError(f"Trusted CE support report changed: {key}")
    if report.get("original_supervision_sha256") != original_supervision_sha256(data):
        raise ValueError("Original v4 supervision identity changed")
    if (report.get("original_supervision_sha256") != plan["original_supervision_sha256"]
            or report.get("mask_sha256") != plan["trusted_mask_semantic_sha256"]):
        raise ValueError("Prepared v4 supervision or mask is not the registered fixed asset")
    return data, mask, boxes, geometry, report


def trusted_ce_mix(base_gce, logits, targets, eligible, beta):
    """Mix CE into already-computed global GCE on only eligible rows."""
    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("logits/targets must be [B,C]")
    if base_gce.shape != (len(logits),) or eligible.shape != base_gce.shape:
        raise ValueError("base_gce/eligible must be [B]")
    if eligible.dtype != torch.bool:
        raise TypeError("eligible must be bool")
    if targets.requires_grad:
        raise ValueError("Frozen supervision targets must not require gradients")
    if not 0.0 <= float(beta) <= 0.25:
        raise ValueError("Fixed v4 beta must be in [0,0.25]")
    if float(beta) == 0.0 or not bool(eligible.any()):
        return base_gce
    ce = -(targets[eligible].float() * F.log_softmax(logits[eligible].float(), dim=-1)).sum(dim=-1)
    result = base_gce.clone()
    result[eligible] = (1.0 - float(beta)) * base_gce[eligible] + float(beta) * ce
    return result


def trusted_beta(step, steps_per_epoch):
    if step < 0 or steps_per_epoch <= 0:
        raise ValueError("Invalid trusted CE schedule position")
    return 0.25 * min(float(step) / max(1, steps_per_epoch - 1), 1.0)


def _candidate_config(checkpoint, plan, name):
    config = copy.deepcopy(checkpoint["config"])
    if checkpoint.get("config", {}).get("project", {}).get("experiment_id") != "PRELIM75_V3_S0":
        raise ValueError("v4 parent experiment id must come from actual S0 metadata")
    config["project"]["experiment_id"] = f"PRELIM75_V4_{name}"
    for key in ("train_csv", "val_csv", "class_mapping", "train_root", "test_root"):
        config["data"][key] = plan[key]
    config["data"]["train_augmentation"] = "clip_center_crop"
    config["train"].update({
        "init_checkpoint": plan["weight_parent"], "require_lineage_for_init_checkpoint": True,
        "epochs": 3, "schedule_epochs": 18, "batch_size": 32,
        "backbone_lr": 1e-6, "backbone_weight_decay": 0.0,
        "head_lr": 1.5e-7, "head_weight_decay": 1e-4, "amp": False,
    })
    config["lineage"] = {
        "enabled": True, "parent_experiment_id": "PRELIM75_V3_S0",
        "parent_train_csv": plan["train_csv"], "parent_val_csv": plan["val_csv"],
        "require_same_train": True, "require_same_val": True,
        "allow_parent_val_in_child_train": True, "allow_parent_train_in_child_val": True,
    }
    config["evaluation"]["selection_policy"] = "last_epoch"
    config["output"]["root"] = plan["output"]
    validate_config(config)
    return config


def _grad_norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().float().pow(2).sum()) for p in parameters if p.grad is not None))


def _classification_terms(logits, targets, weights, eligible, denominator, loss_config, epoch, beta):
    base = _per_sample_loss(logits, targets, loss_config, epoch)
    mixed = trusted_ce_mix(base, logits, targets, eligible, beta)
    base_loss = (base * weights).sum() / denominator
    mixed_loss = (mixed * weights).sum() / denominator
    with torch.no_grad():
        if bool(eligible.any()):
            ce = -(targets[eligible].float() * F.log_softmax(logits[eligible].float(), dim=-1)).sum(1)
            eligible_ce = (ce * weights[eligible]).sum() / denominator
        else:
            eligible_ce = logits.detach().sum() * 0.0
    return base_loss, mixed_loss, eligible_ce


def smoke_v4(plan):
    """Select one common microbatch in an isolated process without optimizer state."""
    output = Path(plan["output"]) / "smoke.json"
    if output.exists():
        raise FileExistsError("PRELIM75 v4 smoke result already exists")
    start = time.monotonic(); device = gpu_setup()
    data, mask, boxes, _, report = read_v4_assets(plan)
    if not report["gate_passed"]:
        candidate_for_smoke = "C0"
    else:
        candidate_for_smoke = "C1_worst_case_beta_0.25"
    model, preprocess, checkpoint, o3, pta = load_composite(plan["weight_parent"], device)
    model.float().train(); o3.float().train(); pta.float().train()
    model.classifier.requires_grad_(True); o3.requires_grad_(True); pta.requires_grad_(True)
    anchor_teacher = official_anchor(device)
    dataset = OfficialImages(data["paths"], plan["train_root"], preprocess)
    generator = torch.Generator().manual_seed(42)
    stream = loader(dataset, 32, plan, shuffle=True, generator=generator)
    batch = next(iter(stream)); indices = batch["index"].numpy()
    flips, scales = fixed_choices(data["paths"], 7)
    denominator = data["weights"][indices].sum().to(device).clamp_min(1e-8)

    def attempt(microbatch):
        model.zero_grad(set_to_none=True); o3.zero_grad(set_to_none=True); pta.zero_grad(set_to_none=True)
        total = None
        for offset in range(0, len(indices), microbatch):
            sub_idx = indices[offset:offset + microbatch]
            images = batch["images"][offset:offset + microbatch].to(device, non_blocking=True)
            selected_flips = flips[sub_idx]
            if selected_flips.any():
                images[torch.tensor(selected_flips, device=device)] = images[torch.tensor(selected_flips, device=device)].flip(3)
            targets = data["targets"][sub_idx].to(device); weights = data["weights"][sub_idx].to(device)
            eligible = torch.from_numpy(np.asarray(mask[sub_idx]).copy()).to(device)
            with torch.no_grad():
                reference = F.normalize(anchor_teacher(images).float(), dim=1)
            global_logits, features = model(images=images, return_features=True)
            _, global_loss, _ = _classification_terms(
                global_logits, targets, weights, eligible, denominator, checkpoint["config"]["loss"], 7,
                0.25 if report["gate_passed"] else 0.0,
            )
            anchor = (1.0 - F.cosine_similarity(features.float(), reference, dim=1)).sum() / len(indices)
            bound = boxes[sub_idx, selected_flips.astype(int), scales[sub_idx]]
            local_logits = adapted_dual_local_view_logits(model, o3, pta, crop_with_bound_boxes(images, bound))
            local_base = _per_sample_loss(local_logits, targets, checkpoint["config"]["loss"], 7)
            local_loss = (local_base * weights).sum() / denominator
            loss = .6 * global_loss + .4 * local_loss + 2.0 * anchor
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite PRELIM75 v4 smoke loss")
            loss.backward(); total = loss.detach() if total is None else total + loss.detach()
        return float(total)

    microbatch = 32; downgraded = False
    try:
        smoke_loss = attempt(microbatch)
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True); o3.zero_grad(set_to_none=True); pta.zero_grad(set_to_none=True)
        import gc
        gc.collect(); torch.cuda.empty_cache(); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        microbatch = 16; downgraded = True; smoke_loss = attempt(microbatch)
    result = {
        "status": "passed", "candidate_scope": candidate_for_smoke,
        "effective_batch_size": 32, "microbatch_size": microbatch,
        "accumulation": 32 // microbatch, "oom_downgrade": downgraded,
        "loss": smoke_loss, "batch_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "eligible_rows_in_batch": int(np.asarray(mask[indices]).sum()),
        "optimizer_created": False, "optimizer_step_performed": False,
        "formal_rng_state_consumed": False, "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
    }
    atomic_json_dump(result, output)
    return result


def train_trusted_ce(plan, name):
    """Train C0 or C1 independently from S0 with identical order and geometry."""
    if name not in ("C0", "C1"):
        raise ValueError("Only PRELIM75 v4 C0/C1 are allowed")
    output = Path(plan["output"]) / name
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); device = gpu_setup()
    data, mask, boxes, geometry, support = read_v4_assets(plan)
    if name == "C1" and not support["gate_passed"]:
        raise ValueError("C1 trusted-support gate is closed")
    smoke = json.loads((Path(plan["output"]) / "smoke.json").read_text())
    if smoke.get("status") != "passed" or smoke.get("microbatch_size") not in (16, 32):
        raise ValueError("Missing or invalid isolated v4 smoke result")
    microbatch = int(smoke["microbatch_size"])
    model, preprocess, checkpoint, o3, pta = load_composite(plan["weight_parent"], device)
    model.float(); o3.float(); pta.float()
    licensed_visual = {n for n, p in model.visual.named_parameters() if p.requires_grad}
    if "conv1.weight" in licensed_visual or "positional_embedding" in licensed_visual:
        raise ValueError("S0 visual permission mask changed")
    model.classifier.requires_grad_(True); o3.requires_grad_(True); pta.requires_grad_(True)
    model.train(); o3.train(); pta.train()
    anchor_teacher = official_anchor(device)
    config = _candidate_config(checkpoint, plan, name)
    run_lineage_audit(
        config, child_train_csv=plan["train_csv"], child_val_csv=plan["val_csv"],
        checkpoint_path=plan["weight_parent"], output_path=output / "split_lineage_audit.json",
    )
    atomic_json_dump(config, output / "resolved_config.json")
    atomic_json_dump(source_manifest(), output / "source_manifest.json")
    frozen = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}
    visual_parameters = [p for p in model.visual.parameters() if p.requires_grad]
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
        "candidate": name, "plan_id": PLAN_ID, "independent_parent_load": True,
        "weight_parent_sha256": plan["weight_parent_sha256"],
        "weight_parent_experiment_id": checkpoint["config"]["project"]["experiment_id"],
        "geometry_origin_sha256": plan["geometry_origin_sha256"],
        "geometry_manifest_sha256": plan["geometry_manifest_sha256"],
        "geometry_boxes_sha256": plan["geometry_boxes_sha256"],
        "mask_file_sha256": support["files_sha256"]["trusted_ce_mask.npy"],
        "mask_semantic_sha256": support["mask_sha256"],
        "original_supervision_sha256": support["original_supervision_sha256"],
        "eligible_rows": support["eligible_rows"], "eligible_groups": support["eligible_unique_groups"],
        "eligible_classes": support["eligible_classes"],
        "epochs": 3, "epoch_numbers": [7, 8, 9], "schedule_epochs": 18,
        "batch_size": 32, "effective_batch_size": 32, "microbatch_size": microbatch,
        "accumulation": 32 // microbatch, "fresh_optimizer": True, "fresh_scheduler": True,
        "global_weight": .6, "local_weight": .4, "anchor_weight": 2.0,
        "global_trusted_ce_beta_max": .25 if name == "C1" else 0.0,
        "local_trusted_ce_beta": 0.0, "gce_q": .5, "classification_temperature": 1.0,
        "supervision": "bitwise_verified_original_w_targets_unchanged",
        "geometry": "native_224_stateless_flip_frozen_V1_boxes",
        "anchor": "frozen_official_CLIP_same_global_pixel_tensor_online",
        "teacher_probabilities_read": False, "recovery_weights_read": False,
        "prior_in_inference": False, "licensed_visual_parameters": sorted(licensed_visual),
        "selection": "fixed_epoch_9_last_epoch",
    }
    atomic_json_dump(recipe, output / "training_recipe.json")
    history = []; optimizer_step = 0; first_step_audit = None; ce_delta_gradient_audit = None
    clipped_steps = 0
    for sample_epoch in (7, 8, 9):
        epoch_start = time.monotonic(); flips, scales = fixed_choices(data["paths"], sample_epoch)
        totals = {
            "global_gce": 0.0, "global_mixed": 0.0, "eligible_global_ce": 0.0,
            "local_gce": 0.0, "anchor": 0.0, "loss": 0.0, "beta": 0.0,
            "samples": 0, "eligible_hits": 0, "eligible_weight_mass": 0.0,
            "gradient_norm_sum": 0.0, "gradient_norm_max": 0.0,
        }
        for step, batch in enumerate(stream):
            indices = batch["index"].numpy()
            denominator = data["weights"][indices].sum().to(device).clamp_min(1e-8)
            batch_eligible = torch.from_numpy(np.asarray(mask[indices]).copy()).to(device)
            beta = trusted_beta(optimizer_step, steps_per_epoch) if name == "C1" else 0.0
            optimizer.zero_grad(set_to_none=True)
            parts = {key: 0.0 for key in ("global_gce", "global_mixed", "eligible_global_ce", "local_gce", "anchor", "loss")}
            for offset in range(0, len(indices), microbatch):
                sub_idx = indices[offset:offset + microbatch]
                images = batch["images"][offset:offset + microbatch].to(device, non_blocking=True)
                selected_flips = flips[sub_idx]
                if selected_flips.any():
                    flip_mask = torch.tensor(selected_flips, device=device)
                    images[flip_mask] = images[flip_mask].flip(3)
                targets = data["targets"][sub_idx].to(device); weights = data["weights"][sub_idx].to(device)
                eligible = batch_eligible[offset:offset + len(sub_idx)]
                with torch.no_grad():
                    reference = F.normalize(anchor_teacher(images).float(), dim=1)
                global_logits, features = model(images=images, return_features=True)
                global_gce, global_mixed, eligible_ce = _classification_terms(
                    global_logits, targets, weights, eligible, denominator,
                    checkpoint["config"]["loss"], sample_epoch, beta,
                )
                anchor = (1.0 - F.cosine_similarity(features.float(), reference, dim=1)).sum() / len(indices)
                bound = boxes[sub_idx, selected_flips.astype(int), scales[sub_idx]]
                local_logits = adapted_dual_local_view_logits(model, o3, pta, crop_with_bound_boxes(images, bound))
                local_base = _per_sample_loss(local_logits, targets, checkpoint["config"]["loss"], sample_epoch)
                local_gce = (local_base * weights).sum() / denominator
                loss = .6 * global_mixed + .4 * local_gce + 2.0 * anchor
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite PRELIM75 v4 continuation loss")
                if (name == "C1" and ce_delta_gradient_audit is None and beta > 0
                        and bool(eligible.any())):
                    probes = torch.autograd.grad(
                        .6 * (global_mixed - global_gce),
                        [model.visual.proj, model.classifier.weight], retain_graph=True,
                    )
                    ce_delta_gradient_audit = {
                        "optimizer_step": optimizer_step, "beta": beta,
                        "visual_proj": float(probes[0].detach().norm()),
                        "shared_head": float(probes[1].detach().norm()),
                    }
                    if any(not math.isfinite(v) or v <= 0 for k, v in ce_delta_gradient_audit.items()
                           if k in ("visual_proj", "shared_head")):
                        raise RuntimeError("Trusted CE delta does not reach permitted global parameters")
                    atomic_json_dump(ce_delta_gradient_audit, output / "ce_delta_gradient_audit.json")
                for key, value in (
                    ("global_gce", global_gce), ("global_mixed", global_mixed),
                    ("eligible_global_ce", eligible_ce), ("local_gce", local_gce),
                    ("anchor", anchor), ("loss", loss),
                ):
                    parts[key] += float(value.detach())
                loss.backward()
            gradient_groups = {
                "visual": _grad_norm(visual_parameters), "head": _grad_norm(head_parameters),
                "o3": _grad_norm(o3.parameters()), "pta": _grad_norm(pta.parameters()),
            }
            frozen_leaks = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
            if optimizer_step == 0:
                first_step_audit = {
                    **{key + "_gradient_norm": value for key, value in gradient_groups.items()},
                    "frozen_gradient_leaks": frozen_leaks, "beta": beta,
                    "batch_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                    **{key: parts[key] for key in ("global_gce", "global_mixed", "local_gce", "anchor", "loss")},
                }
                if beta != 0 or any(value <= 0 or not math.isfinite(value) for value in gradient_groups.values()) or frozen_leaks:
                    raise RuntimeError("PRELIM75 v4 first-step equivalence/gradient audit failed")
                atomic_json_dump(first_step_audit, output / "first_step_gradient_audit.json")
            total_gradient_norm = float(torch.nn.utils.clip_grad_norm_(all_parameters, 1.0, error_if_nonfinite=True))
            clipped_steps += int(total_gradient_norm > 1.0)
            optimizer.step(); scheduler.step(); optimizer_step += 1
            for key in parts:
                totals[key] += parts[key] * len(indices)
            totals["beta"] += beta * len(indices); totals["samples"] += len(indices)
            totals["eligible_hits"] += int(batch_eligible.sum())
            eligible_cpu = torch.from_numpy(np.asarray(mask[indices]).copy())
            totals["eligible_weight_mass"] += float(data["weights"][indices][eligible_cpu].sum())
            totals["gradient_norm_sum"] += total_gradient_norm
            totals["gradient_norm_max"] = max(totals["gradient_norm_max"], total_gradient_norm)
            if step % 100 == 0:
                status = {
                    "status": "training", "candidate": name, "epoch": sample_epoch,
                    "completed_samples": totals["samples"], "optimizer_steps": optimizer_step,
                    "beta": beta, "elapsed_seconds": time.monotonic() - start,
                }
                atomic_json_dump(status, output / "status.json"); print(json.dumps(status), flush=True)
        row = {
            "epoch": sample_epoch,
            **{key: value / totals["samples"] for key, value in totals.items()
               if key not in ("samples", "eligible_hits", "eligible_weight_mass", "gradient_norm_sum", "gradient_norm_max")},
            "samples": totals["samples"], "eligible_hits": totals["eligible_hits"],
            "eligible_weight_mass": totals["eligible_weight_mass"],
            "optimizer_steps": optimizer_step, "updates_this_epoch": steps_per_epoch,
            "gradient_norm_mean": totals["gradient_norm_sum"] / steps_per_epoch,
            "gradient_norm_max": totals["gradient_norm_max"], "clipped_steps_cumulative": clipped_steps,
            "epoch_elapsed_seconds": time.monotonic() - epoch_start,
            "elapsed_seconds": time.monotonic() - start,
            "peak_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
        }
        history.append(row); atomic_json_dump(history, output / "training_history.json"); print(json.dumps(row), flush=True)
    for parameter_name, parameter in model.named_parameters():
        if parameter_name in frozen and not torch.equal(parameter.detach().cpu(), frozen[parameter_name]):
            raise RuntimeError(f"Frozen S0 tensor changed: {parameter_name}")
    if name == "C1" and ce_delta_gradient_audit is None:
        raise RuntimeError("C1 never encountered an eligible trusted CE row")
    payload = saved_candidate(checkpoint, model, config, name, plan, optimizer, scheduler, 9, o3, pta)
    payload["global_step"] = int(checkpoint.get("global_step", 0)) + optimizer_step
    payload["prelim75_training"] = {
        "name": name, "candidate_id": f"PRELIM75_V4_{name}", "plan_id": PLAN_ID,
        "parent_experiment_id": "PRELIM75_V3_S0", "weight_parent_sha256": plan["weight_parent_sha256"],
        "geometry_origin_sha256": plan["geometry_origin_sha256"],
        "geometry_boxes_sha256": plan["geometry_boxes_sha256"],
        "mask_sha256": support["files_sha256"]["trusted_ce_mask.npy"],
        "original_supervision_sha256": support["original_supervision_sha256"],
        "selection": "fixed_epoch_9_last_epoch", "supervision": "unchanged_original_w_targets",
        "global_trusted_ce": name == "C1", "global_ce_beta_max": .25 if name == "C1" else 0.0,
        "local_trusted_ce": False, "recovery_kd": False, "prior_in_inference": False,
        "training_geometry_required_for_inference": False, "upstream_provenance_complete": False,
    }
    _atomic_torch_save(payload, output / "candidate.pt")
    checkpoint_hash = sha256_file(output / "candidate.pt")
    del payload, anchor_teacher
    model.cpu(); o3.cpu(); pta.cpu(); del model, o3, pta; torch.cuda.empty_cache()
    reloaded_model, _, reloaded, reloaded_o3, reloaded_pta = load_composite(output / "candidate.pt", torch.device("cpu"))
    metadata = reloaded.get("prelim75_training", {})
    if metadata.get("weight_parent_sha256") != plan["weight_parent_sha256"] or metadata.get("plan_id") != PLAN_ID:
        raise RuntimeError("Saved v4 candidate lineage changed on strict reload")
    result = {
        "status": "trained_pending_real_diagnostic", "candidate": name, "history": history,
        "first_step_gradient_audit": first_step_audit,
        "ce_delta_gradient_audit": ce_delta_gradient_audit,
        "reload_check": {
            "complete_model_strictly_reloaded": True,
            "shared_head_present": hasattr(reloaded_model, "classifier"),
            "local_feature_adapter_reloaded": reloaded_o3 is not None,
            "part_token_adapter_reloaded": reloaded_pta is not None,
            "training_geometry_accessed_by_reload": False,
            "trusted_mask_accessed_by_reload": False,
        },
        "checkpoint_sha256": checkpoint_hash, "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
        "online_accuracy": None, "validation_scope": "overlap_diagnostic",
    }
    atomic_json_dump(result, output / "training_result.json"); atomic_json_dump(result, output / "status.json")
    return output / "candidate.pt"


@torch.no_grad()
def evaluate_trusted_ce(plan, name):
    if name not in ("C0", "C1"):
        raise ValueError("Unknown PRELIM75 v4 candidate")
    root = Path(plan["output"]) / name; checkpoint_path = root / "candidate.pt"
    output = root / "diagnostic"; output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); device = gpu_setup(); data = supervision(plan)
    model, preprocess, checkpoint, o3, pta = load_composite(checkpoint_path, device)
    model.requires_grad_(False); o3.requires_grad_(False); pta.requires_grad_(False)
    metadata = checkpoint.get("prelim75_training", {})
    if (metadata.get("plan_id") != PLAN_ID or metadata.get("weight_parent_sha256") != plan["weight_parent_sha256"]
            or metadata.get("prior_in_inference") is not False):
        raise ValueError("PRELIM75 v4 candidate lineage mismatch")
    frame = pd.read_csv(plan["val_csv"]); paths = [canonical_sample_path(path) for path in frame.image_path.astype(str)]
    if len(paths) != 10316 or len(set(paths)) != len(paths):
        raise ValueError("Diagnostic row identity mismatch")
    training_index = {path: index for index, path in enumerate(data["paths"])}
    indices = torch.tensor([training_index[path] for path in paths]); labels = torch.tensor(frame.label.to_numpy(), dtype=torch.long)
    if not torch.equal(labels, data["labels"][indices]):
        raise ValueError("Diagnostic labels differ from the frozen fitting list")
    dataset = OfficialImages(paths, plan["train_root"], preprocess); stream = loader(dataset, 64, plan)
    predictions = []
    for step, batch in enumerate(stream):
        images = batch["images"].to(device, non_blocking=True)
        global_logits, local_logits, _ = _teacher_views(model, o3, pta, images)
        scores = fuse_teacher_probabilities(global_logits, local_logits)["probabilities"]
        if scores.shape[1] != 500 or not torch.isfinite(scores).all():
            raise RuntimeError("Invalid PRELIM75 v4 diagnostic scores")
        predictions.append(scores.argmax(1).cpu())
        if step % 20 == 0:
            atomic_json_dump({
                "status": "running", "candidate": name,
                "completed_samples": sum(len(item) for item in predictions),
                "elapsed_seconds": time.monotonic() - start,
            }, output / "status.json")
    prediction = torch.cat(predictions)
    fields = {
        "labels": labels, "clean_probability": data["clean_probability"][indices],
        "pseudo_labels": data["pseudo_label"][indices], "correction_alpha": data["correction_alpha"][indices],
    }
    metrics = prediction_metrics(prediction, **fields, num_classes=500, clean_core_threshold=.7)
    parent = json.loads(Path(plan["parent_diagnostic"]).read_text())
    deltas = {key: 100 * (metrics[key] - parent["metrics"][key]) for key in ("raw_micro", "clean_core_micro")}
    result = {
        "status": "complete", "candidate": name, "validation_scope": "overlap_diagnostic",
        "independent_generalization_claim": False, "metrics": metrics, "delta_pp_vs_S0": deltas,
        "engineering_stop": any(value < -2.0 for value in deltas.values()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parent_checkpoint_sha256": plan["weight_parent_sha256"],
        "parent_diagnostic_sha256": plan["parent_diagnostic_sha256"],
        "val_csv_sha256": plan["val_csv_sha256"], "class_mapping_sha256": plan["class_mapping_sha256"],
        "prior_in_inference": False, "real_candidate_attention_recomputed": True,
        "training_boxes_accessed_for_diagnostic": False,
        "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(), "online_accuracy": None,
    }
    _atomic_torch_save({"paths": paths, "prediction": prediction, **fields}, output / "predictions.pt")
    atomic_json_dump(result, output / "result.json"); atomic_json_dump(result, output / "status.json")
    print(json.dumps(result, indent=2), flush=True); return result


def deliver_trusted_ce(plan, name):
    if name not in ("C0", "C1"):
        raise ValueError("Unknown PRELIM75 v4 candidate")
    root = Path(plan["output"]) / name
    diagnostic = json.loads((root / "diagnostic/result.json").read_text())
    if diagnostic.get("status") != "complete" or diagnostic.get("engineering_stop", True):
        raise ValueError("PRELIM75 v4 candidate stopped by the fixed engineering rule")
    checkpoint = root / "candidate.pt"
    if sha256_file(checkpoint) != diagnostic["checkpoint_sha256"]:
        raise ValueError("PRELIM75 v4 candidate changed after diagnostic")
    output = root / "submission"
    if output.exists():
        raise FileExistsError("Do not overwrite an existing PRELIM75 v4 submission")
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
    manifest = json.loads((output / "manifest.json").read_text())
    manifest.update({
        "prediction_csv_sha256": sha256_file(output / "pred_results.csv"),
        "submission_zip_sha256": sha256_file(output / "submission.zip"),
        "csv_delimiter": "comma_space",
    })
    atomic_json_dump(manifest, output / "manifest.json")
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
            raise ValueError("PRELIM75 v4 ZIP/CSV byte identity failed")
    if len(rows) != 24967 or len({row[0] for row in rows}) != 24967:
        raise ValueError("Official PRELIM75 v4 test coverage mismatch")
    report = {
        "candidate": name, "status": "submission_ready_pending_platform",
        "checkpoint_sha256": sha256_file(checkpoint),
        "csv_sha256": sha256_file(output / "pred_results.csv"),
        "zip_sha256": sha256_file(output / "submission.zip"),
        "inference_manifest_sha256": sha256_file(output / "manifest.json"),
        "resolved_config_sha256": sha256_file(root / "resolved_config.json"),
        "source_manifest_sha256": sha256_file(root / "source_manifest.json"),
        "inference_command": command, "validation_command": check_command, "validation_exit_code": 0,
        "rows": 24967, "unique_prediction_classes": len({row[1] for row in rows}),
        "prediction_coverage_forced_to_500": False, "zip_internal_csv_byte_equal": True,
        "csv_delimiter": "comma_space", "diagnostic": diagnostic, "single_checkpoint": True,
        "training_geometry_required_for_final_inference": False,
        "trusted_mask_required_for_final_inference": False,
        "trust_required_for_final_inference": False, "teacher_checkpoint_required_for_final_inference": False,
        "prior_in_inference": False, "online_accuracy": None, "online_exact_correct": None,
        "actual_platform_upload_time": None,
    }
    atomic_json_dump(report, root / "candidate_report.json"); atomic_json_dump(report, root / "status.json")
    print(json.dumps(report, indent=2), flush=True); return report
