"""Fixed PRELIM75 v5 sampled probability-fusion supervision.

F0 and F1 independently warm-start from the completed v4 C0 checkpoint.  Both
train one global and one sampled local view with classification temperature
1.5.  F0 keeps separate branch GCE, while F1 blends it equally with GCE on the
0.6/0.4 probability mixture.  Original supervision and frozen V1 box geometry
are reused unchanged; no teacher probabilities, recovery mask, trusted-CE mask,
test-fitted prior, or second prediction model enters training or inference.
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
from aegis_clip.prelim75_recovery import _teacher_views
from aegis_clip.prelim75_trusted_ce import load_v1_geometry, original_supervision_sha256
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.trainer import _per_sample_loss, _warmup_cosine


PLAN_ID = "PRELIM75_V5_20260917"
EXPECTED_PROTOCOL = {
    "epochs": 3,
    "schedule_epochs": 18,
    "effective_batch_size": 32,
    "epoch_numbers": [10, 11, 12],
    "global_weight": 0.6,
    "local_weight": 0.4,
    "anchor_weight": 2.0,
    "fusion_blend_F0": 0.0,
    "fusion_blend_F1": 0.5,
    "classification_temperature": 1.5,
    "gce_q": 0.5,
    "epsilon": 1e-7,
    "local_scales": [112, 128, 144, 160],
    "local_scale_weights": [0.2, 0.3, 0.4, 0.1],
    "gpu_budget_seconds": 28800,
}


def load_fusion_plan(path):
    """Load exactly the registered v5 plan and bind all immutable assets."""
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
        "original_supervision_sha256", *path_keys,
        *(key + "_sha256" for key in hash_keys),
    }
    if set(plan) != allowed or plan.get("plan_id") != PLAN_ID or plan.get("seed") != 42:
        raise ValueError("Unknown or incomplete fixed PRELIM75 v5 plan")
    if plan.get("protocol") != EXPECTED_PROTOCOL or plan.get("num_workers") != 4:
        raise ValueError("PRELIM75 v5 protocol differs from the registered fixed plan")
    for key in path_keys:
        plan[key] = str((source.parent / plan[key]).resolve())
    for key in hash_keys:
        if sha256_file(plan[key]) != plan[key + "_sha256"]:
            raise ValueError(f"Frozen PRELIM75 v5 asset hash mismatch: {key}")
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
    if experiment_id != "PRELIM75_V4_C0":
        raise ValueError("Registered v5 parent is not the actual v4 C0")
    if metadata.get("name") != "C0" or metadata.get("plan_id") != "PRELIM75_V4_20260916":
        raise ValueError("Registered parent metadata is not completed v4 C0")
    if (metadata.get("global_trusted_ce") is not False
            or metadata.get("recovery_kd") is not False
            or metadata.get("prior_in_inference") is not False):
        raise ValueError("v5 parent unexpectedly contains CE, recovery KD, or prior")
    if metadata.get("original_supervision_sha256") != plan["original_supervision_sha256"]:
        raise ValueError("v4 C0 parent supervision binding changed")
    if loss.get("name") != "gce" or loss.get("gce_q") != 0.5 or loss.get("epsilon") != 1e-7:
        raise ValueError("C0 GCE definition changed")
    if "local_feature_adapter" not in checkpoint or "part_token_adapter" not in checkpoint:
        raise ValueError("C0 checkpoint is missing one or both local adapters")
    diagnostic = json.loads(Path(plan["parent_diagnostic"]).read_text())
    if (diagnostic.get("status") != "complete" or diagnostic.get("engineering_stop") is not False
            or diagnostic.get("checkpoint_sha256") != plan["weight_parent_sha256"]
            or diagnostic.get("prior_in_inference") is not False):
        raise ValueError("Registered C0 diagnostic is incomplete or mismatched")
    return checkpoint, diagnostic


def _assert_train_only(plan, paths):
    train_root = Path(plan["train_root"]).resolve()
    test_root = Path(plan["test_root"]).resolve()
    for path in paths:
        resolved = (train_root / path).resolve()
        if resolved.is_relative_to(test_root):
            raise ValueError("Test path entered PRELIM75 v5 training")


def preflight(plan):
    checkpoint, diagnostic = _validate_parent(plan)
    data = supervision(plan)
    boxes, geometry = load_v1_geometry(plan, data["paths"])
    supervision_hash = original_supervision_sha256(data)
    if supervision_hash != plan["original_supervision_sha256"]:
        raise ValueError("Original v5 supervision differs from the registered tensors")
    _assert_train_only(plan, data["paths"])
    if len(data["paths"]) != 103218 or data["targets"].shape != (103218, 500):
        raise ValueError("Official v5 training identity changed")
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
        "official_train_rows": len(data["paths"]),
        "positive_weight_rows": int((data["weights"] > 0).sum()),
        "original_supervision_sha256": supervision_hash,
        "teacher_probabilities_read": False,
        "recovery_assets_read": False,
        "trusted_ce_mask_read": False,
        "test_used_for_training": False,
        "prior_in_inference": False,
        "platform_upload_authorized": False,
        "protocol": copy.deepcopy(plan["protocol"]),
    }


def prepare_fusion_assets(plan):
    """Freeze a small identity report; v5 creates no labels, masks, or teacher cache."""
    output = Path(plan["output"]) / "preparation.json"
    if output.exists():
        raise FileExistsError("PRELIM75 v5 preparation already exists")
    checked = preflight(plan)
    checked.update({
        "status": "ready_for_fixed_F0_F1",
        "new_teacher_cache_created": False,
        "new_supervision_created": False,
    })
    atomic_json_dump(checked, output)
    return checked


def read_fusion_assets(plan, verify=True):
    preparation = json.loads((Path(plan["output"]) / "preparation.json").read_text())
    if preparation.get("status") != "ready_for_fixed_F0_F1":
        raise ValueError("Missing valid PRELIM75 v5 preparation")
    data = supervision(plan)
    boxes, geometry = load_v1_geometry(plan, data["paths"], verify=verify)
    supervision_hash = original_supervision_sha256(data)
    if (supervision_hash != plan["original_supervision_sha256"]
            or preparation.get("original_supervision_sha256") != supervision_hash):
        raise ValueError("Prepared PRELIM75 v5 supervision changed")
    _assert_train_only(plan, data["paths"])
    return data, boxes, geometry, preparation


def probability_gce(log_probabilities, targets, *, q=0.5, epsilon=1e-7):
    """Per-sample soft-target GCE from normalized log probabilities."""
    if log_probabilities.ndim != 2 or targets.shape != log_probabilities.shape:
        raise ValueError("log_probabilities/targets must be matching [B,C]")
    if not log_probabilities.is_floating_point() or not targets.is_floating_point():
        raise TypeError("log_probabilities and targets must be floating point")
    if targets.requires_grad:
        raise ValueError("Frozen supervision targets must not require gradients")
    if not (0.0 < float(q) <= 1.0) or not (0.0 < float(epsilon) < 1.0):
        raise ValueError("Invalid GCE q or epsilon")
    if not torch.isfinite(log_probabilities).all() or not torch.isfinite(targets).all():
        raise ValueError("Nonfinite probability-fusion loss input")
    if bool((targets < 0).any()) or not torch.allclose(
        targets.float().sum(1), torch.ones(len(targets), device=targets.device), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Targets must be nonnegative normalized distributions")
    dtype = torch.float32 if log_probabilities.dtype in (torch.float16, torch.bfloat16) else log_probabilities.dtype
    bounded = log_probabilities.to(dtype).clamp_min(math.log(float(epsilon)))
    per_class = -torch.expm1(float(q) * bounded) / float(q)
    return (targets.to(dtype) * per_class).sum(1)


def fusion_gce_terms(global_logits, local_logits, targets, *, temperature=1.5, blend=0.0):
    """Return branch, probability-mixture, and fixed blended per-sample GCE."""
    if (global_logits.ndim != 2 or local_logits.shape != global_logits.shape
            or targets.shape != global_logits.shape):
        raise ValueError("global/local logits and targets must be matching [B,C]")
    if not global_logits.is_floating_point() or not local_logits.is_floating_point():
        raise TypeError("Fusion logits must be floating point")
    if not torch.isfinite(global_logits).all() or not torch.isfinite(local_logits).all():
        raise ValueError("Nonfinite fusion logits")
    if float(temperature) != 1.5 or float(blend) not in (0.0, 0.5):
        raise ValueError("Only fixed v5 temperature/blend values are allowed")
    dtype = torch.float32 if global_logits.dtype in (torch.float16, torch.bfloat16) else global_logits.dtype
    log_global = F.log_softmax(global_logits.to(dtype) / float(temperature), dim=-1)
    log_local = F.log_softmax(local_logits.to(dtype) / float(temperature), dim=-1)
    log_mix = torch.logaddexp(log_global + math.log(0.6), log_local + math.log(0.4))
    global_gce = probability_gce(log_global, targets)
    local_gce = probability_gce(log_local, targets)
    separate_gce = 0.6 * global_gce + 0.4 * local_gce
    mixture_gce = probability_gce(log_mix, targets)
    objective = (1.0 - float(blend)) * separate_gce + float(blend) * mixture_gce
    return {
        "global": global_gce,
        "local": local_gce,
        "separate": separate_gce,
        "fusion": mixture_gce,
        "objective": objective,
        "log_global": log_global,
        "log_local": log_local,
        "log_mix": log_mix,
    }


def _candidate_config(checkpoint, plan, name):
    if checkpoint.get("config", {}).get("project", {}).get("experiment_id") != "PRELIM75_V4_C0":
        raise ValueError("v5 parent experiment id must come from actual C0 metadata")
    config = copy.deepcopy(checkpoint["config"])
    config["project"]["experiment_id"] = f"PRELIM75_V5_{name}"
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
        "enabled": True, "parent_experiment_id": "PRELIM75_V4_C0",
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


def _probe_norm(gradients):
    return math.sqrt(sum(float(g.detach().float().pow(2).sum()) for g in gradients if g is not None))


def _fusion_gradient_probe(fusion_loss, global_logits, local_logits, model, o3, pta):
    o3_parameters = list(o3.parameters())
    pta_parameters = list(pta.parameters())
    probes = [global_logits, local_logits, model.visual.proj, model.classifier.weight]
    probes.extend(o3_parameters); probes.extend(pta_parameters)
    gradients = torch.autograd.grad(fusion_loss, probes, retain_graph=True, allow_unused=True)
    cursor = 4
    result = {
        "global_logits": _probe_norm(gradients[0:1]),
        "local_logits": _probe_norm(gradients[1:2]),
        "visual_proj": _probe_norm(gradients[2:3]),
        "shared_head": _probe_norm(gradients[3:4]),
        "o3": _probe_norm(gradients[cursor:cursor + len(o3_parameters)]),
        "pta": _probe_norm(gradients[cursor + len(o3_parameters):]),
    }
    if any(not math.isfinite(value) or value <= 0 for value in result.values()):
        raise RuntimeError("Fusion-only loss failed to reach both branches and permitted adapters")
    return result


def smoke_v5(plan):
    """Choose one common microbatch using F1's heaviest path, without an optimizer."""
    output = Path(plan["output"]) / "smoke.json"
    if output.exists():
        raise FileExistsError("PRELIM75 v5 smoke result already exists")
    start = time.monotonic(); device = gpu_setup()
    data, boxes, _, _ = read_fusion_assets(plan)
    model, preprocess, checkpoint, o3, pta = load_composite(plan["weight_parent"], device)
    model.float().train(); o3.float().train(); pta.float().train()
    model.classifier.requires_grad_(True); o3.requires_grad_(True); pta.requires_grad_(True)
    anchor_teacher = official_anchor(device)
    dataset = OfficialImages(data["paths"], plan["train_root"], preprocess)
    generator = torch.Generator().manual_seed(42)
    stream = loader(dataset, 32, plan, shuffle=True, generator=generator)
    batch = next(iter(stream)); indices = batch["index"].numpy()
    flips, scales = fixed_choices(data["paths"], 10)
    denominator = data["weights"][indices].sum().to(device).clamp_min(1e-8)
    fusion_audit = None; repository_gce_max_abs_diff = None

    def attempt(microbatch):
        nonlocal fusion_audit, repository_gce_max_abs_diff
        model.zero_grad(set_to_none=True); o3.zero_grad(set_to_none=True); pta.zero_grad(set_to_none=True)
        total = None
        for offset in range(0, len(indices), microbatch):
            sub_idx = indices[offset:offset + microbatch]
            images = batch["images"][offset:offset + microbatch].to(device, non_blocking=True)
            selected_flips = flips[sub_idx]
            if selected_flips.any():
                mask = torch.tensor(selected_flips, device=device)
                images[mask] = images[mask].flip(3)
            targets = data["targets"][sub_idx].to(device); weights = data["weights"][sub_idx].to(device)
            with torch.no_grad():
                reference = F.normalize(anchor_teacher(images).float(), dim=1)
            global_logits, features = model(images=images, return_features=True)
            bound = boxes[sub_idx, selected_flips.astype(int), scales[sub_idx]]
            local_logits = adapted_dual_local_view_logits(model, o3, pta, crop_with_bound_boxes(images, bound))
            terms = fusion_gce_terms(global_logits, local_logits, targets, blend=0.5)
            if repository_gce_max_abs_diff is None:
                repository_global = _per_sample_loss(
                    global_logits / 1.5, targets, checkpoint["config"]["loss"], 10
                )
                repository_local = _per_sample_loss(
                    local_logits / 1.5, targets, checkpoint["config"]["loss"], 10
                )
                repository_gce_max_abs_diff = max(
                    float((terms["global"] - repository_global).abs().max()),
                    float((terms["local"] - repository_local).abs().max()),
                )
                if repository_gce_max_abs_diff > 1e-6:
                    raise RuntimeError("v5 GCE differs from repository semantics at temperature 1.5")
            classification = (terms["objective"] * weights).sum() / denominator
            fusion = (terms["fusion"] * weights).sum() / denominator
            anchor = (1.0 - F.cosine_similarity(features.float(), reference, dim=1)).sum() / len(indices)
            loss = classification + 2.0 * anchor
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite PRELIM75 v5 smoke loss")
            if fusion_audit is None:
                fusion_audit = _fusion_gradient_probe(fusion, global_logits, local_logits, model, o3, pta)
            loss.backward(); total = loss.detach() if total is None else total + loss.detach()
        return float(total)

    microbatch = 32; downgraded = False
    try:
        smoke_loss = attempt(microbatch)
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True); o3.zero_grad(set_to_none=True); pta.zero_grad(set_to_none=True)
        import gc
        gc.collect(); torch.cuda.empty_cache(); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        fusion_audit = None; repository_gce_max_abs_diff = None
        microbatch = 16; downgraded = True; smoke_loss = attempt(microbatch)
    result = {
        "status": "passed", "candidate_scope": "F1_probability_fusion_blend_0.5",
        "effective_batch_size": 32, "microbatch_size": microbatch,
        "accumulation": 32 // microbatch, "oom_downgrade": downgraded,
        "loss": smoke_loss, "batch_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "positive_weight_rows_in_batch": int((data["weights"][indices] > 0).sum()),
        "fusion_gradient_audit": fusion_audit,
        "repository_gce_max_abs_diff": repository_gce_max_abs_diff,
        "optimizer_created": False, "optimizer_step_performed": False,
        "formal_rng_state_consumed": False, "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
    }
    atomic_json_dump(result, output)
    return result


def train_fusion(plan, name):
    """Train F0 or F1 independently from C0 with identical data/view order."""
    if name not in ("F0", "F1"):
        raise ValueError("Only PRELIM75 v5 F0/F1 are allowed")
    blend = 0.0 if name == "F0" else 0.5
    output = Path(plan["output"]) / name
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); device = gpu_setup()
    data, boxes, geometry, preparation = read_fusion_assets(plan)
    smoke = json.loads((Path(plan["output"]) / "smoke.json").read_text())
    if smoke.get("status") != "passed" or smoke.get("microbatch_size") not in (16, 32):
        raise ValueError("Missing or invalid isolated v5 smoke result")
    microbatch = int(smoke["microbatch_size"])
    model, preprocess, checkpoint, o3, pta = load_composite(plan["weight_parent"], device)
    model.float(); o3.float(); pta.float()
    licensed_visual = {n for n, p in model.visual.named_parameters() if p.requires_grad}
    if "conv1.weight" in licensed_visual or "positional_embedding" in licensed_visual:
        raise ValueError("C0 visual permission mask changed")
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
    o3_parameters = list(o3.parameters()); pta_parameters = list(pta.parameters())
    adapter_parameters = o3_parameters + pta_parameters
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
        "original_supervision_sha256": preparation["original_supervision_sha256"],
        "epochs": 3, "epoch_numbers": [10, 11, 12], "schedule_epochs": 18,
        "batch_size": 32, "effective_batch_size": 32, "microbatch_size": microbatch,
        "accumulation": 32 // microbatch, "fresh_optimizer": True, "fresh_scheduler": True,
        "global_weight": 0.6, "local_weight": 0.4, "anchor_weight": 2.0,
        "classification_temperature": 1.5, "gce_q": 0.5, "epsilon": 1e-7,
        "probability_fusion_blend": blend,
        "supervision": "bitwise_verified_original_w_targets_unchanged",
        "geometry": "native_224_stateless_flip_frozen_V1_boxes",
        "anchor": "frozen_official_CLIP_same_global_pixel_tensor_online",
        "teacher_probabilities_read": False, "recovery_assets_read": False,
        "trusted_ce_mask_read": False, "prior_in_inference": False,
        "licensed_visual_parameters": sorted(licensed_visual),
        "selection": "fixed_epoch_12_last_epoch",
    }
    atomic_json_dump(recipe, output / "training_recipe.json")
    history = []; optimizer_step = 0; first_step_audit = None; fusion_gradient_audit = None
    clipped_steps = 0
    for sample_epoch in (10, 11, 12):
        epoch_start = time.monotonic(); flips, scales = fixed_choices(data["paths"], sample_epoch)
        totals = {
            "global_gce": 0.0, "local_gce": 0.0, "separate_gce": 0.0,
            "fusion_gce": 0.0, "classification": 0.0, "anchor": 0.0, "loss": 0.0,
            "samples": 0, "positive_weight_hits": 0, "effective_weight_mass": 0.0,
            "gradient_norm_sum": 0.0, "gradient_norm_max": 0.0,
        }
        for step, batch in enumerate(stream):
            indices = batch["index"].numpy()
            denominator = data["weights"][indices].sum().to(device).clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            parts = {key: 0.0 for key in (
                "global_gce", "local_gce", "separate_gce", "fusion_gce",
                "classification", "anchor", "loss",
            )}
            for offset in range(0, len(indices), microbatch):
                sub_idx = indices[offset:offset + microbatch]
                images = batch["images"][offset:offset + microbatch].to(device, non_blocking=True)
                selected_flips = flips[sub_idx]
                if selected_flips.any():
                    flip_mask = torch.tensor(selected_flips, device=device)
                    images[flip_mask] = images[flip_mask].flip(3)
                targets = data["targets"][sub_idx].to(device); weights = data["weights"][sub_idx].to(device)
                with torch.no_grad():
                    reference = F.normalize(anchor_teacher(images).float(), dim=1)
                global_logits, features = model(images=images, return_features=True)
                bound = boxes[sub_idx, selected_flips.astype(int), scales[sub_idx]]
                local_logits = adapted_dual_local_view_logits(model, o3, pta, crop_with_bound_boxes(images, bound))
                terms = fusion_gce_terms(global_logits, local_logits, targets, blend=blend)
                weighted = {
                    key: (terms[source] * weights).sum() / denominator
                    for key, source in (
                        ("global_gce", "global"), ("local_gce", "local"),
                        ("separate_gce", "separate"), ("fusion_gce", "fusion"),
                        ("classification", "objective"),
                    )
                }
                anchor = (1.0 - F.cosine_similarity(features.float(), reference, dim=1)).sum() / len(indices)
                loss = weighted["classification"] + 2.0 * anchor
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite PRELIM75 v5 continuation loss")
                if name == "F1" and fusion_gradient_audit is None:
                    fusion_gradient_audit = {
                        "optimizer_step": optimizer_step,
                        **_fusion_gradient_probe(
                            weighted["fusion"], global_logits, local_logits, model, o3, pta
                        ),
                    }
                    atomic_json_dump(fusion_gradient_audit, output / "fusion_gradient_audit.json")
                for key, value in (*weighted.items(), ("anchor", anchor), ("loss", loss)):
                    parts[key] += float(value.detach())
                loss.backward()
            gradient_groups = {
                "visual": _grad_norm(visual_parameters), "head": _grad_norm(head_parameters),
                "o3": _grad_norm(o3_parameters), "pta": _grad_norm(pta_parameters),
            }
            frozen_leaks = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
            if optimizer_step == 0:
                first_step_audit = {
                    **{key + "_gradient_norm": value for key, value in gradient_groups.items()},
                    "frozen_gradient_leaks": frozen_leaks, "blend": blend,
                    "batch_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                    **parts,
                }
                if any(value <= 0 or not math.isfinite(value) for value in gradient_groups.values()) or frozen_leaks:
                    raise RuntimeError("PRELIM75 v5 first-step gradient audit failed")
                atomic_json_dump(first_step_audit, output / "first_step_gradient_audit.json")
            total_gradient_norm = float(torch.nn.utils.clip_grad_norm_(all_parameters, 1.0, error_if_nonfinite=True))
            clipped_steps += int(total_gradient_norm > 1.0)
            optimizer.step(); scheduler.step(); optimizer_step += 1
            for key in parts:
                totals[key] += parts[key] * len(indices)
            totals["samples"] += len(indices)
            positive = data["weights"][indices] > 0
            totals["positive_weight_hits"] += int(positive.sum())
            totals["effective_weight_mass"] += float(data["weights"][indices].sum())
            totals["gradient_norm_sum"] += total_gradient_norm
            totals["gradient_norm_max"] = max(totals["gradient_norm_max"], total_gradient_norm)
            if step % 100 == 0:
                status = {
                    "status": "training", "candidate": name, "epoch": sample_epoch,
                    "completed_samples": totals["samples"], "optimizer_steps": optimizer_step,
                    "blend": blend, "elapsed_seconds": time.monotonic() - start,
                }
                atomic_json_dump(status, output / "status.json"); print(json.dumps(status), flush=True)
        row = {
            "epoch": sample_epoch,
            **{key: value / totals["samples"] for key, value in totals.items()
               if key not in ("samples", "positive_weight_hits", "effective_weight_mass",
                              "gradient_norm_sum", "gradient_norm_max")},
            "samples": totals["samples"], "positive_weight_hits": totals["positive_weight_hits"],
            "effective_weight_mass": totals["effective_weight_mass"],
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
            raise RuntimeError(f"Frozen C0 tensor changed: {parameter_name}")
    if name == "F1" and fusion_gradient_audit is None:
        raise RuntimeError("F1 fusion-only gradient audit never ran")
    payload = saved_candidate(checkpoint, model, config, name, plan, optimizer, scheduler, 12, o3, pta)
    payload["global_step"] = int(checkpoint.get("global_step", 0)) + optimizer_step
    payload["prelim75_training"] = {
        "name": name, "candidate_id": f"PRELIM75_V5_{name}", "plan_id": PLAN_ID,
        "parent_experiment_id": "PRELIM75_V4_C0", "weight_parent_sha256": plan["weight_parent_sha256"],
        "geometry_origin_sha256": plan["geometry_origin_sha256"],
        "geometry_boxes_sha256": plan["geometry_boxes_sha256"],
        "original_supervision_sha256": preparation["original_supervision_sha256"],
        "selection": "fixed_epoch_12_last_epoch", "supervision": "unchanged_original_w_targets",
        "classification_temperature": 1.5, "probability_fusion_blend": blend,
        "sampled_probability_fusion": name == "F1",
        "global_trusted_ce": False, "local_trusted_ce": False, "recovery_kd": False,
        "prior_in_inference": False, "training_geometry_required_for_inference": False,
        "upstream_provenance_complete": False,
    }
    _atomic_torch_save(payload, output / "candidate.pt")
    checkpoint_hash = sha256_file(output / "candidate.pt")
    del payload, anchor_teacher
    model.cpu(); o3.cpu(); pta.cpu(); del model, o3, pta; torch.cuda.empty_cache()
    reloaded_model, _, reloaded, reloaded_o3, reloaded_pta = load_composite(output / "candidate.pt", torch.device("cpu"))
    metadata = reloaded.get("prelim75_training", {})
    if (metadata.get("weight_parent_sha256") != plan["weight_parent_sha256"]
            or metadata.get("plan_id") != PLAN_ID
            or metadata.get("probability_fusion_blend") != blend):
        raise RuntimeError("Saved v5 candidate lineage changed on strict reload")
    result = {
        "status": "trained_pending_real_diagnostic", "candidate": name, "history": history,
        "first_step_gradient_audit": first_step_audit,
        "fusion_gradient_audit": fusion_gradient_audit,
        "reload_check": {
            "complete_model_strictly_reloaded": True,
            "shared_head_present": hasattr(reloaded_model, "classifier"),
            "local_feature_adapter_reloaded": reloaded_o3 is not None,
            "part_token_adapter_reloaded": reloaded_pta is not None,
            "training_geometry_accessed_by_reload": False,
            "training_supervision_accessed_by_reload": False,
        },
        "checkpoint_sha256": checkpoint_hash, "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
        "online_accuracy": None, "validation_scope": "overlap_diagnostic",
    }
    atomic_json_dump(result, output / "training_result.json"); atomic_json_dump(result, output / "status.json")
    return output / "candidate.pt"


@torch.no_grad()
def evaluate_fusion(plan, name):
    if name not in ("F0", "F1"):
        raise ValueError("Unknown PRELIM75 v5 candidate")
    root = Path(plan["output"]) / name; checkpoint_path = root / "candidate.pt"
    output = root / "diagnostic"; output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); device = gpu_setup(); data = supervision(plan)
    model, preprocess, checkpoint, o3, pta = load_composite(checkpoint_path, device)
    model.requires_grad_(False); o3.requires_grad_(False); pta.requires_grad_(False)
    metadata = checkpoint.get("prelim75_training", {})
    if (metadata.get("plan_id") != PLAN_ID or metadata.get("weight_parent_sha256") != plan["weight_parent_sha256"]
            or metadata.get("prior_in_inference") is not False):
        raise ValueError("PRELIM75 v5 candidate lineage mismatch")
    frame = pd.read_csv(plan["val_csv"]); paths = [canonical_sample_path(path) for path in frame.image_path.astype(str)]
    if len(paths) != 10316 or len(set(paths)) != len(paths):
        raise ValueError("Diagnostic row identity mismatch")
    training_index = {path: index for index, path in enumerate(data["paths"])}
    indices = torch.tensor([training_index[path] for path in paths]); labels = torch.tensor(frame.label.to_numpy(), dtype=torch.long)
    if not torch.equal(labels, data["labels"][indices]):
        raise ValueError("Diagnostic labels differ from frozen fitting list")
    dataset = OfficialImages(paths, plan["train_root"], preprocess); stream = loader(dataset, 64, plan)
    fused_predictions = []; global_predictions = []; local_predictions = []
    scale_weights = torch.tensor([0.2, 0.3, 0.4, 0.1], device=device).view(1, 1, 4, 1)
    for step, batch in enumerate(stream):
        images = batch["images"].to(device, non_blocking=True)
        global_logits, local_logits, _ = _teacher_views(model, o3, pta, images)
        global_probability = (global_logits.float() / 1.5).softmax(-1).mean(1)
        local_probability = ((local_logits.float() / 1.5).softmax(-1) * scale_weights).sum(2).mean(1)
        fused_probability = 0.6 * global_probability + 0.4 * local_probability
        if not torch.isfinite(fused_probability).all():
            raise RuntimeError("Invalid PRELIM75 v5 diagnostic probabilities")
        global_predictions.append(global_probability.argmax(1).cpu())
        local_predictions.append(local_probability.argmax(1).cpu())
        fused_predictions.append(fused_probability.argmax(1).cpu())
        if step % 20 == 0:
            atomic_json_dump({
                "status": "running", "candidate": name,
                "completed_samples": sum(len(item) for item in fused_predictions),
                "elapsed_seconds": time.monotonic() - start,
            }, output / "status.json")
    predictions = {
        "global": torch.cat(global_predictions),
        "local": torch.cat(local_predictions),
        "fused": torch.cat(fused_predictions),
    }
    fields = {
        "labels": labels, "clean_probability": data["clean_probability"][indices],
        "pseudo_labels": data["pseudo_label"][indices], "correction_alpha": data["correction_alpha"][indices],
    }
    metrics = {key: prediction_metrics(value, **fields, num_classes=500, clean_core_threshold=.7)
               for key, value in predictions.items()}
    parent = json.loads(Path(plan["parent_diagnostic"]).read_text())
    deltas = {key: 100 * (metrics["fused"][key] - parent["metrics"][key])
              for key in ("raw_micro", "clean_core_micro")}
    result = {
        "status": "complete", "candidate": name, "validation_scope": "overlap_diagnostic",
        "independent_generalization_claim": False, "metrics": metrics["fused"],
        "branch_metrics": metrics, "delta_pp_vs_C0": deltas,
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
    _atomic_torch_save({"paths": paths, **predictions, **fields}, output / "predictions.pt")
    atomic_json_dump(result, output / "result.json"); atomic_json_dump(result, output / "status.json")
    print(json.dumps(result, indent=2), flush=True); return result


def deliver_fusion(plan, name):
    if name not in ("F0", "F1"):
        raise ValueError("Unknown PRELIM75 v5 candidate")
    root = Path(plan["output"]) / name
    diagnostic = json.loads((root / "diagnostic/result.json").read_text())
    if diagnostic.get("status") != "complete" or diagnostic.get("engineering_stop", True):
        raise ValueError("PRELIM75 v5 candidate stopped by the fixed engineering rule")
    checkpoint = root / "candidate.pt"
    if sha256_file(checkpoint) != diagnostic["checkpoint_sha256"]:
        raise ValueError("PRELIM75 v5 candidate changed after diagnostic")
    output = root / "submission"
    if output.exists():
        raise FileExistsError("Do not overwrite an existing PRELIM75 v5 submission")
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
            raise ValueError("PRELIM75 v5 ZIP/CSV byte identity failed")
    if len(rows) != 24967 or len({row[0] for row in rows}) != 24967:
        raise ValueError("Official PRELIM75 v5 test coverage mismatch")
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
        "training_supervision_required_for_final_inference": False,
        "teacher_checkpoint_required_for_final_inference": False,
        "prior_in_inference": False, "online_accuracy": None, "online_exact_correct": None,
        "actual_platform_upload_time": None,
    }
    atomic_json_dump(report, root / "candidate_report.json"); atomic_json_dump(report, root / "status.json")
    print(json.dumps(report, indent=2), flush=True); return report
