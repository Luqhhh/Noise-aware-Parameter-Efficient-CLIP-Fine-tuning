"""PRELIM75 v10 fixed AdamW versus standard non-adaptive SAM comparison.

The module intentionally keeps the numerical SAM primitive independent from
the competition pipeline.  The pipeline then binds that primitive to the
frozen L1 parent, original supervision, frozen V1 geometry, three registered
sample epochs, and the existing ten-view no-prior inference protocol.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
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
from aegis_clip.prelim75_cooldown import audit_position
from aegis_clip.prelim75_fusion import fusion_gce_terms
from aegis_clip.prelim75_joint import crop_with_bound_boxes, fixed_choices, official_anchor
from aegis_clip.prelim75_online_geometry import (
    _grad_norm,
    _tensor_sha256,
    forward_train_with_detached_attention,
)
from aegis_clip.prelim75_recovery import _teacher_views
from aegis_clip.prelim75_trusted_ce import load_v1_geometry, original_supervision_sha256
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.trainer import _warmup_cosine


PLAN_ID = "PRELIM75_V10_SAM_20260920"
PARENT_PLAN_ID = "PRELIM75_V7_20260918"
PARENT_EXPERIMENT_ID = "PRELIM75_V7_L1"
CANDIDATES = ("S0", "S1")
SAM_CANDIDATE = "S1"
RHO = 0.05
SAM_EPSILON = 1e-12
RADIUS_TOLERANCE = 5e-6
DEFAULT_STEPS_PER_EPOCH = 3226
TOTAL_UPDATES = 9678
REQUIRED_TRAIN_ROWS = 103218
REQUIRED_DIAGNOSTIC_ROWS = 10316
REQUIRED_TEST_ROWS = 24967
REQUIRED_CLASSES = 500
SAMPLE_EPOCHS = (19, 20, 21)
LOCAL_SCALES = (112, 128, 144, 160)
LOCAL_SCALE_WEIGHTS = (0.2, 0.3, 0.4, 0.1)


EXPECTED_PROTOCOL = {
    "candidates": ["S0", "S1"],
    "epochs": 3,
    "sample_epochs": [19, 20, 21],
    "effective_batch_size": 32,
    "microbatch_size": 16,
    "steps_per_epoch": 3226,
    "total_updates_per_candidate": 9678,
    "global_weight": 0.6,
    "local_weight": 0.4,
    "anchor_weight": 2.0,
    "fusion_blend": 0.5,
    "classification_temperature": 1.5,
    "gce_q": 0.5,
    "epsilon": 1e-7,
    "local_scales": list(LOCAL_SCALES),
    "local_scale_weights": list(LOCAL_SCALE_WEIGHTS),
    "attention_top_k": 5,
    "prior": False,
}
EXPECTED_OPTIMIZER = {
    "type": "AdamW",
    "visual_lr": 1e-6,
    "visual_weight_decay": 0.0,
    "head_lr": 1.5e-7,
    "head_weight_decay": 1e-4,
    "adapters_lr": 3e-6,
    "adapters_weight_decay": 0.0,
    "gradient_clip_norm": 1.0,
    "scheduler": "cosine",
    "schedule_horizon_epochs": 3,
    "floor_ratio": 0.01,
    "warmup_steps": 0,
    "amp": False,
}
EXPECTED_SAM = {
    "candidate": "S1",
    "type": "standard_nonadaptive_global_l2",
    "rho": 0.05,
    "epsilon": 1e-12,
    "actual_radius_tolerance": 5e-6,
    "average_first_and_second_gradients": False,
    "perturb_weight_decay": False,
    "exact_copy_restore": True,
}
EXPECTED_DIAGNOSTIC = {
    "rows": 10316,
    "maximum_drop_pp": 2.0,
    "scope": "training_overlap_engineering_only",
}
EXPECTED_INFERENCE = {
    "batch_size": 64,
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "highest",
}
EXPECTED_BUDGET = {
    "gpu_action_seconds": 28800,
    "maximum_platform_candidates": 2,
    "diagnostic_and_delivery_reserve_seconds": 4000,
    "estimate_uncertainty_multiplier": 1.25,
}
EXPECTED_REFERENCE = {
    "L1_no_prior": 69.2794,
    "G0_legacy_test_batch_prior0.9": 72.4677,
    "target": 75.0,
}

PATH_KEYS = (
    "weight_parent", "parent_training_result", "parent_diagnostic",
    "fallback_submission", "fallback_csv", "fallback_manifest",
    "geometry_manifest", "geometry_paths", "geometry_boxes", "trust",
    "train_csv", "val_csv", "class_mapping", "groups", "train_root",
    "test_root", "output",
)
HASHED_PATH_KEYS = tuple(key for key in PATH_KEYS if key not in ("train_root", "test_root", "output"))


# ---------------------------------------------------------------------------
# Standalone SAM numerical contract
# ---------------------------------------------------------------------------

def _materialize_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    values = list(parameters)
    if not values:
        raise ValueError("SAM requires at least one licensed parameter")
    if len({id(parameter) for parameter in values}) != len(values):
        raise ValueError("SAM parameter list contains aliases")
    storage_pointers: set[tuple[str, int | None, int]] = set()
    for parameter in values:
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError("SAM accepts torch.nn.Parameter values only")
        if not parameter.requires_grad:
            raise ValueError("SAM received an unlicensed frozen parameter")
        if parameter.dtype != torch.float32:
            raise ValueError("PRELIM75 v10 SAM parameters must be FP32")
        pointer = (parameter.device.type, parameter.device.index, parameter.untyped_storage().data_ptr())
        if pointer in storage_pointers:
            raise ValueError("SAM parameters share storage")
        storage_pointers.add(pointer)
    return values


def global_l2_gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Return the shared FP32 L2 norm of all raw gradients."""
    values = _materialize_parameters(parameters)
    per_parameter = []
    for parameter in values:
        if parameter.grad is None:
            raise ValueError("Licensed SAM parameter is missing its gradient")
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("SAM gradient is nonfinite")
        per_parameter.append(torch.linalg.vector_norm(gradient.float()))
    norm = float(torch.linalg.vector_norm(torch.stack(per_parameter)).item())
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("SAM global gradient norm must be positive and finite")
    return norm


@dataclass
class PerturbationState:
    parameters: list[torch.nn.Parameter]
    backups: list[torch.Tensor]
    gradient_norm: float
    theoretical_radius: float
    actual_radius: float
    scale: float
    restored: bool = False


def _distance_from_backups(state: PerturbationState) -> float:
    per_parameter = [
        torch.linalg.vector_norm(parameter.detach() - backup)
        for parameter, backup in zip(state.parameters, state.backups)
    ]
    return float(torch.linalg.vector_norm(torch.stack(per_parameter)).item())


@torch.no_grad()
def apply_standard_sam_perturbation(
    parameters: Iterable[torch.nn.Parameter],
    *,
    rho: float = RHO,
    epsilon: float = SAM_EPSILON,
    radius_tolerance: float = RADIUS_TOLERANCE,
) -> PerturbationState:
    """Copy original FP32 values and add one global non-adaptive L2 perturbation."""
    values = _materialize_parameters(parameters)
    if not math.isfinite(float(rho)) or float(rho) <= 0.0:
        raise ValueError("SAM rho must be positive and finite")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("SAM epsilon must be positive and finite")
    norm = global_l2_gradient_norm(values)
    backups = [parameter.detach().clone() for parameter in values]
    scale = float(rho) / (norm + float(epsilon))
    state = PerturbationState(values, backups, norm, float(rho), float("nan"), scale)
    try:
        for parameter in values:
            parameter.add_(parameter.grad, alpha=scale)
        state.actual_radius = _distance_from_backups(state)
        if not math.isfinite(state.actual_radius):
            raise FloatingPointError("Actual FP32 SAM displacement is nonfinite")
        if abs(state.actual_radius - float(rho)) > float(radius_tolerance):
            raise RuntimeError(
                f"Actual FP32 SAM radius {state.actual_radius:.12g} differs from rho={rho}"
            )
        return state
    except BaseException:
        for parameter, backup in zip(values, backups):
            parameter.copy_(backup)
        state.restored = True
        raise


@torch.no_grad()
def restore_standard_sam_parameters(state: PerturbationState) -> None:
    """Restore by copy, retaining current gradients and requiring bitwise identity."""
    if state.restored:
        raise RuntimeError("SAM perturbation state was already restored")
    for parameter, backup in zip(state.parameters, state.backups):
        parameter.copy_(backup)
    if any(not torch.equal(parameter.detach(), backup) for parameter, backup in zip(state.parameters, state.backups)):
        raise RuntimeError("SAM exact copy restoration failed")
    state.restored = True


def standard_sam_update(
    parameters: Iterable[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    backward_pass: Callable[[int], torch.Tensor],
    *,
    rho: float = RHO,
    epsilon: float = SAM_EPSILON,
    radius_tolerance: float = RADIUS_TOLERANCE,
    clip_norm: float = 1.0,
) -> dict:
    """Execute exactly g1 -> perturb -> g2 -> restore -> clip -> AdamW step.

    ``backward_pass`` must accumulate a complete effective-batch gradient and
    return a detached scalar loss.  The callback is invoked with pass ids 1/2.
    """
    values = _materialize_parameters(parameters)
    optimizer.zero_grad(set_to_none=True)
    first_loss = backward_pass(1)
    if first_loss.ndim or not torch.isfinite(first_loss):
        raise FloatingPointError("First SAM loss must be a finite scalar")
    first_norm = global_l2_gradient_norm(values)
    state = apply_standard_sam_perturbation(
        values, rho=rho, epsilon=epsilon, radius_tolerance=radius_tolerance
    )
    try:
        optimizer.zero_grad(set_to_none=True)
        second_loss = backward_pass(2)
        if second_loss.ndim or not torch.isfinite(second_loss):
            raise FloatingPointError("Second SAM loss must be a finite scalar")
        second_norm = global_l2_gradient_norm(values)
    finally:
        if not state.restored:
            restore_standard_sam_parameters(state)
    clipped_norm = float(torch.nn.utils.clip_grad_norm_(values, clip_norm, error_if_nonfinite=True))
    try:
        optimizer.step()
    except BaseException as exc:
        raise RuntimeError(
            "AdamW step failed after SAM restoration; discard this process and resume only from a complete boundary"
        ) from exc
    return {
        "first_loss": float(first_loss),
        "perturbed_loss": float(second_loss),
        "loss_increase": float(second_loss - first_loss),
        "first_gradient_norm": first_norm,
        "second_gradient_norm": second_norm,
        "actual_radius": state.actual_radius,
        "theoretical_radius": float(rho),
        "restored_bitwise": state.restored,
        "preclip_second_gradient_norm": clipped_norm,
        "gradient_clipped": clipped_norm > float(clip_norm),
    }


# ---------------------------------------------------------------------------
# Fixed plan and asset checks
# ---------------------------------------------------------------------------

def _resolve_plan_paths(plan: dict, source: Path) -> None:
    for key in PATH_KEYS:
        value = plan.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing PRELIM75 v10 path: {key}")
        plan[key] = str((source.parent / value).resolve())


def _validate_plan_schema(plan: dict) -> None:
    allowed = {
        "plan_id", "seed", "reviewed_commit", "num_workers",
        "weight_parent_experiment_id", "geometry_origin_sha256",
        "original_supervision_sha256", "protocol", "optimizer", "sam",
        "diagnostic", "inference", "budget", "reference_platform_percent",
        *PATH_KEYS, *(key + "_sha256" for key in HASHED_PATH_KEYS),
    }
    if set(plan) != allowed:
        raise ValueError(
            f"Unknown/incomplete v10 plan: missing={sorted(allowed-set(plan))}, "
            f"extra={sorted(set(plan)-allowed)}"
        )
    if plan["plan_id"] != PLAN_ID or plan["seed"] != 42 or plan["num_workers"] != 4:
        raise ValueError("PRELIM75 v10 plan identity changed")
    if plan["reviewed_commit"] != "5045e71e69f4ee7de67af6410bbb336c60ecb171":
        raise ValueError("PRELIM75 v10 reviewed commit changed")
    if plan["weight_parent_experiment_id"] != PARENT_EXPERIMENT_ID:
        raise ValueError("PRELIM75 v10 L1 parent id changed")
    for actual, expected, label in (
        (plan["protocol"], EXPECTED_PROTOCOL, "protocol"),
        (plan["optimizer"], EXPECTED_OPTIMIZER, "optimizer"),
        (plan["sam"], EXPECTED_SAM, "sam"),
        (plan["diagnostic"], EXPECTED_DIAGNOSTIC, "diagnostic"),
        (plan["inference"], EXPECTED_INFERENCE, "inference"),
        (plan["budget"], EXPECTED_BUDGET, "budget"),
        (plan["reference_platform_percent"], EXPECTED_REFERENCE, "reference"),
    ):
        if actual != expected:
            raise ValueError(f"PRELIM75 v10 fixed {label} changed")
    for key in HASHED_PATH_KEYS:
        value = plan.get(key + "_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Missing registered SHA-256 for {key}")


def load_v10_plan(path: str | Path) -> dict:
    source = Path(path).resolve()
    if source.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("PRELIM75 v10 config must be YAML")
    plan = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("PRELIM75 v10 config root must be a mapping")
    _validate_plan_schema(plan)
    _resolve_plan_paths(plan, source)
    for key in HASHED_PATH_KEYS:
        if sha256_file(plan[key]) != plan[key + "_sha256"]:
            raise ValueError(f"Frozen PRELIM75 v10 asset hash mismatch: {key}")
    train_root, test_root = Path(plan["train_root"]), Path(plan["test_root"])
    if train_root == test_root or train_root.is_relative_to(test_root) or test_root.is_relative_to(train_root):
        raise ValueError("Official train and test roots must be disjoint")
    plan["parent"] = plan["weight_parent"]
    plan["parent_sha256"] = plan["weight_parent_sha256"]
    plan["_config_path"] = str(source)
    plan["_config_sha256"] = sha256_file(source)
    return plan


def _validate_parent_checkpoint(plan: dict) -> dict:
    checkpoint = torch.load(plan["weight_parent"], map_location="cpu", weights_only=False)
    metadata = checkpoint.get("prelim75_training", {})
    expected = {
        "name": "L1",
        "candidate_id": PARENT_EXPERIMENT_ID,
        "plan_id": PARENT_PLAN_ID,
        "training_geometry_policy": "frozen_V1",
        "comparison_geometry_origin": "V1",
        "geometry_origin_sha256": plan["geometry_origin_sha256"],
        "geometry_boxes_sha256": plan["geometry_boxes_sha256"],
        "original_supervision_sha256": plan["original_supervision_sha256"],
        "supervision_changed": False,
        "inference_teacher_used": False,
        "selection": "fixed_epoch_18_last_epoch",
        "supervision": "unchanged_original_w_targets",
        "classification_temperature": 1.5,
        "probability_fusion_blend": 0.5,
        "global_weight": 0.6,
        "local_weight": 0.4,
        "anchor_weight": 2.0,
        "cosine_horizon_epochs": 3,
        "scheduler_floor_ratio": 0.01,
        "scheduler_warmup_steps": 0,
        "total_updates": 9678,
        "global_trusted_ce": False,
        "local_trusted_ce": False,
        "recovery_kd": False,
        "prior_in_inference": False,
        "training_geometry_required_for_inference": False,
    }
    if checkpoint.get("config", {}).get("project", {}).get("experiment_id") != PARENT_EXPERIMENT_ID:
        raise ValueError("Registered PRELIM75 v10 parent is not v7 L1")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"L1 parent metadata mismatch: {key}")
    if checkpoint.get("epoch") != 18:
        raise ValueError("L1 parent is not the fixed epoch-18 checkpoint")
    if "local_feature_adapter" not in checkpoint or "part_token_adapter" not in checkpoint:
        raise ValueError("L1 parent checkpoint is incomplete")
    return checkpoint


def _assert_train_only(plan: dict, paths: Sequence[str]) -> None:
    train_root, test_root = Path(plan["train_root"]).resolve(), Path(plan["test_root"]).resolve()
    for path in paths:
        resolved = (train_root / path).resolve()
        if not resolved.is_relative_to(train_root) or resolved.is_relative_to(test_root):
            raise ValueError("Non-training path entered PRELIM75 v10 fitting")


def read_v10_assets(plan: dict):
    data = supervision(plan)
    boxes, geometry = load_v1_geometry(plan, data["paths"], verify=True)
    supervision_hash = original_supervision_sha256(data)
    if supervision_hash != plan["original_supervision_sha256"]:
        raise ValueError("PRELIM75 v10 original supervision changed")
    _assert_train_only(plan, data["paths"])
    return data, boxes, geometry


def preflight_v10(plan: dict) -> dict:
    checkpoint = _validate_parent_checkpoint(plan)
    data, boxes, geometry = read_v10_assets(plan)
    parent_training = json.loads(Path(plan["parent_training_result"]).read_text(encoding="utf-8"))
    parent_diagnostic = json.loads(Path(plan["parent_diagnostic"]).read_text(encoding="utf-8"))
    if (
        parent_training.get("status") != "trained_pending_real_diagnostic"
        or parent_training.get("candidate") != "L1"
        or parent_training.get("checkpoint_sha256") != plan["weight_parent_sha256"]
        or parent_training.get("total_updates") != TOTAL_UPDATES
    ):
        raise ValueError("L1 parent training result identity mismatch")
    if (
        parent_diagnostic.get("status") != "complete"
        or parent_diagnostic.get("candidate") != "L1"
        or parent_diagnostic.get("checkpoint_sha256") != plan["weight_parent_sha256"]
        or parent_diagnostic.get("engineering_stop") is not False
        or parent_diagnostic.get("validation_scope") != "overlap_diagnostic"
        or parent_diagnostic.get("metrics", {}).get("samples") != REQUIRED_DIAGNOSTIC_ROWS
    ):
        raise ValueError("L1 parent diagnostic identity mismatch")
    with ZipFile(plan["fallback_submission"]) as archive:
        names = archive.namelist()
        internal_hash = hashlib.sha256(archive.read("pred_results.csv")).hexdigest()
    if names != ["pred_results.csv"] or internal_hash != plan["fallback_csv_sha256"]:
        raise ValueError("L1 fallback ZIP/CSV identity mismatch")
    fallback_manifest = json.loads(Path(plan["fallback_manifest"]).read_text(encoding="utf-8"))
    if (
        fallback_manifest.get("checkpoint_sha256") != plan["weight_parent_sha256"]
        or fallback_manifest.get("prediction_count") != REQUIRED_TEST_ROWS
        or fallback_manifest.get("prior_alignment") is not None
    ):
        raise ValueError("L1 fallback inference manifest changed")
    expected_steps = math.ceil(len(data["paths"]) / 32)
    if (
        len(data["paths"]) != REQUIRED_TRAIN_ROWS
        or data["targets"].shape != (REQUIRED_TRAIN_ROWS, REQUIRED_CLASSES)
        or expected_steps != DEFAULT_STEPS_PER_EPOCH
        or boxes.shape[:3] != (REQUIRED_TRAIN_ROWS, 2, 4)
    ):
        raise ValueError("PRELIM75 v10 registered dataset geometry changed")
    config = checkpoint["config"]
    return {
        "status": "passed",
        "plan_id": PLAN_ID,
        "config_sha256": plan["_config_sha256"],
        "weight_parent_experiment_id": PARENT_EXPERIMENT_ID,
        "weight_parent_sha256": plan["weight_parent_sha256"],
        "parent_training_result_sha256": plan["parent_training_result_sha256"],
        "parent_diagnostic_sha256": plan["parent_diagnostic_sha256"],
        "fallback_submission_sha256": plan["fallback_submission_sha256"],
        "fallback_csv_sha256": plan["fallback_csv_sha256"],
        "fallback_manifest_sha256": plan["fallback_manifest_sha256"],
        "geometry_origin_sha256": geometry["parent_sha256"],
        "geometry_boxes_sha256": plan["geometry_boxes_sha256"],
        "geometry_manifest_sha256": plan["geometry_manifest_sha256"],
        "geometry_paths_sha256": plan["geometry_paths_sha256"],
        "geometry_shape": list(boxes.shape),
        "official_train_rows": len(data["paths"]),
        "positive_weight_rows": int((data["weights"] > 0).sum()),
        "original_supervision_sha256": original_supervision_sha256(data),
        "steps_per_epoch": expected_steps,
        "updates_per_candidate": 3 * expected_steps,
        "total_formal_optimizer_updates": 2 * 3 * expected_steps,
        "parent_train_config": {
            key: config["train"].get(key) for key in (
                "batch_size", "backbone_lr", "backbone_weight_decay",
                "head_lr", "head_weight_decay", "amp",
            )
        },
        "teacher_probabilities_read": False,
        "recovery_assets_read": False,
        "test_used_for_training": False,
        "prior_in_inference": False,
        "platform_upload_authorized": False,
    }


@dataclass(frozen=True)
class V10Schedule:
    steps_per_epoch: int = DEFAULT_STEPS_PER_EPOCH

    @property
    def total_steps(self) -> int:
        return 3 * self.steps_per_epoch

    def multiplier(self, completed_updates: int) -> float:
        if completed_updates < 0 or completed_updates > self.total_steps:
            raise ValueError("PRELIM75 v10 scheduler position outside fixed horizon")
        return float(_warmup_cosine(completed_updates, 0, self.total_steps))

    def lrs_for_bases(self, base_lrs, completed_updates: int) -> tuple[float, ...]:
        multiplier = self.multiplier(completed_updates)
        return tuple(float(value) * multiplier for value in base_lrs)


def build_v10_scheduler(optimizer: torch.optim.Optimizer, spec: V10Schedule):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=spec.multiplier)
    audit_position(optimizer, spec, 0)
    return scheduler


def scheduler_reference(steps_per_epoch: int = DEFAULT_STEPS_PER_EPOCH) -> dict:
    spec = V10Schedule(steps_per_epoch)
    return {
        "status": "passed",
        "steps_per_epoch": steps_per_epoch,
        "total_updates": spec.total_steps,
        "floor_ratio": 0.01,
        "boundaries": {
            label: {"step": step, "multiplier": spec.multiplier(step)}
            for label, step in (("0", 0), ("1M", steps_per_epoch),
                                ("2M", 2 * steps_per_epoch), ("3M", 3 * steps_per_epoch))
        },
        "optimizer_then_scheduler": True,
    }


def prepare_v10(plan: dict) -> dict:
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved_plan.json"
    if resolved_path.exists():
        raise FileExistsError("PRELIM75 v10 output already contains a resolved plan")
    checked = preflight_v10(plan)
    resolved = {key: value for key, value in plan.items() if not key.startswith("_")}
    atomic_json_dump(checked, output / "preflight.json")
    atomic_json_dump(scheduler_reference(), output / "scheduler_reference.json")
    atomic_json_dump(resolved, resolved_path)
    return {"status": "ready_for_v10", "preflight": checked}


# ---------------------------------------------------------------------------
# Model-bound effective batch and gate implementation
# ---------------------------------------------------------------------------

def _candidate_config(checkpoint: dict, plan: dict, name: str) -> dict:
    if name not in CANDIDATES:
        raise ValueError("Unknown PRELIM75 v10 candidate")
    if checkpoint.get("config", {}).get("project", {}).get("experiment_id") != PARENT_EXPERIMENT_ID:
        raise ValueError("PRELIM75 v10 configuration did not originate from L1")
    config = copy.deepcopy(checkpoint["config"])
    config["project"]["experiment_id"] = f"PRELIM75_V10_SAM_{name}"
    for key in ("train_csv", "val_csv", "class_mapping", "train_root", "test_root"):
        config["data"][key] = plan[key]
    config["data"]["train_augmentation"] = "clip_center_crop"
    config["train"].update({
        "init_checkpoint": plan["weight_parent"],
        "require_lineage_for_init_checkpoint": True,
        "epochs": 3,
        "schedule_epochs": 3,
        "batch_size": 32,
        "backbone_lr": 1e-6,
        "backbone_weight_decay": 0.0,
        "head_lr": 1.5e-7,
        "head_weight_decay": 1e-4,
        "amp": False,
    })
    config["lineage"] = {
        "enabled": True,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
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


def _load_training_components(plan: dict, device: torch.device):
    model, preprocess, checkpoint, o3, pta = load_composite(plan["weight_parent"], device)
    model.float(); o3.float(); pta.float()
    licensed_visual = {
        name for name, parameter in model.visual.named_parameters() if parameter.requires_grad
    }
    if "conv1.weight" in licensed_visual or "positional_embedding" in licensed_visual:
        raise ValueError("L1 visual permission mask changed")
    model.classifier.requires_grad_(True)
    o3.requires_grad_(True); pta.requires_grad_(True)
    model.train(); o3.train(); pta.train()
    return model, preprocess, checkpoint, o3, pta, licensed_visual


def _named_trainables(model, o3, pta) -> list[tuple[str, torch.nn.Parameter]]:
    result = []
    result.extend((f"visual.{name}", parameter) for name, parameter in model.visual.named_parameters() if parameter.requires_grad)
    result.extend((f"classifier.{name}", parameter) for name, parameter in model.classifier.named_parameters() if parameter.requires_grad)
    result.extend((f"o3.{name}", parameter) for name, parameter in o3.named_parameters() if parameter.requires_grad)
    result.extend((f"pta.{name}", parameter) for name, parameter in pta.named_parameters() if parameter.requires_grad)
    _materialize_parameters(parameter for _, parameter in result)
    return result


def _make_optimizer(model, o3, pta) -> tuple[torch.optim.AdamW, list[torch.nn.Parameter], dict[str, list[torch.nn.Parameter]]]:
    groups = {
        "visual": [parameter for parameter in model.visual.parameters() if parameter.requires_grad],
        "head": list(model.classifier.parameters()),
        "o3": list(o3.parameters()),
        "pta": list(pta.parameters()),
    }
    optimizer = torch.optim.AdamW([
        {"name": "backbone", "params": groups["visual"], "lr": 1e-6, "weight_decay": 0.0},
        {"name": "head", "params": groups["head"], "lr": 1.5e-7, "weight_decay": 1e-4},
        {"name": "local_adapters", "params": groups["o3"] + groups["pta"], "lr": 3e-6, "weight_decay": 0.0},
    ])
    parameters = groups["visual"] + groups["head"] + groups["o3"] + groups["pta"]
    _materialize_parameters(parameters)
    return optimizer, parameters, groups


def _module_state_audit(model, o3, pta) -> dict:
    stochastic = []
    mutable = []
    for prefix, root in (("model", model), ("o3", o3), ("pta", pta)):
        for name, module in root.named_modules():
            qualified = f"{prefix}.{name}" if name else prefix
            if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.training and module.p > 0:
                stochastic.append(qualified)
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training and module.track_running_stats:
                mutable.append(qualified)
    if stochastic or mutable:
        raise RuntimeError(f"Active stochastic/mutable training modules are not registered: {stochastic}, {mutable}")
    return {"active_dropout": stochastic, "active_batchnorm": mutable}


def _snapshot_buffers(model, o3, pta) -> dict[str, torch.Tensor]:
    result = {}
    for prefix, root in (("model", model), ("o3", o3), ("pta", pta)):
        result.update({f"{prefix}.{name}": value.detach().clone() for name, value in root.named_buffers()})
    return result


def _buffers_equal(snapshot: dict[str, torch.Tensor], model, o3, pta) -> bool:
    current = _snapshot_buffers(model, o3, pta)
    return set(snapshot) == set(current) and all(torch.equal(snapshot[key], current[key]) for key in snapshot)


def _frozen_gradient_leaks(model, o3, pta) -> list[str]:
    leaks = [
        f"model.{name}" for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    leaks.extend(
        f"o3.{name}" for name, parameter in o3.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    )
    leaks.extend(
        f"pta.{name}" for name, parameter in pta.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    )
    return leaks


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prepare_effective_batch(
    batch: dict,
    data: dict,
    boxes: np.ndarray,
    flips: np.ndarray,
    scales: np.ndarray,
    anchor_teacher,
    device: torch.device,
    *,
    microbatch: int,
) -> dict:
    indices = batch["index"].numpy()
    images_cpu = batch["images"].clone()
    references = []
    with torch.no_grad():
        for offset in range(0, len(indices), microbatch):
            sub_idx = indices[offset:offset + microbatch]
            images = images_cpu[offset:offset + microbatch].to(device, non_blocking=True)
            selected_flips = flips[sub_idx]
            if selected_flips.any():
                images[torch.as_tensor(selected_flips, device=device)] = images[
                    torch.as_tensor(selected_flips, device=device)
                ].flip(3)
            references.append(F.normalize(anchor_teacher(images).float(), dim=1).cpu())
    return {
        "indices": indices,
        "images": images_cpu,
        "flips": flips[indices].copy(),
        "scales": scales[indices].copy(),
        "targets": data["targets"][indices].clone(),
        "weights": data["weights"][indices].clone(),
        "references": torch.cat(references),
        "boxes": boxes[indices].copy(),
        "paths": [data["paths"][int(index)] for index in indices],
    }


def _effective_batch_identity(prepared: dict) -> dict:
    flip_scale = torch.as_tensor(
        np.stack((prepared["flips"].astype(np.int64), prepared["scales"])), dtype=torch.int64
    )
    return {
        "batch_indices_sha256": hashlib.sha256(prepared["indices"].tobytes()).hexdigest(),
        "batch_paths_sha256": hashlib.sha256("\n".join(prepared["paths"]).encode()).hexdigest(),
        "batch_images_sha256": _tensor_sha256(prepared["images"]),
        "batch_flip_scale_sha256": _tensor_sha256(flip_scale),
        "batch_targets_sha256": _tensor_sha256(prepared["targets"]),
        "batch_weights_sha256": _tensor_sha256(prepared["weights"]),
        "batch_reference_sha256": _tensor_sha256(prepared["references"]),
        "actual_effective_batch_size": len(prepared["indices"]),
        "classification_denominator": float(prepared["weights"].sum()),
    }


def _backward_effective_batch(
    model,
    o3,
    pta,
    prepared: dict,
    device: torch.device,
    *,
    microbatch: int,
    capture_first_microbatch: bool,
) -> tuple[torch.Tensor, dict, dict | None]:
    indices = prepared["indices"]
    denominator = prepared["weights"].sum().to(device).clamp_min(1e-8)
    totals = {key: 0.0 for key in (
        "global_gce", "local_gce", "separate_gce", "fusion_gce",
        "classification", "anchor", "loss",
    )}
    loss_total = torch.zeros((), dtype=torch.float32, device=device)
    first = None
    for offset in range(0, len(indices), microbatch):
        end = offset + microbatch
        images = prepared["images"][offset:end].to(device, non_blocking=True)
        selected_flips = prepared["flips"][offset:end]
        if selected_flips.any():
            mask = torch.as_tensor(selected_flips, dtype=torch.bool, device=device)
            images[mask] = images[mask].flip(3)
        targets = prepared["targets"][offset:end].to(device)
        weights = prepared["weights"][offset:end].to(device)
        reference = prepared["references"][offset:end].to(device)
        global_logits, features, _ = forward_train_with_detached_attention(model, images)
        if not torch.isfinite(global_logits).all() or not torch.isfinite(features).all():
            raise FloatingPointError("Nonfinite PRELIM75 v10 global output")
        orientation_ids = selected_flips.astype(np.int64)
        row = np.arange(len(orientation_ids))
        bounds = prepared["boxes"][offset:end][row, orientation_ids, prepared["scales"][offset:end], :]
        bounds_tensor = torch.as_tensor(bounds, dtype=torch.int64, device=device)
        local_images = crop_with_bound_boxes(images, bounds_tensor)
        local_logits = adapted_dual_local_view_logits(model, o3, pta, local_images)
        if not torch.isfinite(local_logits).all():
            raise FloatingPointError("Nonfinite PRELIM75 v10 local output")
        terms = fusion_gce_terms(
            global_logits, local_logits, targets, temperature=1.5, blend=0.5
        )
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
            raise FloatingPointError("Nonfinite PRELIM75 v10 loss")
        if capture_first_microbatch and first is None:
            first = {
                "global_logits_sha256": _tensor_sha256(global_logits.detach().cpu()),
                "global_features_sha256": _tensor_sha256(features.detach().cpu()),
                "local_logits_sha256": _tensor_sha256(local_logits.detach().cpu()),
                "fixed_boxes_sha256": _tensor_sha256(bounds_tensor.detach().cpu()),
                "reference_sha256": _tensor_sha256(reference.detach().cpu()),
                "classification": float(weighted["classification"].detach()),
                "anchor": float(anchor.detach()),
                "loss": float(loss.detach()),
            }
        for key, value in (*weighted.items(), ("anchor", anchor), ("loss", loss)):
            totals[key] += float(value.detach())
        loss.backward()
        loss_total = loss_total + loss.detach()
    return loss_total, totals, first


def _first_batch(plan: dict, preprocess, data: dict, boxes: np.ndarray, anchor_teacher, device: torch.device) -> dict:
    dataset = OfficialImages(data["paths"], plan["train_root"], preprocess)
    generator = torch.Generator().manual_seed(42)
    stream = loader(dataset, 32, plan, shuffle=True, generator=generator)
    batch = next(iter(stream))
    flips, scales = fixed_choices(data["paths"], 19)
    return _prepare_effective_batch(
        batch, data, boxes, flips, scales, anchor_teacher, device, microbatch=16
    )


def a0_v10(plan: dict) -> dict:
    output = Path(plan["output"]) / "a0.json"
    if output.exists():
        raise FileExistsError("PRELIM75 v10 A0 already exists")
    start = time.monotonic()
    device = gpu_setup()
    data, boxes, _ = read_v10_assets(plan)
    model, preprocess, _, o3, pta, licensed_visual = _load_training_components(plan, device)
    state_audit = _module_state_audit(model, o3, pta)
    anchor_teacher = official_anchor(device)
    prepared = _first_batch(plan, preprocess, data, boxes, anchor_teacher, device)
    named = _named_trainables(model, o3, pta)
    parameters = [parameter for _, parameter in named]
    buffers = _snapshot_buffers(model, o3, pta)
    identity = _effective_batch_identity(prepared)
    rng_cpu = torch.get_rng_state().clone()
    rng_cuda = torch.cuda.get_rng_state(device).clone()
    model.zero_grad(set_to_none=True); o3.zero_grad(set_to_none=True); pta.zero_grad(set_to_none=True)
    _synchronize(device); first_started = time.monotonic()
    first_loss, first_parts, first_micro = _backward_effective_batch(
        model, o3, pta, prepared, device, microbatch=16, capture_first_microbatch=True
    )
    _synchronize(device); first_seconds = time.monotonic() - first_started
    g1 = global_l2_gradient_norm(parameters)
    leaks_first = _frozen_gradient_leaks(model, o3, pta)
    perturb = apply_standard_sam_perturbation(parameters)
    try:
        model.zero_grad(set_to_none=True); o3.zero_grad(set_to_none=True); pta.zero_grad(set_to_none=True)
        _synchronize(device); second_started = time.monotonic()
        second_loss, second_parts, _ = _backward_effective_batch(
            model, o3, pta, prepared, device, microbatch=16, capture_first_microbatch=False
        )
        _synchronize(device); second_seconds = time.monotonic() - second_started
        g2 = global_l2_gradient_norm(parameters)
        leaks_second = _frozen_gradient_leaks(model, o3, pta)
    finally:
        if not perturb.restored:
            restore_standard_sam_parameters(perturb)
    loss_increase = float(second_loss - first_loss)
    checks = {
        "finite": all(math.isfinite(value) for value in (
            float(first_loss), float(second_loss), loss_increase, g1, g2, perturb.actual_radius
        )),
        "loss_strictly_increased": loss_increase > 0.0,
        "actual_radius_in_tolerance": abs(perturb.actual_radius - RHO) <= RADIUS_TOLERANCE,
        "parameters_restored_bitwise": perturb.restored,
        "frozen_gradient_leaks": leaks_first + leaks_second,
        "buffers_unchanged": _buffers_equal(buffers, model, o3, pta),
        "cpu_rng_unchanged": bool(torch.equal(rng_cpu, torch.get_rng_state())),
        "cuda_rng_unchanged": bool(torch.equal(rng_cuda, torch.cuda.get_rng_state(device))),
    }
    passed = (
        checks["finite"] and checks["loss_strictly_increased"]
        and checks["actual_radius_in_tolerance"] and checks["parameters_restored_bitwise"]
        and not checks["frozen_gradient_leaks"] and checks["buffers_unchanged"]
        and checks["cpu_rng_unchanged"] and checks["cuda_rng_unchanged"]
    )
    result = {
        "status": "passed" if passed else "closed_a0_gate_failed",
        "plan_id": PLAN_ID,
        "checkpoint_sha256": plan["weight_parent_sha256"],
        "sample_epoch": 19,
        "batch_identity": identity,
        "licensed_parameter_names": [name for name, _ in named],
        "licensed_visual_parameters": sorted(licensed_visual),
        "state_audit": state_audit,
        "first_loss": float(first_loss),
        "perturbed_loss": float(second_loss),
        "loss_increase": loss_increase,
        "relative_loss_increase": loss_increase / max(abs(float(first_loss)), 1e-12),
        "g1_global_l2_norm": g1,
        "g2_global_l2_norm": g2,
        "theoretical_radius": RHO,
        "actual_fp32_radius": perturb.actual_radius,
        "radius_tolerance": RADIUS_TOLERANCE,
        "first_pass_parts": first_parts,
        "perturbed_pass_parts": second_parts,
        "first_microbatch": first_micro,
        "checks": checks,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "test_used": False,
        "first_pass_seconds": first_seconds,
        "second_pass_seconds": second_seconds,
        "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
    }
    atomic_json_dump(result, output)
    if not passed:
        raise RuntimeError("PRELIM75 v10 A0 failed closed")
    del model, o3, pta, anchor_teacher
    torch.cuda.empty_cache()
    return result


def _smoke_one(plan: dict, name: str, data: dict, boxes: np.ndarray, device: torch.device) -> dict:
    model, preprocess, _, o3, pta, _ = _load_training_components(plan, device)
    state_audit = _module_state_audit(model, o3, pta)
    anchor_teacher = official_anchor(device)
    prepared = _first_batch(plan, preprocess, data, boxes, anchor_teacher, device)
    optimizer, parameters, groups = _make_optimizer(model, o3, pta)
    spec = V10Schedule()
    scheduler = build_v10_scheduler(optimizer, spec)
    buffers = _snapshot_buffers(model, o3, pta)
    pass_records = {}

    def backward(pass_id: int) -> torch.Tensor:
        loss, parts, first = _backward_effective_batch(
            model, o3, pta, prepared, device, microbatch=16,
            capture_first_microbatch=pass_id == 1,
        )
        pass_records[str(pass_id)] = {"loss": float(loss), "parts": parts, "first_microbatch": first}
        return loss

    _synchronize(device); update_started = time.monotonic()
    if name == "S0":
        optimizer.zero_grad(set_to_none=True)
        first_loss = backward(1)
        gradient_norm = global_l2_gradient_norm(parameters)
        clipped = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
        optimizer.step()
        update = {
            "first_loss": float(first_loss), "first_gradient_norm": gradient_norm,
            "preclip_gradient_norm": clipped, "gradient_clipped": clipped > 1.0,
            "sam": False,
        }
    else:
        update = standard_sam_update(parameters, optimizer, backward)
        update["sam"] = True
    scheduler.step()
    _synchronize(device); update_seconds = time.monotonic() - update_started
    first = pass_records["1"]
    common = {
        **_effective_batch_identity(prepared),
        **(first["first_microbatch"] or {}),
        "full_effective_batch_loss": first["loss"],
        "first_used_lrs": {group["name"]: float(group["initial_lr"]) for group in optimizer.param_groups},
    }
    result = {
        "candidate": name,
        "common_first_pass": common,
        "passes": pass_records,
        "update": update,
        "update_seconds": update_seconds,
        "optimizer_state_entries": len(optimizer.state),
        "scheduler_position": scheduler.last_epoch,
        "frozen_gradient_leaks": _frozen_gradient_leaks(model, o3, pta),
        "buffers_unchanged": _buffers_equal(buffers, model, o3, pta),
        "state_audit": state_audit,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
    }
    if (
        result["optimizer_state_entries"] != len(parameters)
        or result["scheduler_position"] != 1
        or result["frozen_gradient_leaks"]
        or not result["buffers_unchanged"]
    ):
        raise RuntimeError(f"PRELIM75 v10 {name} optimizer smoke failed")
    del model, o3, pta, anchor_teacher, optimizer
    torch.cuda.empty_cache()
    return result


def smoke_v10(plan: dict) -> dict:
    output = Path(plan["output"]) / "smoke.json"
    if output.exists():
        raise FileExistsError("PRELIM75 v10 smoke already exists")
    start = time.monotonic()
    device = gpu_setup()
    data, boxes, _ = read_v10_assets(plan)
    candidates = {name: _smoke_one(plan, name, data, boxes, device) for name in CANDIDATES}
    left, right = candidates["S0"]["common_first_pass"], candidates["S1"]["common_first_pass"]
    exact_fields = (
        "batch_indices_sha256", "batch_paths_sha256", "batch_images_sha256",
        "batch_flip_scale_sha256", "batch_targets_sha256", "batch_weights_sha256",
        "batch_reference_sha256", "global_logits_sha256", "global_features_sha256",
        "local_logits_sha256", "fixed_boxes_sha256", "reference_sha256",
    )
    common_control = {
        "exact_fields": {key: left[key] == right[key] for key in exact_fields},
        "loss_abs_difference": abs(left["full_effective_batch_loss"] - right["full_effective_batch_loss"]),
        "initial_lrs_equal": left["first_used_lrs"] == right["first_used_lrs"],
    }
    common_control["passed"] = (
        all(common_control["exact_fields"].values())
        and common_control["loss_abs_difference"] <= 1e-7
        and common_control["initial_lrs_equal"]
    )
    parent_training = json.loads(Path(plan["parent_training_result"]).read_text(encoding="utf-8"))
    historical_s0_seconds = float(parent_training["elapsed_seconds"])
    measured_ratio = candidates["S1"]["update_seconds"] / max(candidates["S0"]["update_seconds"], 1e-9)
    ratio_for_estimate = max(2.0, measured_ratio)
    uncertainty = float(plan["budget"]["estimate_uncertainty_multiplier"])
    reserve = float(plan["budget"]["diagnostic_and_delivery_reserve_seconds"])
    estimate = {
        "historical_same_machine_L1_training_seconds": historical_s0_seconds,
        "measured_S0_update_seconds": candidates["S0"]["update_seconds"],
        "measured_S1_update_seconds": candidates["S1"]["update_seconds"],
        "measured_ratio": measured_ratio,
        "ratio_used": ratio_for_estimate,
        "uncertainty_multiplier": uncertainty,
        "diagnostic_and_delivery_reserve_seconds": reserve,
        "estimated_S0_training_seconds": historical_s0_seconds * uncertainty,
        "estimated_S1_training_seconds": historical_s0_seconds * ratio_for_estimate * uncertainty,
    }
    estimate["estimated_total_after_gates_seconds"] = (
        estimate["estimated_S0_training_seconds"] + estimate["estimated_S1_training_seconds"] + reserve
    )
    estimate["S1_under_eight_hours"] = estimate["estimated_S1_training_seconds"] < 8 * 3600
    estimate["total_under_budget"] = estimate["estimated_total_after_gates_seconds"] < plan["budget"]["gpu_action_seconds"]
    passed = common_control["passed"] and estimate["S1_under_eight_hours"] and estimate["total_under_budget"]
    result = {
        "status": "passed" if passed else "closed_smoke_or_resource_gate_failed",
        "plan_id": PLAN_ID,
        "effective_batch_size": 32,
        "microbatch_size": 16,
        "accumulation": 2,
        "candidates": candidates,
        "common_control": common_control,
        "resource_estimate": estimate,
        "formal_instances_reloaded_from_L1": True,
        "smoke_weights_discarded": True,
        "test_used": False,
        "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
    }
    atomic_json_dump(result, output)
    if not passed:
        raise RuntimeError("PRELIM75 v10 smoke/resource gate failed closed")
    return result


# ---------------------------------------------------------------------------
# Formal training, diagnostic, and delivery
# ---------------------------------------------------------------------------

def _group_gradient_norms(groups: dict[str, list[torch.nn.Parameter]]) -> dict[str, float]:
    return {name: _grad_norm(parameters) for name, parameters in groups.items()}


def _trace_header(writer: csv.writer) -> None:
    writer.writerow([
        "candidate", "update_index", "sample_epoch", "completed_samples",
        "used_lr_backbone", "used_lr_head", "used_lr_local_adapters",
        "next_lr_backbone", "next_lr_head", "next_lr_local_adapters",
        "first_loss", "perturbed_loss", "loss_increase",
        "g1_global_l2", "g2_global_l2", "actual_radius",
        "preclip_update_gradient_l2", "gradient_clipped", "restored_bitwise",
    ])


def train_v10(plan: dict, name: str) -> Path:
    if name not in CANDIDATES:
        raise ValueError("Only PRELIM75 v10 S0/S1 are allowed")
    gate_root = Path(plan["output"])
    a0 = json.loads((gate_root / "a0.json").read_text(encoding="utf-8"))
    smoke = json.loads((gate_root / "smoke.json").read_text(encoding="utf-8"))
    if a0.get("status") != "passed" or smoke.get("status") != "passed":
        raise ValueError("PRELIM75 v10 formal training requires passed A0 and smoke")
    output = gate_root / name
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    device = gpu_setup()
    data, boxes, _ = read_v10_assets(plan)
    model, preprocess, checkpoint, o3, pta, licensed_visual = _load_training_components(plan, device)
    state_audit = _module_state_audit(model, o3, pta)
    anchor_teacher = official_anchor(device)
    config = _candidate_config(checkpoint, plan, name)
    run_lineage_audit(
        config,
        child_train_csv=plan["train_csv"],
        child_val_csv=plan["val_csv"],
        checkpoint_path=plan["weight_parent"],
        output_path=output / "split_lineage_audit.json",
    )
    atomic_json_dump(config, output / "resolved_config.json")
    atomic_json_dump(source_manifest(), output / "source_manifest.json")
    frozen = {
        parameter_name: parameter.detach().cpu().clone()
        for parameter_name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    initial_buffers = _snapshot_buffers(model, o3, pta)
    named = _named_trainables(model, o3, pta)
    optimizer, parameters, groups = _make_optimizer(model, o3, pta)
    dataset = OfficialImages(data["paths"], plan["train_root"], preprocess)
    generator = torch.Generator().manual_seed(42)
    stream = loader(dataset, 32, plan, shuffle=True, generator=generator)
    if len(stream) != DEFAULT_STEPS_PER_EPOCH:
        raise RuntimeError("PRELIM75 v10 real DataLoader update count changed")
    spec = V10Schedule(len(stream))
    scheduler = build_v10_scheduler(optimizer, spec)
    recipe = {
        "candidate": name,
        "candidate_id": f"PRELIM75_V10_SAM_{name}",
        "plan_id": PLAN_ID,
        "independent_parent_load": True,
        "weight_parent_sha256": plan["weight_parent_sha256"],
        "weight_parent_experiment_id": PARENT_EXPERIMENT_ID,
        "optimizer": "fresh_AdamW",
        "standard_sam": name == SAM_CANDIDATE,
        "sam_type": "standard_nonadaptive_global_l2" if name == SAM_CANDIDATE else None,
        "sam_rho": RHO if name == SAM_CANDIDATE else None,
        "sam_gradient_averaging": False,
        "sam_weight_decay_in_perturbation": False,
        "training_geometry_policy": "frozen_V1",
        "geometry_origin_sha256": plan["geometry_origin_sha256"],
        "geometry_boxes_sha256": plan["geometry_boxes_sha256"],
        "original_supervision_sha256": plan["original_supervision_sha256"],
        "sample_epochs": list(SAMPLE_EPOCHS),
        "steps_per_epoch": len(stream),
        "total_updates": TOTAL_UPDATES,
        "effective_batch_size": 32,
        "microbatch_size": 16,
        "accumulation": 2,
        "scheduler_horizon_updates": TOTAL_UPDATES,
        "scheduler_floor_ratio": 0.01,
        "visual_lr_weight_decay": [1e-6, 0.0],
        "head_lr_weight_decay": [1.5e-7, 1e-4],
        "adapter_lr_weight_decay": [3e-6, 0.0],
        "global_weight": 0.6,
        "local_weight": 0.4,
        "anchor_weight": 2.0,
        "fusion_blend": 0.5,
        "classification_temperature": 1.5,
        "gce_q": 0.5,
        "anchor": "frozen_official_CLIP_same_global_pixel_tensor_reused_between_SAM_passes",
        "selection": "fixed_epoch_21_last_epoch",
        "licensed_visual_parameters": sorted(licensed_visual),
        "licensed_parameter_names": [parameter_name for parameter_name, _ in named],
        "state_audit": state_audit,
        "test_used_for_training": False,
        "prior_in_inference": False,
    }
    atomic_json_dump(recipe, output / "training_recipe.json")

    optimizer_step = 0
    clipped_steps = 0
    sam_nonincreasing_steps = 0
    history = []
    first_step_audit = None
    trace_path = output / "update_trace.csv"
    trace = trace_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(trace)
    _trace_header(writer)
    try:
        for sample_epoch in SAMPLE_EPOCHS:
            epoch_start = time.monotonic()
            flips, scales = fixed_choices(data["paths"], sample_epoch)
            totals = {
                "base_loss": 0.0, "perturbed_loss": 0.0,
                "classification": 0.0, "anchor": 0.0,
                "samples": 0, "positive_weight_hits": 0,
                "effective_weight_mass": 0.0, "gradient_norm_sum": 0.0,
                "gradient_norm_max": 0.0, "actual_radius_sum": 0.0,
                "actual_radius_min": float("inf"), "actual_radius_max": 0.0,
            }
            for step, batch in enumerate(stream):
                prepared = _prepare_effective_batch(
                    batch, data, boxes, flips, scales, anchor_teacher, device, microbatch=16
                )
                pass_records: dict[str, dict] = {}

                def backward(pass_id: int) -> torch.Tensor:
                    loss, parts, first_micro = _backward_effective_batch(
                        model, o3, pta, prepared, device, microbatch=16,
                        capture_first_microbatch=optimizer_step == 0 and pass_id == 1,
                    )
                    pass_records[str(pass_id)] = {
                        "loss": float(loss), "parts": parts, "first_microbatch": first_micro,
                        "gradient_groups": _group_gradient_norms(groups),
                        "frozen_gradient_leaks": _frozen_gradient_leaks(model, o3, pta),
                    }
                    return loss

                before_lrs = audit_position(optimizer, spec, optimizer_step)
                if name == "S0":
                    optimizer.zero_grad(set_to_none=True)
                    base_loss = backward(1)
                    g1 = global_l2_gradient_norm(parameters)
                    preclip = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
                    try:
                        optimizer.step()
                    except BaseException as exc:
                        raise RuntimeError(
                            "S0 AdamW step failed; discard partial update and restart from the latest complete boundary"
                        ) from exc
                    update = {
                        "first_loss": float(base_loss), "perturbed_loss": float(base_loss),
                        "loss_increase": 0.0, "first_gradient_norm": g1,
                        "second_gradient_norm": g1, "actual_radius": 0.0,
                        "preclip_second_gradient_norm": preclip,
                        "gradient_clipped": preclip > 1.0, "restored_bitwise": True,
                    }
                else:
                    update = standard_sam_update(parameters, optimizer, backward)
                    sam_nonincreasing_steps += int(update["loss_increase"] <= 0.0)
                scheduler.step()
                after_lrs = audit_position(optimizer, spec, optimizer_step + 1)
                if pass_records["1"]["frozen_gradient_leaks"]:
                    raise RuntimeError("Frozen parameter received a formal PRELIM75 v10 gradient")
                if optimizer_step == 0:
                    first_step_audit = {
                        "candidate": name,
                        **_effective_batch_identity(prepared),
                        **(pass_records["1"]["first_microbatch"] or {}),
                        "full_effective_batch_loss": pass_records["1"]["loss"],
                        "first_pass_gradient_groups": pass_records["1"]["gradient_groups"],
                        "first_used_lrs": before_lrs,
                        "first_update": update,
                    }
                    required_groups = ("visual", "head", "o3", "pta")
                    if any(
                        not math.isfinite(first_step_audit["first_pass_gradient_groups"][key])
                        or first_step_audit["first_pass_gradient_groups"][key] <= 0.0
                        for key in required_groups
                    ):
                        raise RuntimeError("PRELIM75 v10 first update lacks registered gradient coverage")
                    atomic_json_dump(first_step_audit, output / "first_step_gradient_audit.json")
                writer.writerow([
                    name, optimizer_step, sample_epoch, totals["samples"] + len(prepared["indices"]),
                    *(before_lrs[key] for key in ("backbone", "head", "local_adapters")),
                    *(after_lrs[key] for key in ("backbone", "head", "local_adapters")),
                    update["first_loss"], update["perturbed_loss"], update["loss_increase"],
                    update["first_gradient_norm"], update["second_gradient_norm"],
                    update["actual_radius"], update["preclip_second_gradient_norm"],
                    int(update["gradient_clipped"]), int(update["restored_bitwise"]),
                ])
                if optimizer_step % 50 == 0:
                    trace.flush()
                optimizer_step += 1
                clipped_steps += int(update["gradient_clipped"])
                batch_size = len(prepared["indices"])
                base_parts = pass_records["1"]["parts"]
                totals["base_loss"] += update["first_loss"] * batch_size
                totals["perturbed_loss"] += update["perturbed_loss"] * batch_size
                totals["classification"] += base_parts["classification"] * batch_size
                totals["anchor"] += base_parts["anchor"] * batch_size
                totals["samples"] += batch_size
                positive = prepared["weights"] > 0
                totals["positive_weight_hits"] += int(positive.sum())
                totals["effective_weight_mass"] += float(prepared["weights"].sum())
                update_norm = float(update["preclip_second_gradient_norm"])
                totals["gradient_norm_sum"] += update_norm
                totals["gradient_norm_max"] = max(totals["gradient_norm_max"], update_norm)
                if name == "S1":
                    radius = float(update["actual_radius"])
                    totals["actual_radius_sum"] += radius
                    totals["actual_radius_min"] = min(totals["actual_radius_min"], radius)
                    totals["actual_radius_max"] = max(totals["actual_radius_max"], radius)
                if step % 100 == 0:
                    atomic_json_dump({
                        "status": "training", "candidate": name,
                        "sample_epoch": sample_epoch, "completed_samples": totals["samples"],
                        "optimizer_steps": optimizer_step,
                        "sam_nonincreasing_steps": sam_nonincreasing_steps,
                        "used_lr": after_lrs, "elapsed_seconds": time.monotonic() - start,
                        "at_complete_optimizer_update_boundary": True,
                    }, output / "status.json")
            row = {
                "sample_epoch": sample_epoch,
                "base_loss": totals["base_loss"] / totals["samples"],
                "perturbed_loss": totals["perturbed_loss"] / totals["samples"],
                "classification": totals["classification"] / totals["samples"],
                "anchor": totals["anchor"] / totals["samples"],
                "samples": totals["samples"],
                "positive_weight_hits": totals["positive_weight_hits"],
                "effective_weight_mass": totals["effective_weight_mass"],
                "optimizer_steps": optimizer_step,
                "updates_this_epoch": len(stream),
                "gradient_norm_mean": totals["gradient_norm_sum"] / len(stream),
                "gradient_norm_max": totals["gradient_norm_max"],
                "clipped_steps_cumulative": clipped_steps,
                "sam_nonincreasing_steps_cumulative": sam_nonincreasing_steps,
                "actual_radius_mean": totals["actual_radius_sum"] / len(stream) if name == "S1" else 0.0,
                "actual_radius_min": totals["actual_radius_min"] if name == "S1" else 0.0,
                "actual_radius_max": totals["actual_radius_max"] if name == "S1" else 0.0,
                "scheduler_multiplier": spec.multiplier(optimizer_step),
                "epoch_elapsed_seconds": time.monotonic() - epoch_start,
                "elapsed_seconds": time.monotonic() - start,
                "peak_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
            }
            history.append(row)
            atomic_json_dump(history, output / "training_history.json")
            atomic_json_dump({
                "status": "complete_epoch_boundary", "candidate": name,
                "sample_epoch": sample_epoch, "optimizer_steps": optimizer_step,
                "candidate_checkpoint_saved": False,
                "safe_to_restart_from_parent_only": True,
            }, output / f"epoch_{sample_epoch}_boundary_receipt.json")
            print(json.dumps(row), flush=True)
    except BaseException as exc:
        atomic_json_dump({
            "status": "failed_discard_partial_process",
            "candidate": name,
            "optimizer_steps_at_last_complete_boundary": optimizer_step,
            "exception": repr(exc),
            "candidate_checkpoint_saved": False,
            "must_not_continue_from_in_memory_optimizer": True,
        }, output / "failure.json")
        raise
    finally:
        trace.close()

    if optimizer_step != TOTAL_UPDATES:
        raise RuntimeError(f"PRELIM75 v10 update count mismatch: {optimizer_step} != {TOTAL_UPDATES}")
    for parameter_name, parameter in model.named_parameters():
        if parameter_name in frozen and not torch.equal(parameter.detach().cpu(), frozen[parameter_name]):
            raise RuntimeError(f"Frozen L1 tensor changed during PRELIM75 v10 training: {parameter_name}")
    if not _buffers_equal(initial_buffers, model, o3, pta):
        raise RuntimeError("PRELIM75 v10 model buffers changed during formal training")

    payload = saved_candidate(checkpoint, model, config, name, plan, optimizer, scheduler, 21, o3, pta)
    payload["global_step"] = int(checkpoint.get("global_step", 0)) + optimizer_step
    payload["prelim75_training"] = {
        "name": name,
        "candidate_id": f"PRELIM75_V10_SAM_{name}",
        "plan_id": PLAN_ID,
        "weight_parent_experiment_id": PARENT_EXPERIMENT_ID,
        "weight_parent_sha256": plan["weight_parent_sha256"],
        "training_geometry_policy": "frozen_V1",
        "comparison_geometry_origin": "V1",
        "geometry_origin_sha256": plan["geometry_origin_sha256"],
        "geometry_boxes_sha256": plan["geometry_boxes_sha256"],
        "original_supervision_sha256": plan["original_supervision_sha256"],
        "supervision_changed": False,
        "inference_teacher_used": False,
        "validation_scope": "overlap_diagnostic",
        "upstream_provenance_complete": False,
        "selection": "fixed_epoch_21_last_epoch",
        "supervision": "unchanged_original_w_targets",
        "classification_temperature": 1.5,
        "probability_fusion_blend": 0.5,
        "global_weight": 0.6,
        "local_weight": 0.4,
        "anchor_weight": 2.0,
        "cosine_horizon_epochs": 3,
        "scheduler_floor_ratio": 0.01,
        "scheduler_warmup_steps": 0,
        "sample_epochs": list(SAMPLE_EPOCHS),
        "total_updates": optimizer_step,
        "optimizer": "AdamW",
        "standard_sam": name == "S1",
        "sam_type": "standard_nonadaptive_global_l2" if name == "S1" else None,
        "sam_rho": RHO if name == "S1" else None,
        "global_trusted_ce": False,
        "local_trusted_ce": False,
        "recovery_kd": False,
        "prior_in_inference": False,
        "training_geometry_required_for_inference": False,
    }
    _atomic_torch_save(payload, output / "candidate.pt")
    checkpoint_hash = sha256_file(output / "candidate.pt")
    peak = torch.cuda.max_memory_allocated()
    del payload, anchor_teacher
    model.cpu(); o3.cpu(); pta.cpu()
    del model, o3, pta
    torch.cuda.empty_cache()
    reloaded_model, _, reloaded, reloaded_o3, reloaded_pta = load_composite(
        output / "candidate.pt", torch.device("cpu")
    )
    metadata = reloaded.get("prelim75_training", {})
    if (
        metadata.get("plan_id") != PLAN_ID
        or metadata.get("candidate_id") != f"PRELIM75_V10_SAM_{name}"
        or metadata.get("weight_parent_sha256") != plan["weight_parent_sha256"]
        or metadata.get("total_updates") != TOTAL_UPDATES
        or metadata.get("standard_sam") != (name == "S1")
    ):
        raise RuntimeError("Saved PRELIM75 v10 candidate lineage changed on strict reload")
    result = {
        "status": "trained_pending_real_diagnostic",
        "candidate": name,
        "optimizer": "AdamW",
        "standard_sam": name == "S1",
        "sam_rho": RHO if name == "S1" else None,
        "steps_per_epoch": DEFAULT_STEPS_PER_EPOCH,
        "total_updates": TOTAL_UPDATES,
        "history": history,
        "first_step_gradient_audit": first_step_audit,
        "sam_nonincreasing_steps": sam_nonincreasing_steps,
        "reload_check": {
            "complete_model_strictly_reloaded": True,
            "shared_head_present": hasattr(reloaded_model, "classifier"),
            "local_feature_adapter_reloaded": reloaded_o3 is not None,
            "part_token_adapter_reloaded": reloaded_pta is not None,
            "training_geometry_accessed_by_reload": False,
            "training_supervision_accessed_by_reload": False,
        },
        "checkpoint_sha256": checkpoint_hash,
        "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": peak,
        "online_accuracy": None,
        "validation_scope": "overlap_diagnostic",
    }
    atomic_json_dump(result, output / "training_result.json")
    atomic_json_dump(result, output / "status.json")
    return output / "candidate.pt"


def _inference_gpu_setup() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("PRELIM75 v10 diagnostic requires CUDA")
    from aegis_clip.runtime import set_seed
    set_seed(42, deterministic=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    return torch.device("cuda")


@torch.no_grad()
def evaluate_v10(plan: dict, name: str) -> dict:
    if name not in CANDIDATES:
        raise ValueError("Unknown PRELIM75 v10 candidate")
    root = Path(plan["output"]) / name
    checkpoint_path = root / "candidate.pt"
    output = root / "diagnostic"
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    device = _inference_gpu_setup()
    data = supervision(plan)
    model, preprocess, checkpoint, o3, pta = load_composite(checkpoint_path, device)
    model.requires_grad_(False); o3.requires_grad_(False); pta.requires_grad_(False)
    model.eval(); o3.eval(); pta.eval()
    metadata = checkpoint.get("prelim75_training", {})
    if (
        metadata.get("plan_id") != PLAN_ID
        or metadata.get("candidate_id") != f"PRELIM75_V10_SAM_{name}"
        or metadata.get("weight_parent_sha256") != plan["weight_parent_sha256"]
        or metadata.get("standard_sam") != (name == "S1")
        or metadata.get("training_geometry_policy") != "frozen_V1"
        or metadata.get("supervision_changed") is not False
        or metadata.get("prior_in_inference") is not False
    ):
        raise ValueError("PRELIM75 v10 candidate lineage mismatch")
    frame = pd.read_csv(plan["val_csv"])
    paths = [canonical_sample_path(path) for path in frame.image_path.astype(str)]
    if len(paths) != REQUIRED_DIAGNOSTIC_ROWS or len(set(paths)) != len(paths):
        raise ValueError("PRELIM75 v10 diagnostic row identity mismatch")
    training_index = {path: index for index, path in enumerate(data["paths"])}
    indices = torch.tensor([training_index[path] for path in paths], dtype=torch.long)
    labels = torch.tensor(frame.label.to_numpy(), dtype=torch.long)
    if not torch.equal(labels, data["labels"][indices]):
        raise ValueError("PRELIM75 v10 diagnostic labels changed")
    dataset = OfficialImages(paths, plan["train_root"], preprocess)
    stream = loader(dataset, 64, plan)
    fused_predictions, global_predictions, local_predictions = [], [], []
    scale_weights = torch.tensor(LOCAL_SCALE_WEIGHTS, device=device).view(1, 1, 4, 1)
    for step, batch in enumerate(stream):
        images = batch["images"].to(device, non_blocking=True)
        global_logits, local_logits, _ = _teacher_views(model, o3, pta, images)
        global_probability = (global_logits.float() / 1.5).softmax(-1).mean(1)
        local_probability = ((local_logits.float() / 1.5).softmax(-1) * scale_weights).sum(2).mean(1)
        fused_probability = 0.6 * global_probability + 0.4 * local_probability
        if not torch.isfinite(fused_probability).all():
            raise FloatingPointError("Invalid PRELIM75 v10 diagnostic probability")
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
        "labels": labels,
        "clean_probability": data["clean_probability"][indices],
        "pseudo_labels": data["pseudo_label"][indices],
        "correction_alpha": data["correction_alpha"][indices],
    }
    branch_metrics = {
        key: prediction_metrics(value, **fields, num_classes=500, clean_core_threshold=0.7)
        for key, value in predictions.items()
    }
    parent = json.loads(Path(plan["parent_diagnostic"]).read_text(encoding="utf-8"))
    deltas = {
        key: 100.0 * (branch_metrics["fused"][key] - parent["metrics"][key])
        for key in ("raw_micro", "clean_core_micro")
    }
    engineering_stop = any(value < -2.0 for value in deltas.values())
    result = {
        "status": "complete",
        "candidate": name,
        "standard_sam": name == "S1",
        "sam_rho": RHO if name == "S1" else None,
        "validation_scope": "overlap_diagnostic",
        "independent_generalization_claim": False,
        "metrics": branch_metrics["fused"],
        "branch_metrics": branch_metrics,
        "delta_pp_vs_L1": deltas,
        "engineering_stop": engineering_stop,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parent_checkpoint_sha256": plan["weight_parent_sha256"],
        "parent_diagnostic_sha256": plan["parent_diagnostic_sha256"],
        "val_csv_sha256": plan["val_csv_sha256"],
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "prior_in_inference": False,
        "inference_teacher_used": False,
        "real_candidate_attention_recomputed": True,
        "training_boxes_accessed_for_diagnostic": False,
        "inference_numerics": copy.deepcopy(EXPECTED_INFERENCE),
        "online_accuracy": None,
        "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
    }
    _atomic_torch_save({"paths": paths, **predictions, **fields}, output / "predictions.pt")
    atomic_json_dump(result, output / "result.json")
    atomic_json_dump(result, output / "status.json")
    print(json.dumps(result, indent=2), flush=True)
    del model, o3, pta
    torch.cuda.empty_cache()
    return result


def deliver_v10(plan: dict, name: str) -> dict:
    if name not in CANDIDATES:
        raise ValueError("Unknown PRELIM75 v10 candidate")
    root = Path(plan["output"]) / name
    diagnostic = json.loads((root / "diagnostic/result.json").read_text(encoding="utf-8"))
    if diagnostic.get("status") != "complete" or diagnostic.get("engineering_stop", True):
        raise ValueError("PRELIM75 v10 candidate stopped by the fixed engineering rule")
    checkpoint = root / "candidate.pt"
    if sha256_file(checkpoint) != diagnostic["checkpoint_sha256"]:
        raise ValueError("PRELIM75 v10 candidate changed after diagnostic")
    output = root / "submission"
    if output.exists():
        raise FileExistsError("Do not overwrite an existing PRELIM75 v10 submission")
    command = [
        sys.executable, "-u", "-m", "aegis_clip.cli.infer_prelim75_v10_sam",
        "--checkpoint", str(checkpoint), "--output-dir", str(output),
        "--local-view", "attention_multiscale", "--local-crop-sizes", "112,128,144,160",
        "--local-scale-weights", "0.20,0.30,0.40,0.10", "--local-top-k", "5",
        "--local-weight", "0.40", "--local-temperature", "1.5",
        "--adapt-local-features", "--adapt-part-token-features",
        "--tta", "horizontal_flip", "--tta-fusion", "mean_probabilities",
        "--tta-temperature", "1.5", "--tta-view-weight", "0.50",
        "--acknowledge-local-view-risk", "--acknowledge-tta-risk", "--batch-size", "64",
    ]
    with (root / "inference.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    receipt = json.loads((output / "prelim75_v10_inference_numerics.json").read_text(encoding="utf-8"))
    if receipt.get("settings") != EXPECTED_INFERENCE:
        raise ValueError("PRELIM75 v10 inference numerical context changed")
    rows = []
    with (output / "pred_results.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 2:
                raise ValueError("Generic inference emitted malformed PRELIM75 v10 CSV")
            rows.append((row[0].strip(), row[1].strip()))
    (output / "pred_results.csv").write_text(
        "".join(f"{image_name}, {label}\n" for image_name, label in rows), encoding="utf-8"
    )
    with ZipFile(output / "submission.zip", "w", ZIP_DEFLATED) as archive:
        archive.write(output / "pred_results.csv", arcname="pred_results.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "prediction_csv_sha256": sha256_file(output / "pred_results.csv"),
        "submission_zip_sha256": sha256_file(output / "submission.zip"),
        "csv_delimiter": "comma_space",
        "prelim75_v10_inference_numerics": EXPECTED_INFERENCE,
    })
    atomic_json_dump(manifest, output / "manifest.json")
    check_command = [
        sys.executable, str(repository_root() / "scripts/check_submission.py"),
        "--test_dir", plan["test_root"], "--num-classes", "500",
        "--csv", str(output / "pred_results.csv"), "--zip", str(output / "submission.zip"),
    ]
    with (root / "submission_validation.log").open("w", encoding="utf-8") as log:
        subprocess.run(check_command, check=True, stdout=log, stderr=subprocess.STDOUT)
    raw = (output / "pred_results.csv").read_bytes()
    with ZipFile(output / "submission.zip") as archive:
        if archive.namelist() != ["pred_results.csv"] or archive.read("pred_results.csv") != raw:
            raise ValueError("PRELIM75 v10 ZIP/CSV byte identity failed")
    if len(rows) != REQUIRED_TEST_ROWS or len({row[0] for row in rows}) != REQUIRED_TEST_ROWS:
        raise ValueError("PRELIM75 v10 test coverage mismatch")
    report = {
        "candidate": name,
        "candidate_id": f"PRELIM75_V10_SAM_{name}",
        "status": "submission_ready_pending_platform",
        "standard_sam": name == "S1",
        "sam_rho": RHO if name == "S1" else None,
        "checkpoint_sha256": sha256_file(checkpoint),
        "csv_sha256": sha256_file(output / "pred_results.csv"),
        "zip_sha256": sha256_file(output / "submission.zip"),
        "inference_manifest_sha256": sha256_file(output / "manifest.json"),
        "inference_numerics_receipt_sha256": sha256_file(output / "prelim75_v10_inference_numerics.json"),
        "resolved_config_sha256": sha256_file(root / "resolved_config.json"),
        "source_manifest_sha256": sha256_file(root / "source_manifest.json"),
        "inference_command": command,
        "validation_command": check_command,
        "validation_exit_code": 0,
        "rows": REQUIRED_TEST_ROWS,
        "unique_prediction_classes": len({row[1] for row in rows}),
        "prediction_coverage_forced_to_500": False,
        "zip_internal_csv_byte_equal": True,
        "csv_delimiter": "comma_space",
        "diagnostic": diagnostic,
        "single_checkpoint": True,
        "training_geometry_required_for_final_inference": False,
        "training_supervision_required_for_final_inference": False,
        "anchor_teacher_required_for_final_inference": False,
        "v9_source_recrop_used": False,
        "prior_in_inference": False,
        "online_accuracy": None,
        "online_exact_correct": None,
        "actual_platform_upload_time": None,
    }
    atomic_json_dump(report, root / "candidate_report.json")
    atomic_json_dump(report, root / "status.json")
    print(json.dumps(report, indent=2), flush=True)
    return report
