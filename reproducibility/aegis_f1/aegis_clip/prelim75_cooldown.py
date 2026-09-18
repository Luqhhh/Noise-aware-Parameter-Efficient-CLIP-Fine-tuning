"""PRELIM75 v7 fixed cooldown-schedule comparison.

L0 and L1 independently warm-start from the completed v6 G0 checkpoint.  The
only registered difference is the cosine horizon used by the fresh LambdaLR:
L0 keeps the historical 18-epoch horizon, while L1 decays to the repository
1% floor inside the actual three-epoch training budget.  Loss, supervision,
training boxes, views, batch semantics, optimizer groups, seed, and final
inference remain the v6 G0 protocol.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
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
from aegis_clip.localization import (
    attention_weighted_centers,
    extract_attention_crops,
    forward_features_with_last_block_attention,
)
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
from aegis_clip.prelim75_fusion import fusion_gce_terms
from aegis_clip.prelim75_joint import crop_with_bound_boxes, fixed_choices, official_anchor
from aegis_clip.prelim75_online_geometry import (
    _accumulate_box_drift,
    _cuda_memory_clear,
    _drift_accumulator,
    _drift_summary,
    _empty_positive_safety,
    _first_step_probes,
    _forward_global_with_captured_tokens,
    _grad_norm,
    _module_hook_count,
    _probe_norm,
    _tensor_sha256,
    attention_boxes_for_all_scales,
    attention_crop_boxes,
    forward_train_with_detached_attention,
    replay_last_block_attention,
)
from aegis_clip.prelim75_recovery import _teacher_views
from aegis_clip.prelim75_trusted_ce import load_v1_geometry, original_supervision_sha256
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.trainer import _warmup_cosine


PLAN_ID = "PRELIM75_V7_20260918"
PARENT_PLAN_ID = "PRELIM75_V6_20260918"
PARENT_EXPERIMENT_ID = "PRELIM75_V6_G0"
GRANDPARENT_EXPERIMENT_ID = "PRELIM75_V5_F1"
GRANDPARENT_SHA256 = "44e64a9528653f1d76db6a12b56a22c3ad23851665e8cef45d06f7ece9f52e1b"
CANDIDATES = ("L0", "L1")
CANDIDATE_HORIZONS = {"L0": 18, "L1": 3}
DEFAULT_STEPS_PER_EPOCH = 3226
TOTAL_UPDATES = 3 * DEFAULT_STEPS_PER_EPOCH
FLOOR_RATIO = 0.01
WARMUP_STEPS = 0
REQUIRED_TRAIN_ROWS = 103218
REQUIRED_DIAGNOSTIC_ROWS = 10316
REQUIRED_TEST_ROWS = 24967
REQUIRED_CLASSES = 500
_LOCAL_SCALES = (112, 128, 144, 160)
_LOCAL_SCALE_WEIGHTS = (0.2, 0.3, 0.4, 0.1)
_ATTENTION_TOP_K = 5
_V5_C0_SHA256 = "48d4f4ccec8758f93b126e059790107981c2359a94963d5d0147d49f05613f8a"

EXPECTED_PROTOCOL = {
    "epochs": 3,
    "sample_epochs": [16, 17, 18],
    "effective_batch_size": 32,
    "global_weight": 0.6,
    "local_weight": 0.4,
    "anchor_weight": 2.0,
    "fusion_blend": 0.5,
    "classification_temperature": 1.5,
    "gce_q": 0.5,
    "epsilon": 1e-7,
    "local_scales": list(_LOCAL_SCALES),
    "local_scale_weights": list(_LOCAL_SCALE_WEIGHTS),
    "attention_top_k": _ATTENTION_TOP_K,
    "gpu_budget_seconds": 28800,
}
EXPECTED_SCHEDULER = {
    "floor_ratio": FLOOR_RATIO,
    "warmup_steps": WARMUP_STEPS,
    "candidates": {
        "L0": {"cosine_horizon_epochs": 18},
        "L1": {"cosine_horizon_epochs": 3},
    },
}

_PLAN_PATH_KEYS = (
    "weight_parent",
    "fallback_submission",
    "fallback_csv",
    "parent_diagnostic",
    "geometry_manifest",
    "geometry_paths",
    "geometry_boxes",
    "trust",
    "train_csv",
    "val_csv",
    "class_mapping",
    "groups",
    "train_root",
    "test_root",
    "output",
)
_PLAN_HASH_KEYS = (
    "weight_parent",
    "fallback_submission",
    "fallback_csv",
    "geometry_manifest",
    "geometry_paths",
    "geometry_boxes",
    "trust",
    "train_csv",
    "val_csv",
    "class_mapping",
    "groups",
)


def _resolve_plan_paths(plan: dict, source: Path) -> None:
    for key in _PLAN_PATH_KEYS:
        if key not in plan or not isinstance(plan[key], str) or not plan[key]:
            raise ValueError(f"Missing PRELIM75 v7 path: {key}")
        plan[key] = str((source.parent / plan[key]).resolve())


def _validate_v7_plan_schema(plan: dict) -> None:
    allowed = {
        "plan_id",
        "seed",
        "weight_parent_experiment_id",
        "fallback_platform_percent",
        "num_workers",
        "protocol",
        "scheduler",
        "geometry_origin_sha256",
        "original_supervision_sha256",
        *_PLAN_PATH_KEYS,
        *(key + "_sha256" for key in _PLAN_HASH_KEYS),
    }
    if set(plan) != allowed:
        missing = sorted(allowed - set(plan))
        extra = sorted(set(plan) - allowed)
        raise ValueError(f"Unknown or incomplete fixed PRELIM75 v7 plan: missing={missing}, extra={extra}")
    if plan.get("plan_id") != PLAN_ID or plan.get("seed") != 42:
        raise ValueError("PRELIM75 v7 plan id or seed changed")
    if plan.get("weight_parent_experiment_id") != PARENT_EXPERIMENT_ID:
        raise ValueError("PRELIM75 v7 registered parent experiment id changed")
    if not math.isclose(float(plan.get("fallback_platform_percent", float("nan"))), 69.2274, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("PRELIM75 v7 fallback platform percent changed")
    if plan.get("num_workers") != 4:
        raise ValueError("PRELIM75 v7 requires exactly four data workers")
    if plan.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("PRELIM75 v7 protocol differs from the registered fixed plan")
    if plan.get("scheduler") != EXPECTED_SCHEDULER:
        raise ValueError("PRELIM75 v7 scheduler differs from the registered fixed plan")
    for key in _PLAN_HASH_KEYS:
        if not isinstance(plan.get(key + "_sha256"), str) or len(plan[key + "_sha256"]) != 64:
            raise ValueError(f"PRELIM75 v7 missing sha256 for {key}")


def load_v7_plan(path: str | Path) -> dict:
    source = Path(path).resolve()
    if source.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("PRELIM75 v7 config must be YAML")
    plan = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("PRELIM75 v7 config root must be a mapping")
    _validate_v7_plan_schema(plan)
    _resolve_plan_paths(plan, source)
    for key in _PLAN_HASH_KEYS:
        if sha256_file(plan[key]) != plan[key + "_sha256"]:
            raise ValueError(f"Frozen PRELIM75 v7 asset hash mismatch: {key}")
    train_root = Path(plan["train_root"]).resolve()
    test_root = Path(plan["test_root"]).resolve()
    if train_root == test_root or train_root.is_relative_to(test_root) or test_root.is_relative_to(train_root):
        raise ValueError("Official train and test roots must be disjoint")
    diagnostic_path = Path(plan["parent_diagnostic"]).resolve()
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    _validate_parent_diagnostic(plan, diagnostic)
    plan["parent_diagnostic_sha256"] = sha256_file(diagnostic_path)
    plan["parent_metrics"] = {
        "raw_micro": float(diagnostic["metrics"]["raw_micro"]),
        "clean_core_micro": float(diagnostic["metrics"]["clean_core_micro"]),
    }
    plan["parent"] = plan["weight_parent"]
    plan["parent_sha256"] = plan["weight_parent_sha256"]
    plan["_config_path"] = str(source)
    plan["_config_sha256"] = sha256_file(source)
    return plan


def _validate_parent_diagnostic(plan: dict, diagnostic: dict) -> None:
    required = {
        "status": "complete",
        "candidate": "G0",
        "engineering_stop": False,
        "checkpoint_sha256": plan["weight_parent_sha256"],
        "val_csv_sha256": plan["val_csv_sha256"],
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "prior_in_inference": False,
        "real_candidate_attention_recomputed": True,
        "validation_scope": "overlap_diagnostic",
        "training_boxes_accessed_for_diagnostic": False,
    }
    for key, value in required.items():
        if diagnostic.get(key) != value:
            raise ValueError(f"v6 G0 parent diagnostic identity mismatch: {key}")
    if diagnostic.get("independent_generalization_claim") is not False:
        raise ValueError("v6 G0 parent diagnostic must not claim independent generalization")
    metrics = diagnostic.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("samples") != REQUIRED_DIAGNOSTIC_ROWS:
        raise ValueError("v6 G0 parent diagnostic metrics are incomplete")
    for key in ("raw_micro", "clean_core_micro"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"v6 G0 parent diagnostic metric is invalid: {key}")
    if "online_accuracy" not in diagnostic or diagnostic["online_accuracy"] is not None:
        raise ValueError("v6 G0 parent diagnostic must not borrow a platform score")


def _validate_parent_checkpoint(plan: dict):
    checkpoint = torch.load(plan["weight_parent"], map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    metadata = checkpoint.get("prelim75_training", {})
    experiment_id = config.get("project", {}).get("experiment_id")
    if experiment_id != PARENT_EXPERIMENT_ID:
        raise ValueError("Registered v7 weight parent is not the actual v6 G0")
    if metadata.get("name") != "G0" or metadata.get("candidate_id") != PARENT_EXPERIMENT_ID:
        raise ValueError("Registered v7 parent metadata is not completed v6 G0")
    if metadata.get("plan_id") != PARENT_PLAN_ID:
        raise ValueError("Registered v7 parent metadata has the wrong plan id")
    if metadata.get("weight_parent_experiment_id") != GRANDPARENT_EXPERIMENT_ID:
        raise ValueError("v6 G0 parent metadata does not descend from v5 F1")
    if metadata.get("weight_parent_sha256") != GRANDPARENT_SHA256:
        raise ValueError("v6 G0 parent weight-parent hash changed")
    if metadata.get("geometry_origin_sha256") != plan["geometry_origin_sha256"]:
        raise ValueError("v6 G0 parent geometry origin changed")
    if metadata.get("geometry_boxes_sha256") != plan["geometry_boxes_sha256"]:
        raise ValueError("v6 G0 parent geometry boxes changed")
    if metadata.get("original_supervision_sha256") != plan["original_supervision_sha256"]:
        raise ValueError("v6 G0 parent original supervision changed")
    if metadata.get("classification_temperature") != 1.5:
        raise ValueError("v6 G0 parent classification temperature changed")
    if metadata.get("probability_fusion_blend") != 0.5:
        raise ValueError("v6 G0 parent fusion objective changed")
    if (metadata.get("global_trusted_ce") is not False
            or metadata.get("local_trusted_ce") is not False
            or metadata.get("recovery_kd") is not False):
        raise ValueError("v6 G0 parent unexpectedly contains CE, recovery KD, or trust masks")
    if metadata.get("prior_in_inference") is not False:
        raise ValueError("v6 G0 parent contains an inference prior")
    if metadata.get("training_geometry_policy") != "frozen_V1":
        raise ValueError("v6 G0 parent geometry policy changed")
    if metadata.get("comparison_geometry_origin") != "V1":
        raise ValueError("v6 G0 parent comparison geometry origin changed")
    if metadata.get("supervision_changed") is not False:
        raise ValueError("v6 G0 parent supervision_changed flag changed")
    if metadata.get("inference_teacher_used") is not False:
        raise ValueError("v6 G0 parent inference_teacher_used flag changed")
    if metadata.get("selection") != "fixed_epoch_15_last_epoch":
        raise ValueError("v6 G0 parent checkpoint selection changed")
    if metadata.get("supervision") != "unchanged_original_w_targets":
        raise ValueError("v6 G0 parent supervision metadata changed")
    if metadata.get("training_geometry_required_for_inference") is not False:
        raise ValueError("v6 G0 parent incorrectly requires training geometry for inference")
    if checkpoint.get("epoch") != 15:
        raise ValueError("v6 G0 parent is not the fixed epoch-15 checkpoint")
    if "local_feature_adapter" not in checkpoint or "part_token_adapter" not in checkpoint:
        raise ValueError("v6 G0 parent checkpoint is missing a local adapter")
    return checkpoint


def preflight_v7(plan: dict) -> dict:
    checkpoint = _validate_parent_checkpoint(plan)
    diagnostic = json.loads(Path(plan["parent_diagnostic"]).read_text(encoding="utf-8"))
    data = supervision(plan)
    boxes, geometry = load_v1_geometry(plan, data["paths"])
    supervision_hash = original_supervision_sha256(data)
    if supervision_hash != plan["original_supervision_sha256"]:
        raise ValueError("Original v7 supervision differs from the registered tensors")
    _assert_train_only(plan, data["paths"])
    if len(data["paths"]) != REQUIRED_TRAIN_ROWS or data["targets"].shape != (REQUIRED_TRAIN_ROWS, REQUIRED_CLASSES):
        raise ValueError("Official v7 training identity changed")
    try:
        with ZipFile(plan["fallback_submission"]) as archive:
            fallback_names = archive.namelist()
            fallback_zip_csv_sha256 = hashlib.sha256(archive.read("pred_results.csv")).hexdigest()
    except Exception as exc:
        raise ValueError("v6 G0 fallback submission archive cannot be read") from exc
    if fallback_names != ["pred_results.csv"]:
        raise ValueError("v6 G0 fallback submission structure changed")
    if fallback_zip_csv_sha256 != plan["fallback_csv_sha256"]:
        raise ValueError("v6 G0 fallback ZIP CSV differs from the registered CSV asset")
    if sha256_file(plan["fallback_csv"]) != plan["fallback_csv_sha256"]:
        raise ValueError("v6 G0 fallback CSV asset changed")
    expected_steps = math.ceil(len(data["paths"]) / 32)
    if expected_steps != DEFAULT_STEPS_PER_EPOCH:
        raise ValueError("Registered v7 expected optimizer steps changed")
    return {
        "status": "passed",
        "plan_id": PLAN_ID,
        "config_sha256": plan["_config_sha256"],
        "weight_parent_experiment_id": PARENT_EXPERIMENT_ID,
        "weight_parent_sha256": plan["weight_parent_sha256"],
        "fallback_submission_sha256": plan["fallback_submission_sha256"],
        "fallback_csv_sha256": plan["fallback_csv_sha256"],
        "parent_diagnostic_sha256": sha256_file(plan["parent_diagnostic"]),
        "parent_raw_micro": float(diagnostic["metrics"]["raw_micro"]),
        "parent_clean_core_micro": float(diagnostic["metrics"]["clean_core_micro"]),
        "geometry_origin_sha256": geometry["parent_sha256"],
        "geometry_manifest_sha256": plan["geometry_manifest_sha256"],
        "geometry_boxes_sha256": plan["geometry_boxes_sha256"],
        "geometry_shape": list(boxes.shape),
        "official_train_rows": len(data["paths"]),
        "positive_weight_rows": int((data["weights"] > 0).sum()),
        "original_supervision_sha256": supervision_hash,
        "steps_per_epoch_expected": expected_steps,
        "total_updates_expected": 3 * expected_steps,
        "scheduler": copy.deepcopy(EXPECTED_SCHEDULER),
        "protocol": copy.deepcopy(EXPECTED_PROTOCOL),
        "teacher_probabilities_read": False,
        "recovery_assets_read": False,
        "trusted_ce_mask_read": False,
        "test_used_for_training": False,
        "prior_in_inference": False,
        "platform_upload_authorized": False,
        "training_geometry_policy": "frozen_V1",
        "comparison_geometry_origin": "V1",
    }


def _assert_train_only(plan: dict, paths: list[str]) -> None:
    train_root = Path(plan["train_root"]).resolve()
    test_root = Path(plan["test_root"]).resolve()
    for path in paths:
        resolved = (train_root / path).resolve()
        if resolved.is_relative_to(test_root):
            raise ValueError("Test path entered PRELIM75 v7 training")


def read_v7_assets(plan: dict):
    data = supervision(plan)
    boxes, geometry = load_v1_geometry(plan, data["paths"], verify=True)
    supervision_hash = original_supervision_sha256(data)
    if supervision_hash != plan["original_supervision_sha256"]:
        raise ValueError("v7 supervision changed")
    _assert_train_only(plan, data["paths"])
    return data, boxes, geometry


# ---------------------------------------------------------------------------
# Scheduler semantics and CPU alignment checks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScheduleSpec:
    """A fixed-floor cosine schedule over a candidate-specific horizon."""

    candidate: str
    horizon_epochs: int
    steps_per_epoch: int
    floor_ratio: float = FLOOR_RATIO
    warmup_steps: int = WARMUP_STEPS

    def __post_init__(self) -> None:
        if self.candidate not in CANDIDATES:
            raise ValueError(f"Unknown v7 schedule candidate: {self.candidate}")
        if self.horizon_epochs != CANDIDATE_HORIZONS[self.candidate]:
            raise ValueError("v7 candidate horizon differs from the registered fixed plan")
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        if not 0.0 < float(self.floor_ratio) < 1.0:
            raise ValueError("floor_ratio must be in (0,1)")
        if int(self.warmup_steps) != 0:
            raise ValueError("v7 scheduler has no warmup")

    @property
    def total_steps(self) -> int:
        """Full schedule horizon H in optimizer updates."""
        return int(self.horizon_epochs) * int(self.steps_per_epoch)

    @property
    def cosine_horizon_epochs(self) -> int:
        return int(self.horizon_epochs)

    @property
    def horizon_updates(self) -> int:
        return self.total_steps

    @property
    def training_updates(self) -> int:
        return self.training_steps

    @property
    def training_steps(self) -> int:
        """Actual v7 budget K = 3 * M updates."""
        return 3 * int(self.steps_per_epoch)

    def multiplier(self, completed_updates: int) -> float:
        step = int(completed_updates)
        if step < 0:
            raise ValueError("completed_updates must be nonnegative")
        if step > self.total_steps:
            raise ValueError(
                f"completed_updates={step} beyond schedule horizon {self.total_steps}"
            )
        # Equivalent to the repository's fixed-floor cosine implementation.
        return float(_warmup_cosine(step, self.warmup_steps, self.total_steps))

    def lrs_for_bases(self, base_lrs, completed_updates: int) -> tuple[float, ...]:
        multiplier = self.multiplier(completed_updates)
        return tuple(float(base) * multiplier for base in base_lrs)


def _optimizer_group_names(optimizer: torch.optim.Optimizer) -> list[str]:
    names = []
    for index, group in enumerate(optimizer.param_groups):
        name = group.get("name")
        if name is None:
            raise ValueError(f"v7 optimizer group {index} has no registered name")
        names.append(str(name))
    return names


def _group_initial_lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    result = []
    for group in optimizer.param_groups:
        initial = group.get("initial_lr", group["lr"])
        result.append(float(initial))
    return result


def audit_position(
    optimizer: torch.optim.Optimizer,
    spec: ScheduleSpec,
    completed_updates: int,
    *,
    atol: float = 1e-15,
    rtol: float = 1e-6,
) -> dict[str, float]:
    """Fail closed if the live optimizer LR does not match the frozen schedule."""
    base_lrs = _group_initial_lrs(optimizer)
    expected = spec.lrs_for_bases(base_lrs, completed_updates)
    if len(optimizer.param_groups) != len(expected):
        raise ValueError("Optimizer/schedule group count mismatch")
    actual = tuple(float(group["lr"]) for group in optimizer.param_groups)
    for group, value, target in zip(optimizer.param_groups, actual, expected):
        if not math.isfinite(value) or not math.isclose(value, target, rel_tol=rtol, abs_tol=atol):
            raise RuntimeError(
                f"v7 scheduler position mismatch: group={group.get('name')} "
                f"actual={value!r} expected={target!r} k={completed_updates}"
            )
    names = _optimizer_group_names(optimizer)
    return dict(zip(names, expected))


def build_fresh_scheduler(optimizer: torch.optim.Optimizer, spec: ScheduleSpec):
    """Create a LambdaLR from the candidate's registered horizon, never 18 hardcoded."""
    if not isinstance(spec, ScheduleSpec):
        raise TypeError("spec must be a ScheduleSpec")
    if not optimizer.param_groups:
        raise ValueError("optimizer has no parameter groups")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: spec.multiplier(step)
    )
    audit_position(optimizer, spec, 0)
    return scheduler


def step_and_audit(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    spec: ScheduleSpec,
    completed_updates: int,
) -> dict:
    """Run one real optimizer.scheduler step pair and audit both LR positions."""
    audit_position(optimizer, spec, completed_updates)
    used = tuple(float(group["lr"]) for group in optimizer.param_groups)
    optimizer.step()
    scheduler.step()
    next_expected = spec.multiplier(completed_updates + 1)
    audit_position(optimizer, spec, completed_updates + 1)
    return {
        "completed_updates_before": int(completed_updates),
        "used_lrs": used,
        "next_multiplier": float(next_expected),
        "next_lrs": tuple(float(group["lr"]) for group in optimizer.param_groups),
    }


def _tiny_scheduler_optimizer() -> tuple[torch.optim.Optimizer, dict[str, float]]:
    parameters = [torch.nn.Parameter(torch.zeros(())) for _ in range(3)]
    base_lrs = {"backbone": 1e-6, "head": 1.5e-7, "local_adapters": 3e-6}
    optimizer = torch.optim.AdamW([
        {"name": "backbone", "params": [parameters[0]], "lr": base_lrs["backbone"], "weight_decay": 0.0},
        {"name": "head", "params": [parameters[1]], "lr": base_lrs["head"], "weight_decay": 1e-4},
        {"name": "local_adapters", "params": [parameters[2]], "lr": base_lrs["local_adapters"], "weight_decay": 0.0},
    ])
    return optimizer, base_lrs


def scheduler_alignment_check(steps_per_epoch: int = DEFAULT_STEPS_PER_EPOCH) -> dict:
    """CPU-only exact check of the fixed scheduler semantics used by L0/L1."""
    import warnings

    results: dict[str, dict] = {}
    for candidate in CANDIDATES:
        spec = ScheduleSpec(
            candidate=candidate,
            horizon_epochs=CANDIDATE_HORIZONS[candidate],
            steps_per_epoch=int(steps_per_epoch),
        )
        optimizer, base_lrs = _tiny_scheduler_optimizer()
        scheduler = build_fresh_scheduler(optimizer, spec)
        training_steps = spec.training_steps
        boundaries = {
            "0": 0,
            "1M": spec.steps_per_epoch,
            "2M": 2 * spec.steps_per_epoch,
            "3M": training_steps,
            "horizon_end": spec.total_steps,
        }
        boundary_values = {}
        for label, step in boundaries.items():
            boundary_values[label] = {
                "step": int(step),
                "multiplier": spec.multiplier(step),
                "lrs": list(spec.lrs_for_bases(tuple(base_lrs.values()), step)),
            }
        for step in (
            0, 1, spec.steps_per_epoch, 2 * spec.steps_per_epoch,
            training_steps - 1, training_steps,
            spec.total_steps - 1, spec.total_steps,
        ):
            expected = float(_warmup_cosine(step, 0, spec.total_steps))
            if not math.isclose(spec.multiplier(step), expected, rel_tol=0.0, abs_tol=1e-15):
                raise RuntimeError("v7 ScheduleSpec disagrees with repository _warmup_cosine")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for step in range(spec.total_steps):
                audit_position(optimizer, spec, step)
                optimizer.step()
                scheduler.step()
        audit_position(optimizer, spec, spec.total_steps)
        if not math.isclose(spec.multiplier(spec.total_steps), spec.floor_ratio, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError("v7 schedule horizon did not reach the registered floor")
        values = spec.lrs_for_bases(tuple(base_lrs.values()), training_steps)
        backbone, head, adapter = values
        if not math.isclose(head / backbone, 0.15, rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError("v7 head/backbone LR ratio changed")
        if not math.isclose(adapter / backbone, 3.0, rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError("v7 adapter/backbone LR ratio changed")
        results[candidate] = {
            "horizon_epochs": spec.horizon_epochs,
            "horizon_total_steps": spec.total_steps,
            "training_steps": training_steps,
            "floor_ratio": spec.floor_ratio,
            "warmup_steps": spec.warmup_steps,
            "boundaries": boundary_values,
            "training_end_multiplier": spec.multiplier(training_steps),
            "last_used_multiplier": spec.multiplier(training_steps - 1),
            "horizon_end_multiplier": spec.multiplier(spec.total_steps),
            "base_lrs": base_lrs,
            "first_step_used_lrs": list(spec.lrs_for_bases(tuple(base_lrs.values()), 0)),
            "second_step_used_lrs": list(spec.lrs_for_bases(tuple(base_lrs.values()), 1)),
            "stage_counts": {
                "epochs": 3,
                "steps_per_epoch": int(steps_per_epoch),
                "training_updates": training_steps,
                "horizon_updates": spec.total_steps,
            },
        }
    return {
        "status": "passed",
        "steps_per_epoch": int(steps_per_epoch),
        "candidates": results,
        "reference_floor": 0.01,
        "fixed_order": "optimizer.step() then scheduler.step()",
    }


# ---------------------------------------------------------------------------
# Preparation and training
# ---------------------------------------------------------------------------

def prepare_v7(plan: dict) -> dict:
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "resolved_plan.json").exists():
        raise FileExistsError("PRELIM75 v7 output already contains a resolved plan")
    checked = preflight_v7(plan)
    schedule = scheduler_alignment_check(DEFAULT_STEPS_PER_EPOCH)
    resolved = {
        key: value for key, value in plan.items()
        if not key.startswith("_")
    }
    resolved["parent_diagnostic_sha256"] = plan["parent_diagnostic_sha256"]
    resolved["parent_metrics"] = plan["parent_metrics"]
    resolved["candidates"] = {
        "L0": {"cosine_horizon_epochs": 18},
        "L1": {"cosine_horizon_epochs": 3},
    }
    atomic_json_dump(checked, output / "preflight.json")
    atomic_json_dump(schedule, output / "scheduler_check.json")
    atomic_json_dump(resolved, output / "resolved_plan.json")
    return {"status": "ready_for_v7", "preflight": checked, "scheduler_check": schedule}


def _candidate_config(checkpoint: dict, plan: dict, name: str) -> dict:
    if name not in CANDIDATES:
        raise ValueError(f"Unknown PRELIM75 v7 candidate: {name}")
    if checkpoint.get("config", {}).get("project", {}).get("experiment_id") != PARENT_EXPERIMENT_ID:
        raise ValueError("v7 parent experiment id must come from actual v6 G0 metadata")
    config = copy.deepcopy(checkpoint["config"])
    config["project"]["experiment_id"] = f"PRELIM75_V7_{name}"
    for key in ("train_csv", "val_csv", "class_mapping", "train_root", "test_root"):
        config["data"][key] = plan[key]
    config["data"]["train_augmentation"] = "clip_center_crop"
    config["train"].update({
        "init_checkpoint": plan["weight_parent"],
        "require_lineage_for_init_checkpoint": True,
        "epochs": 3,
        "schedule_epochs": CANDIDATE_HORIZONS[name],
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


def _load_lr_trace_header(writer: csv.writer) -> None:
    writer.writerow([
        "candidate", "update_index", "sample_epoch", "completed_samples",
        "used_lr_backbone", "used_lr_head", "used_lr_local_adapters",
        "next_lr_backbone", "next_lr_head", "next_lr_local_adapters",
        "scheduler_multiplier", "next_multiplier", "gradient_norm", "clipped",
    ])


def _save_first_update_probes(path: Path, model, o3, pta) -> None:
    probes = {
        "visual_proj": model.visual.proj.detach().cpu().clone(),
        "classifier_weight": model.classifier.weight.detach().cpu().clone(),
        "o3": {name: value.detach().cpu().clone() for name, value in o3.state_dict().items()},
        "pta": {name: value.detach().cpu().clone() for name, value in pta.state_dict().items()},
    }
    _atomic_torch_save(probes, path)


def train_cooldown(plan: dict, name: str) -> Path:
    """Train one fixed v7 candidate from v6 G0 with its registered horizon."""
    if name not in CANDIDATES:
        raise ValueError("Only PRELIM75 v7 L0/L1 are allowed")
    output = Path(plan["output"]) / name
    output.mkdir(parents=True, exist_ok=False)
    horizon_epochs = CANDIDATE_HORIZONS[name]
    start = time.monotonic()
    device = gpu_setup()
    data, boxes, geometry = read_v7_assets(plan)
    smoke = json.loads((Path(plan["output"]) / "smoke.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "passed" or smoke.get("candidate_scope") != "L0_and_L1_common":
        raise ValueError("Missing or invalid isolated v7 common smoke result")
    microbatch = int(smoke["microbatch_size"])
    if microbatch not in (16, 32):
        raise ValueError("Invalid v7 common microbatch size")

    model, preprocess, checkpoint, o3, pta = load_composite(plan["weight_parent"], device)
    model.float(); o3.float(); pta.float()
    licensed_visual = {
        parameter_name for parameter_name, parameter in model.visual.named_parameters()
        if parameter.requires_grad
    }
    if "conv1.weight" in licensed_visual or "positional_embedding" in licensed_visual:
        raise ValueError("v6 G0 visual permission mask changed")
    model.classifier.requires_grad_(True)
    o3.requires_grad_(True); pta.requires_grad_(True)
    model.train(); o3.train(); pta.train()
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
    visual_parameters = [parameter for parameter in model.visual.parameters() if parameter.requires_grad]
    head_parameters = list(model.classifier.parameters())
    o3_parameters = list(o3.parameters())
    pta_parameters = list(pta.parameters())
    adapter_parameters = o3_parameters + pta_parameters
    all_parameters = visual_parameters + head_parameters + adapter_parameters
    base_lrs = {"backbone": 1e-6, "head": 1.5e-7, "local_adapters": 3e-6}
    optimizer = torch.optim.AdamW([
        {"name": "backbone", "params": visual_parameters, "lr": base_lrs["backbone"], "weight_decay": 0.0},
        {"name": "head", "params": head_parameters, "lr": base_lrs["head"], "weight_decay": 1e-4},
        {"name": "local_adapters", "params": adapter_parameters, "lr": base_lrs["local_adapters"], "weight_decay": 0.0},
    ])
    dataset = OfficialImages(data["paths"], plan["train_root"], preprocess)
    generator = torch.Generator().manual_seed(42)
    stream = loader(dataset, 32, plan, shuffle=True, generator=generator)
    steps_per_epoch = len(stream)
    expected_steps = math.ceil(len(data["paths"]) / 32)
    if steps_per_epoch != expected_steps or steps_per_epoch != DEFAULT_STEPS_PER_EPOCH:
        raise RuntimeError(
            f"v7 real DataLoader produced {steps_per_epoch} steps; expected {expected_steps}"
        )
    spec = ScheduleSpec(
        candidate=name,
        horizon_epochs=horizon_epochs,
        steps_per_epoch=steps_per_epoch,
        floor_ratio=FLOOR_RATIO,
        warmup_steps=WARMUP_STEPS,
    )
    scheduler = build_fresh_scheduler(optimizer, spec)
    recipe = {
        "candidate": name,
        "plan_id": PLAN_ID,
        "independent_parent_load": True,
        "weight_parent": plan["weight_parent"],
        "weight_parent_sha256": plan["weight_parent_sha256"],
        "weight_parent_experiment_id": PARENT_EXPERIMENT_ID,
        "training_geometry_policy": "frozen_V1",
        "comparison_geometry_origin": "V1",
        "geometry_origin_sha256": plan["geometry_origin_sha256"],
        "geometry_manifest_sha256": plan["geometry_manifest_sha256"],
        "geometry_boxes_sha256": plan["geometry_boxes_sha256"],
        "original_supervision_sha256": plan["original_supervision_sha256"],
        "supervision_changed": False,
        "inference_teacher_used": False,
        "validation_scope": "overlap_diagnostic",
        "upstream_provenance_complete": False,
        "epochs": 3,
        "sample_epochs": [16, 17, 18],
        "steps_per_epoch": steps_per_epoch,
        "total_updates": spec.training_steps,
        "schedule_horizon_updates": spec.total_steps,
        "batch_size": 32,
        "effective_batch_size": 32,
        "microbatch_size": microbatch,
        "accumulation": 32 // microbatch,
        "fresh_optimizer": True,
        "fresh_scheduler": True,
        "cosine_horizon_epochs": horizon_epochs,
        "scheduler_floor_ratio": FLOOR_RATIO,
        "scheduler_warmup_steps": WARMUP_STEPS,
        "global_weight": 0.6,
        "local_weight": 0.4,
        "anchor_weight": 2.0,
        "fusion_blend": 0.5,
        "classification_temperature": 1.5,
        "gce_q": 0.5,
        "epsilon": 1e-7,
        "local_scales": list(_LOCAL_SCALES),
        "local_scale_weights": list(_LOCAL_SCALE_WEIGHTS),
        "attention_top_k": _ATTENTION_TOP_K,
        "anchor": "frozen_official_CLIP_same_global_pixel_tensor_online",
        "selection": "fixed_epoch_18_last_epoch",
        "licensed_visual_parameters": sorted(licensed_visual),
        "scheduler_semantics": "lambda step: _warmup_cosine(step, 0, horizon_epochs*actual_steps_per_epoch)",
    }
    atomic_json_dump(recipe, output / "training_recipe.json")

    history = []
    optimizer_step = 0
    clipped_steps = 0
    first_step_audit = None
    first_step_probes = None
    first_step_global = None
    lr_trace = (output / "lr_trace.csv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(lr_trace)
    _load_lr_trace_header(writer)
    try:
        for sample_epoch in (16, 17, 18):
            epoch_start = time.monotonic()
            flips, scales = fixed_choices(data["paths"], sample_epoch)
            totals = {
                "global_gce": 0.0, "local_gce": 0.0, "separate_gce": 0.0,
                "fusion_gce": 0.0, "classification": 0.0, "anchor": 0.0, "loss": 0.0,
                "samples": 0, "positive_weight_hits": 0, "effective_weight_mass": 0.0,
                "gradient_norm_sum": 0.0, "gradient_norm_max": 0.0,
            }
            drift = _drift_accumulator()
            for step, batch in enumerate(stream):
                indices = batch["index"].numpy()
                denominator = data["weights"][indices].sum().to(device).clamp_min(1e-8)
                optimizer.zero_grad(set_to_none=True)
                parts = {key: 0.0 for key in (
                    "global_gce", "local_gce", "separate_gce", "fusion_gce",
                    "classification", "anchor", "loss",
                )}
                first_microbatch_saved = False
                for offset in range(0, len(indices), microbatch):
                    sub_idx = indices[offset:offset + microbatch]
                    sub_batch_size = len(sub_idx)
                    images = batch["images"][offset:offset + microbatch].to(device, non_blocking=True)
                    selected_flips = flips[sub_idx]
                    selected_scale_ids = scales[sub_idx]
                    if selected_flips.any():
                        flip_mask = torch.tensor(selected_flips, device=device)
                        images[flip_mask] = images[flip_mask].flip(3)
                    targets = data["targets"][sub_idx].to(device)
                    weights = data["weights"][sub_idx].to(device)
                    with torch.no_grad():
                        reference = F.normalize(anchor_teacher(images).float(), dim=1)

                    global_logits, features, attention = forward_train_with_detached_attention(model, images)
                    if not torch.isfinite(global_logits).all() or not torch.isfinite(features).all():
                        raise RuntimeError("Nonfinite v7 global outputs")
                    orientation_ids = selected_flips.astype(np.int64)
                    old_all_scales = torch.as_tensor(
                        boxes[sub_idx, orientation_ids, :, :], dtype=torch.int64, device=device
                    )
                    new_all_scales = attention_boxes_for_all_scales(
                        images, attention, local_scales=_LOCAL_SCALES, top_k=_ATTENTION_TOP_K
                    )
                    row_selector = torch.arange(sub_batch_size, device=device)
                    actual_scale_ids = torch.as_tensor(selected_scale_ids, dtype=torch.long, device=device)
                    actual_bounds = old_all_scales[row_selector, actual_scale_ids, :]
                    local_images = crop_with_bound_boxes(images, actual_bounds)
                    old_cpu = old_all_scales.detach().cpu()
                    new_cpu = new_all_scales.detach().cpu()
                    _accumulate_box_drift(drift, old_cpu, new_cpu)
                    expected_fixed = old_cpu[row_selector.cpu(), actual_scale_ids.cpu(), :]
                    if not torch.equal(expected_fixed, actual_bounds.detach().cpu()):
                        raise RuntimeError("v7 local crop does not use frozen V1 geometry")

                    local_logits = adapted_dual_local_view_logits(model, o3, pta, local_images)
                    if not torch.isfinite(local_logits).all():
                        raise RuntimeError("Nonfinite v7 local outputs")
                    terms = fusion_gce_terms(
                        global_logits, local_logits, targets,
                        temperature=float(EXPECTED_PROTOCOL["classification_temperature"]),
                        blend=float(EXPECTED_PROTOCOL["fusion_blend"]),
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
                    loss = weighted["classification"] + float(EXPECTED_PROTOCOL["anchor_weight"]) * anchor
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite v7 continuation loss")
                    if optimizer_step == 0 and not first_microbatch_saved:
                        global_term = (terms["global"] * weights).sum() / denominator
                        local_term = (terms["local"] * weights).sum() / denominator
                        first_step_probes = _first_step_probes(
                            local_term, weighted["fusion_gce"], global_term, anchor, model, o3, pta
                        )
                        first_step_global = {
                            "global_logits": global_logits.detach().cpu(),
                            "features": features.detach().cpu(),
                            "reference": reference.detach().cpu(),
                            "targets": targets.detach().cpu(),
                            "weights": weights.detach().cpu(),
                            "batch_indices": torch.as_tensor(indices[:microbatch].copy(), dtype=torch.int64),
                            "selected_flips": torch.as_tensor(selected_flips.copy(), dtype=torch.bool),
                            "selected_scales": torch.as_tensor(selected_scale_ids.copy(), dtype=torch.int64),
                        }
                        first_microbatch_saved = True
                    for key, value in (*weighted.items(), ("anchor", anchor), ("loss", loss)):
                        parts[key] += float(value.detach())
                    loss.backward()

                gradient_groups = {
                    "visual": _grad_norm(visual_parameters),
                    "head": _grad_norm(head_parameters),
                    "o3": _grad_norm(o3_parameters),
                    "pta": _grad_norm(pta_parameters),
                }
                frozen_leaks = [
                    parameter_name for parameter_name, parameter in model.named_parameters()
                    if not parameter.requires_grad and parameter.grad is not None
                ]
                batch_has_positive = bool((data["weights"][indices] > 0).any())
                if first_step_probes is None:
                    first_step_probes = {
                        "skipped_empty_positive_batch": True,
                        "reason": "first effective batch contains no positive-weight row",
                    }
                if optimizer_step == 0:
                    batch_flip_scale = torch.as_tensor(
                        np.stack((flips[indices].astype(np.int64), scales[indices])), dtype=torch.int64
                    )
                    first_step_audit = {
                        **{key + "_gradient_norm": value for key, value in gradient_groups.items()},
                        "frozen_gradient_leaks": frozen_leaks,
                        "blend": 0.5,
                        "geometry_policy": "frozen_V1",
                        "cosine_horizon_epochs": horizon_epochs,
                        "scheduler_floor_ratio": FLOOR_RATIO,
                        "first_used_lrs": audit_position(optimizer, spec, 0),
                        "first_next_multiplier": float(spec.multiplier(1)),
                        "empty_positive_effective_batch": not batch_has_positive,
                        "batch_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                        "batch_flip_scale_sha256": _tensor_sha256(batch_flip_scale),
                        "global_logits_sha256": _tensor_sha256(first_step_global["global_logits"]),
                        "global_features_sha256": _tensor_sha256(first_step_global["features"]),
                        "gradient_probes": first_step_probes,
                        **parts,
                    }
                    required_groups = ("visual",) if not batch_has_positive else ("visual", "head", "o3", "pta")
                    if frozen_leaks or any(
                        not math.isfinite(gradient_groups[name]) or gradient_groups[name] <= 0.0
                        for name in required_groups
                    ):
                        raise RuntimeError("PRELIM75 v7 first-step gradient audit failed")
                    if first_step_global is not None:
                        torch.save(first_step_global, output / "first_step_global.pt")
                    atomic_json_dump(first_step_audit, output / "first_step_gradient_audit.json")
                total_gradient_norm = float(torch.nn.utils.clip_grad_norm_(all_parameters, 1.0, error_if_nonfinite=True))
                clipped = total_gradient_norm > 1.0
                clipped_steps += int(clipped)
                before_lrs = audit_position(optimizer, spec, optimizer_step)
                optimizer.step()
                scheduler.step()
                after_expected = spec.multiplier(optimizer_step + 1)
                after_lrs = audit_position(optimizer, spec, optimizer_step + 1)
                writer.writerow([
                    name, optimizer_step, sample_epoch, totals["samples"] + len(indices),
                    *(before_lrs[key] for key in ("backbone", "head", "local_adapters")),
                    *(after_lrs[key] for key in ("backbone", "head", "local_adapters")),
                    spec.multiplier(optimizer_step), after_expected, total_gradient_norm, int(clipped),
                ])
                if optimizer_step % 100 == 0:
                    lr_trace.flush()
                if optimizer_step == 0 and first_step_global is not None:
                    _save_first_update_probes(output / "first_update_probes.pt", model, o3, pta)
                    first_step_audit["first_update_probes_sha256"] = _tensor_sha256(
                        torch.cat([
                            model.visual.proj.detach().cpu().flatten(),
                            model.classifier.weight.detach().cpu().flatten(),
                        ])
                    )
                    atomic_json_dump(first_step_audit, output / "first_step_gradient_audit.json")
                optimizer_step += 1
                for key in parts:
                    totals[key] += parts[key] * len(indices)
                totals["samples"] += len(indices)
                positive = data["weights"][indices] > 0
                totals["positive_weight_hits"] += int(positive.sum())
                totals["effective_weight_mass"] += float(data["weights"][indices].sum())
                totals["gradient_norm_sum"] += total_gradient_norm
                totals["gradient_norm_max"] = max(totals["gradient_norm_max"], total_gradient_norm)
                if step % 100 == 0:
                    atomic_json_dump({
                        "status": "training",
                        "candidate": name,
                        "cosine_horizon_epochs": horizon_epochs,
                        "epoch": sample_epoch,
                        "completed_samples": totals["samples"],
                        "optimizer_steps": optimizer_step,
                        "used_lr": audit_position(optimizer, spec, optimizer_step),
                        "elapsed_seconds": time.monotonic() - start,
                    }, output / "status.json")
            row = {
                "epoch": sample_epoch,
                **{
                    key: value / totals["samples"]
                    for key, value in totals.items()
                    if key not in (
                        "samples", "positive_weight_hits", "effective_weight_mass",
                        "gradient_norm_sum", "gradient_norm_max",
                    )
                },
                "samples": totals["samples"],
                "positive_weight_hits": totals["positive_weight_hits"],
                "effective_weight_mass": totals["effective_weight_mass"],
                "optimizer_steps": optimizer_step,
                "updates_this_epoch": steps_per_epoch,
                "gradient_norm_mean": totals["gradient_norm_sum"] / steps_per_epoch,
                "gradient_norm_max": totals["gradient_norm_max"],
                "clipped_steps_cumulative": clipped_steps,
                "cosine_horizon_epochs": horizon_epochs,
                "scheduler_position": optimizer_step,
                "scheduler_multiplier": spec.multiplier(optimizer_step),
                "geometry_drift": _drift_summary(drift),
                "epoch_elapsed_seconds": time.monotonic() - epoch_start,
                "elapsed_seconds": time.monotonic() - start,
                "peak_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
            }
            history.append(row)
            atomic_json_dump(history, output / "training_history.json")
            print(json.dumps(row), flush=True)
    finally:
        lr_trace.close()
    if optimizer_step != spec.training_steps:
        raise RuntimeError(f"v7 update count mismatch: {optimizer_step} != {spec.training_steps}")
    for parameter_name, parameter in model.named_parameters():
        if parameter_name in frozen and not torch.equal(parameter.detach().cpu(), frozen[parameter_name]):
            raise RuntimeError(f"Frozen v6 G0 tensor changed during v7 training: {parameter_name}")

    payload = saved_candidate(checkpoint, model, config, name, plan, optimizer, scheduler, 18, o3, pta)
    payload["global_step"] = int(checkpoint.get("global_step", 0)) + optimizer_step
    payload["prelim75_training"] = {
        "name": name,
        "candidate_id": f"PRELIM75_V7_{name}",
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
        "selection": "fixed_epoch_18_last_epoch",
        "supervision": "unchanged_original_w_targets",
        "classification_temperature": 1.5,
        "probability_fusion_blend": 0.5,
        "global_weight": 0.6,
        "local_weight": 0.4,
        "anchor_weight": 2.0,
        "cosine_horizon_epochs": horizon_epochs,
        "scheduler_floor_ratio": FLOOR_RATIO,
        "scheduler_warmup_steps": WARMUP_STEPS,
        "total_updates": optimizer_step,
        "global_trusted_ce": False,
        "local_trusted_ce": False,
        "recovery_kd": False,
        "prior_in_inference": False,
        "training_geometry_required_for_inference": False,
    }
    _atomic_torch_save(payload, output / "candidate.pt")
    checkpoint_hash = sha256_file(output / "candidate.pt")
    del payload, anchor_teacher
    model.cpu(); o3.cpu(); pta.cpu()
    del model, o3, pta
    torch.cuda.empty_cache()
    reloaded_model, _, reloaded, reloaded_o3, reloaded_pta = load_composite(
        output / "candidate.pt", torch.device("cpu")
    )
    metadata = reloaded.get("prelim75_training", {})
    if (metadata.get("plan_id") != PLAN_ID
            or metadata.get("weight_parent_sha256") != plan["weight_parent_sha256"]
            or metadata.get("cosine_horizon_epochs") != horizon_epochs
            or metadata.get("total_updates") != spec.training_steps):
        raise RuntimeError("Saved v7 candidate lineage changed on strict reload")
    result = {
        "status": "trained_pending_real_diagnostic",
        "candidate": name,
        "cosine_horizon_epochs": horizon_epochs,
        "scheduler_floor_ratio": FLOOR_RATIO,
        "steps_per_epoch": steps_per_epoch,
        "total_updates": spec.training_steps,
        "schedule_horizon_updates": spec.total_steps,
        "history": history,
        "first_step_gradient_audit": first_step_audit,
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
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
        "online_accuracy": None,
        "validation_scope": "overlap_diagnostic",
    }
    atomic_json_dump(result, output / "training_result.json")
    atomic_json_dump(result, output / "status.json")
    return output / "candidate.pt"


# ---------------------------------------------------------------------------
# Common smoke, diagnostic and delivery
# ---------------------------------------------------------------------------

def smoke_v7(plan: dict) -> dict:
    output = Path(plan["output"]) / "smoke.json"
    if output.exists():
        raise FileExistsError("PRELIM75 v7 smoke result already exists")
    start = time.monotonic()
    device = gpu_setup()
    data, boxes, geometry = read_v7_assets(plan)
    model, preprocess, checkpoint, o3, pta = load_composite(plan["weight_parent"], device)
    model.float(); o3.float(); pta.float()
    model.classifier.requires_grad_(True)
    o3.requires_grad_(True); pta.requires_grad_(True)
    model.train(); o3.train(); pta.train()
    anchor_teacher = official_anchor(device)
    frozen_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    dataset = OfficialImages(data["paths"], plan["train_root"], preprocess)
    generator = torch.Generator().manual_seed(42)
    stream = loader(dataset, 32, plan, shuffle=True, generator=generator)
    batch = next(iter(stream))
    indices = batch["index"].numpy()
    flips, scales = fixed_choices(data["paths"], 16)
    images = batch["images"].to(device, non_blocking=True)
    if flips[indices].any():
        flip_mask = torch.tensor(flips[indices], device=device)
        images[flip_mask] = images[flip_mask].flip(3)
    targets = data["targets"][indices].to(device)
    weights = data["weights"][indices].to(device)
    denominator = weights.sum().clamp_min(1e-8)
    visual_parameters = [parameter for parameter in model.visual.parameters() if parameter.requires_grad]
    head_parameters = list(model.classifier.parameters())
    o3_parameters = list(o3.parameters())
    pta_parameters = list(pta.parameters())
    all_parameters = visual_parameters + head_parameters + o3_parameters + pta_parameters

    hook_count_before = _module_hook_count(model)
    with torch.no_grad():
        native_logits, native_features = model(images=images, return_features=True)
        wrapper_logits, wrapper_features, wrapper_attention = forward_train_with_detached_attention(model, images)
        ref_logits, ref_features, ref_attention = forward_features_with_last_block_attention(model, images)
    alignment = {
        "native_atol": 1e-6,
        "native_rtol": 1e-5,
        "max_abs_logits_diff": float((native_logits - wrapper_logits).abs().max()),
        "max_abs_features_diff": float((native_features - wrapper_features).abs().max()),
        "argmax_equal": bool(torch.equal(native_logits.argmax(1), wrapper_logits.argmax(1))),
        "attention_atol": 1e-6,
        "attention_rtol": 1e-5,
        "max_abs_attention_diff": float((wrapper_attention - ref_attention).abs().max()),
        "inference_max_abs_logits_diff": float((ref_logits - wrapper_logits).abs().max()),
        "inference_max_abs_features_diff": float((ref_features - wrapper_features).abs().max()),
        "hook_count_before": hook_count_before,
        "hook_count_after": _module_hook_count(model),
    }
    try:
        torch.testing.assert_close(native_logits, wrapper_logits, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(native_features, wrapper_features, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(wrapper_attention, ref_attention, atol=1e-6, rtol=1e-5)
    except AssertionError as exc:
        raise RuntimeError("v7 wrapper alignment tolerance failed") from exc
    if not alignment["argmax_equal"]:
        raise RuntimeError("v7 wrapper changed native global argmax")
    box_alignment = {}
    for scale_size in _LOCAL_SCALES:
        _, boxes_reference = extract_attention_crops(
            images, ref_attention, crop_size=int(scale_size), top_k=_ATTENTION_TOP_K
        )
        boxes_wrapper = attention_crop_boxes(
            images, wrapper_attention, crop_size=int(scale_size), top_k=_ATTENTION_TOP_K
        )
        exact = [tuple(map(int, left)) for left in boxes_reference] == [
            tuple(map(int, right)) for right in boxes_wrapper
        ]
        box_alignment[str(scale_size)] = {"exact": bool(exact), "num_boxes": len(boxes_wrapper)}
        if not exact:
            raise RuntimeError(f"v7 wrapper crop boxes differ from inference formula at scale {scale_size}")

    # 32 vs 16 chunking sensitivity for the registered selected-scale boxes.
    chunking_audit = {"checked": False}
    try:
        with torch.no_grad():
            _, _, attention_full = forward_train_with_detached_attention(model, images)
            _, _, attention_low = forward_train_with_detached_attention(model, images[:16])
            _, _, attention_high = forward_train_with_detached_attention(model, images[16:])
            attention_chunked = torch.cat((attention_low, attention_high), dim=0)
            boxes_full = attention_boxes_for_all_scales(images, attention_full, top_k=_ATTENTION_TOP_K)
            boxes_chunked = attention_boxes_for_all_scales(images, attention_chunked, top_k=_ATTENTION_TOP_K)
        box_delta = (boxes_full - boxes_chunked).abs()
        chunking_audit = {
            "checked": True,
            "attention_max_abs_diff": float((attention_full - attention_chunked).abs().max()),
            "boxes_bitwise_equal": bool(torch.equal(boxes_full, boxes_chunked)),
            "boxes_exact_fraction": float((boxes_full == boxes_chunked).all(dim=-1).float().mean()),
            "max_abs_box_delta_px": float(box_delta.max()),
            "fixed_box_tolerance_px": 1.0,
            "within_fixed_tolerance": bool(float(box_delta.max()) <= 1.0),
        }
        if not math.isfinite(chunking_audit["max_abs_box_delta_px"]):
            raise RuntimeError("Nonfinite 32/16 chunking box comparison")
        if not chunking_audit["within_fixed_tolerance"]:
            raise RuntimeError("32/16 chunking changed integer boxes beyond tolerance")
    except torch.cuda.OutOfMemoryError:
        _cuda_memory_clear()
        chunking_audit = {"checked": False, "reason": "cuda_oom_during_chunk_compare"}

    with torch.no_grad():
        _, _, captured_tokens = _forward_global_with_captured_tokens(model, images[:8])
        block = model.visual.transformer.resblocks[-1]
        state_before = {name: value.detach().clone() for name, value in block.state_dict().items()}
        cpu_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(device).clone() if torch.cuda.is_available() else None
        mode_before = (model.training, o3.training, pta.training)
        requires_grad_before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        replay_last_block_attention(block, captured_tokens)
        state_after = {name: value.detach().clone() for name, value in block.state_dict().items()}
    replay_audit = {
        "rng_unchanged": bool(torch.equal(cpu_rng_before, torch.get_rng_state())),
        "model_mode_unchanged": (model.training, o3.training, pta.training) == mode_before,
        "requires_grad_unchanged": all(
            requires_grad_before[name] == parameter.requires_grad
            for name, parameter in model.named_parameters()
        ),
        "replay_state_unchanged": all(
            torch.equal(state_before[name], state_after[name]) for name in state_before
        ),
    }
    if torch.cuda.is_available():
        replay_audit["cuda_rng_unchanged"] = bool(
            torch.equal(cuda_rng_before, torch.cuda.get_rng_state(device))
        )
    if not all(replay_audit.get(key, True) for key in (
        "rng_unchanged", "model_mode_unchanged", "requires_grad_unchanged",
        "replay_state_unchanged", "cuda_rng_unchanged",
    )):
        raise RuntimeError("v7 attention replay changed RNG, mode, or module state")

    def attempt(microbatch: int) -> dict:
        model.zero_grad(set_to_none=True); o3.zero_grad(set_to_none=True); pta.zero_grad(set_to_none=True)
        total_loss = None
        first_audit = None
        empty_positive = None
        batch_has_positive = False
        for offset in range(0, len(indices), microbatch):
            sub_idx = indices[offset:offset + microbatch]
            current_images = batch["images"][offset:offset + microbatch].to(device, non_blocking=True)
            current_flips = flips[sub_idx]
            current_scale_ids = scales[sub_idx]
            if current_flips.any():
                current_flip_mask = torch.tensor(current_flips, device=device)
                current_images[current_flip_mask] = current_images[current_flip_mask].flip(3)
            current_targets = data["targets"][sub_idx].to(device)
            current_weights = data["weights"][sub_idx].to(device)
            current_denominator = current_weights.sum().clamp_min(1e-8)
            batch_has_positive = batch_has_positive or bool((current_weights > 0).any())
            with torch.no_grad():
                reference = F.normalize(anchor_teacher(current_images).float(), dim=1)
            global_logits, features, attention = forward_train_with_detached_attention(model, current_images)
            orientation_ids = current_flips.astype(np.int64)
            old_all_scales = torch.as_tensor(
                boxes[sub_idx, orientation_ids, :, :], dtype=torch.int64, device=device
            )
            row_selector = torch.arange(len(sub_idx), device=device)
            actual_scale_ids = torch.as_tensor(current_scale_ids, dtype=torch.long, device=device)
            actual_bounds = old_all_scales[row_selector, actual_scale_ids, :]
            local_images = crop_with_bound_boxes(current_images, actual_bounds)
            local_logits = adapted_dual_local_view_logits(model, o3, pta, local_images)
            terms = fusion_gce_terms(
                global_logits, local_logits, current_targets,
                temperature=float(EXPECTED_PROTOCOL["classification_temperature"]),
                blend=float(EXPECTED_PROTOCOL["fusion_blend"]),
            )
            weighted = {
                key: (terms[source] * current_weights).sum() / current_denominator
                for key, source in (
                    ("global_gce", "global"), ("local_gce", "local"),
                    ("separate_gce", "separate"), ("fusion_gce", "fusion"),
                    ("classification", "objective"),
                )
            }
            anchor = (1.0 - F.cosine_similarity(features.float(), reference, dim=1)).sum() / len(indices)
            loss = weighted["classification"] + 2.0 * anchor
            if first_audit is None:
                empty_positive = _empty_positive_safety(terms, current_weights)
                probes = (
                    _first_step_probes(
                        (terms["local"] * current_weights).sum() / current_denominator,
                        weighted["fusion_gce"],
                        (terms["global"] * current_weights).sum() / current_denominator,
                        anchor,
                        model, o3, pta,
                    )
                    if bool((current_weights > 0).any())
                    else {"skipped_empty_positive_microbatch": True}
                )
                first_audit = {
                    "gradient_probes": probes,
                    "local_only_term": float(((terms["local"] * current_weights).sum() / current_denominator).detach()),
                    "fusion_only_term": float(weighted["fusion_gce"].detach()),
                    "global_only_term": float(((terms["global"] * current_weights).sum() / current_denominator).detach()),
                    "anchor_term": float(anchor.detach()),
                }
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite v7 smoke loss")
            loss.backward()
            total_loss = loss.detach() if total_loss is None else total_loss + loss.detach()
        gradient_norms = {
            "visual": _grad_norm(visual_parameters),
            "head": _grad_norm(head_parameters),
            "o3": _grad_norm(o3_parameters),
            "pta": _grad_norm(pta_parameters),
        }
        frozen_leaks = [
            name for name, parameter in model.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        ]
        required_groups = ("visual",) if not batch_has_positive else ("visual", "head", "o3", "pta")
        if frozen_leaks or any(
            not math.isfinite(gradient_norms[name]) or gradient_norms[name] <= 0.0
            for name in required_groups
        ):
            raise RuntimeError("v7 smoke gradient coverage failed")
        return {
            "loss": float(total_loss),
            "gradient_norms": gradient_norms,
            "frozen_gradient_leaks": frozen_leaks,
            "first_audit": first_audit,
            "empty_positive": empty_positive,
            "empty_positive_effective_batch": not batch_has_positive,
        }

    try:
        attempt_result = attempt(32)
        microbatch = 32
        downgraded = False
    except torch.cuda.OutOfMemoryError:
        _cuda_memory_clear()
        model.zero_grad(set_to_none=True); o3.zero_grad(set_to_none=True); pta.zero_grad(set_to_none=True)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        attempt_result = attempt(16)
        microbatch = 16
        downgraded = True
    frozen_changed = [
        name for name, parameter in model.named_parameters()
        if name in frozen_before and not torch.equal(parameter.detach().cpu(), frozen_before[name])
    ]
    if frozen_changed:
        raise RuntimeError(f"v7 smoke changed frozen parameters: {frozen_changed}")
    torch.cuda.empty_cache()
    first_audit = attempt_result.pop("first_audit")
    empty_positive = attempt_result.pop("empty_positive")
    result = {
        "status": "passed",
        "candidate_scope": "L0_and_L1_common",
        "effective_batch_size": 32,
        "microbatch_size": microbatch,
        "accumulation": 32 // microbatch,
        "oom_downgrade": downgraded,
        "batch_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "positive_weight_rows_in_batch": int((data["weights"][indices] > 0).sum()),
        "wrapper_alignment": alignment,
        "attention_alignment": {
            "max_abs_diff": alignment["max_abs_attention_diff"],
            "atol": 1e-6,
            "rtol": 1e-5,
        },
        "box_alignment": box_alignment,
        "chunking_audit": chunking_audit,
        "replay_audit": replay_audit,
        "first_step_gradient_audit": first_audit,
        "empty_positive_safety": empty_positive,
        "frozen_parameters_unchanged": True,
        "attempt": attempt_result,
        "optimizer_created": False,
        "optimizer_step_performed": False,
        "formal_rng_state_consumed_by_replay": not replay_audit["rng_unchanged"],
        "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
        "checkpoint_sha256": plan["weight_parent_sha256"],
        "geometry_policy_tested": "frozen_V1",
        "test_used": False,
    }
    atomic_json_dump(result, output)
    del model, o3, pta, anchor_teacher
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def evaluate_v7(plan: dict, name: str) -> dict:
    if name not in CANDIDATES:
        raise ValueError("Unknown PRELIM75 v7 candidate")
    root = Path(plan["output"]) / name
    checkpoint_path = root / "candidate.pt"
    output = root / "diagnostic"
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    device = gpu_setup()
    data = supervision(plan)
    model, preprocess, checkpoint, o3, pta = load_composite(checkpoint_path, device)
    model.requires_grad_(False); o3.requires_grad_(False); pta.requires_grad_(False)
    model.eval(); o3.eval(); pta.eval()
    metadata = checkpoint.get("prelim75_training", {})
    if (metadata.get("plan_id") != PLAN_ID
            or metadata.get("weight_parent_sha256") != plan["weight_parent_sha256"]
            or metadata.get("cosine_horizon_epochs") != CANDIDATE_HORIZONS[name]
            or metadata.get("training_geometry_policy") != "frozen_V1"
            or metadata.get("supervision_changed") is not False
            or metadata.get("inference_teacher_used") is not False
            or metadata.get("prior_in_inference") is not False):
        raise ValueError("PRELIM75 v7 candidate lineage mismatch")
    frame = pd.read_csv(plan["val_csv"])
    paths = [canonical_sample_path(path) for path in frame.image_path.astype(str)]
    if len(paths) != REQUIRED_DIAGNOSTIC_ROWS or len(set(paths)) != len(paths):
        raise ValueError("Diagnostic row identity mismatch")
    training_index = {path: index for index, path in enumerate(data["paths"])}
    indices = torch.tensor([training_index[path] for path in paths], dtype=torch.long)
    labels = torch.tensor(frame.label.to_numpy(), dtype=torch.long)
    if not torch.equal(labels, data["labels"][indices]):
        raise ValueError("Diagnostic labels differ from frozen fitting list")
    dataset = OfficialImages(paths, plan["train_root"], preprocess)
    stream = loader(dataset, 64, plan)
    fused_predictions, global_predictions, local_predictions = [], [], []
    scale_weights = torch.tensor(list(_LOCAL_SCALE_WEIGHTS), device=device).view(1, 1, 4, 1)
    for step, batch in enumerate(stream):
        images = batch["images"].to(device, non_blocking=True)
        global_logits, local_logits, _ = _teacher_views(model, o3, pta, images)
        global_probability = (global_logits.float() / 1.5).softmax(-1).mean(1)
        local_probability = ((local_logits.float() / 1.5).softmax(-1) * scale_weights).sum(2).mean(1)
        fused_probability = 0.6 * global_probability + 0.4 * local_probability
        if not torch.isfinite(fused_probability).all():
            raise RuntimeError("Invalid PRELIM75 v7 diagnostic probabilities")
        global_predictions.append(global_probability.argmax(1).cpu())
        local_predictions.append(local_probability.argmax(1).cpu())
        fused_predictions.append(fused_probability.argmax(1).cpu())
        if step % 20 == 0:
            atomic_json_dump({
                "status": "running",
                "candidate": name,
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
    metrics = {
        key: prediction_metrics(value, **fields, num_classes=REQUIRED_CLASSES, clean_core_threshold=0.7)
        for key, value in predictions.items()
    }
    parent = json.loads(Path(plan["parent_diagnostic"]).read_text(encoding="utf-8"))
    deltas = {
        key: 100.0 * (metrics["fused"][key] - parent["metrics"][key])
        for key in ("raw_micro", "clean_core_micro")
    }
    engineering_stop = any(value < -2.0 for value in deltas.values())
    result = {
        "status": "complete",
        "candidate": name,
        "cosine_horizon_epochs": CANDIDATE_HORIZONS[name],
        "validation_scope": "overlap_diagnostic",
        "independent_generalization_claim": False,
        "metrics": metrics["fused"],
        "branch_metrics": metrics,
        "delta_pp_vs_G0": deltas,
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


def deliver_v7(plan: dict, name: str) -> dict:
    if name not in CANDIDATES:
        raise ValueError("Unknown PRELIM75 v7 candidate")
    root = Path(plan["output"]) / name
    diagnostic = json.loads((root / "diagnostic/result.json").read_text(encoding="utf-8"))
    if diagnostic.get("status") != "complete" or diagnostic.get("engineering_stop", True):
        raise ValueError("PRELIM75 v7 candidate stopped by the fixed engineering rule")
    checkpoint = root / "candidate.pt"
    if sha256_file(checkpoint) != diagnostic["checkpoint_sha256"]:
        raise ValueError("PRELIM75 v7 candidate changed after diagnostic")
    output = root / "submission"
    if output.exists():
        raise FileExistsError("Do not overwrite an existing PRELIM75 v7 submission")
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
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
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
            raise ValueError("PRELIM75 v7 ZIP/CSV byte identity failed")
    if len(rows) != REQUIRED_TEST_ROWS or len({row[0] for row in rows}) != REQUIRED_TEST_ROWS:
        raise ValueError("Official PRELIM75 v7 test coverage mismatch")
    report = {
        "candidate": name,
        "status": "submission_ready_pending_platform",
        "cosine_horizon_epochs": CANDIDATE_HORIZONS[name],
        "checkpoint_sha256": sha256_file(checkpoint),
        "csv_sha256": sha256_file(output / "pred_results.csv"),
        "zip_sha256": sha256_file(output / "submission.zip"),
        "inference_manifest_sha256": sha256_file(output / "manifest.json"),
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
        "teacher_checkpoint_required_for_final_inference": False,
        "prior_in_inference": False,
        "online_accuracy": None,
        "online_exact_correct": None,
        "actual_platform_upload_time": None,
    }
    atomic_json_dump(report, root / "candidate_report.json")
    atomic_json_dump(report, root / "status.json")
    print(json.dumps(report, indent=2), flush=True)
    return report
