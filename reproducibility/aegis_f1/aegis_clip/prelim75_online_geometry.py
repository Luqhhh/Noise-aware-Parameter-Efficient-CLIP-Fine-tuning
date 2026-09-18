"""PRELIM75 v6 current-student online training geometry.

The only new mechanism in this round is a training-only wrapper that keeps the
ordinary global classification graph intact while replaying the final visual
residual block's attention under ``torch.no_grad``.  The resulting CLS-to-patch
attention is used to derive local training crops for G1.  G0 keeps the frozen
V1 crops so that both candidates share the same wrapper, data order, loss,
optimizer settings and final inference protocol.

Training is intentionally limited to the currently merged eager full-finetune
model.  The replay module must have zero attention dropout and no mutable
state.  The box selection is non-differentiable; gradients of the local/fusion
loss still flow through the local classification forward.
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
from aegis_clip.prelim75_recovery import _teacher_views
from aegis_clip.prelim75_trusted_ce import load_v1_geometry, original_supervision_sha256
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.trainer import _warmup_cosine


PLAN_ID = "PRELIM75_V6_20260918"
PARENT_PLAN_ID = "PRELIM75_V5_20260917"
PARENT_EXPERIMENT_ID = "PRELIM75_V5_F1"
SUPERVISION_POLICY = "unchanged_original_w_targets"
EXPECTED_PROTOCOL = {
    "epochs": 3,
    "schedule_epochs": 18,
    "effective_batch_size": 32,
    "epoch_numbers": [13, 14, 15],
    "global_weight": 0.6,
    "local_weight": 0.4,
    "anchor_weight": 2.0,
    "fusion_blend": 0.5,
    "classification_temperature": 1.5,
    "gce_q": 0.5,
    "epsilon": 1e-7,
    "local_scales": [112, 128, 144, 160],
    "local_scale_weights": [0.2, 0.3, 0.4, 0.1],
    "attention_top_k": 5,
    "gpu_budget_seconds": 28800,
}
EXPECTED_D0 = {
    "seed": 42,
    "group_count": 2048,
    "hash_prefix": "prelim75-v6-d0/42/",
    "material_center_distance_px": 8.0,
    "material_fraction": 0.10,
    "minimum_material_groups": 205,
}
D0_MATERIAL_DISTANCE_PX = float(EXPECTED_D0["material_center_distance_px"])
D0_MATERIAL_FRACTION = float(EXPECTED_D0["material_fraction"])
D0_MINIMUM_MATERIAL_GROUPS = int(EXPECTED_D0["minimum_material_groups"])
_LOCAL_SCALES = tuple(int(value) for value in EXPECTED_PROTOCOL["local_scales"])
_LOCAL_SCALE_WEIGHTS = tuple(float(value) for value in EXPECTED_PROTOCOL["local_scale_weights"])
_ATTENTION_TOP_K = int(EXPECTED_PROTOCOL["attention_top_k"])
_REQUIRED_TEST_ROWS = 24967
_REQUIRED_TRAIN_ROWS = 103218
_REQUIRED_DIAGNOSTIC_ROWS = 10316
_REQUIRED_CLASSES = 500


def _resolve_plan_paths(plan, source: Path) -> None:
    path_keys = _PLAN_PATH_KEYS
    for key in path_keys:
        if key not in plan or not isinstance(plan[key], str) or not plan[key]:
            raise ValueError(f"Missing PRELIM75 v6 path: {key}")
        plan[key] = str((source.parent / plan[key]).resolve())


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


def _validate_v6_plan_schema(plan: dict) -> None:
    allowed = {
        "plan_id",
        "seed",
        "num_workers",
        "protocol",
        "d0",
        "geometry_origin_sha256",
        "original_supervision_sha256",
        *_PLAN_PATH_KEYS,
        *(key + "_sha256" for key in _PLAN_HASH_KEYS),
    }
    if set(plan) != allowed:
        missing = sorted(allowed - set(plan))
        extra = sorted(set(plan) - allowed)
        raise ValueError(
            f"Unknown or incomplete fixed PRELIM75 v6 plan: missing={missing}, extra={extra}"
        )
    if plan.get("plan_id") != PLAN_ID or plan.get("seed") != 42:
        raise ValueError("PRELIM75 v6 plan id or seed changed")
    if plan.get("num_workers") != 4:
        raise ValueError("PRELIM75 v6 requires exactly four data workers")
    if plan.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("PRELIM75 v6 protocol differs from the registered fixed plan")
    if plan.get("d0") != EXPECTED_D0:
        raise ValueError("PRELIM75 v6 D0 threshold differs from the registered fixed plan")
    for key in _PLAN_HASH_KEYS:
        if not isinstance(plan.get(key + "_sha256"), str) or len(plan[key + "_sha256"]) != 64:
            raise ValueError(f"PRELIM75 v6 missing sha256 for {key}")


def load_v6_plan(path: str | Path) -> dict:
    """Load exactly the registered v6 plan and bind immutable assets.

    All paths are canonicalized relative to the config.  The F1 diagnostic is
    read and semantically verified here because no pre-registered hash exists
    for it; its actual byte hash is recorded on the returned plan.
    """
    source = Path(path).resolve()
    if source.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("PRELIM75 v6 config must be YAML")
    plan = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("PRELIM75 v6 config root must be a mapping")
    _validate_v6_plan_schema(plan)
    _resolve_plan_paths(plan, source)
    for key in _PLAN_HASH_KEYS:
        if sha256_file(plan[key]) != plan[key + "_sha256"]:
            raise ValueError(f"Frozen PRELIM75 v6 asset hash mismatch: {key}")
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
        "candidate": "F1",
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
            raise ValueError(f"F1 parent diagnostic identity mismatch: {key}")
    if diagnostic.get("independent_generalization_claim") is not False:
        raise ValueError("F1 parent diagnostic must not claim independent generalization")
    metrics = diagnostic.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("samples") != _REQUIRED_DIAGNOSTIC_ROWS:
        raise ValueError("F1 parent diagnostic metrics are incomplete")
    for key in ("raw_micro", "clean_core_micro"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"F1 parent diagnostic metric is invalid: {key}")
    if "online_accuracy" not in diagnostic or diagnostic["online_accuracy"] is not None:
        raise ValueError("F1 parent diagnostic must not borrow a platform score")


def _assert_train_only(plan: dict, paths: list[str]) -> None:
    train_root = Path(plan["train_root"]).resolve()
    test_root = Path(plan["test_root"]).resolve()
    for path in paths:
        resolved = (train_root / path).resolve()
        if resolved.is_relative_to(test_root):
            raise ValueError("Test path entered PRELIM75 v6 training")


def _validate_parent_checkpoint(plan: dict):
    checkpoint = torch.load(plan["weight_parent"], map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    metadata = checkpoint.get("prelim75_training", {})
    loss = config.get("loss", {})
    experiment_id = config.get("project", {}).get("experiment_id")
    if experiment_id != PARENT_EXPERIMENT_ID:
        raise ValueError("Registered v6 weight parent is not the actual v5 F1")
    if metadata.get("name") != "F1" or metadata.get("candidate_id") != PARENT_EXPERIMENT_ID:
        raise ValueError("Registered v6 parent metadata is not completed v5 F1")
    if metadata.get("plan_id") != PARENT_PLAN_ID:
        raise ValueError("Registered v6 parent metadata has the wrong plan id")
    parent_diagnostic = json.loads(Path(plan["parent_diagnostic"]).read_text())
    grandparent_sha = parent_diagnostic.get("parent_checkpoint_sha256")
    if not grandparent_sha or metadata.get("weight_parent_sha256") != grandparent_sha:
        raise ValueError("F1 parent does not descend from the recorded C0 checkpoint")
    if metadata.get("geometry_origin_sha256") != plan["geometry_origin_sha256"]:
        raise ValueError("F1 parent geometry origin changed")
    if metadata.get("geometry_boxes_sha256") != plan["geometry_boxes_sha256"]:
        raise ValueError("F1 parent geometry boxes changed")
    if metadata.get("original_supervision_sha256") != plan["original_supervision_sha256"]:
        raise ValueError("F1 parent original supervision changed")
    if metadata.get("classification_temperature") != 1.5:
        raise ValueError("F1 parent classification temperature changed")
    if metadata.get("probability_fusion_blend") != 0.5 or metadata.get("sampled_probability_fusion") is not True:
        raise ValueError("F1 parent fusion objective changed")
    if (metadata.get("global_trusted_ce") is not False
            or metadata.get("local_trusted_ce") is not False
            or metadata.get("recovery_kd") is not False):
        raise ValueError("F1 parent unexpectedly contains CE, recovery KD, or trust masks")
    if metadata.get("prior_in_inference") is not False:
        raise ValueError("F1 parent contains an inference prior")
    if metadata.get("training_geometry_required_for_inference") is not False:
        raise ValueError("F1 parent incorrectly requires training geometry for inference")
    if metadata.get("selection") != "fixed_epoch_12_last_epoch":
        raise ValueError("F1 parent checkpoint selection changed")
    if metadata.get("supervision") != "unchanged_original_w_targets":
        raise ValueError("F1 parent supervision metadata changed")
    if loss.get("name") != "gce" or float(loss.get("gce_q", -1.0)) != 0.5:
        raise ValueError("F1 parent GCE definition changed")
    if "local_feature_adapter" not in checkpoint or "part_token_adapter" not in checkpoint:
        raise ValueError("F1 parent checkpoint is missing a local adapter")
    return checkpoint


def preflight_v6(plan: dict) -> dict:
    """Read-only fail-closed validation of every asset used by v6."""
    checkpoint = _validate_parent_checkpoint(plan)
    diagnostic = json.loads(Path(plan["parent_diagnostic"]).read_text())
    data = supervision(plan)
    boxes, geometry = load_v1_geometry(plan, data["paths"])
    supervision_hash = original_supervision_sha256(data)
    if supervision_hash != plan["original_supervision_sha256"]:
        raise ValueError("Original v6 supervision differs from the registered tensors")
    _assert_train_only(plan, data["paths"])
    if len(data["paths"]) != _REQUIRED_TRAIN_ROWS or data["targets"].shape != (_REQUIRED_TRAIN_ROWS, _REQUIRED_CLASSES):
        raise ValueError("Official v6 training identity changed")
    fallback_names = None
    try:
        with ZipFile(plan["fallback_submission"]) as archive:
            fallback_names = archive.namelist()
            fallback_zip_csv_sha256 = hashlib.sha256(archive.read("pred_results.csv")).hexdigest()
    except Exception as exc:
        raise ValueError("F1 fallback submission archive cannot be read") from exc
    if fallback_names != ["pred_results.csv"]:
        raise ValueError("F1 fallback submission structure changed")
    if fallback_zip_csv_sha256 != plan["fallback_csv_sha256"]:
        raise ValueError("F1 fallback ZIP CSV differs from the registered CSV asset")
    if sha256_file(plan["fallback_csv"]) != plan["fallback_csv_sha256"]:
        raise ValueError("F1 fallback CSV asset changed")
    checkpoint_epoch = checkpoint.get("epoch")
    if checkpoint_epoch != 12:
        raise ValueError("F1 parent is not the registered fixed epoch-12 checkpoint")
    return {
        "status": "passed",
        "plan_id": PLAN_ID,
        "config_sha256": plan["_config_sha256"],
        "weight_parent_experiment_id": PARENT_EXPERIMENT_ID,
        "weight_parent_sha256": plan["weight_parent_sha256"],
        "parent_diagnostic_sha256": sha256_file(plan["parent_diagnostic"]),
        "parent_raw_micro": float(diagnostic["metrics"]["raw_micro"]),
        "parent_clean_core_micro": float(diagnostic["metrics"]["clean_core_micro"]),
        "fallback_submission_sha256": plan["fallback_submission_sha256"],
        "fallback_csv_sha256": plan["fallback_csv_sha256"],
        "geometry_origin_sha256": geometry["parent_sha256"],
        "geometry_manifest_sha256": plan["geometry_manifest_sha256"],
        "geometry_boxes_sha256": plan["geometry_boxes_sha256"],
        "geometry_shape": list(boxes.shape),
        "official_train_rows": len(data["paths"]),
        "positive_weight_rows": int((data["weights"] > 0).sum()),
        "original_supervision_sha256": supervision_hash,
        "d0": copy.deepcopy(EXPECTED_D0),
        "protocol": copy.deepcopy(EXPECTED_PROTOCOL),
        "teacher_probabilities_read": False,
        "recovery_assets_read": False,
        "trusted_ce_mask_read": False,
        "test_used_for_training": False,
        "prior_in_inference": False,
        "platform_upload_authorized": False,
        "training_geometry_policy_g0": "frozen_V1",
        "training_geometry_policy_g1": "current_student_step",
        "comparison_geometry_origin": "V1",
    }


def prepare_online_geometry(plan: dict) -> dict:
    """Create only the identity preparation record; no labels or caches."""
    output = Path(plan["output"]) / "preparation.json"
    if output.exists():
        raise FileExistsError("PRELIM75 v6 preparation already exists")
    checked = preflight_v6(plan)
    checked.update({
        "status": "ready_for_d0_gate",
        "new_teacher_cache_created": False,
        "new_supervision_created": False,
        "new_geometry_cache_created": False,
    })
    atomic_json_dump(checked, output)
    return checked


def read_geometry_assets(plan: dict, *, verify: bool = True):
    preparation = json.loads((Path(plan["output"]) / "preparation.json").read_text(encoding="utf-8"))
    if preparation.get("status") != "ready_for_d0_gate":
        raise ValueError("Missing valid PRELIM75 v6 preparation")
    data = supervision(plan)
    boxes, geometry = load_v1_geometry(plan, data["paths"], verify=verify)
    supervision_hash = original_supervision_sha256(data)
    if (supervision_hash != plan["original_supervision_sha256"]
            or preparation.get("original_supervision_sha256") != supervision_hash):
        raise ValueError("Prepared PRELIM75 v6 supervision changed")
    _assert_train_only(plan, data["paths"])
    return data, boxes, geometry, preparation


# ---------------------------------------------------------------------------
# D0 fixed geometry-only investment gate
# ---------------------------------------------------------------------------

def rank_d0_groups(data: dict, *, group_count: int = 2048, prefix: str = EXPECTED_D0["hash_prefix"]):
    """Return the fixed group representatives and their deterministic ranks.

    Positive classification weight is used only to avoid samples that receive
    zero classification gradient.  Within each content group the canonical
    lexicographically smallest path represents the group.  Ranking is the
    registered SHA256 hash of ``prefix + canonical_path``; ties fall back to
    the path itself.  Nothing in this function sees model predictions, class
    difficulty, labels outside supervision, test images, or old/new box values.
    """
    paths = list(data["paths"])
    groups = list(data["group_ids"])
    weights = data["weights"]
    if len(paths) != len(groups) or len(paths) != int(weights.numel()):
        raise ValueError("D0 supervision identities have different lengths")
    positive = weights.detach().cpu() > 0
    if positive.ndim != 1:
        raise ValueError("D0 weights must be one-dimensional")
    representative: dict[str, tuple[str, int]] = {}
    for index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
        path = canonical_sample_path(paths[index])
        group = str(groups[index])
        previous = representative.get(group)
        if previous is None or path < previous[0]:
            representative[group] = (path, index)
    ranked = []
    for group, (path, index) in representative.items():
        digest = hashlib.sha256((prefix + path).encode("utf-8")).hexdigest()
        ranked.append((digest, path, group, index))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    selected = ranked[: int(group_count)]
    payload = "".join(
        f"{rank}\t{path}\t{group}\n"
        for rank, (_, path, group, _) in enumerate(selected)
    )
    return {
        "status": "selected" if len(selected) == int(group_count) else "insufficient",
        "available_groups": len(representative),
        "available_positive_rows": int(positive.sum()),
        "group_count_requested": int(group_count),
        "selected_count": len(selected),
        "selected": [
            {"rank": rank, "rank_sha256": digest, "path": path, "group": group, "train_index": index}
            for rank, (digest, path, group, index) in enumerate(selected)
        ],
        "selection_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def d0_selection_for_plan(data: dict, plan: dict) -> dict:
    del plan
    selection = rank_d0_groups(
        data,
        group_count=int(EXPECTED_D0["group_count"]),
        prefix=EXPECTED_D0["hash_prefix"],
    )
    selection["seed"] = int(EXPECTED_D0["seed"])
    selection["minimum_material_groups"] = D0_MINIMUM_MATERIAL_GROUPS
    selection["material_center_distance_px"] = D0_MATERIAL_DISTANCE_PX
    selection["material_fraction"] = D0_MATERIAL_FRACTION
    selection["uses_model_predictions"] = False
    selection["uses_accuracy_or_difficulty"] = False
    selection["uses_test_or_teacher_probabilities"] = False
    return selection


def attention_crop_boxes(
    images: torch.Tensor,
    cls_patch_attention: torch.Tensor,
    *,
    crop_size: int,
    top_k: int = _ATTENTION_TOP_K,
) -> list[tuple[int, int, int, int]]:
    """Derive native-224 integer crop boxes from CLS-patch attention.

    The rounding and clamping intentionally mirror the registered inference
    localization helper ``extract_attention_crops``; actual G1 crops are still
    produced by that existing helper, never by a test replacement.
    """
    if images.ndim != 4:
        raise ValueError("Images must have shape [batch, channels, height, width]")
    batch_size, _, height, width = images.shape
    if cls_patch_attention.shape[0] != batch_size:
        raise ValueError("Image and attention batch sizes do not match")
    if not 1 <= int(crop_size) <= min(height, width):
        raise ValueError("crop_size outside image bounds")
    if batch_size == 0:
        return []
    centers = attention_weighted_centers(
        cls_patch_attention,
        image_height=int(height),
        image_width=int(width),
        top_k=int(top_k),
    )
    max_x = int(width) - int(crop_size)
    max_y = int(height) - int(crop_size)
    boxes: list[tuple[int, int, int, int]] = []
    for center in centers:
        x0 = int(round(float(center[0]) - int(crop_size) / 2.0))
        y0 = int(round(float(center[1]) - int(crop_size) / 2.0))
        x0 = min(max(x0, 0), max_x)
        y0 = min(max(y0, 0), max_y)
        boxes.append((x0, y0, x0 + int(crop_size), y0 + int(crop_size)))
    return boxes


def attention_boxes_by_scale(
    images: torch.Tensor,
    cls_patch_attention: torch.Tensor,
    scale_ids,
    *,
    local_scales: tuple[int, ...] = _LOCAL_SCALES,
    top_k: int = _ATTENTION_TOP_K,
) -> list[tuple[int, int, int, int]]:
    """Return one crop box per row, grouped by scale to preserve row order."""
    if images.ndim != 4 or cls_patch_attention.shape[0] != images.shape[0]:
        raise ValueError("Images and attention must be matching [B,...] tensors")
    if len(local_scales) != 4:
        raise ValueError("Exactly four registered local scales are allowed")
    scale_tensor = torch.as_tensor(scale_ids, dtype=torch.long, device=images.device).flatten()
    if scale_tensor.numel() != images.shape[0]:
        raise ValueError("Scale ids must provide one entry per image row")
    if scale_tensor.numel() and ((scale_tensor < 0).any() or (scale_tensor >= len(local_scales)).any()):
        raise ValueError("Scale id outside registered range")
    boxes: list[tuple[int, int, int, int] | None] = [None] * images.shape[0]
    for scale_id in torch.unique(scale_tensor, sorted=True).tolist():
        rows = torch.nonzero(scale_tensor == int(scale_id), as_tuple=False).flatten()
        group_boxes = attention_crop_boxes(
            images.index_select(0, rows),
            cls_patch_attention.index_select(0, rows),
            crop_size=int(local_scales[int(scale_id)]),
            top_k=top_k,
        )
        for row, box in zip(rows.tolist(), group_boxes):
            boxes[row] = box
    if any(box is None for box in boxes):
        raise RuntimeError("Internal error: unassigned online crop box")
    return [box for box in boxes if box is not None]


def extract_online_local_views(
    images: torch.Tensor,
    cls_patch_attention: torch.Tensor,
    scale_ids,
    *,
    local_scales: tuple[int, ...] = _LOCAL_SCALES,
    top_k: int = _ATTENTION_TOP_K,
) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
    """Crop and resize one local view per row using the existing extractor.

    Rows with mixed scale ids are grouped by scale before calling
    ``extract_attention_crops`` and then restored to the original row order.
    The returned local tensor is row-aligned with the input batch.
    """
    if images.ndim != 4 or cls_patch_attention.shape[0] != images.shape[0]:
        raise ValueError("Images and attention must be matching [B,...] tensors")
    if len(local_scales) != 4:
        raise ValueError("Exactly four registered local scales are allowed")
    scale_tensor = torch.as_tensor(scale_ids, dtype=torch.long, device=images.device).flatten()
    if scale_tensor.numel() != images.shape[0]:
        raise ValueError("Scale ids must provide one entry per image row")
    if scale_tensor.numel() and ((scale_tensor < 0).any() or (scale_tensor >= len(local_scales)).any()):
        raise ValueError("Scale id outside registered range")
    local = torch.empty_like(images)
    boxes: list[tuple[int, int, int, int] | None] = [None] * images.shape[0]
    for scale_id in torch.unique(scale_tensor, sorted=True).tolist():
        rows = torch.nonzero(scale_tensor == int(scale_id), as_tuple=False).flatten()
        crops, group_boxes = extract_attention_crops(
            images.index_select(0, rows),
            cls_patch_attention.index_select(0, rows),
            crop_size=int(local_scales[int(scale_id)]),
            top_k=top_k,
        )
        local.index_copy_(0, rows, crops)
        for row, box in zip(rows.tolist(), group_boxes):
            boxes[row] = box
    if any(box is None for box in boxes):
        raise RuntimeError("Internal error: unassigned online local view")
    return local, [box for box in boxes if box is not None]


def _boxes_to_tensor(boxes) -> torch.Tensor:
    return torch.as_tensor(np.asarray(boxes, dtype=np.float32), dtype=torch.float32)


def geometry_comparison(old_boxes, new_boxes, *, material_distance_px: float = D0_MATERIAL_DISTANCE_PX):
    """Compare two aligned ``[N, 2, 4, 4]`` integer box tensors.

    Reports only geometry: coordinate-change ratios, center-distance
    quantiles, IoU and the fixed material-change predicate.  No model output
    or accuracy signal is consulted.
    """
    old = _boxes_to_tensor(old_boxes)
    new = _boxes_to_tensor(new_boxes)
    if old.shape != new.shape or old.ndim != 4 or old.shape[1:] != (2, 4, 4):
        raise ValueError("Alignment requires two [N,2,4,4] box tensors")
    if old.numel() == 0:
        raise ValueError("Geometry comparison requires at least one group")
    old_centers = (old[..., :2] + old[..., 2:]) / 2.0
    new_centers = (new[..., :2] + new[..., 2:]) / 2.0
    distances = torch.linalg.vector_norm(new_centers - old_centers, dim=-1)  # [N,2,4]
    coordinate_changed = (old != new)
    coordinate_abs_delta = (old - new).abs()

    x0 = torch.maximum(old[..., 0], new[..., 0])
    y0 = torch.maximum(old[..., 1], new[..., 1])
    x1 = torch.minimum(old[..., 2], new[..., 2])
    y1 = torch.minimum(old[..., 3], new[..., 3])
    intersection = (x1 - x0).clamp_min(0.0) * (y1 - y0).clamp_min(0.0)
    old_area = (old[..., 2] - old[..., 0]).clamp_min(0.0) * (old[..., 3] - old[..., 1]).clamp_min(0.0)
    new_area = (new[..., 2] - new[..., 0]).clamp_min(0.0) * (new[..., 3] - new[..., 1]).clamp_min(0.0)
    union = old_area + new_area - intersection
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))

    group_max_distance = distances.amax(dim=(1, 2))
    material = group_max_distance >= float(material_distance_px)
    per_scale: dict[str, dict] = {}
    distance_np = distances.detach().cpu().numpy()
    iou_np = iou.detach().cpu().numpy()
    changed_np = coordinate_changed.detach().cpu().numpy()
    delta_np = coordinate_abs_delta.detach().cpu().numpy()
    for scale_id, scale_size in enumerate(_LOCAL_SCALES):
        scale_distances = distance_np[:, :, scale_id].reshape(-1)
        scale_iou = iou_np[:, :, scale_id].reshape(-1)
        scale_changed = changed_np[:, :, scale_id]
        scale_delta = delta_np[:, :, scale_id]
        per_scale[str(scale_size)] = {
            "center_distance_px": {
                "p50": float(np.quantile(scale_distances, 0.50)),
                "p90": float(np.quantile(scale_distances, 0.90)),
                "p95": float(np.quantile(scale_distances, 0.95)),
                "p99": float(np.quantile(scale_distances, 0.99)),
                "max": float(scale_distances.max()),
                "mean": float(scale_distances.mean()),
                "fraction_ge_8": float((scale_distances >= material_distance_px).mean()),
            },
            "iou": {
                "mean": float(scale_iou.mean()),
                "min": float(scale_iou.min()),
                "p05": float(np.quantile(scale_iou, 0.05)),
            },
            "coordinate_change_ratio": float(scale_changed.mean()),
            "box_change_ratio": float(scale_changed.any(axis=-1).mean()),
            "coordinate_abs_mean_px": float(scale_delta.mean()),
        }
    return {
        "n_groups": int(old.shape[0]),
        "material_center_distance_px": float(material_distance_px),
        "material_change_groups": int(material.sum()),
        "material_change_ratio": float(material.float().mean()),
        "all_boxes_identical": bool(torch.equal(old, new)),
        "group_max_center_distance_px": group_max_distance.detach().cpu(),
        "material_change": material.detach().cpu(),
        "per_scale": per_scale,
        "overall_center_distance_px": {
            "p50": float(np.quantile(distance_np.reshape(-1), 0.50)),
            "p90": float(np.quantile(distance_np.reshape(-1), 0.90)),
            "p95": float(np.quantile(distance_np.reshape(-1), 0.95)),
            "p99": float(np.quantile(distance_np.reshape(-1), 0.99)),
            "max": float(distance_np.max()),
        },
        "overall_iou": {
            "mean": float(iou_np.mean()),
            "min": float(iou_np.min()),
            "p05": float(np.quantile(iou_np, 0.05)),
        },
    }


# ---------------------------------------------------------------------------
# Training-only global wrapper with detached final-block attention
# ---------------------------------------------------------------------------

_ACTIVE_CAPTURES: set[int] = set()


def _final_visual_block(model: torch.nn.Module) -> torch.nn.Module:
    try:
        block = model.visual.transformer.resblocks[-1]
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("Online geometry requires OpenAI CLIP ViT residual blocks") from exc
    if not hasattr(block, "ln_1") or not hasattr(block, "attn"):
        raise ValueError("Final visual block is missing ln_1/attn attention structure")
    return block


def _assert_replay_supported(block: torch.nn.Module) -> None:
    attn = block.attn
    dropout = getattr(attn, "dropout", None)
    if dropout is None:
        raise RuntimeError("Final-block attention does not expose a dropout setting")
    if float(dropout) != 0.0:
        raise RuntimeError("Stochastic final-block attention replay is forbidden")
    if getattr(block, "training", False) and getattr(attn, "training", False) and float(dropout) != 0.0:
        raise RuntimeError("Train-mode attention replay must not consume random state")
    batchnorm_like = [
        module for module in block.modules()
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
    ]
    if batchnorm_like:
        raise RuntimeError("Final-block replay could update BatchNorm running statistics")


def _last_block_attn_mask(block: torch.nn.Module, tokens: torch.Tensor):
    attention_mask = getattr(block, "attn_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=tokens.device, dtype=tokens.dtype)
    return attention_mask


def replay_last_block_attention(block: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Replay only ``ln_1`` and attention for CLS-to-patch weights.

    Must be called under ``torch.no_grad`` by the caller; this function is
    deliberately small because it is shared by the training wrapper and the
    replay-state smoke audit.
    """
    _assert_replay_supported(block)
    hidden = block.ln_1(tokens)
    _, weights = block.attn(
        hidden,
        hidden,
        hidden,
        need_weights=True,
        average_attn_weights=False,
        attn_mask=_last_block_attn_mask(block, hidden),
    )
    if weights.ndim != 4 or weights.shape[-1] < 2:
        raise RuntimeError(f"Unexpected final-block attention shape: {tuple(weights.shape)}")
    # MultiheadAttention returns [batch, heads, target, source].
    return weights[:, :, 0, 1:].float().detach()


def _forward_global_with_captured_tokens(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the ordinary trainable global forward and capture final-block input.

    The returned global logits/features keep their original autograd graph.
    The captured final-block input is ``detach().clone()`` so replaying
    attention cannot backpropagate into it.
    """
    if getattr(model, "_orig_mod", None) is not None:
        raise RuntimeError("Compiled/optimized models are not supported by v6 online geometry")
    block = _final_visual_block(model)
    _assert_replay_supported(block)
    key = id(model)
    if key in _ACTIVE_CAPTURES:
        raise RuntimeError("Concurrent forward on the same model is not supported")
    _ACTIVE_CAPTURES.add(key)
    captured: dict[str, torch.Tensor] = {}

    def capture_input(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        del module
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("Final visual block received no token input")
        tokens = inputs[0]
        if tokens.ndim != 3:
            raise RuntimeError("Final visual block input must be [sequence, batch, width]")
        captured["tokens"] = tokens.detach().clone()

    handle = block.register_forward_pre_hook(capture_input)
    try:
        output = model(images=images, return_features=True)
    finally:
        handle.remove()
        _ACTIVE_CAPTURES.discard(key)
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError("Model did not return (logits, normalized features)")
    logits, features = output
    if "tokens" not in captured:
        raise RuntimeError("Failed to capture final visual-block tokens")
    return logits, features, captured["tokens"]


def forward_train_with_detached_attention(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return trainable global outputs and detached CLS-to-patch attention."""
    block = _final_visual_block(model)
    logits, features, tokens = _forward_global_with_captured_tokens(model, images)
    with torch.no_grad():
        attention = replay_last_block_attention(block, tokens)
    if not torch.isfinite(attention).all():
        raise RuntimeError("Online geometry attention contains non-finite values")
    return logits, features, attention


def run_d0(plan: dict) -> dict:
    """Run the fixed 2,048-group geometry-only gate.

    No local classification, category probability cache, teacher forward, or
    test image is used.  The gate simply measures whether the current student
    attention changes at least 10% of groups by at least 8 native pixels in
    any of the eight old/new boxes.
    """
    root = Path(plan["output"])
    output = root / "D0"
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    device = gpu_setup()
    data, boxes, geometry, _ = read_geometry_assets(plan)
    selection = d0_selection_for_plan(data, plan)
    atomic_json_dump(selection, output / "selection.json")
    if selection["status"] != "selected":
        result = {
            "status": "aborted_insufficient_groups",
            "gate_passed": False,
            "reason": "fewer than 2048 eligible content groups with positive weight",
            "available_groups": selection["available_groups"],
            "selected_count": selection["selected_count"],
            "selection_sha256": selection["selection_sha256"],
            "requested_group_count": int(EXPECTED_D0["group_count"]),
            "test_used": False,
            "training_geometry_only": True,
            "elapsed_seconds": time.monotonic() - start,
        }
        atomic_json_dump(result, output / "result.json")
        atomic_json_dump(result, output / "status.json")
        return result

    selected = selection["selected"]
    global_indices = np.asarray([row["train_index"] for row in selected], dtype=np.int64)
    selected_paths = [row["path"] for row in selected]
    selected_groups = [row["group"] for row in selected]
    if len(set(selected_paths)) != len(selected_paths) or len(set(selected_groups)) != len(selected_groups):
        raise RuntimeError("D0 selection must contain unique groups and representatives")
    old_boxes = np.asarray(boxes)[global_indices]
    new_boxes = np.empty_like(old_boxes, dtype=np.int16)

    model, preprocess, checkpoint, o3, pta = load_composite(plan["weight_parent"], device)
    del o3, pta
    model.float().eval().requires_grad_(False)
    dataset = OfficialImages(selected_paths, plan["train_root"], preprocess)
    stream = loader(dataset, 64, plan)
    positions = 0
    for step, batch in enumerate(stream):
        images = batch["images"].to(device, non_blocking=True)
        expected_start = positions
        expected_end = expected_start + len(images)
        if not torch.equal(batch["index"].cpu(), torch.arange(expected_start, expected_end)):
            raise RuntimeError("D0 dataset order changed")
        with torch.no_grad():
            _, _, attention_original = forward_train_with_detached_attention(model, images)
            _, _, attention_flipped = forward_train_with_detached_attention(model, images.flip(3))
        if not torch.isfinite(attention_original).all() or not torch.isfinite(attention_flipped).all():
            raise RuntimeError("D0 attention contains non-finite values")
        for scale_index, scale_size in enumerate(_LOCAL_SCALES):
            boxes_original = attention_crop_boxes(images, attention_original, crop_size=scale_size)
            boxes_flipped = attention_crop_boxes(images.flip(3), attention_flipped, crop_size=scale_size)
            for row, (box_original, box_flipped) in enumerate(zip(boxes_original, boxes_flipped)):
                new_boxes[expected_start + row, 0, scale_index, :] = box_original
                new_boxes[expected_start + row, 1, scale_index, :] = box_flipped
        positions += len(images)
        if step % 5 == 0:
            atomic_json_dump({
                "status": "running",
                "completed_groups": positions,
                "total_groups": len(selected),
                "elapsed_seconds": time.monotonic() - start,
            }, output / "status.json")
    if positions != len(selected):
        raise RuntimeError("D0 did not process every selected group")

    comparison = geometry_comparison(old_boxes, new_boxes)
    material_groups = int(comparison["material_change_groups"])
    gate_passed = material_groups >= D0_MINIMUM_MATERIAL_GROUPS
    np.save(output / "old_boxes.npy", old_boxes.astype(np.int16, copy=False))
    np.save(output / "new_boxes.npy", new_boxes)
    np.save(output / "group_max_center_distance.npy", comparison["group_max_center_distance_px"].numpy())
    result = {
        "status": "complete_gate_passed" if gate_passed else "complete_gate_not_met",
        "gate_passed": bool(gate_passed),
        "plan_id": PLAN_ID,
        "selection_sha256": selection["selection_sha256"],
        "selected_count": len(selected),
        "available_groups": selection["available_groups"],
        "available_positive_rows": selection["available_positive_rows"],
        "material_center_distance_px": D0_MATERIAL_DISTANCE_PX,
        "material_fraction": D0_MATERIAL_FRACTION,
        "required_material_groups": D0_MINIMUM_MATERIAL_GROUPS,
        "material_change_groups": material_groups,
        "material_change_ratio_vs_n": material_groups / len(selected),
        "gate_definition": "changed if max center distance over 8 boxes >= 8 native px",
        "per_scale": comparison["per_scale"],
        "overall_center_distance_px": comparison["overall_center_distance_px"],
        "overall_iou": comparison["overall_iou"],
        "all_boxes_identical": comparison["all_boxes_identical"],
        "old_boxes_sha256": sha256_file(output / "old_boxes.npy"),
        "new_boxes_sha256": sha256_file(output / "new_boxes.npy"),
        "group_max_center_distance_sha256": sha256_file(output / "group_max_center_distance.npy"),
        "model_checkpoint_sha256": plan["weight_parent_sha256"],
        "geometry_origin_sha256": plan["geometry_origin_sha256"],
        "geometry_boxes_sha256": plan["geometry_boxes_sha256"],
        "classification_forward_run": False,
        "local_classification_run": False,
        "teacher_probabilities_read": False,
        "test_used": False,
        "parameter_update_performed": False,
        "optimizer_created": False,
        "elapsed_seconds": time.monotonic() - start,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
    }
    atomic_json_dump(result, output / "result.json")
    atomic_json_dump(result, output / "status.json")
    del model
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# G0 / G1 training
# ---------------------------------------------------------------------------

def _candidate_config(checkpoint: dict, plan: dict, name: str) -> dict:
    expected_name = "G0" if name == "G0" else "G1"
    if name != expected_name:
        raise ValueError(f"Unknown PRELIM75 v6 candidate: {name}")
    if checkpoint.get("config", {}).get("project", {}).get("experiment_id") != PARENT_EXPERIMENT_ID:
        raise ValueError("v6 parent experiment id must come from actual v5 F1 metadata")
    config = copy.deepcopy(checkpoint["config"])
    config["project"]["experiment_id"] = f"PRELIM75_V6_{name}"
    for key in ("train_csv", "val_csv", "class_mapping", "train_root", "test_root"):
        config["data"][key] = plan[key]
    config["data"]["train_augmentation"] = "clip_center_crop"
    config["train"].update({
        "init_checkpoint": plan["weight_parent"],
        "require_lineage_for_init_checkpoint": True,
        "epochs": 3,
        "schedule_epochs": 18,
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


def _grad_norm(parameters) -> float:
    values = [
        float(parameter.grad.detach().float().pow(2).sum())
        for parameter in parameters
        if parameter.grad is not None
    ]
    return math.sqrt(sum(values)) if values else 0.0


def _probe_norm(gradients) -> float:
    return math.sqrt(sum(float(g.detach().float().pow(2).sum()) for g in gradients if g is not None))


def _gradient_probe(term, probes) -> dict[str, float]:
    gradients = torch.autograd.grad(term, probes, retain_graph=True, allow_unused=True)
    norms = [_probe_norm([gradient]) for gradient in gradients]
    return dict(zip([_probe_name(index) for index in range(len(probes))], norms))


def _probe_name(index: int) -> str:
    # Names are assigned by callers; this helper exists only for stable
    # dictionary ordering in tests.
    return f"probe_{index}"


def _first_step_probes(
    local_term: torch.Tensor,
    fusion_term: torch.Tensor,
    global_term: torch.Tensor,
    anchor_term: torch.Tensor,
    model: torch.nn.Module,
    o3: torch.nn.Module,
    pta: torch.nn.Module,
) -> dict:
    """Independent first-step gradient audits for the four loss paths.

    ``global_only``/``anchor_only`` are the two frozen-branch checks.  The
    local and fusion terms must reach visual.proj, the shared head, and the
    O3/PTA adapter parameter groups in aggregate (matching the registered v5
    gradient-probe semantics).
    """
    probes = [
        model.visual.proj,
        model.classifier.weight,
        *list(o3.parameters()),
        *list(pta.parameters()),
    ]
    names = ["visual_proj", "shared_head"] + [f"o3_{i}" for i in range(len(list(o3.parameters())))] + [
        f"pta_{i}" for i in range(len(list(pta.parameters())))
    ]

    def aggregate(values: dict[str, float], prefix: str) -> float:
        selected = [value for key, value in values.items() if key.startswith(prefix)]
        if any(not math.isfinite(value) for value in selected):
            return float("nan")
        return math.sqrt(sum(value * value for value in selected))

    def require_positive(label: str, values: dict[str, float], name: str) -> float:
        value = values[name]
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"v6 {label} loss failed to reach {name}")
        return value

    result: dict[str, dict[str, float]] = {}
    groups = (
        ("global_only", global_term),
        ("anchor_only", anchor_term),
        ("local_only", local_term),
        ("fusion_only", fusion_term),
    )
    for label, term in groups:
        gradients = torch.autograd.grad(term, probes, retain_graph=True, allow_unused=True)
        values = {
            name: _probe_norm([gradient]) for name, gradient in zip(names, gradients)
        }
        result[label] = values
        require_positive(label, values, "visual_proj")
        if label in ("global_only", "local_only", "fusion_only"):
            require_positive(label, values, "shared_head")
        if label in ("local_only", "fusion_only"):
            o3_aggregate = aggregate(values, "o3_")
            pta_aggregate = aggregate(values, "pta_")
            if not math.isfinite(o3_aggregate) or o3_aggregate <= 0.0:
                raise RuntimeError(f"v6 {label} loss failed to reach O3 in aggregate")
            if not math.isfinite(pta_aggregate) or pta_aggregate <= 0.0:
                raise RuntimeError(f"v6 {label} loss failed to reach PTA in aggregate")
            result[label]["o3_aggregate"] = o3_aggregate
            result[label]["pta_aggregate"] = pta_aggregate
    result["anchor_only"]["shared_head_reached"] = float(
        result["anchor_only"].get("shared_head", 0.0) > 0.0
    )
    return result


def _limits_from_boxes(box_tensor: torch.Tensor):
    return box_tensor.detach().to(torch.int64)


def _drift_accumulator():
    return {
        "pairs": 0,
        "coordinate_changed": 0,
        "coordinate_abs_sum": 0.0,
        "center_distance_sum": 0.0,
        "center_distance_max": 0.0,
        "center_distance_ge_material": 0,
        "iou_sum": 0.0,
        "groups_with_material_change": 0,
    }


def _accumulate_box_drift(accumulator: dict, old_boxes: torch.Tensor, new_boxes: torch.Tensor) -> None:
    if old_boxes.shape != new_boxes.shape or old_boxes.ndim != 3 or old_boxes.shape[-1] != 4:
        raise ValueError("Drift diagnostics require matching [N,K,4] boxes")
    old = old_boxes.detach().to(torch.float32).cpu()
    new = new_boxes.detach().to(torch.float32).cpu()
    changed = old != new
    old_centers = (old[..., :2] + old[..., 2:]) / 2.0
    new_centers = (new[..., :2] + new[..., 2:]) / 2.0
    distances = torch.linalg.vector_norm(new_centers - old_centers, dim=-1)
    x0 = torch.maximum(old[..., 0], new[..., 0]); y0 = torch.maximum(old[..., 1], new[..., 1])
    x1 = torch.minimum(old[..., 2], new[..., 2]); y1 = torch.minimum(old[..., 3], new[..., 3])
    intersection = (x1 - x0).clamp_min(0.0) * (y1 - y0).clamp_min(0.0)
    old_area = (old[..., 2] - old[..., 0]).clamp_min(0.0) * (old[..., 3] - old[..., 1]).clamp_min(0.0)
    new_area = (new[..., 2] - new[..., 0]).clamp_min(0.0) * (new[..., 3] - new[..., 1]).clamp_min(0.0)
    union = old_area + new_area - intersection
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    accumulator["pairs"] += int(distances.numel())
    accumulator["coordinate_changed"] += int(changed.sum())
    accumulator["coordinate_abs_sum"] += float((old - new).abs().sum())
    accumulator["center_distance_sum"] += float(distances.sum())
    accumulator["center_distance_max"] = max(
        accumulator["center_distance_max"], float(distances.max())
    )
    accumulator["center_distance_ge_material"] += int(
        (distances >= D0_MATERIAL_DISTANCE_PX).sum()
    )
    accumulator["iou_sum"] += float(iou.sum())
    accumulator["groups_with_material_change"] += int(
        (distances.amax(dim=1) >= D0_MATERIAL_DISTANCE_PX).sum()
    )


def _drift_summary(accumulator: dict) -> dict:
    pairs = max(accumulator["pairs"], 1)
    return {
        "pairs": accumulator["pairs"],
        "coordinate_change_ratio": accumulator["coordinate_changed"] / (pairs * 4.0),
        "coordinate_abs_mean_px": accumulator["coordinate_abs_sum"] / (pairs * 4.0),
        "center_distance_mean_px": accumulator["center_distance_sum"] / pairs,
        "center_distance_max_px": accumulator["center_distance_max"],
        "fraction_boxes_ge_8px": accumulator["center_distance_ge_material"] / pairs,
        "iou_mean": accumulator["iou_sum"] / pairs,
        "groups_with_material_change": accumulator["groups_with_material_change"],
    }


def attention_boxes_for_all_scales(
    images: torch.Tensor,
    cls_patch_attention: torch.Tensor,
    *,
    local_scales: tuple[int, ...] = _LOCAL_SCALES,
    top_k: int = _ATTENTION_TOP_K,
) -> torch.Tensor:
    """Return boxes for all four registered scales for every row, shape [N,4,4]."""
    if images.ndim != 4 or cls_patch_attention.shape[0] != images.shape[0]:
        raise ValueError("Images and attention must be matching [B,...] tensors")
    all_boxes = []
    for scale in local_scales:
        boxes = attention_crop_boxes(images, cls_patch_attention, crop_size=int(scale), top_k=top_k)
        all_boxes.append(torch.as_tensor(boxes, dtype=torch.int64, device=images.device))
    return torch.stack(all_boxes, dim=1)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def train_online_geometry(plan: dict, name: str) -> Path:
    """Train one fixed v6 candidate (G0 or G1) independently from v5 F1."""
    if name not in ("G0", "G1"):
        raise ValueError("Only the two registered PRELIM75 v6 candidates are allowed")
    output = Path(plan["output"]) / name
    output.mkdir(parents=True, exist_ok=False)
    policy = "frozen_V1" if name == "G0" else "current_student_step"
    start = time.monotonic()
    device = gpu_setup()
    data, boxes, geometry, preparation = read_geometry_assets(plan)
    smoke = json.loads((Path(plan["output"]) / "smoke.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "passed" or smoke.get("candidate_scope") != "G0_and_G1_common":
        raise ValueError("Missing or invalid isolated v6 common smoke result")
    microbatch = int(smoke["microbatch_size"])
    if microbatch not in (16, 32):
        raise ValueError("Invalid v6 common microbatch size")

    model, preprocess, checkpoint, o3, pta = load_composite(plan["weight_parent"], device)
    model.float(); o3.float(); pta.float()
    licensed_visual = {parameter_name for parameter_name, parameter in model.visual.named_parameters()
                       if parameter.requires_grad}
    if "conv1.weight" in licensed_visual or "positional_embedding" in licensed_visual:
        raise ValueError("F1 visual permission mask changed")
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
    frozen = {parameter_name: parameter.detach().cpu().clone()
              for parameter_name, parameter in model.named_parameters() if not parameter.requires_grad}
    visual_parameters = [parameter for parameter in model.visual.parameters() if parameter.requires_grad]
    head_parameters = list(model.classifier.parameters())
    o3_parameters = list(o3.parameters())
    pta_parameters = list(pta.parameters())
    adapter_parameters = o3_parameters + pta_parameters
    all_parameters = visual_parameters + head_parameters + adapter_parameters
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
    recipe = {
        "candidate": name,
        "plan_id": PLAN_ID,
        "independent_parent_load": True,
        "weight_parent": plan["weight_parent"],
        "weight_parent_sha256": plan["weight_parent_sha256"],
        "weight_parent_experiment_id": PARENT_EXPERIMENT_ID,
        "training_geometry_policy": policy,
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
        "epoch_numbers": [13, 14, 15],
        "schedule_epochs": 18,
        "batch_size": 32,
        "effective_batch_size": 32,
        "microbatch_size": microbatch,
        "accumulation": 32 // microbatch,
        "fresh_optimizer": True,
        "fresh_scheduler": True,
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
        "selection": "fixed_epoch_15_last_epoch",
        "licensed_visual_parameters": sorted(licensed_visual),
        "online_geometry": {
            "g0": "frozen_cached_V1",
            "g1": "current_student_detached_final_block_attention",
            "box_gradient": False,
            "supports_eager_sequential_full_finetune_only": True,
        },
    }
    atomic_json_dump(recipe, output / "training_recipe.json")

    history = []
    optimizer_step = 0
    clipped_steps = 0
    first_step_audit = None
    first_step_probes = None
    first_step_global = None
    for sample_epoch in (13, 14, 15):
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
                    raise RuntimeError("Nonfinite v6 global outputs")
                orientation_ids = selected_flips.astype(np.int64)
                old_all_scales = torch.as_tensor(
                    boxes[sub_idx, orientation_ids, :, :], dtype=torch.int64, device=device
                )
                new_all_scales = attention_boxes_for_all_scales(
                    images, attention, local_scales=_LOCAL_SCALES, top_k=_ATTENTION_TOP_K
                )
                row_selector = torch.arange(sub_batch_size, device=device)
                actual_scale_ids = torch.as_tensor(selected_scale_ids, dtype=torch.long, device=device)
                if name == "G0":
                    actual_bounds = old_all_scales[row_selector, actual_scale_ids, :]
                    local_images = crop_with_bound_boxes(images, actual_bounds)
                else:
                    local_images, actual_box_list = extract_online_local_views(
                        images, attention, selected_scale_ids,
                        local_scales=_LOCAL_SCALES, top_k=_ATTENTION_TOP_K,
                    )
                    actual_bounds = new_all_scales[row_selector, actual_scale_ids, :]
                    reported_bounds = torch.as_tensor(
                        np.asarray(actual_box_list, dtype=np.int64), dtype=torch.int64, device=device
                    )
                    if not torch.equal(actual_bounds, reported_bounds):
                        raise RuntimeError("G1 online crop boxes changed during batched extraction")

                # Always accumulate old/new drift over the selected orientation
                # and all four registered scales.  Purely diagnostic; no crop or
                # model output is ever selected from these values.
                old_cpu = old_all_scales.detach().cpu()
                new_cpu = new_all_scales.detach().cpu()
                _accumulate_box_drift(drift, old_cpu, new_cpu)
                if name == "G0":
                    expected_fixed = old_cpu[row_selector.cpu(), actual_scale_ids.cpu(), :]
                    if not torch.equal(expected_fixed, actual_bounds.detach().cpu()):
                        raise RuntimeError("G0 actual crop does not use frozen V1 geometry")

                local_logits = adapted_dual_local_view_logits(model, o3, pta, local_images)
                if not torch.isfinite(local_logits).all():
                    raise RuntimeError("Nonfinite v6 local outputs")
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
                    raise RuntimeError("Nonfinite v6 continuation loss")
                # Empty-positive-batch safety is examined within the common smoke
                # before any optimizer exists; the production loss itself keeps the
                # original anchor term with a clamped denominator exactly as F1 did.
                if optimizer_step == 0 and not first_microbatch_saved:
                    global_term = (terms["global"] * weights).sum() / denominator
                    local_term = (terms["local"] * weights).sum() / denominator
                    if bool((weights > 0).any()):
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
                    "geometry_policy": policy,
                    "empty_positive_effective_batch": not batch_has_positive,
                    "batch_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                    "batch_flip_scale_sha256": _tensor_sha256(batch_flip_scale),
                    "global_logits_sha256": _tensor_sha256(first_step_global["global_logits"]),
                    "global_features_sha256": _tensor_sha256(first_step_global["features"]),
                    "gradient_probes": first_step_probes,
                    **parts,
                }
                required_groups = (
                    ("visual",) if not batch_has_positive
                    else ("visual", "head", "o3", "pta")
                )
                if frozen_leaks or any(
                    not math.isfinite(gradient_groups[name]) or gradient_groups[name] <= 0.0
                    for name in required_groups
                ):
                    raise RuntimeError("PRELIM75 v6 first-step gradient audit failed")
                if first_step_global is not None:
                    torch.save(first_step_global, output / "first_step_global.pt")
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
                atomic_json_dump({
                    "status": "training",
                    "candidate": name,
                    "geometry_policy": policy,
                    "epoch": sample_epoch,
                    "completed_samples": totals["samples"],
                    "optimizer_steps": optimizer_step,
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
            "geometry_drift": _drift_summary(drift),
            "epoch_elapsed_seconds": time.monotonic() - epoch_start,
            "elapsed_seconds": time.monotonic() - start,
            "peak_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
        }
        history.append(row)
        atomic_json_dump(history, output / "training_history.json")
        print(json.dumps(row), flush=True)
    for parameter_name, parameter in model.named_parameters():
        if parameter_name in frozen and not torch.equal(parameter.detach().cpu(), frozen[parameter_name]):
            raise RuntimeError(f"Frozen F1 tensor changed during v6 training: {parameter_name}")
    if first_step_audit is None or first_step_probes is None:
        raise RuntimeError("v6 first-step audit never ran")
    payload = saved_candidate(checkpoint, model, config, name, plan, optimizer, scheduler, 15, o3, pta)
    payload["global_step"] = int(checkpoint.get("global_step", 0)) + optimizer_step
    payload["prelim75_training"] = {
        "name": name,
        "candidate_id": f"PRELIM75_V6_{name}",
        "plan_id": PLAN_ID,
        "weight_parent_experiment_id": PARENT_EXPERIMENT_ID,
        "weight_parent_sha256": plan["weight_parent_sha256"],
        "training_geometry_policy": policy,
        "comparison_geometry_origin": "V1",
        "geometry_origin_sha256": plan["geometry_origin_sha256"],
        "geometry_boxes_sha256": plan["geometry_boxes_sha256"],
        "original_supervision_sha256": plan["original_supervision_sha256"],
        "supervision_changed": False,
        "inference_teacher_used": False,
        "validation_scope": "overlap_diagnostic",
        "upstream_provenance_complete": False,
        "selection": "fixed_epoch_15_last_epoch",
        "supervision": SUPERVISION_POLICY,
        "classification_temperature": 1.5,
        "probability_fusion_blend": 0.5,
        "sampled_probability_fusion": True,
        "global_weight": 0.6,
        "local_weight": 0.4,
        "anchor_weight": 2.0,
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
    reloaded_model, _, reloaded, reloaded_o3, reloaded_pta = load_composite(output / "candidate.pt", torch.device("cpu"))
    metadata = reloaded.get("prelim75_training", {})
    if (metadata.get("plan_id") != PLAN_ID
            or metadata.get("weight_parent_sha256") != plan["weight_parent_sha256"]
            or metadata.get("training_geometry_policy") != policy
            or metadata.get("comparison_geometry_origin") != "V1"
            or metadata.get("supervision_changed") is not False
            or metadata.get("inference_teacher_used") is not False):
        raise RuntimeError("Saved v6 candidate lineage changed on strict reload")
    result = {
        "status": "trained_pending_real_diagnostic",
        "candidate": name,
        "geometry_policy": policy,
        "history": history,
        "first_step_gradient_audit": first_step_audit,
        "first_step_probes": first_step_probes,
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


def _module_hook_count(model: torch.nn.Module) -> int:
    return sum(len(module._forward_pre_hooks) for module in model.modules())


def _cuda_memory_clear() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _box_tensor_to_boxes(box_tensor: torch.Tensor) -> list[tuple[int, int, int, int]]:
    array = box_tensor.detach().cpu().numpy().astype(np.int64)
    return [tuple(int(value) for value in row) for row in array]


def _empty_positive_safety(terms: dict, weights: torch.Tensor) -> dict:
    zero_weights = torch.zeros_like(weights)
    denominator = zero_weights.sum().clamp_min(1e-8)
    classification = (terms["objective"] * zero_weights).sum() / denominator
    anchor_proxy = terms["objective"].sum() * 0.0
    return {
        "classification_zero": float(classification.detach()) == 0.0,
        "classification_finite": bool(torch.isfinite(classification)),
        "anchor_proxy_finite": bool(torch.isfinite(anchor_proxy)),
    }


def smoke_v6(plan: dict) -> dict:
    """Choose one common microbatch and verify the v6 integration contract."""
    output = Path(plan["output"]) / "smoke.json"
    if output.exists():
        raise FileExistsError("PRELIM75 v6 smoke result already exists")
    start = time.monotonic()
    device = gpu_setup()
    data, boxes, geometry, _ = read_geometry_assets(plan)
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
    flips, scales = fixed_choices(data["paths"], 13)
    images = batch["images"].to(device, non_blocking=True)
    if flips[indices].any():
        flip_mask = torch.tensor(flips[indices], device=device)
        images[flip_mask] = images[flip_mask].flip(3)
    targets = data["targets"][indices].to(device)
    weights = data["weights"][indices].to(device)
    denominator = weights.sum().clamp_min(1e-8)
    visual_parameters = [parameter for parameter in model.visual.parameters() if parameter.requires_grad]
    head_parameters = list(model.classifier.parameters())
    o3_parameters = list(o3.parameters()); pta_parameters = list(pta.parameters())
    all_parameters = visual_parameters + head_parameters + o3_parameters + pta_parameters

    # --- wrapper value/attention/box alignment --------------------------------
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
        raise RuntimeError("v6 wrapper alignment tolerance failed") from exc
    if not alignment["argmax_equal"]:
        raise RuntimeError("v6 wrapper changed native global argmax")
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
        box_alignment[str(scale_size)] = {
            "exact": bool(exact),
            "num_boxes": len(boxes_wrapper),
        }
        if not exact:
            raise RuntimeError(f"v6 wrapper crop boxes differ from inference formula at scale {scale_size}")

    # --- 32 vs 16 chunking sensitivity of integer boxes ----------------------
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
            raise RuntimeError(
                "32/16 chunking changed integer online boxes beyond the fixed 1px tolerance"
            )
    except torch.cuda.OutOfMemoryError:
        _cuda_memory_clear()
        chunking_audit = {"checked": False, "reason": "cuda_oom_during_chunk_compare"}

    # --- G1 dependency on current attention -----------------------------------
    probe_count = min(8, len(indices))
    original_probe_boxes = attention_crop_boxes(
        images[:probe_count], wrapper_attention[:probe_count], crop_size=160, top_k=_ATTENTION_TOP_K
    )
    perturbed_attention = torch.zeros_like(wrapper_attention[:probe_count])
    high_patch_start = int(wrapper_attention.shape[-1] - 5)
    original_center_x = float(np.mean([(box[0] + box[2]) / 2.0 for box in original_probe_boxes]))
    target_indices = (
        list(range(high_patch_start, int(wrapper_attention.shape[-1])))
        if original_center_x < 112.0 else list(range(0, 5))
    )
    perturbed_attention[:, :, target_indices] = 1.0
    perturbed_boxes = attention_crop_boxes(
        images[:probe_count], perturbed_attention, crop_size=160, top_k=_ATTENTION_TOP_K
    )
    attention_dependency = {
        "probe_count": probe_count,
        "perturbed_boxes_differ": bool(any(left != right for left, right in zip(original_probe_boxes, perturbed_boxes))),
    }
    if not attention_dependency["perturbed_boxes_differ"]:
        raise RuntimeError("v6 G1 boxes did not respond to a changed attention tensor")

    # --- replay state and RNG isolation ---------------------------------------
    with torch.no_grad():
        _, _, captured_tokens = _forward_global_with_captured_tokens(model, images[:probe_count])
        block = _final_visual_block(model)
        state_before = {name: tensor.detach().clone() for name, tensor in block.state_dict().items()}
        cpu_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(device).clone() if torch.cuda.is_available() else None
        mode_before = (model.training, o3.training, pta.training)
        requires_grad_before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        replay_last_block_attention(block, captured_tokens)
        state_after = {name: tensor.detach().clone() for name, tensor in block.state_dict().items()}
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
        "rng_unchanged", "model_mode_unchanged", "requires_grad_unchanged", "replay_state_unchanged", "cuda_rng_unchanged",
    )):
        raise RuntimeError("v6 attention replay changed RNG, mode, or module state")

    # --- executable common microbatch -----------------------------------------
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
            local_images, actual_boxes = extract_online_local_views(
                current_images, attention, scales[sub_idx],
                local_scales=_LOCAL_SCALES, top_k=_ATTENTION_TOP_K,
            )
            independent_boxes = attention_boxes_by_scale(
                current_images, attention, scales[sub_idx],
                local_scales=_LOCAL_SCALES, top_k=_ATTENTION_TOP_K,
            )
            if [tuple(map(int, box)) for box in actual_boxes] != independent_boxes:
                raise RuntimeError("v6 grouped online crops do not match their boxes")
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
                raise RuntimeError("Nonfinite v6 smoke loss")
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
        required_groups = (
            ("visual",) if not batch_has_positive
            else ("visual", "head", "o3", "pta")
        )
        if frozen_leaks or any(
            not math.isfinite(gradient_norms[name]) or gradient_norms[name] <= 0.0
            for name in required_groups
        ):
            raise RuntimeError("v6 smoke gradient coverage failed")
        return {
            "loss": float(total_loss),
            "gradient_norms": gradient_norms,
            "frozen_gradient_leaks": frozen_leaks,
            "first_audit": first_audit,
            "empty_positive": empty_positive,
            "empty_positive_effective_batch": not batch_has_positive,
        }

    extra_geometry_rows = None
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
    frozen_unclipped = [
        name for name, parameter in model.named_parameters()
        if name in frozen_before and not torch.equal(parameter.detach().cpu(), frozen_before[name])
    ]
    if frozen_unclipped:
        raise RuntimeError(f"v6 smoke changed frozen parameters: {frozen_unclipped}")
    torch.cuda.empty_cache()
    first_audit = attempt_result.pop("first_audit")
    empty_positive = attempt_result.pop("empty_positive")
    result = {
        "status": "passed",
        "candidate_scope": "G0_and_G1_common",
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
        "attention_dependency": attention_dependency,
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
        "geometry_policy_tested": "current_student_step",
        "old_box_cache_required_for_g1": False,
        "test_used": False,
    }
    atomic_json_dump(result, output)
    del model, o3, pta, anchor_teacher
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Fixed overlap diagnostic and submission delivery
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_v6(plan: dict, name: str) -> dict:
    if name not in ("G0", "G1"):
        raise ValueError("Unknown PRELIM75 v6 candidate")
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
    expected_policy = "frozen_V1" if name == "G0" else "current_student_step"
    if (metadata.get("plan_id") != PLAN_ID
            or metadata.get("weight_parent_sha256") != plan["weight_parent_sha256"]
            or metadata.get("training_geometry_policy") != expected_policy
            or metadata.get("comparison_geometry_origin") != "V1"
            or metadata.get("supervision_changed") is not False
            or metadata.get("inference_teacher_used") is not False
            or metadata.get("prior_in_inference") is not False):
        raise ValueError("PRELIM75 v6 candidate lineage mismatch")
    frame = pd.read_csv(plan["val_csv"])
    paths = [canonical_sample_path(path) for path in frame.image_path.astype(str)]
    if len(paths) != _REQUIRED_DIAGNOSTIC_ROWS or len(set(paths)) != len(paths):
        raise ValueError("Diagnostic row identity mismatch")
    training_index = {path: index for index, path in enumerate(data["paths"])}
    indices = torch.tensor([training_index[path] for path in paths], dtype=torch.long)
    labels = torch.tensor(frame.label.to_numpy(), dtype=torch.long)
    if not torch.equal(labels, data["labels"][indices]):
        raise ValueError("Diagnostic labels differ from frozen fitting list")
    dataset = OfficialImages(paths, plan["train_root"], preprocess)
    stream = loader(dataset, 64, plan)
    fused_predictions = []
    global_predictions = []
    local_predictions = []
    scale_weights = torch.tensor(list(_LOCAL_SCALE_WEIGHTS), device=device).view(1, 1, 4, 1)
    for step, batch in enumerate(stream):
        images = batch["images"].to(device, non_blocking=True)
        global_logits, local_logits, _ = _teacher_views(model, o3, pta, images)
        global_probability = (global_logits.float() / 1.5).softmax(-1).mean(1)
        local_probability = ((local_logits.float() / 1.5).softmax(-1) * scale_weights).sum(2).mean(1)
        fused_probability = 0.6 * global_probability + 0.4 * local_probability
        if not torch.isfinite(fused_probability).all():
            raise RuntimeError("Invalid PRELIM75 v6 diagnostic probabilities")
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
        key: prediction_metrics(value, **fields, num_classes=_REQUIRED_CLASSES, clean_core_threshold=0.7)
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
        "geometry_policy": expected_policy,
        "validation_scope": "overlap_diagnostic",
        "independent_generalization_claim": False,
        "metrics": metrics["fused"],
        "branch_metrics": metrics,
        "delta_pp_vs_F1": deltas,
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


def deliver_v6(plan: dict, name: str) -> dict:
    if name not in ("G0", "G1"):
        raise ValueError("Unknown PRELIM75 v6 candidate")
    root = Path(plan["output"]) / name
    diagnostic = json.loads((root / "diagnostic/result.json").read_text(encoding="utf-8"))
    if diagnostic.get("status") != "complete" or diagnostic.get("engineering_stop", True):
        raise ValueError("PRELIM75 v6 candidate stopped by the fixed engineering rule")
    checkpoint = root / "candidate.pt"
    if sha256_file(checkpoint) != diagnostic["checkpoint_sha256"]:
        raise ValueError("PRELIM75 v6 candidate changed after diagnostic")
    output = root / "submission"
    if output.exists():
        raise FileExistsError("Do not overwrite an existing PRELIM75 v6 submission")
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
        "".join(f"{image_name}, {label}\n" for image_name, label in rows),
        encoding="utf-8",
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
            raise ValueError("PRELIM75 v6 ZIP/CSV byte identity failed")
    if len(rows) != _REQUIRED_TEST_ROWS or len({row[0] for row in rows}) != _REQUIRED_TEST_ROWS:
        raise ValueError("Official PRELIM75 v6 test coverage mismatch")
    report = {
        "candidate": name,
        "status": "submission_ready_pending_platform",
        "geometry_policy": diagnostic["geometry_policy"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "csv_sha256": sha256_file(output / "pred_results.csv"),
        "zip_sha256": sha256_file(output / "submission.zip"),
        "inference_manifest_sha256": sha256_file(output / "manifest.json"),
        "resolved_config_sha256": sha256_file(root / "resolved_config.json"),
        "source_manifest_sha256": sha256_file(root / "source_manifest.json"),
        "inference_command": command,
        "validation_command": check_command,
        "validation_exit_code": 0,
        "rows": _REQUIRED_TEST_ROWS,
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


# Convenience names mirroring the proposal's mathematical notation and
# common integration wording.
attention_box = attention_crop_boxes
extract_attention_boxes = attention_crop_boxes
extract_grouped_attention_crops = extract_online_local_views
compute_geometry_metrics = geometry_comparison
compute_d0_metrics = geometry_comparison
select_d0_groups = rank_d0_groups
run_d0_gate = run_d0
forward_with_detached_attention = forward_train_with_detached_attention

# Compatibility aliases for callers that prefer the file's historical role.
load_online_geometry_plan = load_v6_plan
preflight_online_geometry = preflight_v6
prepare_online_geometry_assets = prepare_online_geometry
smoke_online_geometry = smoke_v6
train_online_geometry_candidate = train_online_geometry
evaluate_online_geometry = evaluate_v6
deliver_online_geometry = deliver_v6
