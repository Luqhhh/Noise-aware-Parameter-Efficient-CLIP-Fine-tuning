#!/usr/bin/env python3
"""Generate the REMATCH750_SEARCH_V5 conditional manifest and trial configs.

Every manifest row is a *slot*, not a run.  The generated YAML files are
loadable so implementation can be checked and bound before execution, but the
V5 declaration layer refuses to execute rows whose mechanism is still blocked.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/rematch750_search_v5"
BASE_CONFIG = ROOT / "configs/rematch750_search_v4/F05.yaml"
MANIFEST = ROOT / "search_manifest.json"
V5_OUTPUT = "../../outputs/rematch750_search_v5"
LP_CHECKPOINT = "/workspace/noise/outputs/rematch750_npu/RM_LP/seed42/checkpoints/best.pt"
F05_CHECKPOINT = (
    "/workspace/noise-worktrees/rematch750_search_v4/outputs/"
    "rematch750_search_v4/RM_V4_F05/seed42/checkpoints/best.pt"
)
F05_FEATURES = "../../outputs/rematch750_search_v5/assets/RM_V4_F05_encoder_features"
F05_FEATURES_LOCAL = ROOT / "outputs/rematch750_search_v5/assets/RM_V4_F05_encoder_features"

BASE: dict[str, Any] = {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def training_trial(
    trial_id: str,
    family: str,
    mechanism: str,
    *,
    search: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    implementation_status: str = "implemented",
    dependencies: list[str] | None = None,
    notes: str = "",
    parent_kind: str = "shared_lp",
    kind: str = "training",
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = deep_merge(BASE, overrides or {})
    project = cfg.setdefault("project", {})
    project.update(
        protocol="rematch750_search_v5",
        experiment_id=f"RM_V5_{trial_id}",
        trial_id=trial_id,
        family=family,
        mechanism=mechanism,
        implementation_status=implementation_status,
        parent_kind=parent_kind,
        search=dict(search),
    )
    if dependencies:
        project["dependencies"] = list(dependencies)
    if parent_kind == "shared_lp":
        cfg.setdefault("train", {})["init_checkpoint"] = LP_CHECKPOINT
    cfg.setdefault("output", {})["root"] = V5_OUTPUT
    cfg.setdefault("train", {})["device"] = "npu:UNASSIGNED"
    cfg["project"]["seed"] = 42
    meta = {
        "trial_id": trial_id,
        "family": family,
        "kind": kind,
        "mechanism": mechanism,
        "parent_kind": parent_kind,
        "implementation_status": implementation_status,
        "dependencies": list(dependencies or []),
        "notes": notes,
    }
    return cfg, meta


def _microbatch_candidates() -> list[int]:
    return [256, 128, 64]


def _set_microbatch(cfg: dict[str, Any], microbatch: int) -> None:
    train = cfg.setdefault("train", {})
    train["batch_size"] = int(microbatch)
    train["grad_accum_steps"] = 1024 // int(microbatch)
    train["effective_batch_size"] = 1024


def _visual_fp32(cfg: dict[str, Any], *, resolution: int, microbatch: int,
                 optimizer: str, sam_rho: float | None = None,
                 gsam_alpha: float | None = None) -> dict[str, Any]:
    cfg["model"]["input_resolution"] = int(resolution)
    cfg["data"]["train_augmentation"] = "weak_rrc_flip"
    cfg["loss"].update(
        name="gce",
        gce_q=0.5,
        ce_warmup_epochs=2,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        feature_distillation_weight=2.0,
    )
    train = cfg["train"]
    train.update(
        amp=False,
        epochs=16,
        schedule_epochs=16,
        backbone_lr=1.2e-05,
        head_lr=4.0e-04,
        lr_warmup_epochs=0,
        max_grad_norm=1.0,
    )
    _set_microbatch(cfg, microbatch)
    if optimizer == "sam":
        train["sam"] = {
            "enabled": True,
            "mode": "standard_global_l2",
            "rho": float(sam_rho if sam_rho is not None else 0.025),
        }
    elif optimizer == "gsam":
        train["sam"] = {
            "enabled": True,
            "mode": "gsam_constant_rho",
            "rho": float(sam_rho if sam_rho is not None else 0.025),
            "gsam_alpha": float(gsam_alpha if gsam_alpha is not None else 0.2),
        }
    else:
        train.pop("sam", None)
    return cfg


def _visual_amp(cfg: dict[str, Any], *, resolution: int, microbatch: int) -> dict[str, Any]:
    cfg["model"]["input_resolution"] = int(resolution)
    cfg["data"]["train_augmentation"] = "weak_rrc_flip"
    cfg["loss"].update(
        name="gce",
        gce_q=0.5,
        ce_warmup_epochs=2,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        feature_distillation_weight=2.0,
    )
    cfg["train"].update(amp=True, epochs=16, schedule_epochs=16)
    _set_microbatch(cfg, microbatch)
    cfg["train"].pop("sam", None)
    return cfg


def _h_config(trial_id: str, mechanism: str, classifier: str, sampler: str,
              init: str) -> tuple[dict[str, Any], dict[str, Any]]:
    features_ready = (F05_FEATURES_LOCAL / "manifest.json").is_file()
    status = "implemented" if features_ready else "pending_artifact"
    deps = [] if features_ready else ["F05_encoder_320_feature_cache"]
    cfg, meta = training_trial(
        trial_id,
        "H",
        mechanism,
        search={
            "feature_source": "F05_encoder",
            "reference_resolution": 320,
            "classifier_mode": classifier,
            "sampler_mode": sampler,
            "init": init,
            "micro_batch_candidates": [1024],
            "smoke_required": False,
        },
        implementation_status=status,
        dependencies=deps,
        notes="Frozen F05 visual encoder; 20-epoch cached head refit.",
        parent_kind="frozen_backbone_head",
        kind="cached_head",
    )
    cfg["model"].update(
        input_resolution=320,
        peft_mode="frozen",
        use_cached_training=True,
        classifier_mode=classifier,
        cosine_scale_init=20.0,
        cosine_scale_trainable=True,
    )
    cfg["features"] = {
        "tensor_path": f"{F05_FEATURES}/features.pt",
        "paths_path": f"{F05_FEATURES}/image_paths.json",
        "manifest_path": f"{F05_FEATURES}/manifest.json",
    }
    cfg["project"]["parent_experiment_id"] = "RM_V4_F05"
    cfg["loss"].update(
        name="cross_entropy",
        gce_q=0.5,
        ce_warmup_epochs=0,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        feature_distillation_weight=0.0,
    )
    sampler_map = {
        "natural": "none",
        "sqrt_class_balanced": "sqrt_class_balanced",
        "class_balanced": "class_balanced",
    }
    cfg["longtail"].update(
        sampler_mode=sampler_map[sampler],
        loss_reweighting="none",
        balanced_softmax_tau=0.0,
    )
    cfg["train"].update(
        device="npu:UNASSIGNED",
        amp=True,
        epochs=20,
        schedule_epochs=20,
        batch_size=1024,
        grad_accum_steps=1,
        effective_batch_size=1024,
        head_lr=4.0e-4,
        backbone_lr=0.0,
        init_weights_mode="all" if init == "inherit" else "visual_only",
    )
    if init == "inherit":
        cfg["train"]["init_checkpoint"] = F05_CHECKPOINT
    else:
        cfg["train"].pop("init_checkpoint", None)
    return cfg, meta


def build_trials() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    trials: list[tuple[dict[str, Any], dict[str, Any]]] = []

    # R: fixed and staged resolution.
    for trial_id, resolution, microbatch in (
        ("R01", 352, 256),
        ("R02", 384, 128),
        ("R03", 416, 128),
        ("R04", 448, 64),
    ):
        cfg, meta = training_trial(
            trial_id,
            "R",
            f"fixed_{resolution}px",
            search={
                "resolution": resolution,
                "staged_resolution": False,
                "micro_batch_candidates": [microbatch, 128, 64],
                "smoke_required": True,
                "reference_resolution": 224,
            },
            implementation_status="implemented",
            notes="Fixed-resolution full fine-tune; microbatch must pass smoke before run.",
        )
        _visual_amp(cfg, resolution=resolution, microbatch=microbatch)
        meta["notes"] = (
            "Fixed-resolution full fine-tune; declared microbatch is a placeholder "
            "and the runner must retain the first smoke-passing candidate."
        )
        trials.append((cfg, meta))

    for trial_id, early, late in (("R05", 224, 384), ("R06", 320, 448)):
        cfg, meta = training_trial(
            trial_id,
            "R",
            f"staged_{early}_to_{late}",
            search={
                "resolution": late,
                "staged_resolution": True,
                "micro_batch_candidates": [256, 128, 64],
                "smoke_required": True,
                "reference_resolution": 224,
            },
            implementation_status="implemented",
            dependencies=[],
            notes="Epoch 9 switch; positional embeddings are interpolated per-forward and optimizer moments stay shape-compatible.",
        )
        _visual_amp(cfg, resolution=early, microbatch=256)
        cfg["train"]["staged_resolution"] = {
            "enabled": True,
            "early_resolution": int(early),
            "late_resolution": int(late),
            "switch_epoch": 9,
        }
        cfg["model"]["input_resolution"] = int(early)
        trials.append((cfg, meta))

    # S: FP32 AdamW controls and SAM radii.
    s_micro = {320: 256, 384: 128, 448: 64}
    s_index = 0
    for resolution in (320, 384, 448):
        for optimizer, rho in (("adamw", None), ("sam", 0.025), ("sam", 0.05)):
            s_index += 1
            trial_id = f"S{s_index:02d}"
            status = "implemented"
            deps = []
            mechanism = "fp32_adamw" if optimizer == "adamw" else f"fp32_sam_rho{rho:g}"
            cfg, meta = training_trial(
                trial_id,
                "S",
                mechanism,
                search={
                    "resolution": resolution,
                    "precision": "fp32",
                    "optimizer": optimizer,
                    "sam_rho": rho,
                    "micro_batch_candidates": [s_micro[resolution], 128, 64],
                    "smoke_required": True,
                    "reference_resolution": 224,
                },
                implementation_status=status,
                dependencies=deps,
                notes="FP32 paired control; SAM rows require the effective-batch implementation.",
            )
            _visual_fp32(
                cfg,
                resolution=resolution,
                microbatch=s_micro[resolution],
                optimizer=optimizer,
                sam_rho=rho,
            )
            if optimizer == "sam":
                # The V5 declaration layer enforces this fail-closed detail.
                cfg["train"]["grad_accum_steps"] = 1024 // s_micro[resolution]
            trials.append((cfg, meta))

    # K: anchor strength and same-view teacher.
    k_index = 0
    for resolution in (320, 384):
        for condition, anchor, schedule in (
            ("anchor0", 0.0, False),
            ("anchor0.5", 0.5, False),
            ("schedule2to0.2", 2.0, True),
        ):
            k_index += 1
            trial_id = f"K{k_index:02d}"
            status = "implemented"
            deps = []
            cfg, meta = training_trial(
                trial_id,
                "K",
                f"{resolution}px_{condition}",
                search={
                    "resolution": resolution,
                    "anchor_weight": anchor,
                    "anchor_schedule": schedule,
                    "teacher": "fixed_reference",
                    "micro_batch_candidates": [256, 128, 64],
                    "smoke_required": True,
                    "reference_resolution": 224,
                },
                implementation_status=status,
                dependencies=deps,
                notes="Fixed reference-feature anchor unless teacher/schedule is declared.",
            )
            _visual_amp(cfg, resolution=resolution, microbatch=256 if resolution == 320 else 128)
            cfg["loss"]["feature_distillation_weight"] = float(anchor)
            if schedule:
                cfg["loss"]["anchor_schedule"] = {
                    "enabled": True,
                    "hold_epochs": 2,
                    "start_weight": 2.0,
                    "end_epoch": 16,
                    "end_weight": 0.2,
                }
            trials.append((cfg, meta))
    for resolution in (320, 384, 448):
        k_index += 1
        trial_id = f"K{k_index:02d}"
        cfg, meta = training_trial(
            trial_id,
            "K",
            f"{resolution}px_same_view_official_teacher",
            search={
                "resolution": resolution,
                "anchor_weight": 2.0,
                "anchor_schedule": False,
                "teacher": "same_view_official",
                "micro_batch_candidates": [256, 128, 64],
                "smoke_required": True,
                "reference_resolution": 224,
            },
            implementation_status="implemented",
            dependencies=[],
            notes="Teacher replays the same augmented view resized to the official 224px encoder; no cached center-crop reuse.",
        )
        _visual_amp(cfg, resolution=resolution, microbatch=256 if resolution == 320 else 128)
        cfg["loss"]["feature_distillation_weight"] = 2.0
        cfg["train"]["same_view_teacher"] = {
            "enabled": True,
            "source": "official_clip",
            "resolution": 224,
        }
        trials.append((cfg, meta))

    # V: training geometry.
    v_index = 0
    for resolution in (320, 384):
        for geometry, label, status, deps in (
            ("rrc_area_min", "rrc_area0.5", "implemented", []),
            ("rrc_area_min", "rrc_area0.8", "implemented", []),
            ("letterbox", "letterbox", "implemented", []),
            ("late_rrc_window", "late_rrc_0.9_1.0", "implemented", []),
        ):
            v_index += 1
            trial_id = f"V{v_index:02d}"
            geometry_search: dict[str, Any] = {
                "resolution": resolution,
                "geometry": geometry,
                "micro_batch_candidates": [256, 128, 64],
                "smoke_required": True,
                "reference_resolution": 224,
            }
            if geometry == "rrc_area_min":
                area = 0.5 if label.endswith("0.5") else 0.8
                geometry_search["rrc_scale_min"] = area
                geometry_search["rrc_scale_max"] = 1.0
            elif geometry == "late_rrc_window":
                geometry_search["late_rrc_scale_min"] = 0.9
                geometry_search["late_rrc_scale_max"] = 1.0
            cfg, meta = training_trial(
                trial_id,
                "V",
                f"{resolution}px_{label}",
                search=geometry_search,
                implementation_status=status,
                dependencies=deps,
                notes="Compare the whole training+inference recipe, not only train geometry.",
            )
            _visual_amp(cfg, resolution=resolution, microbatch=256 if resolution == 320 else 128)
            if geometry == "rrc_area_min":
                cfg["data"]["rrc_scale_min"] = geometry_search["rrc_scale_min"]
                cfg["data"]["rrc_scale_max"] = 1.0
            elif geometry == "letterbox":
                cfg["data"]["train_augmentation"] = "clip_letterbox"
            elif geometry == "late_rrc_window":
                cfg["data"]["late_rrc"] = {
                    "enabled": True,
                    "start_epoch": 13,
                    "scale_min": 0.9,
                    "scale_max": 1.0,
                }
            trials.append((cfg, meta))

    # L: attention-guided local details.
    l_specs = [
        ("L01", 320, 0.35, 0.25),
        ("L02", 320, 0.35, 0.50),
        ("L03", 320, 0.55, 0.25),
        ("L04", 320, 0.55, 0.50),
        ("L05", 384, 0.55, 0.25),
        ("L06", 384, 0.55, 0.50),
    ]
    for trial_id, resolution, area, local_weight in l_specs:
        cfg, meta = training_trial(
            trial_id,
            "L",
            f"{resolution}px_area{area:g}_lambda{local_weight:g}",
            search={
                "resolution": resolution,
                "area_fraction": area,
                "local_supervision_weight": local_weight,
                "local_start_epoch": 5,
                "confidence_gate": 0.70,
                "micro_batch_candidates": [256, 128, 64],
                "smoke_required": True,
                "reference_resolution": 224,
            },
            implementation_status="implemented",
            dependencies=[],
            notes="Global-only epochs 1-4; confidence-gated local GCE; no local anchor/KL; activation-rate ledger emitted.",
        )
        _visual_amp(cfg, resolution=resolution, microbatch=256 if resolution == 320 else 128)
        cfg["model"]["peft_mode"] = "full_finetune"
        crop_size = int(round(resolution * (area ** 0.5)))
        crop_size = max(1, min(crop_size, resolution - 1))
        top_patches = max(1, min(5, (resolution // 32) ** 2))
        cfg["loss"]["attention_local_training"] = {
            "enabled": True,
            "crop_size": crop_size,
            "top_patches": top_patches,
            "local_supervision_weight": local_weight,
            "consistency_weight": 0.0,
            "temperature": 1.0,
            "start_epoch": 5,
            "confidence_gate": 0.70,
            "area_fraction": area,
        }
        trials.append((cfg, meta))

    # Q: GSAM branch, blocked on effective-batch SAM.
    q_specs = [
        ("Q01", 320, 0.1),
        ("Q02", 320, 0.2),
        ("Q03", 320, 0.4),
        ("Q04", 384, 0.1),
        ("Q05", 384, 0.2),
        ("Q06", 384, 0.4),
    ]
    for trial_id, resolution, alpha in q_specs:
        cfg, meta = training_trial(
            trial_id,
            "Q",
            f"{resolution}px_gsam_alpha{alpha:g}",
            search={
                "resolution": resolution,
                "gsam_alpha": alpha,
                "sam_rho": 0.025,
                "micro_batch_candidates": [256, 128, 64],
                "smoke_required": True,
                "reference_resolution": 224,
            },
            implementation_status="implemented",
            dependencies=[],
            notes="Pair with S-group rho=0.025 at the same resolution; alpha=0 degenerates to SAM.",
        )
        _visual_fp32(
            cfg,
            resolution=resolution,
            microbatch=256 if resolution == 320 else 128,
            optimizer="gsam",
            gsam_alpha=alpha,
        )
        trials.append((cfg, meta))

    # N: supervision repair only with new valid evidence.
    n_specs = [
        ("N01", 320, 0.2, 0.8, 0.10),
        ("N02", 320, 0.4, 0.8, 0.10),
        ("N03", 384, 0.2, 0.8, 0.10),
        ("N04", 384, 0.4, 0.8, 0.10),
    ]
    for trial_id, resolution, rho, gate, max_fraction in n_specs:
        cfg, meta = training_trial(
            trial_id,
            "N",
            f"{resolution}px_soft_repair_rho{rho:g}",
            search={
                "resolution": resolution,
                "repair_rho": rho,
                "confidence_gate": gate,
                "max_relabel_fraction": max_fraction,
                "evidence_required": True,
                "micro_batch_candidates": [256, 128, 64],
                "smoke_required": True,
                "reference_resolution": 224,
            },
            implementation_status="conditional_pending_evidence",
            dependencies=["new_training_side_valid_evidence", "activation_ledger"],
            notes="Skip and reallocate budget unless target/gradient activity is demonstrated.",
        )
        _visual_amp(cfg, resolution=resolution, microbatch=256 if resolution == 320 else 128)
        cfg["trust"].update(
            enabled=True,
            bundle_path=f"../../outputs/rematch750_search_v5/assets/quality_asset/{trial_id}.pt",
            selection_threshold=gate,
            minimum_sample_weight=1.0,
        )
        cfg.setdefault("train", {}).update(
            quality_asset_mode="soft_repair",
            repair_rho=rho,
            repair_confidence_gate=gate,
            max_relabel_fraction=max_fraction,
        )
        trials.append((cfg, meta))

    # H: 12 F05-encoder cached-head points.
    h_index = 0
    for classifier in ("linear", "cosine"):
        for sampler in ("natural", "sqrt_class_balanced", "class_balanced"):
            for init in ("inherit", "reinit"):
                h_index += 1
                trial_id = f"H{h_index:02d}"
                mechanism = f"{classifier}_{sampler}_{init}"
                cfg, meta = _h_config(
                    trial_id, mechanism, classifier, sampler, init
                )
                trials.append((cfg, meta))

    # X: compatibility combinations.  These are declarative placeholders until
    # component families produce valid winners.
    component_pairs = [
        ("X01", ["best_R", "best_K"]),
        ("X02", ["best_R", "best_S"]),
        ("X03", ["best_R", "best_V"]),
        ("X04", ["best_R", "best_L"]),
        ("X05", ["best_S", "best_K"]),
        ("X06", ["best_S", "best_V"]),
        ("X07", ["best_Q", "best_K"]),
        ("X08", ["best_visual", "best_H"]),
    ]
    for trial_id, components in component_pairs:
        cfg, meta = training_trial(
            trial_id,
            "X",
            "+".join(components),
            search={
                "component_trials": components,
                "reference_resolution": 224,
                "micro_batch_candidates": [256, 128, 64],
                "smoke_required": True,
            },
            implementation_status="blocked_dependency",
            dependencies=components,
            notes="Generate only after both component winners exist and compatibility is re-tested.",
        )
        cfg["project"]["experiment_id"] = f"RM_V5_{trial_id}"
        trials.append((cfg, meta))

    expected = 48 + 12 + 8
    if len(trials) != expected:
        raise AssertionError(f"expected {expected} V5 slots, generated {len(trials)}")
    families = {}
    for _, meta in trials:
        families[meta["family"]] = families.get(meta["family"], 0) + 1
    expected_families = {"R": 6, "S": 9, "K": 9, "V": 8, "L": 6, "Q": 6, "N": 4, "H": 12, "X": 8}
    if families != expected_families:
        raise AssertionError(f"family coverage mismatch: {families}")
    return trials


def main() -> None:
    global BASE
    BASE = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    trials = build_trials()

    from aegis_clip.config import validate_config
    from aegis_clip.rematch_search_v5 import validate_declaration

    manifest_trials = []
    for cfg, meta in trials:
        validate_config(copy.deepcopy(cfg))
        validate_declaration(copy.deepcopy(cfg))
        path = CONFIG_DIR / f"{meta['trial_id']}.yaml"
        text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        path.write_text(text, encoding="utf-8")
        row = dict(meta)
        row.update(
            config=str(path.relative_to(ROOT)),
            config_sha256=sha256_text(text),
            seed=int(cfg["project"].get("seed", 42)),
            effective_batch_size=int(
                cfg["train"].get("grad_accum_steps", 1)
                * cfg["train"].get("batch_size", 1024)
            ),
            micro_batch_size=int(cfg["train"].get("batch_size", 1024)),
            epochs=int(cfg["train"].get("epochs", 16)),
            status="planned_not_run",
        )
        manifest_trials.append(row)

    manifest = {
        "schema_version": 1,
        "protocol": "rematch750_search_v5",
        "name": "REMATCH750_SEARCH_V5_CONDITIONAL_68",
        "status": "planned_not_executed",
        "baseline": {
            "candidate": "RM_V4_F05",
            "config": "configs/rematch750_search_v4/F05.yaml",
            "config_sha256": "d5cf5ae16ee174cc2a6e3e75a84ce4a631255314f66f192a705197d714f21f9a",
            "checkpoint": F05_CHECKPOINT,
            "checkpoint_sha256": "efcc6cb933cad305eab15109f88046b714f697f02c7228a6beea4d2f5603a70a",
            "local_macro": 0.7443073987960815,
            "local_micro": 0.7544354796409607,
            "selected_epoch": 14,
            "platform_score": 65.4711,
            "platform_score_fraction": 0.654711,
            "platform_correct_predictions": 24515,
            "platform_correct_predictions_basis": "inferred from the reported 4-decimal score and 37,444 test rows; count was not separately supplied",
            "platform_receipt": "results/rematch750_f05_platform_20260924.json",
            "artifact_audit": "pending_real_artifact_hashes",
        },
        "historical_anchor": {
            "candidate": "RM_V3_B1024_E16_LR4",
            "local_macro": 0.7234723567962646,
            "local_micro": 0.7341398000717163,
        },
        "known_platform": {
            "RM_V4_F05": {
                "score": 65.4711,
                "correct": 24515,
                "correct_basis": "inferred_from_reported_4_decimal_score",
                "total": 37444,
                "status": "current_platform_leader",
                "receipt": "results/rematch750_f05_platform_20260924.json",
            },
            "RM_V4_F03": {
                "score": 64.41352419613288,
                "correct": 24119,
                "total": 37444,
                "status": "previous_platform_reference",
                "receipt": "results/rematch750_f03_platform_20260923.json",
            },
            "RM_FULL": {
                "score": 61.6360,
                "correct": None,
                "total": 37444,
                "status": "overlap_diagnostic_not_holdout_rank",
                "receipt": "results/rematch750_full01_platform_20260923.json",
            },
        },
        "platform_goal": {
            "target_percent": 70.0,
            "required_correct_predictions": 26211,
            "f05_correct_predictions": 24515,
            "additional_correct_predictions_needed": 1696,
            "reached": False,
        },
        "first_wave": [
            "R02", "R04", "R01", "R03", "S01", "S02",
            "K02", "K03", "K07", "V02", "L01", "H01",
        ],
        "budget": {
            "visual_training_points": 48,
            "cached_head_points": 12,
            "max_combinations": 8,
            "max_additional_confirmation_training_runs": 6,
        },
        "family_counts": {
            "R": 6, "S": 9, "K": 9, "V": 8, "L": 6,
            "Q": 6, "N": 4, "H": 12, "X": 8,
        },
        "thresholds": {
            "family_significant_macro_delta_pp": 0.3,
            "future_strong_candidate_macro_delta_pp_vs_f05": 1.0,
            "future_strong_candidate_micro_delta_pp_vs_f05": 0.5,
            "target70_priority_macro_delta_pp_vs_f05": 2.0,
            "target70_priority_micro_delta_pp_vs_f05": 1.5,
            "confirmation_seeds": [3407, 2026],
            "confirmation_mean_macro_delta_pp": 0.8,
            "confirmation_mean_micro_delta_pp": 0.4,
        },
        "trials": manifest_trials,
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifest_trials)} V5 conditional slots")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))
    main()
