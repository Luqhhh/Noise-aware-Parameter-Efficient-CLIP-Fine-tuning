#!/usr/bin/env python3
"""Generate the deterministic F05 focus bundle.

The bundle keeps exactly seven first-stage units: C0, N1, A0, P0, D1, D2 and
D3.  C0/N1 are ordinary Aegis training configs.  A0/P0/D1/D2 are lightweight
descriptors consumed by the focus probes and queue (they intentionally do not
pretend to be full training configs).  ``--check`` verifies byte-for-byte
equality with the committed files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "f05_focus"
BASE_CONFIG = ROOT / "configs" / "rematch750_search_v4" / "F05.yaml"
F05_MACRO = 0.7443073987960815
F05_MICRO = 0.7544354796409607
F05_PLATFORM = 65.4711


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict):
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise TypeError(f"cannot merge mapping into {key}")
            _deep_update(child, value)
        else:
            target[key] = value


def _base_training_config() -> dict[str, Any]:
    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(BASE_CONFIG)
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("F05 base config is not a mapping")
    return config


def _yaml_bytes(config: dict[str, Any]) -> bytes:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode("utf-8")


def _descriptor(
    *,
    experiment_id: str,
    direction: str,
    stage: str,
    cost: str,
    purpose: str,
    mechanism: dict[str, Any],
    audit: dict[str, Any] | None = None,
    promotion: dict[str, Any] | None = None,
    references: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "direction": direction,
        "stage": stage,
        "cost": cost,
        "purpose": purpose,
        "parent": {
            "kind": "f05_checkpoint",
            "checkpoint_env": "F05_CHECKPOINT",
        },
        "mechanism": mechanism,
        "promotion": {
            "macro_delta_pp_min": 0.30,
            "micro_delta_pp_min": -0.10,
            "strong_macro_delta_pp": 0.50,
            "breakthrough_macro_delta_pp": 1.00,
        },
        "platform": {"submission_allowed": False, "test_tuning_allowed": False},
    }
    if audit:
        payload["audit"] = audit
    if promotion is not None:
        payload["promotion"] = promotion
    if references:
        payload["references"] = references
    return payload


def _render_training_configs() -> dict[Path, bytes]:
    rendered: dict[Path, bytes] = {}

    common_project = {
        "data_version": "20260921",
        "seed": 42,
        "stage": "repechage",
        "protocol": "rematch750_search_v4",
        "implementation_status": "implemented",
        "parent_kind": "shared_lp",
        "parent_experiment_id": "RM_V4_F05",
    }
    c0 = copy.deepcopy(_base_training_config())
    c0["project"] = {
        **common_project,
        "experiment_id": "C0_F05_CUDA",
        "trial_id": "C0",
        "family": "C",
        "mechanism": "cuda_f05_control",
        "search": {"focus_stage": "round1", "paired_baseline": "NPU_F05"},
    }
    c0["output"]["root"] = "../../outputs/f05_focus"
    c0["train"].update(
        {
            "device": "cuda:0",
            "optimizer_impl": "foreach",
            "gradient_norm_impl": "foreach",
            "npu_pin_memory": False,
        }
    )
    rendered[CONFIG_DIR / "C0_cuda_control.yaml"] = _yaml_bytes(c0)

    n1 = copy.deepcopy(c0)
    n1["project"].update(
        {
            "experiment_id": "N1_HP_DROP",
            "trial_id": "N1",
            "family": "N",
            "mechanism": "high_precision_noise_drop",
            "search": {
                "focus_stage": "round1",
                "paired_control": "C0_F05_CUDA",
                "manifest": "artifacts/f05_focus/hp_noise_manifest.csv",
            },
        }
    )
    n1["trust"] = {
        "enabled": False,
        "sample_weight_path": "../../artifacts/f05_focus/hp_noise_manifest.csv",
    }
    rendered[CONFIG_DIR / "N1_hp_noise_drop.yaml"] = _yaml_bytes(n1)
    return rendered


def _render_descriptors() -> dict[Path, bytes]:
    rendered: dict[Path, bytes] = {}
    promotion = {
        "macro_delta_pp_min": 0.30,
        "micro_delta_pp_min": -0.10,
        "strong_macro_delta_pp": 0.50,
        "breakthrough_macro_delta_pp": 1.00,
    }

    rendered[CONFIG_DIR / "A0_m1.yaml"] = _yaml_bytes(
        _descriptor(
            experiment_id="A0_M1",
            direction="attention-local",
            stage="round0",
            cost="low",
            purpose="training-free migration of F05 global + attention local fusion",
            mechanism={
                "name": "attention_guided_global_local_probability_fusion",
                "input_resolution": 320,
                "crop_size": 224,
                "top_k": 5,
                "fusions": [
                    {"global_weight": 0.50, "local_weight": 0.50, "tag": "A0-50"},
                    {"global_weight": 0.60, "local_weight": 0.40, "tag": "A0-40"},
                ],
                "flip": False,
                "reuse_existing_checkpoint": True,
            },
            audit={
                "native_global_vs_attention_pipeline_global": {
                    "require_prediction_agreement": 1.0,
                    "require_max_abs_logit_diff": 0.0,
                    "acceptance_tolerance": 1.0e-5,
                }
            },
            promotion=promotion,
            references={
                "preliminary_m1_raw_gain_pp": 0.85,
                "frozen_local_view": "320px -> crop224 -> top5",
            },
        )
    )
    rendered[CONFIG_DIR / "P0_prior.yaml"] = _yaml_bytes(
        _descriptor(
            experiment_id="P0_PRIOR",
            direction="prior",
            stage="round0",
            cost="very_low",
            purpose="compliant validation-fitted uniform-prior calibration",
            mechanism={
                "name": "validation_fitted_ipf_prior_bias",
                "target_prior": "uniform",
                "strengths": [0.25, 0.50, 0.75, 1.00],
                "selection": "validation_only",
                "test_soft_marginal_fitting": False,
                "apply_frozen_bias": True,
            },
            audit={
                "fitted_on": "current_stage_validation",
                "test_data_used_for_fit_or_selection": False,
            },
            promotion=promotion,
            references={
                "preliminary_prior_platform_gain_pp_range": [1.8, 3.5],
                "legacy_forbidden_api": "align_logits_to_prior(test_logits)",
            },
        )
    )
    adapter_common = {
        "frozen_parent": "F05_CHECKPOINT",
        "train_subset": "artifacts/f05_focus/clean070.csv",
        "frozen_clean_probability_threshold": 0.70,
        "local_geometry": {"input_resolution": 320, "crop_size": 160, "top_k": 5},
        "backbone_frozen": True,
        "combine_rule": "additive feature residual around native F05 local features",
    }
    rendered[CONFIG_DIR / "D1_o3.yaml"] = _yaml_bytes(
        _descriptor(
            experiment_id="D1_O3",
            direction="dual-adapter",
            stage="round2",
            cost="medium_low",
            purpose="validate local feature adapter on F05",
            mechanism={
                **adapter_common,
                "adapter": "BottleneckLocalFeatureAdapter",
                "feature_dim": 512,
                "bottleneck_dim": 32,
                "residual_scale": 0.25,
                "dropout": 0.10,
            },
            audit={
                "epoch_zero_must_equal_f05_m1": True,
                "global_path_frozen": True,
                "same_subset_as_d2": True,
            },
            promotion=promotion,
        )
    )
    rendered[CONFIG_DIR / "D2_pta.yaml"] = _yaml_bytes(
        _descriptor(
            experiment_id="D2_PTA",
            direction="dual-adapter",
            stage="round2",
            cost="medium_low",
            purpose="validate part-token adapter on F05",
            mechanism={
                **adapter_common,
                "adapter": "PartTokenResidualAdapter",
                "feature_dim": 512,
                "bottleneck_dim": 64,
                "residual_scale": 0.25,
                "dropout": 0.10,
                "part_top_patches": 8,
                "part_temperature": 0.07,
            },
            audit={
                "epoch_zero_must_equal_f05_m1": True,
                "global_path_frozen": True,
                "same_subset_as_d1": True,
                "part_top_patches_frozen": 8,
            },
            promotion=promotion,
        )
    )
    return rendered


def _render_manifest(rendered: dict[Path, bytes]) -> bytes:
    config_names = {
        "C0_cuda_control.yaml": "C0",
        "N1_hp_noise_drop.yaml": "N1",
        "A0_m1.yaml": "A0",
        "P0_prior.yaml": "P0",
        "D1_o3.yaml": "D1",
        "D2_pta.yaml": "D2",
    }
    config_hashes = {
        name: _sha256(rendered[CONFIG_DIR / name]) for name in config_names
    }
    manifest = {
        "schema_version": 1,
        "protocol": "F05_FOCUS_SEVEN_UNIT_STAGE1",
        "status": "executed_through_round_2_no_promotion",
        "baseline": {
            "experiment_id": "RM_V4_F05",
            "macro": F05_MACRO,
            "micro": F05_MICRO,
            "platform_score_percent": F05_PLATFORM,
        },
        "fixed_recipe": {
            "input_resolution": 320,
            "epochs": 16,
            "effective_batch_size": 1024,
            "head_lr": 4.0e-4,
            "backbone_lr": 1.2e-5,
            "weight_decay": 1.0e-4,
            "loss": "ce_warmup_2_then_gce_q0.5",
            "feature_distillation_weight": 2.0,
            "seed": 42,
            "selection": "raw_macro_then_raw_micro",
        },
        "promotion": {
            "fail": "macro_delta_pp <= 0",
            "weak": "0 < macro_delta_pp < 0.30",
            "promote": "macro_delta_pp >= 0.30 and micro_delta_pp >= -0.10",
            "strong": "macro_delta_pp >= 0.50 and micro_delta_pp >= 0",
            "breakthrough": "macro_delta_pp >= 1.00",
            "seed_confirmation": "2 of 3 seeds same direction after seed42 strong",
        },
        # Registered amendment 2026-09-25. The previous condition
        # `prediction_empty_classes == 0` was logically unsatisfiable: the
        # F05 + same-local-view baseline used as the candidate reference already
        # leaves 4 classes empty, so even the identity candidate was rejected.
        # See the preregistration for the full argument and the disclosure that
        # the round-2 results had already been observed when this was written.
        "safety_gate": {
            "registered_amendment": "2026-09-25",
            "preregistration": "docs/f05_focus_round2_safety_gate_preregistration_20260925.md",
            "baseline": "adapter=None evaluated on the same validation cache (F05 + same local view)",
            "per_epoch_eligibility": [
                "finite metrics",
                "prediction_empty_classes(candidate) <= prediction_empty_classes(baseline)",
                "trusted_macro(candidate) >= trusted_macro(baseline)",
                "raw_micro(candidate) >= raw_micro(baseline) - 0.001",
                "local_feature_drift(candidate) <= 0.01",
            ],
            "final_gate": [
                "clean_core_micro_delta_pp >= 0.20",
                "trusted_macro_delta_pp >= 0.0",
                "raw_micro_delta_pp >= -0.10",
                "local_feature_drift <= 0.01",
                "prediction_empty_classes(candidate) <= prediction_empty_classes(baseline)",
                "reference audits pass (center bit-exact, M1 fusion <= 4e-6, epoch zero bit-exact)",
            ],
            "superseded_condition": "prediction_empty_classes == 0",
        },
        "units": [
            {
                "id": "C0",
                "direction": "CUDA control",
                "change": "rerun F05 on CUDA",
                "cost": "high",
                "round": 1,
                "config": "configs/f05_focus/C0_cuda_control.yaml",
                "config_sha256": config_hashes["C0_cuda_control.yaml"],
                "kind": "train",
            },
            {
                "id": "N1",
                "direction": "high-precision noise drop",
                "change": "F05 recipe + strict CL+kNN reject",
                "cost": "high",
                "round": 1,
                "config": "configs/f05_focus/N1_hp_noise_drop.yaml",
                "config_sha256": config_hashes["N1_hp_noise_drop.yaml"],
                "kind": "train",
                "paired_control": "C0",
            },
            {
                "id": "A0",
                "direction": "attention-local",
                "change": "F05 checkpoint + M1",
                "cost": "low",
                "round": 0,
                "config": "configs/f05_focus/A0_m1.yaml",
                "config_sha256": config_hashes["A0_m1.yaml"],
                "kind": "probe",
            },
            {
                "id": "P0",
                "direction": "prior",
                "change": "F05 + validation-fitted prior",
                "cost": "very_low",
                "round": 0,
                "config": "configs/f05_focus/P0_prior.yaml",
                "config_sha256": config_hashes["P0_prior.yaml"],
                "kind": "probe",
            },
            {
                "id": "D1",
                "direction": "Dual Adapter",
                "change": "F05 + O3",
                "cost": "medium_low",
                "round": 2,
                "config": "configs/f05_focus/D1_o3.yaml",
                "config_sha256": config_hashes["D1_o3.yaml"],
                "kind": "adapter",
            },
            {
                "id": "D2",
                "direction": "Dual Adapter",
                "change": "F05 + PTA",
                "cost": "medium_low",
                "round": 2,
                "config": "configs/f05_focus/D2_pta.yaml",
                "config_sha256": config_hashes["D2_pta.yaml"],
                "kind": "adapter",
            },
            {
                "id": "D3",
                "direction": "Dual Adapter",
                "change": "F05 + O3 + PTA",
                "cost": "low",
                "round": 3,
                "config": None,
                "config_sha256": None,
                "kind": "compose",
                "depends_on": ["D1", "D2"],
            },
        ],
        "outputs": {
            "C0": "outputs/f05_focus/C0_F05_CUDA",
            "N1": "outputs/f05_focus/N1_HP_DROP",
            "A0": "outputs/f05_focus/A0_M1",
            "P0": "outputs/f05_focus/P0_PRIOR",
            "D1": "outputs/f05_focus/D1_O3",
            "D2": "outputs/f05_focus/D2_PTA",
            "D3": "outputs/f05_focus/D3_DUAL",
        },
        "artifacts": {
            "hp_noise_manifest": "artifacts/f05_focus/hp_noise_manifest.csv",
            "clean070": "artifacts/f05_focus/clean070.csv",
            "prior_bias": "artifacts/f05_focus/prior_bias.pt",
            "f05_val_logits": "artifacts/f05_focus/f05_val_logits.pt",
            "f05_attention_cache": "artifacts/f05_focus/f05_attention_cache.pt",
        },
        "execution_queue": [
            {"round": 0, "gpu0": "A0", "gpu1": "P0"},
            {"round": 1, "gpu0": "C0", "gpu1": "N1"},
            # D2 has no GPU1 runner or artifacts; it was executed on GPU0 after
            # D1 released the card (deploy/f05_focus_gpu0/run_d2_adapter.sh).
            {"round": 2, "gpu0": "D1, D2", "gpu1": "none (D2 moved to GPU0)"},
            {"round": 3, "gpu0": "D3", "gpu1": "strongest second seed"},
            {"round": 4, "gpu0": "strongest seed3407", "gpu1": "strongest seed2026"},
        ],
        "forbidden": [
            "LR/WD/SAM/warmup/GCE-q/epoch sweeps before seven units finish",
            "DDP across the two GPUs",
            "changing effective batch size",
            "N x A x D x P Cartesian combinations",
            "test-batch prior fitting",
        ],
    }
    return json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")


def render_bundle() -> dict[Path, bytes]:
    rendered = _render_training_configs()
    rendered.update(_render_descriptors())
    rendered[CONFIG_DIR / "manifest.json"] = _render_manifest(rendered)
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed bundle byte-for-byte instead of writing it",
    )
    args = parser.parse_args()
    rendered = render_bundle()
    failures: list[str] = []
    for path, payload in rendered.items():
        if args.check:
            if not path.is_file() or path.read_bytes() != payload:
                failures.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    if failures:
        print("bundle differs:", ", ".join(failures), file=sys.stderr)
        return 1
    print(json.dumps({"files": len(rendered), "check": bool(args.check)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
