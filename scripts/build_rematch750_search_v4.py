#!/usr/bin/env python3
"""Generate the REMATCH750_SEARCH_V4 trial manifest and per-trial configs.

The manifest is a research point list.  It is intentionally separate from the
training CLI: every training command is a concrete YAML file generated here.
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
CONFIG_DIR = ROOT / "configs/rematch750_search_v4"
BASE_CONFIG = ROOT / "configs/rematch750_v3_b1024_e16_lr4.yaml"
MANIFEST = ROOT / "search_manifest.json"
V3_CHECKPOINT = (
    "/workspace/noise-worktrees/rematch750_v3_scan/outputs/codex/"
    "rematch750_v3_tradeoff/RM_V3_B1024_E16_LR4/seed42/checkpoints/best.pt"
)
LP_CHECKPOINT = "/workspace/noise/outputs/rematch750_npu/RM_LP/seed42/checkpoints/best.pt"
V4_OUTPUT = "../outputs/rematch750_search_v4"


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


def rebase_relative_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Configs live one level below configs/, so rewrite repo-relative ../ paths."""
    path_fields = (
        ("data", "class_mapping"),
        ("data", "dataset_manifest"),
        ("data", "train_csv"),
        ("data", "val_csv"),
        ("data", "train_root"),
        ("data", "test_root"),
        ("features", "tensor_path"),
        ("features", "paths_path"),
        ("features", "manifest_path"),
        ("trust", "bundle_path"),
        ("train", "init_checkpoint"),
        ("output", "root"),
    )
    for section, key in path_fields:
        value = config.get(section, {}).get(key)
        if isinstance(value, str) and value.startswith("../"):
            config[section][key] = "../" + value
    return config


def training_overrides(
    *,
    loss: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    longtail: dict[str, Any] | None = None,
    elr: dict[str, Any] | None = None,
    train: dict[str, Any] | None = None,
    trust: dict[str, Any] | None = None,
) -> dict[str, Any]:
    override: dict[str, Any] = {}
    for key, value in (
        ("loss", loss),
        ("model", model),
        ("data", data),
        ("longtail", longtail),
        ("elr", elr),
        ("train", train),
        ("trust", trust),
    ):
        if value is not None:
            override[key] = value
    return override


def normalize_trial(
    trial_id: str,
    family: str,
    mechanism: str,
    overrides: dict[str, Any],
    *,
    parent_kind: str = "shared_lp",
    implementation_status: str = "implemented",
    dependencies: list[str] | None = None,
    notes: str = "",
    base_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = deep_merge(base_config or BASE_CONFIG_DATA, overrides)
    project = base.setdefault("project", {})
    project.update(
        protocol="rematch750_search_v4",
        trial_id=trial_id,
        family=family,
        mechanism=mechanism,
        parent_kind=parent_kind,
        implementation_status=implementation_status,
        search=dict(project.get("search", {})),
    )
    if dependencies:
        project["dependencies"] = list(dependencies)
    project["experiment_id"] = f"RM_V4_{trial_id}"
    base.setdefault("output", {})["root"] = V4_OUTPUT
    base.setdefault("train", {})["device"] = "npu:UNASSIGNED"
    if parent_kind == "shared_lp":
        base["train"]["init_checkpoint"] = LP_CHECKPOINT
    base = rebase_relative_paths(base)
    return base, {
        "trial_id": trial_id,
        "family": family,
        "kind": "training",
        "mechanism": mechanism,
        "parent_kind": parent_kind,
        "implementation_status": implementation_status,
        "dependencies": list(dependencies or []),
        "notes": notes,
    }


def head_config(
    base: dict[str, Any],
    *,
    trial_id: str,
    mechanism: str,
    loss_name: str,
    gce_q: float,
    sampler_mode: str,
    classifier_mode: str,
    derived_parent: bool,
    parent_kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = copy.deepcopy(base)
    cfg["project"].update(
        protocol="rematch750_search_v4",
        trial_id=trial_id,
        family="H",
        mechanism=mechanism,
        parent_kind=parent_kind,
        parent_experiment_id="RM_V3_B1024_E16_LR4",
        implementation_status="pending_v3_feature_cache",
        search={"reference_resolution": 224},
    )
    cfg["project"]["experiment_id"] = f"RM_V4_{trial_id}"
    cfg["output"]["root"] = V4_OUTPUT
    cfg["model"].update(
        peft_mode="frozen",
        use_cached_training=True,
        classifier_mode=classifier_mode,
        cosine_scale_init=20.0,
        cosine_scale_trainable=True,
    )
    cfg["features"] = {
        "tensor_path": "../outputs/rematch750_search_v4/assets/RM_V3_B1024_E16_LR4_features/features.pt",
        "paths_path": "../outputs/rematch750_search_v4/assets/RM_V3_B1024_E16_LR4_features/image_paths.json",
        "manifest_path": "../outputs/rematch750_search_v4/assets/RM_V3_B1024_E16_LR4_features/manifest.json",
    }
    cfg["loss"].update(
        name=loss_name,
        gce_q=float(gce_q),
        ce_warmup_epochs=0 if loss_name != "gce" else 2,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        feature_distillation_weight=0.0,
    )
    cfg["longtail"].update(
        sampler_mode=sampler_mode,
        loss_reweighting="none",
        balanced_softmax_tau=0.0,
    )
    cfg["train"].update(
        device="npu:UNASSIGNED",
        epochs=20,
        schedule_epochs=20,
        batch_size=1024,
        grad_accum_steps=1,
        head_lr=4.0e-4,
        backbone_lr=0.0,
        init_checkpoint=V3_CHECKPOINT if derived_parent else None,
        init_weights_mode="all" if derived_parent else "visual_only",
    )
    if not derived_parent:
        cfg["train"].pop("init_checkpoint", None)
    cfg = rebase_relative_paths(cfg)
    return cfg, {
        "trial_id": trial_id,
        "family": "H",
        "kind": "cached_head",
        "mechanism": mechanism,
        "parent_kind": parent_kind,
        "implementation_status": "pending_v3_feature_cache",
        "dependencies": ["V3_feature_cache"],
        "notes": "Frozen V3 encoder features; head-only 20-epoch refit.",
    }


# ---------------------------------------------------------------------------
# Trial recipes
# ---------------------------------------------------------------------------

def build_trials() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    base = BASE_CONFIG_DATA
    trials: list[tuple[dict[str, Any], dict[str, Any]]] = []

    # A: augmentation and mixed supervision.
    for idx, (loss_name, alpha) in enumerate(
        (("gce", 0.2), ("gce", 0.4), ("gce", 0.8),
         ("cross_entropy", 0.2), ("cross_entropy", 0.4), ("cross_entropy", 0.8)),
        start=1,
    ):
        trial_id = f"A{idx:02d}"
        label = "GCE" if loss_name == "gce" else "soft_CE"
        overrides = training_overrides(
            loss={
                "name": loss_name,
                "gce_q": 0.5,
                "mixup_alpha": alpha,
                "mixup_probability": 0.5,
                "ce_warmup_epochs": 2,
                "feature_distillation_weight": 2.0,
            }
        )
        trials.append(normalize_trial(trial_id, "A", f"{label}_mixup_a{alpha}", overrides))

    for idx, magnitude in ((7, 5), (8, 9)):
        trial_id = f"A{idx:02d}"
        overrides = training_overrides(
            loss={"name": "gce", "gce_q": 0.5, "mixup_alpha": 0.0,
                  "mixup_probability": 0.0, "ce_warmup_epochs": 2,
                  "feature_distillation_weight": 2.0},
            data={"train_augmentation": "weak_rrc_flip_randaugment",
                  "randaugment_num_ops": 2, "randaugment_magnitude": magnitude},
        )
        trials.append(normalize_trial(trial_id, "A", f"GCE_randaug_n2_m{magnitude}", overrides))

    trials.append(normalize_trial(
        "A09", "A", "GCE_cutmix_a1_p05",
        training_overrides(loss={
            "name": "gce", "gce_q": 0.5, "cutmix_alpha": 1.0,
            "cutmix_probability": 0.5, "mixup_alpha": 0.0,
            "mixup_probability": 0.0, "ce_warmup_epochs": 2,
            "feature_distillation_weight": 2.0,
        }),
    ))
    trials.append(normalize_trial(
        "A10", "A", "GCE_mixup_a04_randaug_n2_m5",
        training_overrides(
            loss={"name": "gce", "gce_q": 0.5, "mixup_alpha": 0.4,
                  "mixup_probability": 0.5, "ce_warmup_epochs": 2,
                  "feature_distillation_weight": 2.0},
            data={"train_augmentation": "weak_rrc_flip_randaugment",
                  "randaugment_num_ops": 2, "randaugment_magnitude": 5},
        ),
    ))

    # B: noise-robust objectives.
    b_specs = [
        ("B01", "CE_control", {"name": "cross_entropy", "ce_warmup_epochs": 0}),
        ("B02", "CE_label_smoothing_005", {"name": "cross_entropy", "label_smoothing": 0.05, "ce_warmup_epochs": 2}),
        ("B03", "CE_label_smoothing_010", {"name": "cross_entropy", "label_smoothing": 0.10, "ce_warmup_epochs": 2}),
        ("B04", "GCE_q03", {"name": "gce", "gce_q": 0.3, "ce_warmup_epochs": 2}),
        ("B05", "GCE_q07", {"name": "gce", "gce_q": 0.7, "ce_warmup_epochs": 2}),
        ("B06", "SCE_ce01_rce1", {"name": "sce", "sce_ce_weight": 0.1, "sce_rce_weight": 1.0,
                                  "sce_min_probability": 1.0e-4, "ce_warmup_epochs": 2}),
        ("B07", "SCE_ce05_rce1", {"name": "sce", "sce_ce_weight": 0.5, "sce_rce_weight": 1.0,
                                  "sce_min_probability": 1.0e-4, "ce_warmup_epochs": 2}),
    ]
    for trial_id, mechanism, loss in b_specs:
        trials.append(normalize_trial(
            trial_id, "B", mechanism,
            training_overrides(loss={**loss, "mixup_alpha": 0.0,
                                     "mixup_probability": 0.0,
                                     "feature_distillation_weight": 2.0}),
        ))
    for trial_id, weight in (("B08", 1.0), ("B09", 3.0), ("B10", 5.0)):
        trials.append(normalize_trial(
            trial_id, "B", f"ELR_lambda_{weight:g}",
            training_overrides(
                loss={"name": "cross_entropy", "ce_warmup_epochs": 2,
                      "mixup_alpha": 0.0, "mixup_probability": 0.0,
                      "feature_distillation_weight": 2.0},
                elr={"enabled": True, "momentum": 0.7,
                     "target_weight": weight, "warmup_epochs": 2, "ramp_epochs": 0},
            ),
        ))

    # C: supervised repair / training-pool self-training.  The quality asset is
    # deliberately a separate prerequisite; configs are generated but fail
    # closed until that asset exists.
    c_index = 0
    for rho in (0.2, 0.4):
        for gate in (0.8, 0.9):
            c_index += 1
            trial_id = f"C{c_index:02d}"
            trials.append(normalize_trial(
                trial_id, "C", f"soft_repair_rho{rho}_gate{gate}",
                training_overrides(
                    loss={"name": "gce", "gce_q": 0.5, "ce_warmup_epochs": 2,
                          "mixup_alpha": 0.0, "mixup_probability": 0.0},
                    trust={"enabled": True, "bundle_path": "../outputs/rematch750_search_v4/assets/quality_asset/trust_bundle.pt",
                           "selection_threshold": gate, "minimum_sample_weight": 1.0},
                    train={"quality_asset_mode": "soft_repair", "repair_rho": rho,
                           "repair_confidence_gate": gate, "max_relabel_fraction": 0.2},
                ),
                implementation_status="pending_quality_asset",
                dependencies=["quality_asset_v1"],
            ))
    for gate in (0.9, 0.95):
        for fraction in (0.05, 0.10):
            c_index += 1
            trial_id = f"C{c_index:02d}"
            trials.append(normalize_trial(
                trial_id, "C", f"hard_relabel_gate{gate}_frac{fraction:g}",
                training_overrides(
                    loss={"name": "gce", "gce_q": 0.5, "ce_warmup_epochs": 2,
                          "mixup_alpha": 0.0, "mixup_probability": 0.0},
                    trust={"enabled": True, "bundle_path": "../outputs/rematch750_search_v4/assets/quality_asset/trust_bundle.pt",
                           "selection_threshold": gate, "minimum_sample_weight": 1.0},
                    train={"quality_asset_mode": "hard_relabel", "repair_confidence_gate": gate,
                           "max_relabel_fraction": fraction},
                ),
                implementation_status="pending_quality_asset",
                dependencies=["quality_asset_v1"],
            ))
    for fraction in (0.10, 0.20):
        for consistency in (0.5, 1.0):
            c_index += 1
            trial_id = f"C{c_index:02d}"
            trials.append(normalize_trial(
                trial_id, "C", f"unlabel_frac{fraction:g}_consistency{consistency:g}",
                training_overrides(
                    loss={"name": "gce", "gce_q": 0.5, "ce_warmup_epochs": 2,
                          "mixup_alpha": 0.0, "mixup_probability": 0.0},
                    trust={"enabled": True, "bundle_path": "../outputs/rematch750_search_v4/assets/quality_asset/trust_bundle.pt",
                           "selection_threshold": 0.9, "minimum_sample_weight": 1.0},
                    train={"quality_asset_mode": "unlabel_consistency",
                           "max_relabel_fraction": fraction,
                           "consistency_weight": consistency,
                           "pseudo_confidence_gate": 0.9,
                           "minimum_supervised_fraction": 0.8},
                ),
                implementation_status="pending_quality_asset",
                dependencies=["quality_asset_v1"],
            ))

    # D: representation protection and unfreeze structure.
    anchor_specs = [("D01", 0.0), ("D02", 0.5), ("D03", 1.0), ("D04", 4.0), ("D05", 8.0)]
    for trial_id, weight in anchor_specs:
        trials.append(normalize_trial(
            trial_id, "D", f"feature_anchor_{weight:g}",
            training_overrides(loss={
                "name": "gce", "gce_q": 0.5, "ce_warmup_epochs": 2,
                "mixup_alpha": 0.0, "mixup_probability": 0.0,
                "feature_distillation_weight": weight,
            }),
        ))
    for idx, blocks in ((6, 2), (7, 4), (8, 8)):
        trial_id = f"D{idx:02d}"
        trials.append(normalize_trial(
            trial_id, "D", f"unfreeze_last_{blocks}_blocks",
            training_overrides(
                model={"peft_mode": "full_finetune", "unfreeze_last_n_blocks": blocks},
                loss={"name": "gce", "gce_q": 0.5, "ce_warmup_epochs": 2,
                      "mixup_alpha": 0.0, "mixup_probability": 0.0,
                      "feature_distillation_weight": 2.0},
            ),
        ))
    for idx, decay in ((9, 0.65), (10, 0.80)):
        trial_id = f"D{idx:02d}"
        trials.append(normalize_trial(
            trial_id, "D", f"visual_layer_decay_{decay:g}",
            training_overrides(
                model={"peft_mode": "full_finetune", "unfreeze_last_n_blocks": 0},
                train={"visual_layer_decay": decay},
                loss={"name": "gce", "gce_q": 0.5, "ce_warmup_epochs": 2,
                      "mixup_alpha": 0.0, "mixup_probability": 0.0,
                      "feature_distillation_weight": 2.0},
            ),
        ))

    # E: long-tail classification and late balancing.
    for idx, tau in ((1, 0.5), (2, 1.0)):
        trial_id = f"E{idx:02d}"
        trials.append(normalize_trial(
            trial_id, "E", f"balanced_softmax_tau_{tau:g}",
            training_overrides(
                longtail={"balanced_softmax_tau": tau, "sampler_mode": "none",
                          "loss_reweighting": "none"},
                loss={"name": "gce", "gce_q": 0.5, "ce_warmup_epochs": 2,
                      "mixup_alpha": 0.0, "mixup_probability": 0.0,
                      "feature_distillation_weight": 2.0},
            ),
        ))
    for idx, alpha in ((3, 0.25), (4, 0.5)):
        trial_id = f"E{idx:02d}"
        trials.append(normalize_trial(
            trial_id, "E", f"late_sampler_power_{alpha:g}",
            training_overrides(
                longtail={"balanced_softmax_tau": 0.0, "sampler_mode": "none",
                          "loss_reweighting": "none",
                          "late_sampler_mode": "class_power",
                          "late_sampler_start_epoch": 13,
                          "late_sampler_alpha": alpha,
                          "late_sampler_num_samples": 133815},
                loss={"name": "gce", "gce_q": 0.5, "ce_warmup_epochs": 2,
                      "mixup_alpha": 0.0, "mixup_probability": 0.0,
                      "feature_distillation_weight": 2.0},
            ),
        ))
    for idx, alpha in ((5, 0.25), (6, 0.5)):
        trial_id = f"E{idx:02d}"
        trials.append(normalize_trial(
            trial_id, "E", f"late_reweight_clipped_power_{alpha:g}",
            training_overrides(
                longtail={"balanced_softmax_tau": 0.0, "sampler_mode": "none",
                          "loss_reweighting": "none",
                          "late_reweight_mode": "class_power_clipped",
                          "late_reweight_start_epoch": 13,
                          "late_reweight_alpha": alpha,
                          "late_reweight_clip_min": 0.5,
                          "late_reweight_clip_max": 2.0},
                loss={"name": "gce", "gce_q": 0.5, "ce_warmup_epochs": 2,
                      "mixup_alpha": 0.0, "mixup_probability": 0.0,
                      "feature_distillation_weight": 2.0},
            ),
        ))

    # F: higher-resolution full fine-tuning.  `batch_size` is the microbatch;
    # `grad_accum_steps` restores the fixed effective batch of 1024.
    f_specs = [
        ("F01", 256, "weak_rrc_flip", 512, 2),
        ("F02", 256, "A_best", 512, 2),
        ("F03", 288, "weak_rrc_flip", 512, 2),
        ("F04", 288, "A_best", 512, 2),
        ("F05", 320, "weak_rrc_flip", 256, 4),
        ("F06", 320, "A_best", 256, 4),
    ]
    for trial_id, resolution, augmentation, microbatch, accum in f_specs:
        aug_name = "weak_rrc_flip" if augmentation == "weak_rrc_flip" else "weak_rrc_flip"
        overrides = training_overrides(
            model={"input_resolution": resolution},
            data={"train_augmentation": aug_name},
            train={"batch_size": microbatch, "grad_accum_steps": accum,
                   "effective_batch_size": 1024},
            loss={"name": "gce", "gce_q": 0.5, "ce_warmup_epochs": 2,
                  "mixup_alpha": 0.0, "mixup_probability": 0.0,
                  "feature_distillation_weight": 2.0},
        )
        dependencies = ["A_best"] if augmentation == "A_best" else []
        impl = "implemented"
        trials.append(normalize_trial(
            trial_id, "F", f"{resolution}px_{augmentation}", overrides,
            implementation_status=impl, dependencies=dependencies,
            notes="Gradient accumulation verified; A_best placeholder is replaced after A group is complete.",
        ))

    # G: standard non-adaptive SAM.
    g_specs = [
        ("G01", 0.025, "gce"), ("G02", 0.05, "gce"),
        ("G03", 0.025, "A_best"), ("G04", 0.05, "A_best"),
    ]
    for trial_id, rho, recipe in g_specs:
        loss_name = "gce" if recipe == "gce" else "cross_entropy"
        overrides = training_overrides(
            loss={"name": loss_name, "gce_q": 0.5, "ce_warmup_epochs": 2,
                  "mixup_alpha": 0.0, "mixup_probability": 0.0,
                  "feature_distillation_weight": 2.0},
            train={"sam": {"enabled": True, "mode": "standard_global_l2", "rho": rho}},
        )
        dependencies = ["A_best"] if recipe == "A_best" else []
        trials.append(normalize_trial(
            trial_id, "G", f"SAM_rho{rho:g}_{recipe}", overrides,
            implementation_status="pending_sam_integration",
            dependencies=dependencies,
        ))

    # H: 24 frozen-V3 feature head points.
    h_index = 0
    for classifier_mode, classifier_tag in (("linear", "linear"), ("cosine", "cosine")):
        for sampler_mode, sampler_tag in (
            ("none", "natural"),
            ("sqrt_class_balanced", "sqrt_bal"),
            ("class_balanced", "class_bal"),
        ):
            for loss_name, loss_tag in (("cross_entropy", "ce"), ("gce", "gce05")):
                for derived_parent, init_tag in ((False, "reinit"), (True, "parent")):
                    h_index += 1
                    trial_id = f"H{h_index:02d}"
                    mechanism = (
                        f"{classifier_tag}_{sampler_tag}_{loss_tag}_{init_tag}"
                    )
                    cfg, meta = head_config(
                        base,
                        trial_id=trial_id,
                        mechanism=mechanism,
                        loss_name=loss_name,
                        gce_q=0.5,
                        sampler_mode=sampler_mode,
                        classifier_mode=classifier_mode,
                        derived_parent=derived_parent,
                        parent_kind="frozen_backbone_head",
                    )
                    trials.append((cfg, meta))

    if len(trials) != 82:
        raise AssertionError(f"expected 82 trial points, generated {len(trials)}")
    return trials


def main() -> None:
    global BASE_CONFIG_DATA
    BASE_CONFIG_DATA = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    trials = build_trials()

    manifest_trials = []
    for cfg, meta in trials:
        # Validate the config without requiring assets on the current host.
        from aegis_clip.config import validate_config

        validate_config(copy.deepcopy(cfg))
        path = CONFIG_DIR / f"{meta['trial_id']}.yaml"
        text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        path.write_text(text, encoding="utf-8")
        meta = dict(meta)
        meta.update(
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
        manifest_trials.append(meta)

    manifest = {
        "schema_version": 1,
        "protocol": "rematch750_search_v4",
        "name": "REMATCH750_SEARCH_V4_LOCAL_FIRST_65",
        "status": "planned_not_executed",
        "baseline": {
            "candidate": "RM_V3_B1024_E16_LR4",
            "config": "configs/rematch750_v3_b1024_e16_lr4.yaml",
            "config_sha256": "db802b635a908204bbb1c9e946a0e6e2648a4f24dc9a42a1a61ffd1c4ba3c0a5",
            "checkpoint_sha256": "a8de990eb0fd347e52dd5387d6cd986a30268ed316111dce6ce8d9e4c42c2f27",
            "local_macro": 0.7234723567962646,
            "local_micro": 0.7341398000717163,
            "platform_score": None,
        },
        "first_wave": ["A01", "A02", "A04", "A07", "B02", "B08", "D02", "D08", "E01", "E03", "C01", "F01"],
        "budget": {
            "training_method_points": 58,
            "cached_head_points": 24,
            "max_combinations": 8,
            "max_additional_confirmation_training_runs": 11,
        },
        "thresholds": {
            "local_significant_macro_delta_pp": 2.0,
            "local_significant_micro_delta_pp": 1.0,
            "local_platform_candidate_macro_delta_pp": 3.0,
            "local_platform_candidate_micro_delta_pp": 2.0,
            "local_platform_candidate_macro_floor": 0.753472,
            "local_platform_candidate_micro_floor": 0.754140,
        },
        "trials": manifest_trials,
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifest_trials)} trial configs to {CONFIG_DIR}")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))
    main()
