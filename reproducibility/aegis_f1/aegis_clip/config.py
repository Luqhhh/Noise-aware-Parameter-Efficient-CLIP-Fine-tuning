"""Strict configuration loading and cross-field validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is ambiguous or internally inconsistent."""


TOP_LEVEL_KEYS = {
    "project",
    "data",
    "features",
    "model",
    "trust",
    "elr",
    "loss",
    "train",
    "evaluation",
    "output",
    "lineage",
    "promotion",
    "clean_routing",
    "prototype_contrastive",
    "dynamic_trust",
}

REQUIRED_SECTIONS = {
    "project",
    "data",
    "features",
    "model",
    "trust",
    "loss",
    "train",
    "evaluation",
    "output",
}

PEFT_MODES = {
    "frozen",
    "feature_adapter",
    "visual_ln",
    "ln_post_proj",
    "visual_lora",
    "visual_mlp_adapter",
    "visual_prompt",
}
CLASSIFIER_MODES = {"linear", "anchored_residual"}
LOSS_NAMES = {"cross_entropy", "double_softmax_cross_entropy", "gce"}
SELECTOR_METRICS = {
    "raw_micro",
    "raw_macro",
    "trusted_micro",
    "trusted_macro",
    "proxy_micro",
    "proxy_macro",
    "clean_core_micro",
    "clean_core_macro",
}
COMPETITION_STAGES = {"preliminary", "repechage", "semifinal"}
# Team Phase 4 configs use an internal ablation label while retaining the same
# competition-compliance checks for data, backbone and test usage below.
INTERNAL_EXPERIMENT_STAGES = {"p4_ablation"}
PROJECT_STAGES = COMPETITION_STAGES | INTERNAL_EXPERIMENT_STAGES


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, validate it, and resolve paths relative to its file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a mapping")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path)
    _resolve_paths(config, config_path.parent)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail closed on unknown sections and invalid cross-field combinations."""
    public_keys = {key for key in config if not key.startswith("_")}
    unknown = public_keys - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"Unknown top-level sections: {sorted(unknown)}")
    missing = REQUIRED_SECTIONS - public_keys
    if missing:
        raise ConfigError(f"Missing required sections: {sorted(missing)}")

    project = config["project"]
    data = config["data"]
    model = config["model"]
    trust = config["trust"]
    loss = config["loss"]
    train = config["train"]
    evaluation = config["evaluation"]

    if not str(project.get("experiment_id", "")).strip():
        raise ConfigError("project.experiment_id must be non-empty")
    if project.get("stage") not in PROJECT_STAGES:
        raise ConfigError(
            f"project.stage must be one of {sorted(PROJECT_STAGES)}"
        )
    if data.get("external_data") is not False:
        raise ConfigError("data.external_data must be false for competition compliance")
    if data.get("test_usage") != "inference_only":
        raise ConfigError("data.test_usage must be 'inference_only'")
    if int(data.get("expected_official_train_samples", 0)) <= 0:
        raise ConfigError("data.expected_official_train_samples must be positive")
    if int(data.get("expected_test_samples", 0)) <= 0:
        raise ConfigError("data.expected_test_samples must be positive")
    if data.get("train_augmentation", "clip_center_crop") not in {
        "clip_center_crop",
        "weak_rrc_flip",
    }:
        raise ConfigError(
            "data.train_augmentation must be clip_center_crop or weak_rrc_flip"
        )
    if model.get("backbone") != "ViT-B/32":
        raise ConfigError("Only OpenAI CLIP ViT-B/32 is competition-compliant")
    if model.get("pretrained") != "openai":
        raise ConfigError("model.pretrained must be 'openai'")
    if model.get("peft_mode", "frozen") not in PEFT_MODES:
        raise ConfigError(f"model.peft_mode must be one of {sorted(PEFT_MODES)}")
    classifier_mode = model.get("classifier_mode", "linear")
    if classifier_mode not in CLASSIFIER_MODES:
        raise ConfigError(
            f"model.classifier_mode must be one of {sorted(CLASSIFIER_MODES)}"
        )
    if classifier_mode == "anchored_residual":
        residual_scale = float(model.get("classifier_residual_scale", 0.25))
        if not 0.0 < residual_scale <= 1.0:
            raise ConfigError(
                "model.classifier_residual_scale must be in (0, 1]"
            )
        if not train.get("init_checkpoint"):
            raise ConfigError(
                "anchored_residual requires train.init_checkpoint for its frozen base"
            )
    if int(model.get("num_classes", 0)) <= 1:
        raise ConfigError("model.num_classes must be greater than one")
    input_resolution = int(model.get("input_resolution", 224))
    if not 224 <= input_resolution <= 448 or input_resolution % 32 != 0:
        raise ConfigError(
            "model.input_resolution must be a multiple of 32 in [224,448]"
        )

    if loss.get("name") not in LOSS_NAMES:
        raise ConfigError(f"loss.name must be one of {sorted(LOSS_NAMES)}")
    q = float(loss.get("gce_q", 0.5))
    if loss.get("name") == "gce" and not 0.0 < q <= 1.0:
        raise ConfigError("loss.gce_q must be in (0, 1]")
    if float(loss.get("class_prior_adjustment_tau", 0.0)) < 0.0:
        raise ConfigError("loss.class_prior_adjustment_tau must be non-negative")
    active_forgetting = loss.get("active_forgetting", {})
    if active_forgetting.get("enabled", False):
        if not trust.get("enabled", False):
            raise ConfigError("active_forgetting requires trust.enabled=true")
        threshold = float(
            active_forgetting.get("maximum_clean_probability", 0.05)
        )
        if not 0.0 <= threshold < 1.0:
            raise ConfigError(
                "active_forgetting.maximum_clean_probability must be in [0,1)"
            )
        unlearning_weight = float(
            active_forgetting.get("unlearning_weight", 0.001)
        )
        negative_weight = float(
            active_forgetting.get("negative_learning_weight", 0.1)
        )
        if unlearning_weight < 0.0 or negative_weight < 0.0:
            raise ConfigError("active_forgetting weights must be non-negative")
        if unlearning_weight == 0.0 and negative_weight == 0.0:
            raise ConfigError("active_forgetting requires at least one positive weight")
        if int(active_forgetting.get("start_epoch", 1)) <= 0:
            raise ConfigError("active_forgetting.start_epoch must be positive")
        if float(loss.get("mixup_probability", 0.0)) > 0.0:
            raise ConfigError("active_forgetting requires mixup_probability=0")
    attention_local = loss.get("attention_local_training", {})
    if attention_local.get("enabled", False):
        if model.get("peft_mode") not in {"visual_lora", "visual_mlp_adapter"}:
            raise ConfigError(
                "attention_local_training requires online visual LoRA or MLP adapters"
            )
        if bool(model.get("use_cached_training", False)):
            raise ConfigError("attention_local_training requires online images")
        if not trust.get("enabled", False):
            raise ConfigError("attention_local_training requires trust.enabled=true")
        if float(loss.get("mixup_probability", 0.0)) > 0.0:
            raise ConfigError("attention_local_training requires mixup_probability=0")
        input_resolution = int(model.get("input_resolution", 224))
        crop_size = int(attention_local.get("crop_size", 160))
        if not 0 < crop_size < input_resolution:
            raise ConfigError(
                "attention_local_training.crop_size must be smaller than the input"
            )
        patch_side = input_resolution // 32
        top_patches = int(attention_local.get("top_patches", 5))
        if not 1 <= top_patches <= patch_side * patch_side:
            raise ConfigError("attention_local_training.top_patches is out of range")
        local_weight = float(
            attention_local.get("local_supervision_weight", 0.5)
        )
        if not 0.0 < local_weight < 1.0:
            raise ConfigError(
                "attention_local_training.local_supervision_weight must be in (0,1)"
            )
        if float(attention_local.get("consistency_weight", 0.25)) < 0.0:
            raise ConfigError(
                "attention_local_training.consistency_weight must be non-negative"
            )
        if float(attention_local.get("temperature", 1.0)) <= 0.0:
            raise ConfigError(
                "attention_local_training.temperature must be positive"
            )
        incompatible = {
            "active_forgetting": active_forgetting.get("enabled", False),
            "adaptive_cap": loss.get("adaptive_cap", {}).get("enabled", False),
            "contrastive": loss.get("contrastive", {}).get("enabled", False),
            "cyclic_filter": loss.get("cyclic_filter", {}).get("enabled", False),
            "snscl": loss.get("snscl", {}).get("enabled", False),
        }
        enabled_conflicts = [name for name, enabled in incompatible.items() if enabled]
        if enabled_conflicts:
            raise ConfigError(
                "attention_local_training has incompatible losses: "
                + ", ".join(enabled_conflicts)
            )
    dual_gce = loss.get("dual_gce", {})
    if dual_gce.get("enabled", False):
        if loss.get("name") != "gce":
            raise ConfigError("loss.dual_gce requires loss.name=gce")
        if not 0.0 < float(dual_gce.get("suspicious_fraction", 0.2)) < 0.5:
            raise ConfigError("dual_gce.suspicious_fraction must be in (0, 0.5)")
        clean_q = float(dual_gce.get("clean_q", q))
        suspicious_q = float(dual_gce.get("suspicious_q", 1.0))
        if not 0.0 < clean_q <= suspicious_q <= 1.0:
            raise ConfigError("dual_gce requires 0 < clean_q <= suspicious_q <= 1")
        if float(loss.get("mixup_probability", 0.0)) > 0.0:
            raise ConfigError("dual_gce requires mixup_probability=0")
    contrastive = loss.get("contrastive", {})
    if contrastive.get("enabled", False):
        if model.get("peft_mode") != "feature_adapter" or not bool(
            model.get("use_cached_training", False)
        ):
            raise ConfigError(
                "The preregistered contrastive gate requires cached feature_adapter"
            )
        if float(loss.get("mixup_probability", 0.0)) > 0.0:
            raise ConfigError("contrastive gate requires mixup_probability=0")
        if float(contrastive.get("weight", 0.0)) <= 0.0:
            raise ConfigError("contrastive.weight must be positive")
        if float(contrastive.get("temperature", 0.0)) <= 0.0:
            raise ConfigError("contrastive.temperature must be positive")
        if float(contrastive.get("feature_noise_std", 0.0)) <= 0.0:
            raise ConfigError("contrastive.feature_noise_std must be positive")
        if not 0.0 <= float(contrastive.get("trusted_threshold", 0.70)) <= 1.0:
            raise ConfigError("contrastive.trusted_threshold must be in [0,1]")
    snscl = loss.get("snscl", {})
    if snscl.get("enabled", False):
        if contrastive.get("enabled", False):
            raise ConfigError("snscl cannot be combined with the legacy contrastive gate")
        if model.get("peft_mode") not in {"visual_ln", "ln_post_proj", "visual_lora"}:
            raise ConfigError("snscl requires trainable online visual PEFT")
        if bool(model.get("use_cached_training", False)):
            raise ConfigError("snscl requires online image training")
        if not config.get("trust", {}).get("enabled", False):
            raise ConfigError("snscl requires trust.enabled=true")
        if float(loss.get("mixup_probability", 0.0)) > 0.0:
            raise ConfigError("snscl requires mixup_probability=0 for stable anchors")
        for name in ("contrastive_weight", "kl_weight", "temperature", "module_lr"):
            if float(snscl.get(name, 0.0)) <= 0.0:
                raise ConfigError(f"snscl.{name} must be positive")
        for name in ("hidden_dim", "projection_dim", "queue_size"):
            if int(snscl.get(name, 0)) <= 0:
                raise ConfigError(f"snscl.{name} must be positive")
        if not 0.0 <= float(snscl.get("reliability_threshold", 0.5)) <= 1.0:
            raise ConfigError("snscl.reliability_threshold must be in [0,1]")
        if not 0.0 <= float(snscl.get("label_moving_average", 0.99)) < 1.0:
            raise ConfigError("snscl.label_moving_average must be in [0,1)")
        if not 0.0 < float(snscl.get("initial_std", 0.05)) < 1.0:
            raise ConfigError("snscl.initial_std must be in (0,1)")
        if not 0.0 < float(snscl.get("mean_residual_scale", 0.1)) <= 1.0:
            raise ConfigError("snscl.mean_residual_scale must be in (0,1]")
    cyclic_filter = loss.get("cyclic_filter", {})
    if cyclic_filter.get("enabled", False):
        if model.get("peft_mode") != "frozen" or not bool(
            model.get("use_cached_training", False)
        ):
            raise ConfigError(
                "The preregistered cyclic gate requires cached frozen features"
            )
        if float(loss.get("mixup_probability", 0.0)) > 0.0:
            raise ConfigError("cyclic_filter requires mixup_probability=0")
        if dual_gce.get("enabled", False):
            raise ConfigError("cyclic_filter cannot be combined with dual_gce")
        if loss.get("adaptive_cap", {}).get("enabled", False):
            raise ConfigError("cyclic_filter cannot be combined with adaptive_cap")
        cycle_epochs = int(cyclic_filter.get("cycle_epochs", 0))
        if cycle_epochs < 2 or int(train.get("epochs", 0)) % cycle_epochs != 0:
            raise ConfigError(
                "cyclic_filter requires cycle_epochs >=2 and complete cycles"
            )
        if int(train.get("early_stop_patience", 0)) != 0:
            raise ConfigError("cyclic_filter requires early_stop_patience=0")
        if not 0.0 < float(cyclic_filter.get("maximum_delta", 0.0)) <= 1.0:
            raise ConfigError("cyclic_filter.maximum_delta must be in (0,1]")
        for name in ("remove_fraction", "maximum_class_fraction"):
            if not 0.0 < float(cyclic_filter.get(name, 0.0)) < 0.5:
                raise ConfigError(f"cyclic_filter.{name} must be in (0,0.5)")
        if int(cyclic_filter.get("minimum_kept_per_class", 0)) < 1:
            raise ConfigError(
                "cyclic_filter.minimum_kept_per_class must be positive"
            )
        if not 0.0 < float(
            cyclic_filter.get("protect_clean_threshold", 0.999)
        ) <= 1.0:
            raise ConfigError(
                "cyclic_filter.protect_clean_threshold must be in (0,1]"
            )

    workers = int(train.get("num_workers", 0))
    timeout = int(train.get("loader_timeout", 0))
    if workers == 0 and timeout != 0:
        raise ConfigError("train.loader_timeout must be 0 when num_workers is 0")
    if workers > 0 and int(train.get("prefetch_factor", 1)) <= 0:
        raise ConfigError("train.prefetch_factor must be positive")
    if int(train.get("epochs", 0)) <= 0:
        raise ConfigError("train.epochs must be positive")
    schedule_epochs = int(train.get("schedule_epochs", train.get("epochs", 0)))
    if schedule_epochs < int(train["epochs"]):
        raise ConfigError("train.schedule_epochs must be at least train.epochs")
    if int(train.get("batch_size", 0)) <= 0:
        raise ConfigError("train.batch_size must be positive")
    if float(train.get("amp_initial_scale", 65536.0)) <= 0.0:
        raise ConfigError("train.amp_initial_scale must be positive")
    selector_metric = evaluation.get("selector_metric", "proxy_macro")
    if selector_metric not in SELECTOR_METRICS:
        raise ConfigError(
            f"evaluation.selector_metric must be one of {sorted(SELECTOR_METRICS)}"
        )
    selection_policy = evaluation.get("selection_policy", "best_selector")
    if selection_policy not in {"best_selector", "last_epoch"}:
        raise ConfigError(
            "evaluation.selection_policy must be best_selector or last_epoch"
        )

    peft_mode = model.get("peft_mode", "frozen")
    if peft_mode in {
        "visual_ln",
        "ln_post_proj",
        "visual_lora",
        "visual_mlp_adapter",
        "visual_prompt",
    }:
        if bool(model.get("use_cached_training", False)):
            raise ConfigError(
                "Visual PEFT requires online images; cached-only training is invalid"
            )
        if float(train.get("backbone_lr", 0.0)) <= 0.0:
            raise ConfigError("Visual PEFT requires train.backbone_lr > 0")
    if peft_mode == "visual_lora":
        if int(model.get("lora_last_n_blocks", 0)) <= 0:
            raise ConfigError("visual_lora requires model.lora_last_n_blocks > 0")
        if int(model.get("lora_rank", 0)) <= 0:
            raise ConfigError("visual_lora requires model.lora_rank > 0")
        if float(model.get("lora_alpha", 0.0)) <= 0.0:
            raise ConfigError("visual_lora requires model.lora_alpha > 0")
        if not bool(model.get("lora_adapt_qv", True)) and not bool(
            model.get("lora_adapt_out", True)
        ):
            raise ConfigError("visual_lora must adapt Q/V, output, or both")
    if peft_mode == "visual_mlp_adapter":
        if int(model.get("visual_adapter_last_n_blocks", 0)) <= 0:
            raise ConfigError(
                "visual_mlp_adapter requires visual_adapter_last_n_blocks > 0"
            )
        if int(model.get("visual_adapter_bottleneck", 0)) <= 0:
            raise ConfigError(
                "visual_mlp_adapter requires visual_adapter_bottleneck > 0"
            )
        scale = float(model.get("visual_adapter_scale", 0.0))
        if not 0.0 < scale <= 1.0:
            raise ConfigError("visual_adapter_scale must be in (0,1]")
        dropout = float(model.get("visual_adapter_dropout", -1.0))
        if not 0.0 <= dropout < 1.0:
            raise ConfigError("visual_adapter_dropout must be in [0,1)")
    if peft_mode == "visual_prompt":
        last_n_blocks = int(model.get("visual_prompt_last_n_blocks", 0))
        if not 1 <= last_n_blocks <= 12:
            raise ConfigError("visual_prompt_last_n_blocks must be in [1,12]")
        if int(model.get("visual_prompt_num_tokens", 0)) <= 0:
            raise ConfigError("visual_prompt_num_tokens must be positive")
        prompt_dropout = float(model.get("visual_prompt_dropout", -1.0))
        if not 0.0 <= prompt_dropout < 1.0:
            raise ConfigError("visual_prompt_dropout must be in [0,1)")
    if peft_mode == "feature_adapter":
        if not bool(model.get("use_cached_training", False)):
            raise ConfigError("feature_adapter requires cached feature training")
        if int(model.get("adapter_dim", 0)) <= 0:
            raise ConfigError("feature_adapter requires model.adapter_dim > 0")
        adapter_scale = float(model.get("adapter_scale", 1.0))
        if not 0.0 < adapter_scale <= 1.0:
            raise ConfigError("feature_adapter model.adapter_scale must be in (0, 1]")

    trust = config.get("trust", {})
    if trust.get("enabled", False) and not trust.get("bundle_path"):
        raise ConfigError("trust.bundle_path is required when trust.enabled=true")
    selection_threshold = trust.get("selection_threshold")
    if selection_threshold is not None and not 0.0 <= float(
        selection_threshold
    ) <= 1.0:
        raise ConfigError("trust.selection_threshold must be in [0,1]")
    rejected_weight = float(trust.get("rejected_sample_weight", 0.0))
    if not 0.0 <= rejected_weight <= 1.0:
        raise ConfigError("trust.rejected_sample_weight must be in [0,1]")
    if dual_gce.get("enabled", False) and not trust.get("enabled", False):
        raise ConfigError("dual_gce requires trust.enabled=true for OOF scores")
    if cyclic_filter.get("enabled", False) and not trust.get("enabled", False):
        raise ConfigError("cyclic_filter requires trust.enabled=true")

    elr = config.get("elr", {})
    if elr.get("enabled", False):
        if not 0.0 < float(elr.get("momentum", 0.9)) < 1.0:
            raise ConfigError("elr.momentum must be in (0, 1)")
        if float(elr.get("target_weight", 3.0)) < 0.0:
            raise ConfigError("elr.target_weight must be non-negative")
        if int(elr.get("warmup_epochs", 5)) < 0 or int(
            elr.get("ramp_epochs", 5)
        ) < 0:
            raise ConfigError("elr warmup/ramp epochs must be non-negative")
        if float(loss.get("mixup_probability", 0.0)) > 0.0:
            raise ConfigError("ELR requires mixup_probability=0 for sample alignment")
        if cyclic_filter.get("enabled", False):
            raise ConfigError("cyclic_filter cannot be combined with ELR")

    for key in ("train_csv", "val_csv", "class_mapping", "train_root", "test_root"):
        if key not in data:
            raise ConfigError(f"data.{key} is required")


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a serialisable configuration without private runtime keys."""
    return {key: copy.deepcopy(value) for key, value in config.items() if not key.startswith("_")}


def _resolve_paths(config: dict[str, Any], base: Path) -> None:
    path_keys = {
        ("data", "train_csv"),
        ("data", "val_csv"),
        ("data", "class_mapping"),
        ("data", "train_root"),
        ("data", "test_root"),
        ("features", "tensor_path"),
        ("features", "paths_path"),
        ("features", "manifest_path"),
        ("trust", "bundle_path"),
        ("trust", "groups_path"),
        ("train", "init_checkpoint"),
        ("lineage", "parent_train_csv"),
        ("lineage", "parent_val_csv"),
        ("output", "root"),
    }
    for section, key in path_keys:
        if section not in config or key not in config[section]:
            continue
        value = config[section][key]
        if value in (None, ""):
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = base / path
        config[section][key] = str(path.resolve())
