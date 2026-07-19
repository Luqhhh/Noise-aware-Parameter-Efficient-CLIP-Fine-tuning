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
}
CLASSIFIER_MODES = {"linear", "anchored_residual"}
LOSS_NAMES = {"cross_entropy", "gce"}
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
    loss = config["loss"]
    train = config["train"]
    evaluation = config["evaluation"]

    if not str(project.get("experiment_id", "")).strip():
        raise ConfigError("project.experiment_id must be non-empty")
    if project.get("stage") not in COMPETITION_STAGES:
        raise ConfigError(
            f"project.stage must be one of {sorted(COMPETITION_STAGES)}"
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

    if loss.get("name") not in LOSS_NAMES:
        raise ConfigError(f"loss.name must be one of {sorted(LOSS_NAMES)}")
    q = float(loss.get("gce_q", 0.5))
    if loss.get("name") == "gce" and not 0.0 < q <= 1.0:
        raise ConfigError("loss.gce_q must be in (0, 1]")
    if float(loss.get("class_prior_adjustment_tau", 0.0)) < 0.0:
        raise ConfigError("loss.class_prior_adjustment_tau must be non-negative")
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
    selector_metric = evaluation.get("selector_metric", "proxy_macro")
    if selector_metric not in SELECTOR_METRICS:
        raise ConfigError(
            f"evaluation.selector_metric must be one of {sorted(SELECTOR_METRICS)}"
        )

    peft_mode = model.get("peft_mode", "frozen")
    if peft_mode in {"visual_ln", "ln_post_proj", "visual_lora"}:
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
