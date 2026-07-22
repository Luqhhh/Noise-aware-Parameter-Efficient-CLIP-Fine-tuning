from copy import deepcopy
from pathlib import Path

import pytest

from aegis_clip.config import ConfigError, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_all_shipped_configs_are_valid() -> None:
    configs = sorted((ROOT / "configs").glob("*.yaml"))
    assert len(configs) >= 5
    for path in configs:
        config = load_config(path)
        assert config["model"]["backbone"] == "ViT-B/32"
        assert Path(config["output"]["root"]).is_absolute()


def test_unknown_section_fails_closed() -> None:
    config = load_config(ROOT / "configs" / "a0_fulldata_anchor.yaml")
    config["surprise"] = {}
    with pytest.raises(ConfigError, match="Unknown"):
        validate_config(config)


def test_team_internal_ablation_stage_keeps_full_validation() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "a0_fulldata_anchor.yaml"))
    config["project"]["stage"] = "p4_ablation"
    validate_config(config)
    config["data"]["external_data"] = True
    with pytest.raises(ConfigError, match="external_data"):
        validate_config(config)


def test_visual_peft_cannot_train_from_cached_features() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "a0_fulldata_anchor.yaml"))
    config["model"]["peft_mode"] = "visual_ln"
    config["model"]["use_cached_training"] = True
    config["train"]["backbone_lr"] = 1.0e-6
    with pytest.raises(ConfigError, match="online images"):
        validate_config(config)


def test_input_resolution_must_match_vit_b32_patch_grid() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "a0_fulldata_anchor.yaml"))
    config["model"]["input_resolution"] = 250
    with pytest.raises(ConfigError, match="input_resolution"):
        validate_config(config)


def test_visual_lora_requires_a_real_low_rank_target() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "a0_fulldata_anchor.yaml"))
    config["model"].update(
        {
            "peft_mode": "visual_lora",
            "use_cached_training": False,
            "lora_last_n_blocks": 4,
            "lora_rank": 8,
            "lora_alpha": 8.0,
            "lora_adapt_qv": False,
            "lora_adapt_out": False,
        }
    )
    config["train"]["backbone_lr"] = 2.0e-5
    with pytest.raises(ConfigError, match="adapt Q/V"):
        validate_config(config)


def test_visual_mlp_adapter_requires_valid_structure() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "a0_fulldata_anchor.yaml"))
    config["model"].update(
        {
            "peft_mode": "visual_mlp_adapter",
            "use_cached_training": False,
            "visual_adapter_last_n_blocks": 6,
            "visual_adapter_bottleneck": 64,
            "visual_adapter_scale": 0.1,
            "visual_adapter_dropout": 0.1,
        }
    )
    config["train"]["backbone_lr"] = 1.0e-4
    validate_config(config)
    config["model"]["visual_adapter_bottleneck"] = 0
    with pytest.raises(ConfigError, match="bottleneck"):
        validate_config(config)


def test_visual_prompt_requires_valid_structure() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "a0_fulldata_anchor.yaml"))
    config["model"].update(
        {
            "peft_mode": "visual_prompt",
            "use_cached_training": False,
            "visual_prompt_last_n_blocks": 12,
            "visual_prompt_num_tokens": 5,
            "visual_prompt_dropout": 0.0,
        }
    )
    config["train"]["backbone_lr"] = 1.0e-3
    validate_config(config)
    config["model"]["visual_prompt_num_tokens"] = 0
    with pytest.raises(ConfigError, match="num_tokens"):
        validate_config(config)


def test_feature_adapter_requires_cached_features() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "a0_fulldata_anchor.yaml"))
    config["model"]["peft_mode"] = "feature_adapter"
    config["model"]["adapter_dim"] = 64
    config["model"]["use_cached_training"] = False
    with pytest.raises(ConfigError, match="cached feature"):
        validate_config(config)


def test_schedule_must_cover_all_training_epochs() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "a0_fulldata_anchor.yaml"))
    config["train"]["epochs"] = 10
    config["train"]["schedule_epochs"] = 9
    with pytest.raises(ConfigError, match="schedule_epochs"):
        validate_config(config)


def test_amp_initial_scale_must_be_positive() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "a0_fulldata_anchor.yaml"))
    config["train"]["amp_initial_scale"] = 0.0
    with pytest.raises(ConfigError, match="amp_initial_scale"):
        validate_config(config)


def test_checkpoint_selection_policy_fails_closed() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "a0_fulldata_anchor.yaml"))
    config["evaluation"]["selection_policy"] = "platform_best"
    with pytest.raises(ConfigError, match="selection_policy"):
        validate_config(config)


def test_anchored_residual_requires_valid_scale_and_initial_checkpoint() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "a0_fulldata_anchor.yaml"))
    config["model"]["classifier_mode"] = "anchored_residual"
    config["model"]["classifier_residual_scale"] = 0.25
    with pytest.raises(ConfigError, match="init_checkpoint"):
        validate_config(config)

    config["train"]["init_checkpoint"] = "base.pt"
    config["model"]["classifier_residual_scale"] = 0.0
    with pytest.raises(ConfigError, match="classifier_residual_scale"):
        validate_config(config)


def test_elr_rejects_mixup_that_breaks_sample_alignment() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "b0_cached_nomix.yaml"))
    config["elr"] = {"enabled": True, "momentum": 0.9, "target_weight": 3.0}
    config["loss"]["mixup_probability"] = 0.2
    with pytest.raises(ConfigError, match="sample alignment"):
        validate_config(config)


def test_dual_gce_requires_oof_trust_and_ordered_q() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "b0_cached_nomix.yaml"))
    config["loss"]["dual_gce"] = {
        "enabled": True,
        "suspicious_fraction": 0.2,
        "clean_q": 0.5,
        "suspicious_q": 1.0,
    }
    with pytest.raises(ConfigError, match="trust.enabled"):
        validate_config(config)
    config["trust"]["enabled"] = True
    config["loss"]["dual_gce"]["clean_q"] = 1.0
    config["loss"]["dual_gce"]["suspicious_q"] = 0.5
    with pytest.raises(ConfigError, match="clean_q"):
        validate_config(config)


def test_active_forgetting_requires_trust_positive_weight_and_no_mixup() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "b0_cached_nomix.yaml"))
    config["loss"]["active_forgetting"] = {
        "enabled": True,
        "maximum_clean_probability": 0.05,
        "unlearning_weight": 0.001,
        "negative_learning_weight": 0.1,
        "start_epoch": 1,
    }
    with pytest.raises(ConfigError, match="trust.enabled"):
        validate_config(config)
    config["trust"]["enabled"] = True
    config["trust"]["bundle_path"] = "trust.pt"
    config["loss"]["active_forgetting"]["unlearning_weight"] = 0.0
    config["loss"]["active_forgetting"]["negative_learning_weight"] = 0.0
    with pytest.raises(ConfigError, match="positive weight"):
        validate_config(config)
    config["loss"]["active_forgetting"]["negative_learning_weight"] = 0.1
    config["loss"]["mixup_probability"] = 0.2
    with pytest.raises(ConfigError, match="mixup_probability"):
        validate_config(config)
    config["loss"]["mixup_probability"] = 0.0
    validate_config(config)


def test_attention_local_training_requires_online_trusted_visual_peft() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "b0_cached_nomix.yaml"))
    config["loss"]["attention_local_training"] = {
        "enabled": True,
        "crop_size": 160,
        "top_patches": 5,
        "local_supervision_weight": 0.5,
        "consistency_weight": 0.25,
        "temperature": 1.0,
    }
    with pytest.raises(ConfigError, match="visual LoRA"):
        validate_config(config)
    config["model"].update(
        {
            "peft_mode": "visual_lora",
            "use_cached_training": False,
            "lora_last_n_blocks": 4,
            "lora_rank": 8,
            "lora_alpha": 8.0,
            "lora_adapt_qv": True,
            "lora_adapt_out": True,
        }
    )
    config["train"]["backbone_lr"] = 1.0e-5
    with pytest.raises(ConfigError, match="trust.enabled"):
        validate_config(config)
    config["trust"]["enabled"] = True
    config["trust"]["bundle_path"] = "trust.pt"
    validate_config(config)
    config["loss"]["mixup_probability"] = 0.2
    with pytest.raises(ConfigError, match="mixup_probability"):
        validate_config(config)


def test_attention_local_training_rejects_invalid_crop_and_weights() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "f1_visual_lora_clean_core.yaml"))
    config["loss"]["attention_local_training"] = {
        "enabled": True,
        "crop_size": 224,
        "top_patches": 5,
        "local_supervision_weight": 0.5,
        "consistency_weight": 0.25,
        "temperature": 1.0,
    }
    with pytest.raises(ConfigError, match="crop_size"):
        validate_config(config)
    config["loss"]["attention_local_training"]["crop_size"] = 160
    config["loss"]["attention_local_training"]["local_supervision_weight"] = 1.0
    with pytest.raises(ConfigError, match="local_supervision_weight"):
        validate_config(config)


def test_contrastive_gate_requires_cached_adapter_and_no_mixup() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "b0_cached_nomix.yaml"))
    config["loss"]["contrastive"] = {
        "enabled": True,
        "weight": 1.0,
        "temperature": 0.1,
        "feature_noise_std": 0.01,
        "trusted_threshold": 0.7,
    }
    with pytest.raises(ConfigError, match="feature_adapter"):
        validate_config(config)
    config["model"].update(
        {
            "peft_mode": "feature_adapter",
            "adapter_dim": 64,
            "use_cached_training": True,
        }
    )
    config["loss"]["mixup_probability"] = 0.2
    with pytest.raises(ConfigError, match="mixup_probability"):
        validate_config(config)


def test_snscl_requires_online_visual_peft_trust_and_unmixed_anchors() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "b0_cached_nomix.yaml"))
    config["loss"]["snscl"] = {
        "enabled": True,
        "contrastive_weight": 0.1,
        "kl_weight": 0.001,
        "temperature": 0.07,
        "module_lr": 0.0001,
        "hidden_dim": 32,
        "projection_dim": 16,
        "queue_size": 4,
    }
    with pytest.raises(ConfigError, match="visual PEFT"):
        validate_config(config)
    config["model"].update(
        {
            "peft_mode": "visual_lora",
            "use_cached_training": False,
            "lora_last_n_blocks": 1,
            "lora_rank": 2,
            "lora_alpha": 2.0,
            "lora_adapt_qv": True,
            "lora_adapt_out": True,
        }
    )
    config["train"]["backbone_lr"] = 1.0e-5
    with pytest.raises(ConfigError, match="trust.enabled"):
        validate_config(config)
    config["trust"]["enabled"] = True
    config["trust"]["bundle_path"] = "trust.pt"
    config["loss"]["mixup_probability"] = 0.2
    with pytest.raises(ConfigError, match="mixup_probability"):
        validate_config(config)
    config["loss"]["mixup_probability"] = 0.0
    validate_config(config)


def test_cyclic_filter_requires_complete_frozen_cached_cycles() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "b0_cached_nomix.yaml"))
    config["trust"]["enabled"] = True
    config["loss"]["cyclic_filter"] = {
        "enabled": True,
        "cycle_epochs": 15,
        "maximum_delta": 0.25,
        "remove_fraction": 0.02,
        "maximum_class_fraction": 0.10,
        "minimum_kept_per_class": 5,
    }
    config["train"]["epochs"] = 44
    config["train"]["early_stop_patience"] = 0
    with pytest.raises(ConfigError, match="complete cycles"):
        validate_config(config)
    config["train"]["epochs"] = 45
    validate_config(config)
