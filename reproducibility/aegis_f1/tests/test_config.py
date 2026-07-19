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


def test_visual_peft_cannot_train_from_cached_features() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "a0_fulldata_anchor.yaml"))
    config["model"]["peft_mode"] = "visual_ln"
    config["model"]["use_cached_training"] = True
    config["train"]["backbone_lr"] = 1.0e-6
    with pytest.raises(ConfigError, match="online images"):
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
