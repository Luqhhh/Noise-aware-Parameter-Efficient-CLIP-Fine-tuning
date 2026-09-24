"""Contract tests for REMATCH750_SEARCH_V5 conditional slots."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from PIL import Image
import torchvision.transforms as T

from aegis_clip.config import load_config
from aegis_clip.rematch_search_v5 import (
    MechanismBlockedError,
    validate_declaration,
)
from aegis_clip.trainer import _training_preprocess


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs/rematch750_search_v5"


def test_v5_manifest_has_conditional_slot_coverage():
    manifest = json.loads((ROOT / "search_manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "rematch750_search_v5"
    assert manifest["status"] == "planned_not_executed"
    assert len(manifest["trials"]) == 68
    assert manifest["budget"] == {
        "visual_training_points": 48,
        "cached_head_points": 12,
        "max_combinations": 8,
        "max_additional_confirmation_training_runs": 6,
    }
    counts = manifest["family_counts"]
    assert counts == {"R": 6, "S": 9, "K": 9, "V": 8, "L": 6, "Q": 6, "N": 4, "H": 12, "X": 8}
    assert manifest["baseline"]["candidate"] == "RM_V4_F05"
    assert manifest["baseline"]["platform_score"] is None
    assert manifest["first_wave"] == [
        "R02", "R04", "R01", "R03", "S01", "S02",
        "K02", "K03", "K07", "V02", "L01", "H01",
    ]
    by_id = {trial["trial_id"]: trial for trial in manifest["trials"]}
    assert by_id["R01"]["implementation_status"] == "implemented"
    assert by_id["S02"]["implementation_status"] == "blocked_implementation"
    assert by_id["K07"]["implementation_status"] == "blocked_implementation"
    assert by_id["L01"]["implementation_status"] == "blocked_implementation"
    assert by_id["H01"]["implementation_status"] == "pending_artifact"
    assert by_id["X08"]["implementation_status"] == "blocked_dependency"


def test_all_v5_configs_load_and_declare():
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(paths) == 68
    for path in paths:
        config = load_config(path)
        declaration = validate_declaration(config)
        assert declaration["trial_id"] == path.stem
        assert config["project"]["protocol"] == "rematch750_search_v5"
        assert config["train"]["device"] == "npu:UNASSIGNED"


@pytest.mark.parametrize("trial_id", ["S02", "S03", "K07", "L01", "Q01", "H01", "X01"])
def test_blocked_v5_slots_fail_closed_when_execution_is_requested(trial_id):
    config = load_config(CONFIG_DIR / f"{trial_id}.yaml")
    with pytest.raises(MechanismBlockedError):
        validate_declaration(config, require_implemented=True)


def test_v5_unknown_search_field_is_not_silently_ignored():
    config = load_config(CONFIG_DIR / "R01.yaml")
    config["project"]["search"] = copy.deepcopy(config["project"]["search"])
    config["project"]["search"]["mystery_axis"] = 1
    with pytest.raises(ValueError, match="unknown project.search keys"):
        validate_declaration(config)


def test_v5_rrc_area_bounds_are_forwarded_to_train_transform():
    base = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform = _training_preprocess(
        base,
        {
            "train_augmentation": "weak_rrc_flip",
            "rrc_scale_min": 0.8,
            "rrc_scale_max": 1.0,
            "rrc_ratio_min": 0.85,
            "rrc_ratio_max": 1.15,
        },
        input_resolution=224,
    )
    tensor = transform(Image.new("RGB", (320, 240), "blue"))
    assert tensor.shape == (3, 224, 224)


def test_v5_staged_resolution_is_declared_blocked():
    config = load_config(CONFIG_DIR / "R05.yaml")
    with pytest.raises(MechanismBlockedError):
        validate_declaration(config, require_implemented=True)
