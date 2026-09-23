from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from aegis_clip.config import load_config


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs" / "rematch750_wd_relax"
EXPECTED = {
    "WD00": ("RM_WD00_CONTROL_1E4", 1.0e-4, 1.0e-4),
    "WD01": ("RM_WD01_BACKBONE0", 0.0, 1.0e-4),
    "WD02": ("RM_WD02_HEAD0", 1.0e-4, 0.0),
    "WD03": ("RM_WD03_BOTH0", 0.0, 0.0),
}


def _comparison_view(config):
    result = copy.deepcopy(config)
    result.pop("_config_path", None)
    result["project"].pop("experiment_id")
    result["output"].pop("root")
    result["train"].pop("backbone_weight_decay")
    result["train"].pop("head_weight_decay")
    return result


def test_factorial_configs_parse_and_only_change_registered_fields():
    configs = {name: load_config(CONFIG_DIR / f"{name}.yaml") for name in EXPECTED}
    control_view = _comparison_view(configs["WD00"])
    for name, (experiment_id, backbone_wd, head_wd) in EXPECTED.items():
        config = configs[name]
        assert config["project"]["experiment_id"] == experiment_id
        assert config["train"]["backbone_weight_decay"] == backbone_wd
        assert config["train"]["head_weight_decay"] == head_wd
        assert config["train"]["device"] == "npu:UNASSIGNED"
        assert _comparison_view(config) == control_view


def test_control_matches_v3_recipe_except_identity_and_output():
    baseline = load_config(ROOT / "configs" / "rematch750_v3_b1024_e16_lr4.yaml")
    control = load_config(CONFIG_DIR / "WD00.yaml")
    assert _comparison_view(control) == _comparison_view(baseline)


def test_manifest_hashes_and_factorial_cells_are_complete():
    manifest = json.loads((CONFIG_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ready_for_npu"
    assert manifest["platform_submission"] is False
    assert {trial["name"] for trial in manifest["trials"]} == set(EXPECTED)
    cells = set()
    for trial in manifest["trials"]:
        config_path = ROOT / trial["config"]
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == trial["config_sha256"]
        cells.add((trial["backbone_weight_decay"], trial["head_weight_decay"]))
    assert cells == {(1.0e-4, 1.0e-4), (0.0, 1.0e-4), (1.0e-4, 0.0), (0.0, 0.0)}


def test_runtime_materialization_is_explicit_and_reloads(tmp_path):
    import importlib.util

    script = ROOT / "scripts" / "run_rematch750_wd_relax.py"
    spec = importlib.util.spec_from_file_location("wd_runner", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    runtime = module.materialize_config(CONFIG_DIR / "WD01.yaml", "npu:0", tmp_path)
    loaded = load_config(runtime)
    assert loaded["train"]["device"] == "npu:0"
    assert Path(loaded["data"]["train_csv"]).is_absolute()
    assert Path(loaded["output"]["root"]).is_absolute()
    assert module.materialize_config(CONFIG_DIR / "WD01.yaml", "npu:0", tmp_path) == runtime
    runtime.write_text("stale: true\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="different content"):
        module.materialize_config(CONFIG_DIR / "WD01.yaml", "npu:0", tmp_path)
    with pytest.raises(ValueError, match="device must be explicit"):
        module.materialize_config(CONFIG_DIR / "WD02.yaml", "npu:UNASSIGNED", tmp_path / "bad")
