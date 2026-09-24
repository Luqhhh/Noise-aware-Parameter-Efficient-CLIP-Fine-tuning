from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from aegis_clip.config import load_config


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs" / "rematch750_f05_transfer"
BASE_CONFIG = ROOT / "configs" / "rematch750_head_l2sp" / "HL00.yaml"
IDENTITY_PATHS = {
    "output.root",
    "project.experiment_id",
    "project.protocol",
    "project.trial_id",
    "project.mechanism",
    "project.implementation_status",
}
EXPECTED_VARIABLES = {
    "ET01": {"loss.feature_distillation_weight"},
    "ET02": {"loss.gce_q"},
    "ET03": {"longtail.balanced_softmax_tau"},
    "ET04": {"longtail.balanced_softmax_tau"},
    "ET05": {"train.backbone_lr"},
    "ET06": {"train.backbone_lr"},
    "ET07": {"train.head_lr"},
    "ET08": {"train.head_lr"},
    "ET09": {"train.lr_warmup_epochs"},
    "ET10": {"train.epochs", "train.schedule_epochs"},
}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if key == "_config_path":
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, path))
        return result
    return {prefix: value}


def _diff_paths(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    flat_left = _flatten(left)
    flat_right = _flatten(right)
    keys = set(flat_left) | set(flat_right)
    return {key for key in keys if flat_left.get(key) != flat_right.get(key)}


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_configs_parse_and_are_single_variable_f05_transfers():
    baseline = load_config(BASE_CONFIG)
    for trial, variables in EXPECTED_VARIABLES.items():
        config = load_config(CONFIG_DIR / f"{trial}.yaml")
        assert config["project"]["trial_id"] == trial
        assert config["project"]["protocol"] == "rematch750_f05_transfer"
        assert config["model"]["input_resolution"] == 320
        assert config["model"]["classifier_mode"] == "linear"
        assert config["train"]["batch_size"] == 256
        assert config["train"]["grad_accum_steps"] == 4
        assert config["train"]["effective_batch_size"] == 1024
        assert config["train"]["device"] == "npu:UNASSIGNED"
        assert config["evaluation"]["record_initial"] is True
        assert _diff_paths(baseline, config) == IDENTITY_PATHS | variables


def test_priority_one_values_match_preregistered_evidence_transfer():
    et01 = load_config(CONFIG_DIR / "ET01.yaml")
    et02 = load_config(CONFIG_DIR / "ET02.yaml")
    et03 = load_config(CONFIG_DIR / "ET03.yaml")
    et04 = load_config(CONFIG_DIR / "ET04.yaml")
    assert et01["loss"]["feature_distillation_weight"] == 1.0
    assert et02["loss"]["gce_q"] == 0.7
    assert et03["longtail"]["balanced_softmax_tau"] == 0.5
    assert et04["longtail"]["balanced_softmax_tau"] == 1.0


def test_builder_reproduces_committed_bundle_byte_for_byte():
    builder = _import(
        ROOT / "scripts" / "build_rematch750_f05_transfer.py", "f05_transfer_builder"
    )
    rendered = builder.render_bundle(ROOT)
    assert len(rendered) == 11
    for path, payload in rendered.items():
        assert path.read_bytes() == payload


def test_manifest_hashes_and_promotion_policy_are_bound():
    manifest = json.loads((CONFIG_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ready_for_npu"
    assert manifest["platform_submission"] is False
    assert manifest["baseline"]["trial"] == "HL00"
    assert manifest["promotion"]["macro_delta_pp"] == 0.30
    assert manifest["promotion"]["minimum_directional_seeds"] == 2
    assert {trial["name"] for trial in manifest["trials"]} == set(
        EXPECTED_VARIABLES
    )
    for trial in manifest["trials"]:
        path = ROOT / trial["config"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == trial["config_sha256"]


def test_runtime_materialization_is_explicit_hash_bound_and_idempotent(tmp_path):
    runner = _import(
        ROOT / "scripts" / "run_rematch750_f05_transfer.py", "f05_transfer_runner"
    )
    source = CONFIG_DIR / "ET05.yaml"
    runtime = runner.materialize_config(source, "npu:0", tmp_path)
    loaded = load_config(runtime)
    assert loaded["train"]["device"] == "npu:0"
    assert Path(loaded["data"]["train_csv"]).is_absolute()
    assert Path(loaded["output"]["root"]).is_absolute()
    assert runner.materialize_config(source, "npu:0", tmp_path) == runtime
    runtime.write_text("stale: true\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="different content"):
        runner.materialize_config(source, "npu:0", tmp_path)
    with pytest.raises(ValueError, match="device must be explicit"):
        runner.materialize_config(
            CONFIG_DIR / "ET06.yaml", "npu:UNASSIGNED", tmp_path / "bad"
        )
