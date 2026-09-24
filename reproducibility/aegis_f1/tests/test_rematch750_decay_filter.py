from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from aegis_clip.config import ConfigError, load_config, validate_config
from aegis_clip.model import AegisCLIP


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs" / "rematch750_decay_filter"
BASE_CONFIG = ROOT / "configs" / "rematch750_head_l2sp" / "HL00.yaml"
EXPECTED = {
    "DF01": ("matrix_only", "all"),
    "DF02": ("all", "matrix_only"),
    "DF03": ("matrix_only", "matrix_only"),
}
IDENTITY_PATHS = {
    "output.root",
    "project.experiment_id",
    "project.protocol",
    "project.trial_id",
    "project.mechanism",
    "project.implementation_status",
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


class TinyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.class_embedding = nn.Parameter(torch.ones(4))
        self.positional_embedding = nn.Parameter(torch.ones(2, 4))
        self.ln_pre = nn.LayerNorm(4)
        self.ln_post = nn.LayerNorm(4)
        self.proj = nn.Parameter(torch.ones(4, 4))


def _build_model() -> AegisCLIP:
    model = AegisCLIP(
        TinyVisual(),
        num_classes=3,
        feature_dim=4,
        peft_mode="full_finetune",
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
    return model


def _parameter_names(model: nn.Module) -> dict[int, str]:
    return {id(parameter): name for name, parameter in model.named_parameters()}


def test_configs_parse_and_form_strict_two_by_two_factorial():
    baseline = load_config(BASE_CONFIG)
    for trial, (head_filter, backbone_filter) in EXPECTED.items():
        config = load_config(CONFIG_DIR / f"{trial}.yaml")
        assert config["project"]["trial_id"] == trial
        assert config["project"]["protocol"] == "rematch750_decay_filter"
        assert config["train"]["head_weight_decay_filter"] == head_filter
        assert config["train"]["backbone_weight_decay_filter"] == backbone_filter
        assert config["train"]["head_weight_decay"] == 1.0e-4
        assert config["train"]["backbone_weight_decay"] == 1.0e-4
        assert config["train"]["device"] == "npu:UNASSIGNED"
        assert _diff_paths(baseline, config) == IDENTITY_PATHS | {
            "train.head_weight_decay_filter",
            "train.backbone_weight_decay_filter",
        }


def test_default_all_mode_preserves_historical_two_groups():
    model = _build_model()
    groups = model.parameter_groups(
        head_lr=4.0e-4,
        head_weight_decay=1.0e-4,
        backbone_lr=1.2e-5,
        backbone_weight_decay=1.0e-4,
    )
    assert [group["name"] for group in groups] == ["head", "visual"]
    assert all(group["weight_decay"] == 1.0e-4 for group in groups)


def test_matrix_only_groups_cover_trainable_parameters_once_and_by_shape():
    model = _build_model()
    groups = model.parameter_groups(
        head_lr=4.0e-4,
        head_weight_decay=1.0e-4,
        backbone_lr=1.2e-5,
        backbone_weight_decay=1.0e-4,
        head_weight_decay_filter="matrix_only",
        backbone_weight_decay_filter="matrix_only",
    )
    names = _parameter_names(model)
    grouped = [parameter for group in groups for parameter in group["params"]]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in trainable
    }

    by_group = {
        group["name"]: {names[id(parameter)] for parameter in group["params"]}
        for group in groups
    }
    assert "classifier.weight" in by_group["head"]
    assert "classifier.bias" in by_group["head_no_decay"]
    assert "visual.proj" in by_group["visual"]
    assert "visual.class_embedding" in by_group["visual_no_decay"]
    assert "visual.ln_pre.weight" in by_group["visual_no_decay"]
    assert "visual.ln_post.bias" in by_group["visual_no_decay"]
    for group in groups:
        expected_decay = 0.0 if group["name"].endswith("_no_decay") else 1.0e-4
        assert group["weight_decay"] == expected_decay
        for parameter in group["params"]:
            assert (parameter.ndim < 2) == group["name"].endswith("_no_decay")


def test_zero_gradient_adamw_decays_only_matrix_parameters():
    model = _build_model()
    groups = model.parameter_groups(
        head_lr=0.1,
        head_weight_decay=0.2,
        backbone_lr=0.1,
        backbone_weight_decay=0.2,
        head_weight_decay_filter="matrix_only",
        backbone_weight_decay_filter="matrix_only",
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.zeros_like(parameter)
    torch.optim.AdamW(groups).step()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            assert not torch.equal(parameter, before[name]), name
            assert torch.allclose(parameter, before[name] * 0.98), name
        else:
            assert torch.equal(parameter, before[name]), name


def test_invalid_filter_fails_closed_in_config_and_model():
    config = copy.deepcopy(load_config(CONFIG_DIR / "DF03.yaml"))
    config["train"]["head_weight_decay_filter"] = "bias_only"
    with pytest.raises(ConfigError, match="head_weight_decay_filter"):
        validate_config(config)

    model = _build_model()
    with pytest.raises(ValueError, match="Unsupported head"):
        model.parameter_groups(
            head_lr=4.0e-4,
            head_weight_decay=1.0e-4,
            backbone_lr=1.2e-5,
            backbone_weight_decay=1.0e-4,
            head_weight_decay_filter="bias_only",
        )


def test_builder_reproduces_committed_bundle_byte_for_byte():
    builder = _import(
        ROOT / "scripts" / "build_rematch750_decay_filter.py",
        "decay_filter_builder",
    )
    rendered = builder.render_bundle(ROOT)
    assert len(rendered) == 4
    for path, payload in rendered.items():
        assert path.read_bytes() == payload


def test_manifest_hashes_and_promotion_policy_are_bound():
    manifest = json.loads((CONFIG_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ready_for_npu"
    assert manifest["platform_submission"] is False
    assert manifest["baseline"]["trial"] == "HL00"
    assert manifest["promotion"]["macro_delta_pp"] == 0.30
    assert manifest["promotion"]["minimum_directional_seeds"] == 2
    assert {trial["name"] for trial in manifest["trials"]} == set(EXPECTED)
    for trial in manifest["trials"]:
        path = ROOT / trial["config"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == trial["config_sha256"]


def test_runtime_materialization_is_explicit_hash_bound_and_idempotent(tmp_path):
    runner = _import(
        ROOT / "scripts" / "run_rematch750_decay_filter.py",
        "decay_filter_runner",
    )
    source = CONFIG_DIR / "DF03.yaml"
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
            CONFIG_DIR / "DF02.yaml", "npu:UNASSIGNED", tmp_path / "bad"
        )
