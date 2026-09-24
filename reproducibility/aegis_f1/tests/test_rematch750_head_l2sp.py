from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from aegis_clip.config import load_config
from aegis_clip.model import AegisCLIP


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs" / "rematch750_head_l2sp"
EXPECTED = {
    "HL00": ("RM_HL00_F05_CONTROL", "linear", 1.0e-4),
    "HL01": ("RM_HL01_L2SP_WD1E4", "anchored_residual", 1.0e-4),
    "HL02": ("RM_HL02_L2SP_WD1E3", "anchored_residual", 1.0e-3),
    "HL03": ("RM_HL03_L2SP_WD1E2", "anchored_residual", 1.0e-2),
}


def _comparison_view(config):
    result = copy.deepcopy(config)
    result.pop("_config_path", None)
    result["project"].pop("experiment_id")
    result["model"].pop("classifier_mode", None)
    result["model"].pop("classifier_residual_scale", None)
    result["train"].pop("head_weight_decay")
    return result


def test_configs_parse_and_only_change_registered_fields():
    configs = {name: load_config(CONFIG_DIR / f"{name}.yaml") for name in EXPECTED}
    control_view = _comparison_view(configs["HL00"])
    for name, (experiment_id, mode, head_wd) in EXPECTED.items():
        config = configs[name]
        assert config["project"]["experiment_id"] == experiment_id
        assert config["model"]["classifier_mode"] == mode
        assert config["train"]["head_weight_decay"] == head_wd
        assert config["model"]["input_resolution"] == 320
        assert config["train"]["batch_size"] == 256
        assert config["train"]["grad_accum_steps"] == 4
        assert config["train"]["effective_batch_size"] == 1024
        assert config["train"]["device"] == "npu:UNASSIGNED"
        assert config["evaluation"]["evaluate_initial_checkpoint"] is True
        assert config["evaluation"]["record_initial"] is True
        assert _comparison_view(config) == control_view


class TinyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.positional_embedding = nn.Parameter(torch.zeros(2, 4))
        self.ln_pre = nn.LayerNorm(4)
        self.ln_post = nn.LayerNorm(4)
        self.proj = nn.Parameter(torch.eye(4))


def test_l2sp_epoch0_identity_and_optimizer_scope():
    torch.manual_seed(7)
    linear = AegisCLIP(
        TinyVisual(),
        num_classes=3,
        feature_dim=4,
        peft_mode="full_finetune",
        classifier_mode="linear",
    )
    anchored = AegisCLIP(
        TinyVisual(),
        num_classes=3,
        feature_dim=4,
        peft_mode="full_finetune",
        classifier_mode="anchored_residual",
        classifier_residual_scale=1.0,
    )
    incompatible = anchored.load_state_dict(linear.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "classifier.residual_weight",
        "classifier.residual_bias",
    }
    assert incompatible.unexpected_keys == []

    features = torch.randn(8, 4)
    with torch.no_grad():
        assert torch.equal(linear(features=features), anchored(features=features))

    groups = anchored.parameter_groups(
        head_lr=4.0e-4,
        head_weight_decay=1.0e-2,
        backbone_lr=1.2e-5,
        backbone_weight_decay=1.0e-4,
    )
    head_group = next(group for group in groups if group["name"] == "head")
    assert head_group["weight_decay"] == 1.0e-2
    assert {id(parameter) for parameter in head_group["params"]} == {
        id(anchored.classifier.residual_weight),
        id(anchored.classifier.residual_bias),
    }
    assert anchored.classifier.weight.requires_grad is False
    assert anchored.classifier.bias.requires_grad is False

    base_weight = anchored.classifier.weight.detach().clone()
    residual_weight = anchored.classifier.residual_weight.detach().clone()
    optimizer = torch.optim.AdamW(groups)
    loss = anchored(features=features).square().mean()
    loss.backward()
    optimizer.step()
    assert torch.equal(anchored.classifier.weight, base_weight)
    assert not torch.equal(anchored.classifier.residual_weight, residual_weight)


def test_runtime_materialization_is_explicit_and_reloads(tmp_path):
    script = ROOT / "scripts" / "run_rematch750_head_l2sp.py"
    spec = importlib.util.spec_from_file_location("head_l2sp_runner", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    runtime = module.materialize_config(CONFIG_DIR / "HL02.yaml", "npu:0", tmp_path)
    loaded = load_config(runtime)
    assert loaded["train"]["device"] == "npu:0"
    assert Path(loaded["data"]["train_csv"]).is_absolute()
    assert Path(loaded["output"]["root"]).is_absolute()
    assert module.materialize_config(CONFIG_DIR / "HL02.yaml", "npu:0", tmp_path) == runtime
    runtime.write_text("stale: true\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="different content"):
        module.materialize_config(CONFIG_DIR / "HL02.yaml", "npu:0", tmp_path)
    with pytest.raises(ValueError, match="device must be explicit"):
        module.materialize_config(
            CONFIG_DIR / "HL03.yaml", "npu:UNASSIGNED", tmp_path / "bad"
        )


def test_manifest_hashes_are_bound():
    manifest = json.loads((CONFIG_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ready_for_npu"
    assert manifest["platform_submission"] is False
    assert {trial["name"] for trial in manifest["trials"]} == set(EXPECTED)
    for trial in manifest["trials"]:
        config_path = ROOT / trial["config"]
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == trial["config_sha256"]
