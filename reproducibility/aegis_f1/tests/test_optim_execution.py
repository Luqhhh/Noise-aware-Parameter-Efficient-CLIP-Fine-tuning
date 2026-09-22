import math
from pathlib import Path

import pytest
import torch

from aegis_clip.optim import build_adamw, gradient_norm


@pytest.mark.parametrize("implementation", ["sum_squares", "vector", "foreach"])
def test_group_norm_preserves_scale_and_nonfinite_detection(implementation):
    a = torch.nn.Parameter(torch.zeros(2))
    b = torch.nn.Parameter(torch.zeros(1))
    a.grad = torch.tensor([3., 4.])
    b.grad = torch.tensor([12.])
    assert gradient_norm([a, b], implementation) == 13
    assert gradient_norm([], implementation) == 0
    a.grad[0] = float("inf")
    assert math.isinf(gradient_norm([a, b], implementation))


def test_optimizer_rejects_wrong_device_and_unknown_implementation():
    parameter = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match="requires an NPU"):
        build_adamw([parameter], torch.device("cpu"), "npu_fused_adamw")
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_adamw([parameter], torch.device("cpu"), "typo")


def test_default_optimizer_preserves_group_hyperparameters():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = build_adamw([dict(params=[parameter], lr=3e-6, weight_decay=1e-4)], torch.device("cpu"))
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 3e-6
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4
    assert optimizer.defaults["foreach"] is None


@pytest.mark.parametrize("key,value", [("optimizer_impl", "typo"),
                                       ("gradient_norm_impl", "typo"),
                                       ("optimizer_impl", "npu_fused_adamw")])
def test_config_rejects_invalid_execution_options(key, value):
    from aegis_clip.config import ConfigError, load_config, validate_config
    config = load_config(Path(__file__).resolve().parents[3] / "configs/rematch750_ft.yaml")
    config["train"][key] = value
    with pytest.raises(ConfigError):
        validate_config(config)
