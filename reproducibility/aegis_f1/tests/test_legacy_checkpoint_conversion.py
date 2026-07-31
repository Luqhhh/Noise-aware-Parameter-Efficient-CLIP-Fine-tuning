import pytest
import torch

from aegis_clip.cli.convert_legacy_linear_checkpoint import _require_exact_state


def test_exact_state_accepts_bit_identical_tensors() -> None:
    source = {
        "weight": torch.tensor([[1.0, 2.0]]),
        "bias": torch.tensor([3.0]),
    }
    target = {name: value.clone() for name, value in source.items()}

    _require_exact_state(source, target)


def test_exact_state_rejects_value_change() -> None:
    source = {"weight": torch.tensor([1.0])}
    target = {"weight": torch.tensor([1.0 + 1.0e-6])}

    with pytest.raises(ValueError, match="tensor changed"):
        _require_exact_state(source, target)


def test_exact_state_rejects_key_change() -> None:
    with pytest.raises(ValueError, match="keys changed"):
        _require_exact_state(
            {"weight": torch.tensor([1.0])},
            {"other": torch.tensor([1.0])},
        )
