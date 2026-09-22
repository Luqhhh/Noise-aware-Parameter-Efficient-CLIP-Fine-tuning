from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from aegis_clip import device, checkpoint
from aegis_clip.config import load_config


def test_cuda_fallback_must_be_explicit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        device.resolve_device("cuda")
    assert device.resolve_device("cuda", allow_cuda_fallback=True).type == "cpu"


def test_npu_import_failure_never_falls_back(monkeypatch):
    def missing(name):
        raise ImportError(name)
    monkeypatch.setattr(device.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="torch_npu/CANN"):
        device.resolve_device("npu:0", allow_cuda_fallback=True)


def test_npu_device_selection_and_unavailable(monkeypatch):
    backend = SimpleNamespace(is_available=lambda: True, set_device=Mock())
    monkeypatch.setattr(torch, "npu", backend, raising=False)
    monkeypatch.setattr(device.importlib, "import_module", lambda name: backend)
    monkeypatch.setattr(torch, "device", lambda name: SimpleNamespace(type="npu", name=name))
    result = device.resolve_device("npu:2")
    backend.set_device.assert_called_once_with(result)
    assert result.name == "npu:2"
    backend.is_available = lambda: False
    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        device.resolve_device("npu:2")


def test_amp_is_accelerator_only():
    for kind in ("cuda", "npu", "cpu"):
        assert device.amp_enabled(SimpleNamespace(type=kind), True) == (kind != "cpu")
        assert not device.amp_enabled(SimpleNamespace(type=kind), False)


def test_npu_rng_restore_is_not_silently_dropped(monkeypatch):
    import random
    import numpy as np
    state = dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(), npu=[torch.tensor([1], dtype=torch.uint8)])
    monkeypatch.setattr(checkpoint, "npu_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="NPU training RNG"):
        checkpoint._restore_rng(state)
    backend = SimpleNamespace(set_rng_state_all=Mock())
    monkeypatch.setattr(torch, "npu", backend, raising=False)
    monkeypatch.setattr(checkpoint, "npu_initialized", lambda: True)
    checkpoint._restore_rng(state)
    backend.set_rng_state_all.assert_called_once()


@pytest.mark.parametrize("mode", ["lp", "ft"])
def test_npu_recipe_preserves_training_hyperparameters(mode):
    root = Path(__file__).resolve().parents[3]
    cpu = load_config(root / f"configs/rematch750_{mode}.yaml")
    npu = load_config(root / f"configs/rematch750_{mode}_npu.yaml")
    for config in (cpu, npu):
        config.pop("_config_path")
        config["project"].pop("experiment_id")
        config["train"].pop("device")
        config["train"].pop("init_checkpoint", None)
        config["output"].pop("root")
        for key in ("train_csv", "val_csv", "class_mapping", "dataset_manifest"):
            config["data"].pop(key)
        config.pop("features")
    assert cpu == npu


def test_rematch_cache_dispatches_configured_npu(monkeypatch):
    from aegis_clip.cli import rematch, cache_features
    root = Path(__file__).resolve().parents[3]
    target = SimpleNamespace(type="npu")
    resolve = Mock(return_value=target)
    cache = Mock(return_value={"ok": True})
    monkeypatch.setattr(rematch, "resolve_device", resolve)
    monkeypatch.setattr(cache_features, "cache_stage_features", cache)
    assert rematch.execute("cache", root / "configs/rematch750_lp_npu.yaml") == {"ok": True}
    resolve.assert_called_once_with("npu:0")
    assert cache.call_args.kwargs["device"] is target


def test_npu_run_cannot_launch_legacy_cuda_queue(monkeypatch):
    from aegis_clip.cli import rematch
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(rematch, "resolve_device", lambda name: SimpleNamespace(type="npu"))
    with pytest.raises(ValueError, match="explicit NPU"):
        rematch.execute("run", root / "configs/rematch750_lp_npu.yaml")
