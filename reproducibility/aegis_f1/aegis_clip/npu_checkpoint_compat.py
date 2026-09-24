"""Compatibility shim for loading Ascend NPU checkpoints on non-NPU hosts.

NPU checkpoints written by ``torch_npu`` may contain two pickle globals:

* ``torch_npu.utils.storage._rebuild_npu_tensor``
* ``torch_npu.npu._format.Format``

The actual tensor payloads are ordinary CPU/GPU-compatible storages once
unpickled, so for inference/training-initialisation on CUDA we can rebuild
them through PyTorch's standard tensor rebuild path and ignore the NPU
format marker.  The shim is installed only when the real ``torch_npu`` package
is unavailable; it never overrides a working NPU environment.
"""

from __future__ import annotations

import importlib
import sys
import threading
import types


_LOCK = threading.Lock()
_INSTALLED = False


class _Format:
    """Placeholder for ``torch_npu.npu._format.Format``."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"NPUFormat(args={self.args!r}, kwargs={self.kwargs!r})"


def _rebuild_npu_tensor(*args, **kwargs):
    """Rebuild an NPU-marked tensor through PyTorch's standard tensor path."""
    import torch

    if len(args) < 6:
        raise RuntimeError(
            "NPU checkpoint compatibility shim received too few rebuild args"
        )
    return torch._utils._rebuild_tensor_v2(*args[:6], **kwargs)


def ensure_npu_checkpoint_stubs() -> bool:
    """Install fake ``torch_npu`` modules if the real package is absent.

    Returns ``True`` when the shim is active, ``False`` when a real
    ``torch_npu`` package was found and left untouched.
    """
    global _INSTALLED
    with _LOCK:
        try:
            importlib.import_module("torch_npu")
            return False
        except ImportError:
            pass
        if _INSTALLED:
            return True

        storage_mod = types.ModuleType("torch_npu.utils.storage")
        storage_mod._rebuild_npu_tensor = _rebuild_npu_tensor
        utils_mod = types.ModuleType("torch_npu.utils")
        utils_mod.storage = storage_mod
        format_mod = types.ModuleType("torch_npu.npu._format")
        format_mod.Format = _Format
        npu_mod = types.ModuleType("torch_npu.npu")
        npu_mod._format = format_mod
        root_mod = types.ModuleType("torch_npu")
        root_mod.utils = utils_mod
        root_mod.npu = npu_mod

        sys.modules.update(
            {
                "torch_npu": root_mod,
                "torch_npu.utils": utils_mod,
                "torch_npu.utils.storage": storage_mod,
                "torch_npu.npu": npu_mod,
                "torch_npu.npu._format": format_mod,
            }
        )
        _INSTALLED = True
        return True
