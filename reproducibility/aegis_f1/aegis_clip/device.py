"""Explicit accelerator selection without silently moving NPU work to CPU."""
from __future__ import annotations

import importlib

import torch


def resolve_device(requested: str, *, allow_cuda_fallback: bool = False) -> torch.device:
    if str(requested).split(":", 1)[0] == "npu":
        try:
            importlib.import_module("torch_npu")
        except ImportError as exc:
            raise RuntimeError("NPU requested: install the matching torch_npu/CANN stack") from exc
        device = torch.device(requested)
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable; refusing CPU fallback")
        torch.npu.set_device(device)
        return device
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        if allow_cuda_fallback:
            return torch.device("cpu")
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported training device: {requested}")
    return device


def amp_enabled(device: torch.device, requested: bool) -> bool:
    return bool(requested) and device.type in {"cuda", "npu"}


def npu_initialized() -> bool:
    backend = getattr(torch, "npu", None)
    return backend is not None and backend.is_initialized()
