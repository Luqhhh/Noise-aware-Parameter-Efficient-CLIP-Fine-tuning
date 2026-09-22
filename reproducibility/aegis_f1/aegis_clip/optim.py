"""Explicit optimizer/norm execution choices; defaults preserve existing recipes."""
from __future__ import annotations

import torch


def build_adamw(groups, device, implementation="default"):
    if implementation == "npu_fused_adamw":
        if device.type != "npu":
            raise ValueError("npu_fused_adamw requires an NPU")
        from torch_npu.optim import NpuFusedAdamW
        return NpuFusedAdamW(groups)
    if implementation == "foreach":
        return torch.optim.AdamW(groups, foreach=True)
    if implementation != "default":
        raise ValueError(f"Unknown optimizer implementation: {implementation}")
    return torch.optim.AdamW(groups, **({"foreach": False} if device.type == "npu" else {}))


def gradient_norm(parameters, implementation="sum_squares"):
    grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    if implementation == "sum_squares":
        return float(torch.stack([g.pow(2).sum() for g in grads]).sum().sqrt())
    if implementation == "foreach":
        return float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))))
    if implementation == "vector":
        return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])))
    raise ValueError(f"Unknown gradient norm implementation: {implementation}")
