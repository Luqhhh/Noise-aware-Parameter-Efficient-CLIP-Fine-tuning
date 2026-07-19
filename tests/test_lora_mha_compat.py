"""Regression test for wrapping MultiheadAttention.out_proj with LoRA."""

import torch
import torch.nn as nn

from common.lora import LoRALinear


def test_lora_out_proj_is_compatible_with_multihead_attention():
    attn = nn.MultiheadAttention(embed_dim=8, num_heads=2)
    attn = attn.to(dtype=torch.float64)
    wrapper = LoRALinear(attn.out_proj, r=2, alpha=4)
    attn.out_proj = wrapper

    x = torch.randn(5, 2, 8, dtype=torch.float64)
    output, _ = attn(x, x, x, need_weights=False)
    output.sum().backward()

    assert output.shape == x.shape
    assert wrapper.lora_A.dtype == wrapper.base.weight.dtype
    assert wrapper.lora_B.dtype == wrapper.base.weight.dtype
    assert wrapper.lora_A.grad is not None
    assert wrapper.lora_B.grad is not None
