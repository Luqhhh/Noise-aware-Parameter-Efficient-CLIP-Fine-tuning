"""LoRA (Low-Rank Adaptation) for linear layers.

Standard LoRA::

    h = W x + (alpha / r) * B A x

where A ∈ R^{r × in}, B ∈ R^{out × r} are low-rank adapters.
At init, A ∼ N(0, σ²) and B = 0 so the adapter is an identity perturbation.

Supports merge/unmerge for inference and state-dict round-trip for checkpointing.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """LoRA wrapper around an existing nn.Linear layer.

    The original weight is frozen; only A and B are trainable.

    Args:
        base: The original nn.Linear layer to wrap.
        r: Rank of the low-rank decomposition.
        alpha: Scaling factor (output is scaled by alpha / r).
        dropout: Dropout probability on the LoRA path (0 = disabled).
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 4,
        alpha: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0

        in_features = base.in_features
        out_features = base.out_features

        # Freeze base weight
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        # Kaiming uniform init for A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Zero-init for B (so LoRA starts as identity perturbation)
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._merged:
            return F.linear(x, self.base.weight, self.base.bias)

        result = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_out

    def merge(self):
        """Merge LoRA weights into the base weight for inference."""
        if not self._merged:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base.weight.data.add_(delta)
            self._merged = True

    def unmerge(self):
        """Remove LoRA weights from base weight (train mode)."""
        if self._merged:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base.weight.data.sub_(delta)
            self._merged = False

    def trainable_parameters(self):
        """Yield trainable LoRA parameters."""
        yield self.lora_A
        yield self.lora_B
        for p in self.lora_dropout.parameters():
            yield p

    def trainable_parameter_count(self) -> int:
        """Number of trainable parameters in this LoRA adapter."""
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora_to_block(
    block: nn.Module,
    r: int = 4,
    alpha: int = 8,
    dropout: float = 0.0,
    target_modules: tuple = ("out_proj",),
) -> list[LoRALinear]:
    """Apply LoRA to specific linear layers in a transformer block.

    Args:
        block: A CLIP transformer ResBlock.
        r: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: LoRA dropout.
        target_modules: Which linear sub-modules to wrap (e.g., "out_proj").

    Returns:
        List of LoRALinear wrappers created.
    """
    lora_layers = []

    for name in target_modules:
        # Navigate: block.attn.<name>
        attn = block.attn
        if not hasattr(attn, name):
            logger.warning(f"Module 'attn.{name}' not found in block, skipping LoRA.")
            continue

        base_layer = getattr(attn, name)
        if not isinstance(base_layer, nn.Linear):
            logger.warning(
                f"attn.{name} is {type(base_layer).__name__}, not nn.Linear; skipping."
            )
            continue

        lora = LoRALinear(base_layer, r=r, alpha=alpha, dropout=dropout)
        setattr(attn, name, lora)
        lora_layers.append(lora)
        logger.info(
            "LoRA r=%d alpha=%d applied to attn.%s (%d → %d)",
            r, alpha, name, base_layer.in_features, base_layer.out_features,
        )

    return lora_layers
