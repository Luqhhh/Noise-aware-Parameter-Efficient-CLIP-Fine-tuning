"""Parameter-Efficient Fine-Tuning configuration.

Provides a single entry point ``configure_peft(model, peft_config)`` that
applies the requested PEFT strategy and returns parameter group info for
the optimizer.

C implements LoRA internals; A provides the hook interface and config parsing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch.nn as nn

logger = logging.getLogger(__name__)

KNOWN_PEFT_TYPES = {
    "linear_head_only",
    "ln_post_and_proj",
    "visual_layernorm_only",
    "last_block_lora",
}


def configure_peft(model: nn.Module, peft_config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a PEFT configuration to a model.

    Args:
        model: The model (must have a ``visual`` attribute for CLIP-based models).
        peft_config: Dict with keys:
            type (str): One of ``KNOWN_PEFT_TYPES``.
            train_ln_post (bool): For ``ln_post_and_proj``.
            train_visual_proj (bool): For ``ln_post_and_proj``.
            train_visual_layernorm (bool): For ``visual_layernorm_only``.
            lora_rank (int): For ``last_block_lora``.
            lora_alpha (int): For ``last_block_lora``.
            lora_dropout (float): For ``last_block_lora``.
            backbone_lr (float): LR for trainable backbone params.
            backbone_weight_decay (float): WD for trainable backbone params.

    Returns:
        Dict with keys:
            type: str — resolved PEFT type.
            trainable_param_count: int.
            trainable_param_names: list[str] (first 20).
            param_groups: list[dict] for optimizer.
    """
    peft_type = peft_config.get("type", "linear_head_only")
    if peft_type not in KNOWN_PEFT_TYPES:
        raise ValueError(
            f"Unknown peft.type: {peft_type}. Known: {sorted(KNOWN_PEFT_TYPES)}"
        )

    # Always start from frozen
    if hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad_(False)

    # Head is always trainable
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad_(True)

    backbone_params = []

    if peft_type == "linear_head_only":
        # Default — nothing extra to unfreeze
        pass

    elif peft_type == "ln_post_and_proj":
        if not hasattr(model, "visual"):
            raise ValueError("Model has no 'visual' attribute — cannot apply PEFT")
        if peft_config.get("train_ln_post", True):
            for p in model.visual.ln_post.parameters():
                p.requires_grad_(True)
                backbone_params.append(p)
        if peft_config.get("train_visual_proj", True):
            for p in model.visual.proj.parameters():
                p.requires_grad_(True)
                backbone_params.append(p)

    elif peft_type == "visual_layernorm_only":
        if not hasattr(model, "visual"):
            raise ValueError("Model has no 'visual' attribute — cannot apply PEFT")
        for name, module in model.visual.named_modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad_(True)
                    backbone_params.append(p)

    elif peft_type == "last_block_lora":
        # Placeholder — C implements LoRA module
        logger.warning(
            "last_block_lora: LoRA not yet implemented. "
            "Falling back to linear_head_only."
        )

    # Build param group info
    head_params = list(model.classifier.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [
        n for n, p in model.named_parameters() if p.requires_grad
    ]

    bb_lr = peft_config.get("backbone_lr", 1e-5)
    bb_wd = peft_config.get("backbone_weight_decay", 0.01)

    param_groups = [
        {"name": "head", "params": head_params},
    ]
    if backbone_params:
        param_groups.append({
            "name": "backbone",
            "params": backbone_params,
            "lr": bb_lr,
            "weight_decay": bb_wd,
        })

    result = {
        "type": peft_type,
        "trainable_param_count": len(trainable),
        "trainable_param_names": trainable_names[:20],
        "param_groups": param_groups,
    }

    logger.info(
        "PEFT: type=%s, trainable=%d/%d params",
        peft_type, result["trainable_param_count"],
        sum(p.numel() for p in model.parameters()),
    )

    return result
