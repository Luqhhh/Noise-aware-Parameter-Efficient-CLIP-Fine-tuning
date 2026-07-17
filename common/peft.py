"""Parameter-Efficient Fine-Tuning configuration.

Applies PEFT strategies to a CLIPLinearClassifier model from config alone.
Freezes all parameters first, then selectively unfreezes or attaches adapters
according to ``peft.type``.

Supported types:

    linear_head_only      — Only the classifier head is trainable (default).
    ln_post_and_proj      — Head + visual.ln_post + visual.proj.
    visual_layernorm_only — Head + all visual LayerNorm (blocks + ln_post).
    last_block_lora       — Head + LoRA on last transformer block's attn.out_proj.

Usage in train.py::

    from common.peft import apply_peft, build_peft_param_groups, get_peft_config

    peft_cfg = get_peft_config(config)
    peft_info = apply_peft(model, peft_cfg)
    param_groups = build_peft_param_groups(
        model, peft_cfg, head_lr=train_cfg["lr"],
        head_weight_decay=train_cfg.get("weight_decay", 0.0),
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

KNOWN_PEFT_TYPES = {
    "linear_head_only",
    "ln_post_and_proj",
    "visual_layernorm_only",
    "last_block_lora",
}


def get_peft_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract resolved PEFT config from full project config."""
    return config.get("peft", {})


def apply_peft(
    model: nn.Module,
    peft_cfg: Optional[Dict[str, Any]] = None,
) -> dict:
    """Apply PEFT configuration to a model.

    Freezes **all** parameters first, then selectively unfreezes or attaches
    adapters.

    Args:
        model: A ``CLIPLinearClassifier`` instance.
        peft_cfg: PEFT config dict, or None/{} for default ``linear_head_only``.

    Returns:
        Audit dict with keys: peft_type, trainable_param_names,
        trainable_param_count, frozen_param_count, lora_layers.
    """
    if peft_cfg is None:
        peft_cfg = {}

    peft_type = peft_cfg.get("type", "linear_head_only")
    if peft_type not in KNOWN_PEFT_TYPES:
        raise ValueError(
            f"Unknown peft.type: {peft_type!r}. Known: {sorted(KNOWN_PEFT_TYPES)}"
        )

    # ── Step 1: freeze everything ──────────────────────────────────────
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = []

    # ── Step 2: unfreeze classifier head (unless explicitly frozen for diagnostics) ──
    freeze_classifier = peft_cfg.get("freeze_classifier", False)
    if not freeze_classifier:
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        logger.info("PEFT: classifier frozen (freeze_classifier=True)")

    # ── Step 3: dispatch by type ───────────────────────────────────────
    if peft_type == "linear_head_only":
        pass

    elif peft_type == "ln_post_and_proj":
        _unfreeze_ln_post_and_proj(model)

    elif peft_type == "visual_layernorm_only":
        _unfreeze_visual_layernorm(model)

    elif peft_type == "last_block_lora":
        lora_layers = _apply_last_block_lora(model, peft_cfg)

    # ── Step 4: audit ──────────────────────────────────────────────────
    trainable_names = [
        name for name, p in model.named_parameters() if p.requires_grad
    ]
    trainable_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_count = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    total_count = trainable_count + frozen_count

    logger.info("PEFT type: %s", peft_type)
    logger.info("  Trainable: %d / %d (%.2f%%)",
                trainable_count, total_count,
                100.0 * trainable_count / max(total_count, 1))
    logger.info("  Frozen:    %d", frozen_count)

    return {
        "peft_type": peft_type,
        "trainable_param_names": trainable_names,
        "trainable_param_count": trainable_count,
        "frozen_param_count": frozen_count,
        "lora_layers": lora_layers,
    }


def build_peft_param_groups(
    model: nn.Module,
    peft_cfg: Dict[str, Any],
    head_lr: float,
    head_weight_decay: float,
) -> List[dict]:
    """Build optimizer parameter groups from PEFT config.

    Groups returned depend on ``peft.type``:

    - ``linear_head_only``: [head]
    - ``ln_post_and_proj``: [head, backbone]
    - ``visual_layernorm_only``: [head, backbone]
    - ``last_block_lora``: [head, lora]

    Args:
        model: PEFT-configured model.
        peft_cfg: PEFT config dict.
        head_lr: Learning rate for the classifier head.
        head_weight_decay: Weight decay for the classifier head.

    Returns:
        List of param group dicts, each with keys: name, params, lr, weight_decay.
    """
    peft_type = peft_cfg.get("type", "linear_head_only")

    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "classifier" in n
    ]

    groups = [{
        "name": "head",
        "params": head_params,
        "lr": head_lr,
        "weight_decay": head_weight_decay,
    }]

    if peft_type in ("ln_post_and_proj", "visual_layernorm_only"):
        backbone_lr = peft_cfg.get("backbone_lr", 1e-5)
        backbone_wd = peft_cfg.get("backbone_weight_decay", 0.01)
        backbone_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "classifier" not in n and "lora_" not in n
        ]
        if backbone_params:
            groups.append({
                "name": "backbone",
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": backbone_wd,
            })

    elif peft_type == "last_block_lora":
        lora_lr = peft_cfg.get("lora", {}).get("lora_lr", 1e-5)
        lora_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "lora_" in n
        ]
        if lora_params:
            groups.append({
                "name": "lora",
                "params": lora_params,
                "lr": lora_lr,
                "weight_decay": 0.0,  # LoRA typically uses no weight decay
            })

    # Log groups
    for g in groups:
        param_count = sum(p.numel() for p in g["params"])
        logger.info(
            "  %-12s: %8d params, lr=%.1e, wd=%.1e",
            g["name"], param_count, g.get("lr", head_lr),
            g.get("weight_decay", head_weight_decay),
        )

    return groups


# ── Internal helpers ────────────────────────────────────────────────────


def _unfreeze_ln_post_and_proj(model: nn.Module):
    """Unfreeze visual.ln_post and visual.proj."""
    visual = model.visual

    for param in visual.ln_post.parameters():
        param.requires_grad = True
    logger.info("PEFT: unfrozen visual.ln_post (weight + bias)")

    proj = getattr(visual, "proj", None)
    if proj is not None:
        proj.requires_grad = True
        logger.info("PEFT: unfrozen visual.proj")
    else:
        logger.warning("PEFT: visual.proj not found")


def _unfreeze_visual_layernorm(model: nn.Module):
    """Unfreeze all LayerNorm in transformer blocks + ln_pre + ln_post."""
    visual = model.visual

    # ln_pre
    ln_pre = getattr(visual, "ln_pre", None)
    if ln_pre is not None:
        for param in ln_pre.parameters():
            param.requires_grad = True
        logger.info("PEFT: unfrozen visual.ln_pre")

    # Transformer block LayerNorms (ln_1, ln_2 in each block)
    blocks = visual.transformer.resblocks
    ln_count = 0
    for i, block in enumerate(blocks):
        for ln_name in ("ln_1", "ln_2"):
            ln = getattr(block, ln_name, None)
            if ln is not None:
                for param in ln.parameters():
                    param.requires_grad = True
                ln_count += 1
    logger.info("PEFT: unfrozen %d LayerNorms across %d blocks", ln_count, len(blocks))

    # ln_post
    for param in visual.ln_post.parameters():
        param.requires_grad = True
    logger.info("PEFT: unfrozen visual.ln_post")


def _apply_last_block_lora(model: nn.Module, peft_cfg: dict) -> list:
    """Apply LoRA to the last transformer block's attention out_proj.

    Returns list of LoRALinear wrappers for state-dict serialisation.
    """
    from common.lora import apply_lora_to_block

    lora_cfg = peft_cfg.get("lora", {})
    r = lora_cfg.get("rank", 4)
    alpha = lora_cfg.get("alpha", 8)
    dropout = lora_cfg.get("dropout", 0.0)
    target_block_idx = lora_cfg.get("target_block", 11)

    blocks = model.visual.transformer.resblocks
    num_blocks = len(blocks)

    if not 0 <= target_block_idx < num_blocks:
        raise ValueError(
            f"LoRA target_block must be in [0, {num_blocks - 1}], "
            f"got {target_block_idx}"
        )

    target_block = blocks[target_block_idx]
    lora_layers = apply_lora_to_block(
        target_block, r=r, alpha=alpha, dropout=dropout,
        target_modules=("out_proj",),
    )

    logger.info(
        "PEFT: LoRA applied to block %d/%d attn.out_proj (r=%d, alpha=%d)",
        target_block_idx, num_blocks, r, alpha,
    )

    return lora_layers


# ── Audit utilities ─────────────────────────────────────────────────────


def capture_reference_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Capture frozen parameter state for epoch-0 equivalence check."""
    return {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if not param.requires_grad
    }


def audit_frozen_parameters(
    model: nn.Module,
    reference_state: Dict[str, torch.Tensor],
) -> Tuple[int, int, List[str]]:
    """Verify frozen parameters haven't changed since reference_state.

    Returns:
        (ok_count, changed_count, changed_names).
    """
    changed = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            if name in reference_state:
                if not torch.equal(param.data, reference_state[name].data):
                    changed.append(name)
            # If name not in reference, it's newly frozen — not a violation

    ok_count = sum(
        1 for n, p in model.named_parameters()
        if not p.requires_grad and n not in changed
    )
    return ok_count, len(changed), changed
