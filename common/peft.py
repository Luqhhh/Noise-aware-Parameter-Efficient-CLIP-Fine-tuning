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
import torch.nn.functional as F

logger = logging.getLogger(__name__)

KNOWN_PEFT_TYPES = {
    "linear_head_only",
    "ln_post_and_proj",
    "visual_layernorm_only",
    "last_block_lora",
    "visual_lora",
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

    elif peft_type == "visual_lora":
        lora_layers = _apply_visual_lora(model, peft_cfg)

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
    backbone_lr: Optional[float] = None,
    backbone_weight_decay: Optional[float] = None,
) -> List[dict]:
    """Build optimizer parameter groups from PEFT config.

    Groups returned depend on ``peft.type``:

    - ``linear_head_only``: [head]
    - ``ln_post_and_proj``: [head, backbone]
    - ``visual_layernorm_only``: [head, backbone]
    - ``last_block_lora``: [head, lora]
    - ``visual_lora``: [head, lora]

    Args:
        model: PEFT-configured model.
        peft_cfg: PEFT config dict.
        head_lr: Learning rate for the classifier head.
        head_weight_decay: Weight decay for the classifier head.
        backbone_lr: Learning rate for backbone/LoRA parameters
            (used by ``visual_lora``; for other types the LR is read
            from *peft_cfg*).
        backbone_weight_decay: Weight decay for backbone/LoRA parameters.

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
        backbone_lr_val = peft_cfg.get("backbone_lr", 1e-5)
        backbone_wd = peft_cfg.get("backbone_weight_decay", 0.01)
        backbone_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "classifier" not in n and "lora_" not in n
        ]
        if backbone_params:
            groups.append({
                "name": "backbone",
                "params": backbone_params,
                "lr": backbone_lr_val,
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

    elif peft_type == "visual_lora":
        # LoRA LR from backbone_lr arg (plan-specified), fallback to peft_cfg
        if backbone_lr is not None:
            lora_lr = backbone_lr
        else:
            lora_lr = peft_cfg.get("lora", {}).get("lora_lr", 1e-5)
        lora_wd = backbone_weight_decay if backbone_weight_decay is not None else 0.0

        lora_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "lora_" in n
        ]
        if lora_params:
            groups.append({
                "name": "lora",
                "params": lora_params,
                "lr": lora_lr,
                "weight_decay": lora_wd,
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


def _apply_visual_lora(model: nn.Module, peft_cfg: dict) -> list:
    """Apply LoRA to the last N transformer blocks' attention projections.

    Targets Q, V (via fused ``in_proj_weight`` split) and ``out_proj``
    according to the config flags.  Each adapted attention module is
    replaced with a :class:`_QKVLoRAPatchedAttention` — a proper
    ``nn.Module`` that survives ``deepcopy`` correctly.

    Config keys (all under the ``peft`` section):

    - ``lora_last_n_blocks``: int (default 4) — number of final blocks to adapt.
    - ``lora_rank``: int (default 8).
    - ``lora_alpha``: int (default 8).
    - ``lora_adapt_qv``: bool (default True) — adapt Q and V projections.
    - ``lora_adapt_out``: bool (default True) — adapt output projection.
    """
    n_last = peft_cfg.get("lora_last_n_blocks", 4)
    r = peft_cfg.get("lora_rank", 8)
    alpha = peft_cfg.get("lora_alpha", 8)
    adapt_qv = peft_cfg.get("lora_adapt_qv", True)
    adapt_out = peft_cfg.get("lora_adapt_out", True)

    blocks = model.visual.transformer.resblocks
    num_blocks = len(blocks)

    if n_last > num_blocks:
        raise ValueError(
            f"lora_last_n_blocks ({n_last}) exceeds total blocks ({num_blocks})"
        )
    start_block = num_blocks - n_last

    all_lora_layers: list = []

    for block_idx in range(start_block, num_blocks):
        block = blocks[block_idx]
        orig_attn = block.attn

        # Replace attention with proper nn.Module wrapper
        patched = _QKVLoRAPatchedAttention(
            orig_attn, r=r, alpha=alpha,
            adapt_qv=adapt_qv, adapt_out=adapt_out,
        )
        block.attn = patched

        # Collect LoRA layers for state-dict tracking
        if adapt_out:
            all_lora_layers.append(patched.out_proj)
            logger.info(
                "LoRA out_proj: block %d/%d (r=%d, alpha=%d)",
                block_idx, num_blocks, r, alpha,
            )
        if adapt_qv:
            all_lora_layers.extend([patched.q_proj, patched.v_proj])
            logger.info(
                "LoRA Q/V: block %d/%d (r=%d, alpha=%d) — "
                "fused in_proj_weight split; Q/V adapted, K frozen",
                block_idx, num_blocks, r, alpha,
            )

    return all_lora_layers


class _QKVLoRAPatchedAttention(nn.Module):
    """Replace a CLIP MultiheadAttention with LoRA on Q and V.

    Instead of using ``use_separate_proj_weight=True`` (which drops Q/K/V
    biases in the PyTorch API), this module dynamically reconstructs a
    modified fused ``in_proj_weight`` that includes LoRA deltas on the Q
    and V slices while preserving K and the original bias structure.

    This is a proper ``nn.Module`` — ``deepcopy`` copies the entire
    submodule tree correctly.

    Parameters
    ----------
    orig_attn:
        The original ``nn.MultiheadAttention`` module.
    r:
        LoRA rank.
    alpha:
        LoRA scaling factor.
    adapt_qv:
        If True, inject LoRA on Q/V via modified fused ``in_proj_weight``.
    adapt_out:
        If True, wrap ``out_proj`` with ``LoRALinear``.
    """

    def __init__(self, orig_attn: nn.Module, r: int, alpha: int,
                 adapt_qv: bool = True, adapt_out: bool = True):
        from common.lora import LoRALinear

        super().__init__()
        self.embed_dim: int = orig_attn.embed_dim
        self.num_heads: int = orig_attn.num_heads
        self.dropout: float = orig_attn.dropout
        self.add_zero_attn: bool = orig_attn.add_zero_attn

        # Register bias_k / bias_v as buffers if present
        if orig_attn.bias_k is not None:
            self.register_buffer("bias_k", orig_attn.bias_k.data.clone())
        else:
            self.bias_k = None
        if orig_attn.bias_v is not None:
            self.register_buffer("bias_v", orig_attn.bias_v.data.clone())
        else:
            self.bias_v = None

        embed_dim = self.embed_dim
        device = orig_attn.in_proj_weight.device
        dtype = orig_attn.in_proj_weight.dtype
        has_bias = orig_attn.in_proj_bias is not None

        # ── out_proj: optionally wrap with LoRA ──────────────────
        if adapt_out and isinstance(orig_attn.out_proj, nn.Linear):
            self.out_proj = LoRALinear(orig_attn.out_proj, r=r, alpha=alpha)
        else:
            self.out_proj = orig_attn.out_proj

        # ── Q / V LoRA on fused in_proj_weight ───────────────────
        # Store the base fused weight as a registered buffer so that
        # weight decay / freezing work correctly.  The effective
        # in_proj_weight in forward = base + LoRA delta on Q & V.
        if adapt_qv:
            fused_w = orig_attn.in_proj_weight.data.clone()   # [3*E, E]
            q_base = fused_w[:embed_dim, :].clone()
            k_base = fused_w[embed_dim:2 * embed_dim, :].clone()
            v_base = fused_w[2 * embed_dim:, :].clone()

            # Create independent nn.Linear for Q, K, V so LoRA hooks
            # have standard weight/bias parameters.
            q_linear = nn.Linear(embed_dim, embed_dim, bias=False).to(
                device=device, dtype=dtype)
            k_linear = nn.Linear(embed_dim, embed_dim, bias=False).to(
                device=device, dtype=dtype)
            v_linear = nn.Linear(embed_dim, embed_dim, bias=False).to(
                device=device, dtype=dtype)

            q_linear.weight.data.copy_(q_base)
            k_linear.weight.data.copy_(k_base)
            v_linear.weight.data.copy_(v_base)

            self.q_proj = LoRALinear(q_linear, r=r, alpha=alpha)
            self.k_proj = k_linear        # plain nn.Linear, frozen
            self.v_proj = LoRALinear(v_linear, r=r, alpha=alpha)

            # Store in_proj_bias as a registered buffer
            if has_bias:
                self.register_buffer(
                    "in_proj_bias", orig_attn.in_proj_bias.data.clone(),
                )
            else:
                self.in_proj_bias = None

            for p in self.k_proj.parameters():
                p.requires_grad_(False)
        else:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.register_buffer(
                "in_proj_weight_orig", orig_attn.in_proj_weight.data.clone(),
            )
            if has_bias:
                self.register_buffer(
                    "in_proj_bias", orig_attn.in_proj_bias.data.clone(),
                )
            else:
                self.in_proj_bias = None

    # ── Property: effective fused in_proj_weight ───────────────────
    def _get_effective_in_proj_weight(self) -> torch.Tensor:
        """Build the fused ``[3*E, E]`` weight with LoRA deltas on Q/V."""
        if self.q_proj is not None:
            embed_dim = self.embed_dim
            q_eff = self.q_proj.weight   # base + LoRA delta
            k_eff = self.k_proj.weight   # frozen base
            v_eff = self.v_proj.weight   # base + LoRA delta
            return torch.cat([q_eff, k_eff, v_eff], dim=0)
        else:
            return getattr(self, "in_proj_weight_orig")

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights: bool = True, attn_mask=None,
                average_attn_weights: bool = True,
                is_causal: bool = False):
        """Patched forward using modified fused ``in_proj_weight``."""
        return F.multi_head_attention_forward(
            query, key, value,
            self.embed_dim, self.num_heads,
            self._get_effective_in_proj_weight(),
            getattr(self, "in_proj_bias", None),
            self.bias_k, self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )


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
