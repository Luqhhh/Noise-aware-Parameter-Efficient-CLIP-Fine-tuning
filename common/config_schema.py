"""Minimal config schema validation — fail-closed on unknown fields.

Does NOT validate every nested value.  Only checks that top-level keys are
known and that a small set of required structural keys are present.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Known top-level config sections.  Unknown keys in these sections are
# ignored (non-strict) — we only enforce at the top level.
KNOWN_TOP_KEYS = {
    "experiment",
    "data",
    "model",
    "loss",
    "train",
    "eval",
    "output",
    "sample_weighting",
    "head_ema",
    "peft",
    "teacher",
    "cache",
    "runtime",       # written back by resolve_runtime_args
}

# Keys that MUST be present in every config
REQUIRED_TOP_KEYS = {"experiment", "data", "model", "train", "eval", "output"}

# Known sample_weighting types
KNOWN_WEIGHT_TYPES = {
    "none", "static_manifest", "ema_loss", "prototype",
    "prototype_ema_hybrid", "oof_manifest", "relabel_manifest",
}

# Known loss names
KNOWN_LOSS_NAMES = {"cross_entropy", "gce", "label_smoothing"}

# Known PEFT types
KNOWN_PEFT_TYPES = {
    "linear_head_only", "ln_post_and_proj",
    "visual_layernorm_only", "last_block_lora",
}


def validate_config(config: dict) -> List[str]:
    """Validate config structure and return warnings.

    Raises ``ValueError`` on hard failures (unknown top-level key,
    missing required key).

    Args:
        config: Full project config dictionary.

    Returns:
        List of warning strings (non-fatal issues).
    """
    warnings: List[str] = []

    # ── Top-level keys ──
    unknown = set(config.keys()) - KNOWN_TOP_KEYS
    if unknown:
        raise ValueError(
            f"Unknown top-level config keys: {sorted(unknown)}. "
            f"Known keys: {sorted(KNOWN_TOP_KEYS)}"
        )

    missing = REQUIRED_TOP_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")

    # ── experiment section ──
    exp = config.get("experiment", {})
    if "id" not in exp:
        warnings.append("experiment.id is missing — using default")
    if exp.get("head_type") not in (None, "linear", "cosine"):
        raise ValueError(f"Unknown head_type: {exp['head_type']}")

    # ── model section ──
    model = config.get("model", {})
    if model.get("clip_model_name", "ViT-B/32") != "ViT-B/32":
        raise ValueError(
            f"Only ViT-B/32 is allowed, got {model.get('clip_model_name')}"
        )

    # ── loss section ──
    loss_cfg = config.get("loss", {})
    if loss_cfg:
        loss_name = loss_cfg.get("name", "cross_entropy")
        if loss_name not in KNOWN_LOSS_NAMES and "schedule" not in loss_cfg:
            warnings.append(f"Unknown loss name: {loss_name}")

    # ── sample_weighting section ──
    sw = config.get("sample_weighting", {})
    if sw:
        sw_type = sw.get("type", "none")
        if sw_type not in KNOWN_WEIGHT_TYPES:
            raise ValueError(
                f"Unknown sample_weighting.type: {sw_type}. "
                f"Known: {sorted(KNOWN_WEIGHT_TYPES)}"
            )

    # ── peft section ──
    peft = config.get("peft", {})
    if peft:
        peft_type = peft.get("type", "linear_head_only")
        if peft_type not in KNOWN_PEFT_TYPES:
            raise ValueError(
                f"Unknown peft.type: {peft_type}. "
                f"Known: {sorted(KNOWN_PEFT_TYPES)}"
            )

    # ── train section ──
    train = config.get("train", {})
    if train.get("epochs", 0) <= 0:
        raise ValueError("train.epochs must be > 0")

    return warnings
