"""Unified model build → PEFT → strict checkpoint loading.

Provides a single entry point used by training, evaluation, inference,
and TTA scripts so that ``visual_lora`` (and future PEFT types) are
applied consistently before weights are loaded.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _sha256_hex(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_and_load_model(
    config: Dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
    *,
    build_model_fn=None,
    strict: bool = True,
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """Build a model, apply PEFT, and strict-load checkpoint weights.

    This is the canonical entry point for all scripts that need to
    reconstruct a trained model.  It ensures that PEFT transformations
    (especially ``visual_lora``) are applied *before* weight loading so
    that the LoRA wrappers capture the correct parent base weights.

    Parameters
    ----------
    config:
        Full project config dict (from ``load_config``).
    checkpoint_path:
        Path to the ``.pt`` checkpoint file.
    device:
        Torch device.
    build_model_fn:
        Optional function ``(config, device) -> (model, preprocess)``.
        When ``None``, defaults to ``experiments.baseline.model.build_model``.
    strict:
        Whether to require exact key match (default ``True``).

    Returns
    -------
    (model, preprocess, load_info)
        *model*: the PEFT-configured, weight-loaded model in eval mode.
        *preprocess*: the CLIP preprocessing transform.
        *load_info*: dict with keys ``missing_keys``, ``unexpected_keys``,
          ``checkpoint_sha256``, ``checkpoint_epoch``, ``parent_best_val_acc``.

    Raises
    ------
    FileNotFoundError:
        If *checkpoint_path* does not exist.
    RuntimeError:
        If ``strict=True`` and keys do not exactly match.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ── 1. Build model ──────────────────────────────────────────
    if build_model_fn is None:
        from experiments.baseline.model import build_model as _default_build
        build_model_fn = _default_build

    model, preprocess = build_model_fn(config, device)

    # ── 2. Apply PEFT (MUST happen before weight loading) ───────
    peft_cfg = config.get("peft", {})
    peft_type = peft_cfg.get("type", "linear_head_only")
    if peft_type != "linear_head_only":
        from common.peft import apply_peft
        peft_info = apply_peft(model, peft_cfg)
        logger.info("PEFT applied: type=%s, trainable=%d",
                     peft_type, peft_info["trainable_param_count"])

    # ── 3. Load checkpoint (strict) ─────────────────────────────
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get("model_state_dict", checkpoint)

    missing, unexpected = model.load_state_dict(model_state, strict=strict)

    if strict:
        if missing:
            raise RuntimeError(
                f"Strict load failed: {len(missing)} missing keys. "
                f"First 5: {missing[:5]}"
            )
        if unexpected:
            raise RuntimeError(
                f"Strict load failed: {len(unexpected)} unexpected keys. "
                f"First 5: {unexpected[:5]}"
            )
        logger.info("Checkpoint strict-loaded: 0 missing, 0 unexpected")
    else:
        if missing:
            logger.warning("Missing keys (%d): %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:5])

    model.eval()

    # ── 4. Audit info ───────────────────────────────────────────
    ckpt_sha = _sha256_hex(ckpt_path)
    load_info: Dict[str, Any] = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "parent_best_val_acc": checkpoint.get("best_val_acc", None),
    }
    logger.info("Checkpoint SHA-256: %s", ckpt_sha)

    return model, preprocess, load_info
