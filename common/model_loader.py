"""Unified model build → PEFT → strict checkpoint loading.

Provides a single entry point used by training, evaluation, inference,
and TTA scripts so that ``visual_lora`` (and future PEFT types) are
applied consistently before weights are loaded.

Also provides pre-model-building parent verification and two distinct
semantic-signature checks:

1. **init_compat** — child config vs parent checkpoint (for
   ``--init-checkpoint``).  Parent is typically ``linear_head_only``
   while child is ``visual_lora``; only CLIP model, num_classes, and
   head type must match.

2. **inference_rebuild** — runtime config vs checkpoint config (for
   evaluate / infer / TTA loading).  All PEFT parameters must match
   exactly because the checkpoint was produced by the same experiment.
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Known-good A2 parent constants ────────────────────────────────────
_A2_CKPT_SHA = "74ad2856e4449a42397edbda599ae79e8a4c6a6fa923624ef4e91a35e20a2a4c"
_A2_TRAIN_SHA = "646fc7b90b7c244a402f6376d966f40148b5b278dad29cce0c6955c92a1b6666"
_A2_VAL_SHA = "607e019165912bb0639efb456b7e8dea122b3e8579a2344dedb8109798921eae"
_A2_EXPERIMENT_ID = "NR_CL_KNN_DROP"


def _sha256_hex(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────
# Pre-model-building parent verification
# ──────────────────────────────────────────────────────────────────────


def verify_parent_checkpoint(
    init_checkpoint_path: str,
    child_experiment_id: str,
) -> Dict[str, Any]:
    """Verify a parent checkpoint BEFORE any model is built or loaded.

    Checks file existence, artifact manifest, SHA-256, experiment ID,
    and (for NR_COMBINED_UPGRADE) train/val split SHAs.

    Returns a lineage dict for writing into the child's artifacts.
    """
    ckpt_path = Path(init_checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Parent checkpoint not found: {ckpt_path}")

    manifest_path = ckpt_path.parent / "artifact_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Parent artifact manifest not found: {manifest_path}. "
            f"Every checkpoint must have an artifact_manifest.json."
        )

    manifest = _json.loads(manifest_path.read_text())
    parent_exp_id = manifest.get("experiment_id", "unknown")
    expected_sha = manifest.get("checkpoint_sha256")
    if not expected_sha:
        raise RuntimeError(
            "Parent artifact_manifest.json has no checkpoint_sha256 field."
        )

    actual_sha = _sha256_hex(ckpt_path)
    if expected_sha != actual_sha:
        raise RuntimeError(
            f"Parent checkpoint SHA-256 mismatch!\n"
            f"  Expected (manifest): {expected_sha}\n"
            f"  Actual:              {actual_sha}"
        )

    lineage = {
        "parent_experiment_id": parent_exp_id,
        "parent_checkpoint_path": str(ckpt_path),
        "parent_checkpoint_sha256": actual_sha,
        "parent_train_csv_sha256": manifest.get("train_csv_sha256", ""),
        "parent_val_csv_sha256": manifest.get("val_csv_sha256", ""),
        "parent_best_val_acc": manifest.get("best_val_acc"),
    }

    # ── NR_COMBINED_UPGRADE hard-gates ──────────────────────────
    if child_experiment_id == "NR_COMBINED_UPGRADE":
        if parent_exp_id != _A2_EXPERIMENT_ID:
            raise RuntimeError(
                f"NR_COMBINED_UPGRADE requires parent {_A2_EXPERIMENT_ID}, "
                f"got {parent_exp_id}"
            )
        if actual_sha != _A2_CKPT_SHA:
            raise RuntimeError(
                f"NR_COMBINED_UPGRADE requires exact A2 checkpoint.\n"
                f"  Expected: {_A2_CKPT_SHA}\n"
                f"  Actual:   {actual_sha}"
            )
        if lineage["parent_train_csv_sha256"] != _A2_TRAIN_SHA:
            raise RuntimeError(
                f"A2 train CSV SHA mismatch.\n"
                f"  Expected: {_A2_TRAIN_SHA}\n"
                f"  Actual:   {lineage['parent_train_csv_sha256']}"
            )
        if lineage["parent_val_csv_sha256"] != _A2_VAL_SHA:
            raise RuntimeError(
                f"A2 val CSV SHA mismatch.\n"
                f"  Expected: {_A2_VAL_SHA}\n"
                f"  Actual:   {lineage['parent_val_csv_sha256']}"
            )
        logger.info("A2 parent verification PASSED: experiment=%s, SHA-256, splits — all OK",
                     parent_exp_id)
    else:
        logger.info("Parent verification PASSED: experiment=%s, SHA-256 OK",
                     parent_exp_id)

    return lineage


# ──────────────────────────────────────────────────────────────────────
# Semantic signature checks
# ──────────────────────────────────────────────────────────────────────


def verify_init_compat(
    child_config: Dict[str, Any],
    parent_checkpoint_config: Dict[str, Any],
) -> List[str]:
    """Check that *child_config* is compatible with a *parent* checkpoint.

    This is used for ``--init-checkpoint`` flows.  Only structural
    compatibility is checked (CLIP model, num_classes, head type).
    PEFT types are expected to differ (parent is typically
    ``linear_head_only`` while child is ``visual_lora``).

    Returns a list of error messages (empty = OK).
    """
    errors = []
    pc = parent_checkpoint_config or {}
    cc = child_config

    pm = pc.get("model", {})
    cm = cc.get("model", {})
    pe = pc.get("experiment", {})
    ce = cc.get("experiment", {})

    # CLIP model name
    p_clip = pm.get("clip_model_name", "ViT-B/32")
    c_clip = cm.get("clip_model_name", "ViT-B/32")
    if p_clip != c_clip:
        errors.append(f"CLIP model: parent={p_clip!r}, child={c_clip!r}")

    # num_classes
    p_nc = pm.get("num_classes")
    c_nc = cm.get("num_classes")
    if p_nc is not None and c_nc is not None and p_nc != c_nc:
        errors.append(f"num_classes: parent={p_nc}, child={c_nc}")

    # head type
    p_head = pe.get("head_type", "linear")
    c_head = ce.get("head_type", "linear")
    if p_head != c_head:
        errors.append(f"head_type: parent={p_head!r}, child={c_head!r}")

    return errors


def verify_inference_rebuild(
    runtime_config: Dict[str, Any],
    checkpoint_config: Dict[str, Any],
) -> List[str]:
    """Check that *runtime_config* matches a *trained checkpoint* config.

    This is used for eval / infer / TTA loading.  The runtime config
    must exactly match the checkpoint's config because the checkpoint
    was produced by the same experiment.

    Returns a list of error messages (empty = OK).
    """
    errors = []
    rc = runtime_config
    cc = checkpoint_config or {}

    def _cmp(key_path, label, cv, rv):
        if cv is not None and rv is not None and cv != rv:
            errors.append(f"{label}: ckpt={cv!r}, runtime={rv!r}")

    # Model section
    cm = cc.get("model", {})
    rm = rc.get("model", {})
    _cmp("model.clip_model_name", "CLIP model",
         cm.get("clip_model_name"), rm.get("clip_model_name"))
    _cmp("model.num_classes", "num_classes",
         cm.get("num_classes"), rm.get("num_classes"))

    # Head type
    ce = cc.get("experiment", {})
    re = rc.get("experiment", {})
    _cmp("experiment.head_type", "head_type",
         ce.get("head_type"), re.get("head_type"))

    # PEFT section
    cp = cc.get("peft", {})
    rp = rc.get("peft", {})
    _cmp("peft.type", "PEFT type",
         cp.get("type"), rp.get("type"))
    _cmp("peft.lora_rank", "LoRA rank",
         cp.get("lora_rank") or (cp.get("lora", {}) or {}).get("rank"),
         rp.get("lora_rank") or (rp.get("lora", {}) or {}).get("rank"))
    _cmp("peft.lora_alpha", "LoRA alpha",
         cp.get("lora_alpha") or (cp.get("lora", {}) or {}).get("alpha"),
         rp.get("lora_alpha") or (rp.get("lora", {}) or {}).get("alpha"))
    _cmp("peft.lora_last_n_blocks", "LoRA last_n_blocks",
         cp.get("lora_last_n_blocks"), rp.get("lora_last_n_blocks"))
    _cmp("peft.lora_adapt_qv", "LoRA adapt_qv",
         cp.get("lora_adapt_qv"), rp.get("lora_adapt_qv"))
    _cmp("peft.lora_adapt_out", "LoRA adapt_out",
         cp.get("lora_adapt_out"), rp.get("lora_adapt_out"))

    return errors


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────


def build_and_load_model(
    config: Dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
    *,
    build_model_fn=None,
    strict: bool = True,
    verify_rebuild: bool = True,
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """Build a model, apply PEFT, and strict-load checkpoint weights.

    This is the canonical entry point for all scripts that need to
    reconstruct a trained model.

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
    strict:
        Whether to require exact key match (default ``True``).
    verify_rebuild:
        If True (default), compare the checkpoint's embedded config
        against *config* via :func:`verify_inference_rebuild` and
        hard-fail on mismatch.

    Returns
    -------
    (model, preprocess, load_info)
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ── 0. Pre-load semantic check ──────────────────────────────
    # Read the checkpoint's embedded config BEFORE building the model
    # so we can fail fast on config mismatch without GPU work.
    # Use map_location="cpu" to avoid GPU memory allocation during
    # config-only pre-read.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", {})

    if verify_rebuild and ckpt_config:
        rebuild_errs = verify_inference_rebuild(config, ckpt_config)
        if rebuild_errs:
            raise RuntimeError(
                "Inference-rebuild signature mismatch between runtime config "
                "and checkpoint config:\n  " + "\n  ".join(rebuild_errs)
            )

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

    # ── 3. Load checkpoint weights (strict) ─────────────────────
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
