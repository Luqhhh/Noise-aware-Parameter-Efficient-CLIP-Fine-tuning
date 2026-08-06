"""Atomic, resumable checkpoints with one model construction path."""

from __future__ import annotations

import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aegis_clip.config import public_config
from aegis_clip.model import AegisCLIP, build_model, interpolate_visual_positional_embedding


def save_checkpoint(
    path: str | Path,
    *,
    model: AegisCLIP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_selector: float,
    config: dict[str, Any],
    metrics: dict[str, Any],
    adaptive_cap_state: dict[str, Any] | None,
    data_generator_state: torch.Tensor,
    elr_state_dict: dict[str, Any] | None = None,
    training_aux_state: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format_version": 1,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_selector": float(best_selector),
        "model_state_dict": model.state_dict(),
        "effective_model_spec": model.effective_spec(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "adaptive_cap_state": adaptive_cap_state,
        "elr_state_dict": elr_state_dict,
        "training_aux_state": training_aux_state,
        "data_generator_state": data_generator_state,
        "config": public_config(config),
        "metrics": metrics,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    _atomic_torch_save(payload, path)


def load_initial_weights(
    model: AegisCLIP, path: str | Path, device: torch.device
) -> dict[str, Any]:
    # Keep non-model state (notably CPU RNG tensors) on CPU. ``load_state_dict``
    # copies model and optimizer tensors to their parameter devices safely.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    if state is None:
        raise ValueError("Checkpoint does not contain model weights")
    state = dict(state)
    # Resolution-adaptive init: when the model runs at a different input
    # resolution than the checkpoint, bicubically interpolate the source
    # visual position embedding so the backbone patch grid matches the target.
    # This lets a 224px parent initialise a higher-resolution child.
    if "visual.positional_embedding" in state and hasattr(model, "visual"):
        target = getattr(model.visual, "positional_embedding", None)
        if target is not None and tuple(target.shape) != tuple(
            state["visual.positional_embedding"].shape
        ):
            source = state["visual.positional_embedding"]
            target_tokens = int(target.shape[0] - 1)
            source_tokens = int(source.shape[0] - 1)
            target_size = int(round(math.sqrt(target_tokens)))
            source_size = int(round(math.sqrt(source_tokens)))
            if (
                target_size * target_size == target_tokens
                and source_size * source_size == source_tokens
            ):
                state["visual.positional_embedding"] = (
                    interpolate_visual_positional_embedding(
                        source, (target_size, target_size)
                    )
                )
    if getattr(model, "peft_mode", None) in {
        "visual_lora",
        "visual_lora_last_mlp",
        "visual_lora_mlp_lora",
        "visual_lora_mlp_adapter",
    }:
        _remap_base_weights_for_parametrized_model(model, state)
    elif getattr(model, "peft_mode", None) == "full_finetune" and any(
        ".parametrizations." in name for name in state
    ):
        # A LoRA parent checkpoint stores its effective weights as
        # ``parametrizations.<name>.<index>.*`` tensors.  The full-finetune
        # target owns plain weights, so materialise the LoRA updates into the
        # effective base weights before loading.
        state = _merge_parametrized_state_for_plain_model(
            state, checkpoint.get("effective_model_spec", {})
        )
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing: set[str] = set()
    if getattr(model, "peft_mode", None) == "feature_adapter":
        allowed_missing.update(
            name
            for name in model.state_dict()
            if name.startswith("feature_adapter.")
        )
    if getattr(model, "peft_mode", None) in {
        "visual_mlp_adapter",
        "visual_lora_mlp_adapter",
    }:
        allowed_missing.update(
            name for name in model.state_dict() if ".adaptmlp." in name
        )
    if getattr(model, "peft_mode", None) == "visual_prompt":
        allowed_missing.update(
            name
            for name in model.state_dict()
            if name.startswith("visual.visual_prompt.")
        )
    if getattr(model, "classifier_mode", None) == "anchored_residual":
        allowed_missing.update(
            {"classifier.residual_weight", "classifier.residual_bias"}
        )
    if getattr(model, "peft_mode", None) in {
        "visual_lora",
        "visual_lora_last_mlp",
        "visual_lora_mlp_lora",
        "visual_lora_mlp_adapter",
    }:
        allowed_missing.update(
            name
            for name in model.state_dict()
            if ".parametrizations." in name and not name.endswith(".original")
        )
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "Initial checkpoint is architecture-incompatible: "
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return checkpoint


def _remap_base_weights_for_parametrized_model(
    model: AegisCLIP, state: dict[str, Any]
) -> None:
    """Map a frozen checkpoint's weights onto LoRA parametrization originals."""
    marker = ".parametrizations."
    suffix = ".original"
    for target_name in model.state_dict():
        if marker not in target_name or not target_name.endswith(suffix):
            continue
        prefix, remainder = target_name.split(marker, maxsplit=1)
        parameter_name = remainder[: -len(suffix)]
        source_name = f"{prefix}.{parameter_name}"
        if target_name not in state and source_name in state:
            state[target_name] = state.pop(source_name)


def _merge_parametrized_state_for_plain_model(
    state: dict[str, torch.Tensor],
    effective_spec: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Collapse LoRA parametrizations into effective plain weight tensors.

    A checkpoint saved from a LoRA model stores ``.original`` base weights plus
    per-parametrization ``lora_A``/``lora_B`` (additive) or ``q_A``/``q_B``/
    ``v_A``/``v_B`` (QV attention) updates.  This returns a state dict whose
    base weight names hold ``original + scaling * update`` and which drops every
    parametrization tensor, so a plain-weight (``full_finetune``) model can be
    initialised from the same parent exactly.
    """

    state = dict(state)
    marker = ".parametrizations."
    suffix = ".original"

    attention_scaling = 1.0
    mlp_scaling = 1.0
    if effective_spec:
        try:
            attention_scaling = float(
                effective_spec["lora_alpha"]
            ) / float(effective_spec["lora_rank"])
            mlp_scaling = float(
                effective_spec["mlp_lora_alpha"]
            ) / float(effective_spec["mlp_lora_rank"])
        except (KeyError, TypeError, ZeroDivisionError, ValueError):
            attention_scaling = 1.0
            mlp_scaling = 1.0

    original_names = [
        name
        for name in state
        if marker in name and name.endswith(suffix)
    ]
    merged: dict[str, torch.Tensor] = {}
    for original_name in original_names:
        root = original_name[: -len(suffix)]
        prefix, parameter_name = root.split(marker, maxsplit=1)
        base_name = f"{prefix}.{parameter_name}"
        keys = {
            key: state[key]
            for key in state
            if key.startswith(root + ".") and key != original_name
        }
        leaf = lambda part: next(
            (value for key, value in keys.items() if key.endswith("." + part)),
            None,
        )

        if leaf("lora_A") is not None and leaf("lora_B") is not None:
            scaling = (
                mlp_scaling if ".mlp." in root else attention_scaling
            )
            update = leaf("lora_B") @ leaf("lora_A")
            merged[base_name] = (
                state[original_name] + scaling * update
            )
        elif (
            leaf("q_A") is not None
            and leaf("q_B") is not None
            and leaf("v_A") is not None
            and leaf("v_B") is not None
        ):
            q_update = leaf("q_B") @ leaf("q_A")
            v_update = leaf("v_B") @ leaf("v_A")
            update = torch.cat(
                [
                    q_update,
                    torch.zeros_like(q_update),
                    v_update,
                ],
                dim=0,
            )
            merged[base_name] = (
                state[original_name]
                + attention_scaling * update
            )
        else:
            # Unknown parametrisation shape: keep the base weight untouched so
            # the strict load can report the mismatch instead of silently
            # dropping it.
            merged[base_name] = state[original_name]

    # Remove every parametrisation leaf (originals and update tensors) and
    # install the merged plain weights.
    state = {
        name: value
        for name, value in state.items()
        if marker not in name
    }
    state.update(merged)
    return state


def resume_checkpoint(
    path: str | Path,
    *,
    model: AegisCLIP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    adaptive_cap: Any = None,
    elr_regularizer: Any = None,
    data_generator: torch.Generator | None = None,
    training_auxiliary: Any = None,
) -> dict[str, Any]:
    checkpoint = load_initial_weights(model, path, device)
    required = {
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "rng_state",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Resume checkpoint missing state: {sorted(missing)}")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if adaptive_cap is not None and checkpoint.get("adaptive_cap_state") is not None:
        adaptive_cap.load_state_dict(checkpoint["adaptive_cap_state"])
    if elr_regularizer is not None:
        if checkpoint.get("elr_state_dict") is None:
            raise ValueError("Resume checkpoint is missing ELR state")
        elr_regularizer.load_state_dict(checkpoint["elr_state_dict"])
    if data_generator is not None:
        if checkpoint.get("data_generator_state") is None:
            raise ValueError("Resume checkpoint is missing data_generator_state")
        data_generator.set_state(checkpoint["data_generator_state"].cpu())
    if training_auxiliary is not None:
        if checkpoint.get("training_aux_state") is None:
            raise ValueError("Resume checkpoint is missing training auxiliary state")
        training_auxiliary.load_state_dict(checkpoint["training_aux_state"])
    _restore_rng(checkpoint["rng_state"])
    return checkpoint


def build_from_checkpoint(
    path: str | Path,
    device: torch.device,
    config_override: dict[str, Any] | None = None,
) -> tuple[AegisCLIP, Any, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Aegis checkpoint is missing its resolved config")
    model, preprocess = build_model(config_override or config, device)
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise ValueError("Aegis checkpoint is missing model_state_dict")
    model.load_state_dict(state, strict=True)
    return model, preprocess, checkpoint


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
