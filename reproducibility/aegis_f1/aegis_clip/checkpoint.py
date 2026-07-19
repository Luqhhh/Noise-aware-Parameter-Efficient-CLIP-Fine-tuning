"""Atomic, resumable checkpoints with one model construction path."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aegis_clip.config import public_config
from aegis_clip.model import AegisCLIP, build_model


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
    if getattr(model, "peft_mode", None) == "visual_lora":
        _remap_base_weights_for_parametrized_model(model, state)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing: set[str] = set()
    if getattr(model, "peft_mode", None) == "feature_adapter":
        allowed_missing.update(
            name
            for name in model.state_dict()
            if name.startswith("feature_adapter.")
        )
    if getattr(model, "classifier_mode", None) == "anchored_residual":
        allowed_missing.update(
            {"classifier.residual_weight", "classifier.residual_bias"}
        )
    if getattr(model, "peft_mode", None) == "visual_lora":
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
    _restore_rng(checkpoint["rng_state"])
    return checkpoint


def build_from_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[AegisCLIP, Any, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Aegis checkpoint is missing its resolved config")
    model, preprocess = build_model(config, device)
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
