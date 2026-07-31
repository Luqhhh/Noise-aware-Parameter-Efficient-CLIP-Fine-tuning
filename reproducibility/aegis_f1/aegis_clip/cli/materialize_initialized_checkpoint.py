"""Materialize a strict, inference-ready epoch-0 checkpoint from a parent."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from aegis_clip.checkpoint import build_from_checkpoint, load_initial_weights
from aegis_clip.config import load_config, public_config
from aegis_clip.model import build_model
from aegis_clip.runtime import sha256_file


def materialize_initialized_checkpoint(
    config_path: str | Path, output_path: str | Path
) -> Path:
    config = load_config(config_path)
    source = config["train"].get("init_checkpoint")
    if not source:
        raise ValueError("train.init_checkpoint is required")
    source = Path(source).resolve()
    destination = Path(output_path).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(config, device)
    parent = load_initial_weights(model, source, device)
    payload = {
        "format_version": 1,
        "epoch": 0,
        "global_step": 0,
        "best_selector": float("-inf"),
        "model_state_dict": model.state_dict(),
        "effective_model_spec": model.effective_spec(),
        "config": public_config(config),
        "metrics": {"status": "initialized_untrained"},
        "initialization_lineage": {
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": sha256_file(source),
            "source_epoch": int(parent.get("epoch", -1)),
            "zero_initialized_visual_adapter": config["model"].get("peft_mode")
            == "visual_mlp_adapter",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    # Fail before any expensive inference if the emitted checkpoint cannot be
    # reconstructed through the single production loading path.
    rebuilt, _, _ = build_from_checkpoint(destination, device)
    if rebuilt.effective_spec() != model.effective_spec():
        raise RuntimeError("materialized checkpoint changed the effective model spec")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(materialize_initialized_checkpoint(args.config, args.output))


if __name__ == "__main__":
    main()
