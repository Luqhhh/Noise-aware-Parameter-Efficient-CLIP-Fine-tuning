"""Convert an exact legacy frozen-CLIP linear checkpoint to Aegis format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from aegis_clip.checkpoint import _atomic_torch_save, build_from_checkpoint
from aegis_clip.config import load_config, public_config
from aegis_clip.data import load_class_mapping
from aegis_clip.model import build_model
from aegis_clip.runtime import atomic_json_dump, sha256_file


def _require_exact_state(
    source_state: dict[str, Any], target_state: dict[str, Any]
) -> None:
    if set(source_state) != set(target_state):
        missing = sorted(set(source_state) - set(target_state))
        added = sorted(set(target_state) - set(source_state))
        raise ValueError(
            f"State keys changed during conversion: missing={missing}, added={added}"
        )
    for name in source_state:
        source = torch.as_tensor(source_state[name]).cpu()
        target = torch.as_tensor(target_state[name]).cpu()
        if source.dtype != target.dtype or source.shape != target.shape:
            raise ValueError(f"State metadata changed during conversion: {name}")
        if not torch.equal(source, target):
            raise ValueError(f"State tensor changed during conversion: {name}")


def convert_legacy_linear_checkpoint(
    source_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    *,
    expected_source_sha256: str,
) -> tuple[Path, Path]:
    source_path = Path(source_path).resolve()
    destination = Path(output_path).resolve()
    actual_source_sha256 = sha256_file(source_path)
    if actual_source_sha256 != str(expected_source_sha256):
        raise ValueError(
            "Legacy source checkpoint SHA-256 mismatch: "
            f"expected={expected_source_sha256}, actual={actual_source_sha256}"
        )
    config = load_config(config_path)
    if config["model"].get("peft_mode") != "frozen":
        raise ValueError("Legacy conversion requires model.peft_mode=frozen")
    if config["model"].get("classifier_mode", "linear") != "linear":
        raise ValueError("Legacy conversion requires a linear classifier")

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = source.get("model_state_dict", source.get("model_state"))
    if not isinstance(source_state, dict):
        raise ValueError("Legacy checkpoint has no model state dictionary")
    if source.get("head_type", "linear") != "linear":
        raise ValueError("Legacy checkpoint is not a linear-head model")
    legacy_model = source.get("config", {}).get("model", {})
    if not (
        legacy_model.get("freeze_clip") is True
        and int(legacy_model.get("unfreeze_last_n_blocks", 0)) == 0
        and not bool(legacy_model.get("train_ln_post", False))
        and not bool(legacy_model.get("train_visual_proj", False))
    ):
        raise ValueError("Legacy checkpoint is not a frozen-CLIP model")

    class_to_idx, _ = load_class_mapping(config["data"]["class_mapping"])
    embedded_mapping = {
        str(name): int(index) for name, index in source.get("class_to_idx", {}).items()
    }
    if embedded_mapping and embedded_mapping != class_to_idx:
        raise ValueError("Legacy checkpoint class mapping differs from Aegis config")

    model, _ = build_model(config, torch.device("cpu"))
    model.load_state_dict(source_state, strict=True)
    converted_state = model.state_dict()
    _require_exact_state(source_state, converted_state)
    payload = {
        "format_version": 1,
        "epoch": int(source.get("epoch", -1)),
        "global_step": int(source.get("global_step", 0)),
        "best_selector": float(source.get("best_val_acc", 0.0)),
        "model_state_dict": converted_state,
        "effective_model_spec": model.effective_spec(),
        "config": public_config(config),
        "metrics": {
            "source_best_val_acc": source.get("best_val_acc"),
            "source_best_raw_val_acc": source.get("best_raw_val_acc"),
            "source_best_ema_val_acc": source.get("best_ema_val_acc"),
        },
        "source_lineage": {
            "checkpoint": str(source_path),
            "checkpoint_sha256": actual_source_sha256,
            "experiment_id": source.get("config", {})
            .get("experiment", {})
            .get("id"),
            "selection_source": source.get("selection_source"),
            "epoch_selection_split": source.get("epoch_selection_split"),
        },
    }
    _atomic_torch_save(payload, destination)

    rebuilt, _, rebuilt_payload = build_from_checkpoint(
        destination, torch.device("cpu")
    )
    _require_exact_state(converted_state, rebuilt.state_dict())
    report = {
        "status": "passed",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": actual_source_sha256,
        "converted_checkpoint": str(destination),
        "converted_checkpoint_sha256": sha256_file(destination),
        "state_tensor_count": len(converted_state),
        "state_keys_exact": True,
        "state_tensors_bit_exact": True,
        "rebuild_strict": True,
        "class_mapping_exact": True,
        "effective_model_spec": rebuilt_payload["effective_model_spec"],
    }
    report_path = destination.with_suffix(".conversion.json")
    atomic_json_dump(report, report_path)
    return destination, report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    checkpoint, report = convert_legacy_linear_checkpoint(
        args.source,
        args.config,
        args.output,
        expected_source_sha256=args.expected_source_sha256,
    )
    print(checkpoint)
    print(report)


if __name__ == "__main__":
    main()
