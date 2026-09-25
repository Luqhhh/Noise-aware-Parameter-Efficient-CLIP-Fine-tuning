#!/usr/bin/env python3
"""Create a path-overridden runtime YAML from a checkpoint config.

This is intended for the second worker: the checkpoint carries the original
absolute paths from the machine that trained it, while GPU1 may store the same
dataset under a different root.  The generated YAML can be passed to:

* ``scripts/run_a0_m1_probe.py --config``
* ``aegis_clip.cli.cache_validation_logits --config-override``
* ``aegis_clip.cli.cache_local_adapter_features --config-override``
* ``aegis_clip.cli.cache_part_token_adapter_features --config-override``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml


def _abs(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experiment-id", default="RM_V5_C0_GPU1")
    parser.add_argument("--output-root", default="outputs/f05_focus_gpu1")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    stage_dir = _abs(args.stage_dir)
    train_root = _abs(args.train_root)
    test_root = _abs(args.test_root)
    for path in (checkpoint_path, stage_dir, train_root, test_root):
        if path is None or not path.exists():
            raise FileNotFoundError(path)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint does not contain a resolved config")
    config = {key: value for key, value in config.items() if not key.startswith("_")}

    for name in (
        "class_to_idx.json",
        "dataset_manifest.json",
        "train_dev.csv",
        "val_dev.csv",
        "features/features.pt",
        "features/image_paths.json",
        "features/manifest.json",
    ):
        if not (stage_dir / name).is_file():
            raise FileNotFoundError(stage_dir / name)

    config["data"].update(
        {
            "class_mapping": str(stage_dir / "class_to_idx.json"),
            "dataset_manifest": str(stage_dir / "dataset_manifest.json"),
            "train_csv": str(stage_dir / "train_dev.csv"),
            "val_csv": str(stage_dir / "val_dev.csv"),
            "train_root": str(train_root),
            "test_root": str(test_root),
        }
    )
    config["features"].update(
        {
            "tensor_path": str(stage_dir / "features/features.pt"),
            "paths_path": str(stage_dir / "features/image_paths.json"),
            "manifest_path": str(stage_dir / "features/manifest.json"),
        }
    )
    config["train"]["device"] = str(args.device)
    config["output"]["root"] = str(_abs(args.output_root))
    config["project"]["experiment_id"] = str(args.experiment_id)
    config["project"]["gpu1_path_override"] = {
        "source_checkpoint": str(checkpoint_path),
        "stage_dir": str(stage_dir),
        "train_root": str(train_root),
        "test_root": str(test_root),
    }

    destination = _abs(args.output)
    if destination is None:
        raise ValueError("--output is required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "runtime_config": str(destination),
                "checkpoint": str(checkpoint_path),
                "experiment_id": config["project"]["experiment_id"],
                "input_resolution": config["model"].get("input_resolution"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
