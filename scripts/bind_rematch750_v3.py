"""Bind a V3 template to an idle physical NPU and shared current-stage assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


LP_SHA256 = "d5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b"
SHARED_PATHS = {
    "data": ("class_mapping", "dataset_manifest", "test_root", "train_csv", "train_root", "val_csv"),
    "features": ("manifest_path", "paths_path", "tensor_path"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bind(template: Path, output: Path, shared_root: Path, physical_device: int) -> dict:
    if physical_device < 0:
        raise ValueError("Physical NPU index must be nonnegative")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(physical_device):
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must select exactly the physical device being bound")
    if output.suffixes[-2:] != [".local", ".yaml"] or output.exists():
        raise ValueError("Use a new ignored *.local.yaml output; existing bindings are immutable")
    config = yaml.safe_load(template.read_text())
    if config["train"]["device"] != "npu:UNASSIGNED":
        raise ValueError("Expected an unbound V3 template")
    if config["project"]["experiment_id"] not in {
        "RM_V3_B64", "RM_V3_B64_LR2", "RM_V3_B128_E16", "RM_V3_B1024_E16_LR4",
    }:
        raise ValueError("Unexpected V3 candidate")
    shared_root = shared_root.resolve()
    for section, keys in SHARED_PATHS.items():
        for key in keys:
            value = config[section][key]
            if not value.startswith("../"):
                raise ValueError(f"Expected a shared relative path: {section}.{key}")
            path = shared_root / value[3:]
            if not path.exists():
                raise FileNotFoundError(path)
            config[section][key] = str(path)
    init = Path(config["train"]["init_checkpoint"])
    if not init.is_file() or digest(init) != LP_SHA256:
        raise ValueError("NPU LP initialization is missing or has the wrong SHA-256")
    # With one physical card visible, torch_npu addresses it as logical npu:0.
    config["train"]["device"] = "npu:0"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    return {
        "candidate": config["project"]["experiment_id"],
        "physical_device": physical_device,
        "logical_device": "npu:0",
        "template_sha256": digest(template),
        "config_sha256": digest(output),
        "init_checkpoint_sha256": LP_SHA256,
        "shared_root": str(shared_root),
        "bound_config": str(output.resolve()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shared-root", type=Path, required=True)
    p.add_argument("--physical-device", type=int, required=True)
    args = p.parse_args()
    print(json.dumps(bind(args.template, args.output, args.shared_root, args.physical_device), indent=2))


if __name__ == "__main__":
    main()
