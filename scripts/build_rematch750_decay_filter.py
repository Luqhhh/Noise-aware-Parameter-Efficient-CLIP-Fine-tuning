#!/usr/bin/env python3
"""Build the REMATCH750 AdamW decay-filter factorial deterministically."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "rematch750_head_l2sp" / "HL00.yaml"
OUTPUT_DIR = ROOT / "configs" / "rematch750_decay_filter"
TRIALS: tuple[dict[str, Any], ...] = (
    {
        "name": "DF01",
        "experiment_id": "RM_DF01_HEAD_MATRIX_ONLY",
        "head_filter": "matrix_only",
        "backbone_filter": "all",
    },
    {
        "name": "DF02",
        "experiment_id": "RM_DF02_VISUAL_MATRIX_ONLY",
        "head_filter": "all",
        "backbone_filter": "matrix_only",
    },
    {
        "name": "DF03",
        "experiment_id": "RM_DF03_BOTH_MATRIX_ONLY",
        "head_filter": "matrix_only",
        "backbone_filter": "matrix_only",
    },
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_bundle(root: Path = ROOT) -> dict[Path, bytes]:
    base_path = root / BASE_CONFIG.relative_to(ROOT)
    output_dir = root / OUTPUT_DIR.relative_to(ROOT)
    base_bytes = base_path.read_bytes()
    base = yaml.safe_load(base_bytes)
    rendered: dict[Path, bytes] = {}
    manifest_trials: list[dict[str, Any]] = []
    for trial in TRIALS:
        config = copy.deepcopy(base)
        config["project"].update(
            {
                "experiment_id": trial["experiment_id"],
                "protocol": "rematch750_decay_filter",
                "trial_id": trial["name"],
                "mechanism": (
                    f"head_{trial['head_filter']}+visual_{trial['backbone_filter']}"
                ),
                "implementation_status": "implemented",
            }
        )
        config["output"]["root"] = "../../outputs/xjn/rematch750_decay_filter"
        config["train"]["head_weight_decay_filter"] = trial["head_filter"]
        config["train"]["backbone_weight_decay_filter"] = trial[
            "backbone_filter"
        ]
        payload = yaml.safe_dump(
            config, sort_keys=False, allow_unicode=True
        ).encode("utf-8")
        path = output_dir / f"{trial['name']}.yaml"
        rendered[path] = payload
        manifest_trials.append(
            {
                "name": trial["name"],
                "experiment_id": trial["experiment_id"],
                "config": str(path.relative_to(root)),
                "config_sha256": _sha256(payload),
                "head_weight_decay_filter": trial["head_filter"],
                "backbone_weight_decay_filter": trial["backbone_filter"],
                "status": "ready_for_npu",
            }
        )

    manifest = {
        "schema_version": 1,
        "protocol": "REMATCH750_DECAY_FILTER_2X2",
        "owner": "xjn",
        "status": "ready_for_npu",
        "created_at": "2026-09-24",
        "baseline": {
            "trial": "HL00",
            "experiment_id": "RM_HL00_F05_CONTROL",
            "config": "configs/rematch750_head_l2sp/HL00.yaml",
            "config_sha256": _sha256(base_bytes),
            "head_weight_decay_filter": "all",
            "backbone_weight_decay_filter": "all",
        },
        "hypothesis": (
            "AdamW decay on one-dimensional bias/norm/token parameters may hurt "
            "F05 transfer; isolate head and visual scopes in a 2x2 factorial."
        ),
        "lineage": {
            "parent_experiment_id": "RM_LP",
            "parent_checkpoint_sha256": (
                "d5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b"
            ),
            "train_csv_sha256": (
                "615d510481e9df610371e84cbdeeb0058cc9a176989786a8ccf4eb30235b92c6"
            ),
            "val_csv_sha256": (
                "d91106df365b62f9bf56eff51474c222925301546642fa427f6778258cb833ab"
            ),
            "class_mapping_sha256": (
                "6f6fdc3eef8e526d3cde89da6d98690603e26ee8cae5de7e7e90c743bf3a551c"
            ),
            "seed": 42,
        },
        "fixed_recipe": {
            "input_resolution": 320,
            "effective_batch_size": 1024,
            "epochs": 16,
            "head_lr": 4.0e-4,
            "backbone_lr": 1.2e-5,
            "head_weight_decay": 1.0e-4,
            "backbone_weight_decay": 1.0e-4,
            "selector_metric": "raw_macro",
        },
        "filter_semantics": {
            "all": "every trainable tensor receives the configured AdamW decay",
            "matrix_only": (
                "ndim>=2 receives configured decay; ndim<2 receives zero decay"
            ),
        },
        "trials": manifest_trials,
        "promotion": {
            "reference": "same-revision HL00",
            "macro_delta_pp": 0.30,
            "micro_max_regression_pp": 0.10,
            "confirmation_seeds": [3407, 2026],
            "minimum_directional_seeds": 2,
        },
        "runner": "scripts/run_rematch750_decay_filter.py",
        "platform_submission": False,
    }
    rendered[output_dir / "manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_bundle()
    failures: list[str] = []
    for path, payload in rendered.items():
        if args.check:
            if not path.exists() or path.read_bytes() != payload:
                failures.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    if failures:
        raise SystemExit("bundle differs: " + ", ".join(failures))
    print(
        json.dumps(
            {
                "status": "verified" if args.check else "written",
                "files": len(rendered),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
