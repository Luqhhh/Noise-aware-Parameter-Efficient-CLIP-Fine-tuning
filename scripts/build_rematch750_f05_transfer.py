#!/usr/bin/env python3
"""Build the F05 evidence-transfer search bundle deterministically."""
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
OUTPUT_DIR = ROOT / "configs" / "rematch750_f05_transfer"


TRIALS: tuple[dict[str, Any], ...] = (
    {
        "name": "ET01",
        "experiment_id": "RM_ET01_ANCHOR1",
        "mechanism": "feature_anchor_1.0",
        "priority": 1,
        "overrides": {"loss": {"feature_distillation_weight": 1.0}},
        "prior": "V4 D03: +0.1293pp macro / +0.1142pp micro vs V3",
    },
    {
        "name": "ET02",
        "experiment_id": "RM_ET02_GCE_Q07",
        "mechanism": "gce_q_0.7",
        "priority": 1,
        "overrides": {"loss": {"gce_q": 0.7}},
        "prior": "V4 B05: +0.0135pp macro / +0.0134pp micro vs V3",
    },
    {
        "name": "ET03",
        "experiment_id": "RM_ET03_BALSOFT_TAU05",
        "mechanism": "balanced_softmax_tau_0.5",
        "priority": 1,
        "overrides": {"longtail": {"balanced_softmax_tau": 0.5}},
        "prior": "V4 E01: +0.1126pp macro / +0.0134pp micro vs V3",
    },
    {
        "name": "ET04",
        "experiment_id": "RM_ET04_BALSOFT_TAU10",
        "mechanism": "balanced_softmax_tau_1.0",
        "priority": 1,
        "overrides": {"longtail": {"balanced_softmax_tau": 1.0}},
        "prior": "V4 E02: +0.1831pp macro / -0.0202pp micro vs V3",
    },
    {
        "name": "ET05",
        "experiment_id": "RM_ET05_BBLR_8E6",
        "mechanism": "backbone_lr_8e-6",
        "priority": 2,
        "overrides": {"train": {"backbone_lr": 8.0e-6}},
        "prior": "unexplored F05-local lower-side bracket",
    },
    {
        "name": "ET06",
        "experiment_id": "RM_ET06_BBLR_16E6",
        "mechanism": "backbone_lr_1.6e-5",
        "priority": 2,
        "overrides": {"train": {"backbone_lr": 1.6e-5}},
        "prior": "unexplored F05-local upper-side bracket",
    },
    {
        "name": "ET07",
        "experiment_id": "RM_ET07_HEADLR_2E4",
        "mechanism": "head_lr_2e-4",
        "priority": 2,
        "overrides": {"train": {"head_lr": 2.0e-4}},
        "prior": "unexplored F05-local lower-side bracket",
    },
    {
        "name": "ET08",
        "experiment_id": "RM_ET08_HEADLR_8E4",
        "mechanism": "head_lr_8e-4",
        "priority": 2,
        "overrides": {"train": {"head_lr": 8.0e-4}},
        "prior": "unexplored F05-local upper-side bracket",
    },
    {
        "name": "ET09",
        "experiment_id": "RM_ET09_WARMUP1",
        "mechanism": "lr_warmup_1_epoch",
        "priority": 3,
        "overrides": {"train": {"lr_warmup_epochs": 1}},
        "prior": "high-resolution optimization-stability probe",
    },
    {
        "name": "ET10",
        "experiment_id": "RM_ET10_E20",
        "mechanism": "cosine_20_epochs",
        "priority": 3,
        "overrides": {"train": {"epochs": 20, "schedule_epochs": 20}},
        "prior": "F05 selected epoch 14/16; tests a slower cosine tail",
    },
)


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict):
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise TypeError(f"cannot merge mapping into {key}")
            _deep_update(child, value)
        else:
            target[key] = value


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
                "protocol": "rematch750_f05_transfer",
                "trial_id": trial["name"],
                "mechanism": trial["mechanism"],
                "implementation_status": "implemented",
            }
        )
        config["output"]["root"] = "../../outputs/xjn/rematch750_f05_transfer"
        _deep_update(config, trial["overrides"])
        payload = yaml.safe_dump(
            config, sort_keys=False, allow_unicode=True
        ).encode("utf-8")
        path = output_dir / f"{trial['name']}.yaml"
        rendered[path] = payload
        manifest_trials.append(
            {
                "name": trial["name"],
                "experiment_id": trial["experiment_id"],
                "mechanism": trial["mechanism"],
                "priority": trial["priority"],
                "config": str(path.relative_to(root)),
                "config_sha256": _sha256(payload),
                "single_variable_overrides": trial["overrides"],
                "evidence": trial["prior"],
                "status": "ready_for_npu",
            }
        )

    manifest = {
        "schema_version": 1,
        "protocol": "REMATCH750_F05_EVIDENCE_TRANSFER",
        "owner": "xjn",
        "status": "ready_for_npu",
        "created_at": "2026-09-24",
        "baseline": {
            "trial": "HL00",
            "experiment_id": "RM_HL00_F05_CONTROL",
            "config": "configs/rematch750_head_l2sp/HL00.yaml",
            "config_sha256": _sha256(base_bytes),
            "historical_reference": "RM_V4_F05",
            "historical_local_macro": 0.7443073987960815,
            "historical_local_micro": 0.7544354796409607,
            "historical_platform_score_percent": 65.4711,
        },
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
            "micro_batch_size": 256,
            "grad_accum_steps": 4,
            "effective_batch_size": 1024,
            "epochs": 16,
            "head_lr": 4.0e-4,
            "backbone_lr": 1.2e-5,
            "loss": "ce_warmup_2_then_gce_q0.5",
            "feature_distillation_weight": 2.0,
            "selector_metric": "raw_macro",
        },
        "trials": manifest_trials,
        "promotion": {
            "reference": "same-revision HL00",
            "macro_delta_pp": 0.30,
            "micro_max_regression_pp": 0.10,
            "confirmation_seeds": [3407, 2026],
            "minimum_directional_seeds": 2,
            "combinations": "only after an individual factor passes promotion",
        },
        "runner": "scripts/run_rematch750_f05_transfer.py",
        "platform_submission": False,
    }
    rendered[output_dir / "manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="verify committed bundle byte-for-byte"
    )
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
