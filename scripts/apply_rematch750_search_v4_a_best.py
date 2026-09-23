#!/usr/bin/env python3
"""Select the best completed A-group augmentation and apply it to F02/F04/F06."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/rematch750_search_v4"
RESULT_DIR = ROOT / "results/rematch750_search_v4"
A_TRIALS = [f"A{i:02d}" for i in range(1, 11)]
F_TRIALS = ["F02", "F04", "F06"]
AUGMENT_KEYS = (
    "train_augmentation",
    "randaugment_num_ops",
    "randaugment_magnitude",
)
LOSS_AUGMENT_KEYS = (
    "mixup_alpha",
    "mixup_probability",
    "cutmix_alpha",
    "cutmix_probability",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_result(trial_id: str) -> dict | None:
    path = RESULT_DIR / f"{trial_id}.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "passed_training":
        return None
    if result.get("raw_macro") is None or result.get("raw_micro") is None:
        return None
    return result


def select_best_a() -> dict:
    results = []
    for trial_id in A_TRIALS:
        result = load_result(trial_id)
        if result is not None:
            results.append(result)
    if len(results) != len(A_TRIALS):
        missing = [trial for trial in A_TRIALS if load_result(trial) is None]
        raise RuntimeError(f"A group incomplete: missing {missing}")
    ranked = sorted(
        results,
        key=lambda item: (float(item["raw_macro"]), float(item["raw_micro"])),
        reverse=True,
    )
    return ranked[0]


def apply_best_a(best: dict) -> dict:
    source_config_path = CONFIG_DIR / f"{best['trial_id']}.yaml"
    source = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    augmentation = {
        key: source["data"][key]
        for key in AUGMENT_KEYS
        if key in source["data"]
    }
    loss_augmentation = {
        key: source["loss"][key]
        for key in LOSS_AUGMENT_KEYS
        if key in source["loss"]
    }
    records = {}
    for trial_id in F_TRIALS:
        path = CONFIG_DIR / f"{trial_id}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["data"].update(augmentation)
        config["loss"].update(loss_augmentation)
        config["project"]["augmentation_source"] = best["trial_id"]
        config["project"]["augmentation_source_metrics"] = {
            "raw_macro": best["raw_macro"],
            "raw_micro": best["raw_micro"],
        }
        text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        path.write_text(text, encoding="utf-8")
        records[trial_id] = {
            "config_path": str(path.relative_to(ROOT)),
            "config_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "applied_data_augmentation": augmentation,
            "applied_loss_augmentation": loss_augmentation,
        }
    return {
        "source_a_trial": best["trial_id"],
        "source_a_metrics": {
            "raw_macro": best["raw_macro"],
            "raw_micro": best["raw_micro"],
        },
        "source_a_config_sha256": sha256_file(source_config_path),
        "applied": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--marker", default=str(ROOT / "outputs/rematch750_search_v4/queue/A_BEST_APPLIED.json"))
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    args = parser.parse_args()
    marker = Path(args.marker)
    while True:
        try:
            best = select_best_a()
        except RuntimeError as exc:
            print(f"waiting for A group: {exc}", flush=True)
            if not args.watch:
                raise
            time.sleep(max(float(args.interval_seconds), 1.0))
            continue
        payload = apply_best_a(best)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
