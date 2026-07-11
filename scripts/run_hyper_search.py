from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from common.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse trials with an existing eval_results.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser.parse_args()


def format_float(value: float) -> str:
    return f"{value:.0e}".replace("+", "")


def run_command(command, dry_run=False):
    print("+", " ".join(map(str, command)))
    if not dry_run:
        subprocess.run(command, check=True)


def main():
    args = parse_args()
    base_config = load_config(args.config)

    search_cfg = base_config.get("hyper_search")
    if not search_cfg:
        raise ValueError(
            f"No hyper_search section found in {args.config}"
        )

    lr_values = search_cfg["lr_values"]
    wd_values = search_cfg["wd_values"]

    experiment_id = base_config["experiment"]["id"]
    base_save_dir = Path(base_config["train"]["save_dir"])
    experiment_root = base_save_dir.parent
    search_root = experiment_root / "search"
    generated_root = search_root / "generated_configs"
    generated_root.mkdir(parents=True, exist_ok=True)

    if base_config["model"].get("use_cached_features", False):
        cache_dir = Path(base_config["cache"]["cache_dir"])
        if not (cache_dir / "manifest.json").exists():
            if args.dry_run:
                print(
                    "Feature cache not found; dry-run will still preview "
                    f"trial commands. Build cache before real runs: "
                    f"python scripts/cache_features.py --config {args.config}"
                )
            else:
                raise FileNotFoundError(
                    f"Feature cache not found: {cache_dir}\n"
                    f"Run: python scripts/cache_features.py "
                    f"--config {args.config}"
                )

    split_dir = Path(base_config["data"]["split_dir"])
    required_split_files = [
        split_dir / "train.csv",
        split_dir / "val.csv",
        split_dir / "class_to_idx.json",
        split_dir / "idx_to_class.json",
    ]
    if not all(path.exists() for path in required_split_files):
        run_command(
            [
                sys.executable,
                "scripts/split_data.py",
                "--config",
                args.config,
            ],
            dry_run=args.dry_run,
        )

    rows = []

    for lr in lr_values:
        for wd in wd_values:
            trial_name = (
                f"lr_{format_float(float(lr))}"
                f"__wd_{format_float(float(wd))}"
            )
            trial_root = search_root / trial_name
            result_path = (
                trial_root / "checkpoints" / "eval_results.json"
            )

            trial_config = copy.deepcopy(base_config)
            trial_config["train"]["lr"] = float(lr)
            trial_config["train"]["weight_decay"] = float(wd)
            trial_config["train"]["save_dir"] = str(
                trial_root / "checkpoints"
            )
            trial_config["output"]["log_dir"] = str(
                trial_root / "logs"
            )
            trial_config["output"]["submission_dir"] = str(
                trial_root / "submissions"
            )
            trial_config.setdefault("runtime", {})
            trial_config["runtime"]["search_trial"] = trial_name

            config_path = generated_root / f"{trial_name}.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    trial_config,
                    f,
                    sort_keys=False,
                    allow_unicode=True,
                )

            if not (args.skip_existing and result_path.exists()):
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "experiments.baseline.train",
                        "--config",
                        str(config_path),
                    ],
                    dry_run=args.dry_run,
                )

            if args.dry_run:
                continue

            if not result_path.exists():
                raise FileNotFoundError(
                    f"Trial finished without result: {result_path}"
                )

            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)

            rows.append(
                {
                    "experiment_id": experiment_id,
                    "trial": trial_name,
                    "lr": float(lr),
                    "weight_decay": float(wd),
                    "best_val_acc": result["best_val_acc"],
                    "dev_best_epoch": result["dev_best_epoch"],
                    "head_type": result["head_type"],
                    "augmentation_preset": result[
                        "augmentation_preset"
                    ],
                    "config_path": str(config_path),
                    "checkpoint_path": str(
                        trial_root / "checkpoints" / "best.pt"
                    ),
                }
            )

    if args.dry_run:
        return

    rows.sort(key=lambda row: row["best_val_acc"], reverse=True)

    csv_path = search_root / "search_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    with open(
        search_root / "best_trial.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    best_config = load_config(best["config_path"])
    with open(
        search_root / "best_config.yaml",
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            best_config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Best trial: {best}")
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
