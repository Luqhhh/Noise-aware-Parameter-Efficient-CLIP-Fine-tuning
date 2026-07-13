#!/usr/bin/env python3
"""Register a completed experiment in results/phase3_experiments.csv.

Reads ``eval_results.json`` and ``artifact_manifest.json`` from an experiment
output directory and appends a row to the experiment registry.

Usage:
    python tools/register_experiment.py \\
        --experiment-dir outputs/gce_q07/seed42 \\
        --status candidate \\
        --platform-score 0.589578
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REGISTRY_PATH = "results/phase3_experiments.csv"

COLUMNS = [
    "experiment_id", "parent_experiment", "wave", "priority",
    "commit_sha", "config_path", "output_dir", "split_seed", "train_seed",
    "loss_name", "loss_parameters", "sample_weighting", "augmentation",
    "head_ema", "trainable_parameters", "best_epoch", "checkpoint_sha256",
    "train_split_sha256", "val_split_sha256", "raw_micro", "raw_macro",
    "raw_bottom10", "trusted_micro", "trusted_macro",
    "trusted_class_balanced", "trust_weighted_accuracy", "rejected_micro",
    "prediction_change_vs_parent", "platform_score",
    "platform_delta_vs_ref", "platform_delta_vs_parent", "status", "notes",
]

VALID_STATUSES = {
    "planned", "running", "failed", "protocol_invalid", "local_rejected",
    "candidate", "seed_validation", "submitted", "platform_rejected",
    "platform_best", "closed",
}


def parse_args():
    p = argparse.ArgumentParser(description="Register experiment in phase3 CSV")
    p.add_argument("--experiment-dir", required=True,
                   help="Path to experiment output directory")
    p.add_argument("--status", default="candidate",
                   help=f"Experiment status. Valid: {sorted(VALID_STATUSES)}")
    p.add_argument("--platform-score", type=float, default=None,
                   help="Platform accuracy (if submitted)")
    p.add_argument("--platform-delta-vs-ref", type=float, default=None)
    p.add_argument("--platform-delta-vs-parent", type=float, default=None)
    p.add_argument("--notes", default="", help="Free-text notes")
    p.add_argument("--wave", default="", help="Phase/wave identifier")
    p.add_argument("--priority", default="P1", help="P0/P1/P2")
    p.add_argument("--parent-experiment", default=None)
    return p.parse_args()


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    if args.status not in VALID_STATUSES:
        logger.error("Invalid status: %s. Valid: %s", args.status, sorted(VALID_STATUSES))
        sys.exit(1)

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        logger.error("Experiment dir not found: %s", exp_dir)
        sys.exit(1)

    eval_json = _load_json(exp_dir / "eval_results.json")
    manifest_json = _load_json(exp_dir / "artifact_manifest.json")

    # Build row from available data
    row = {c: "" for c in COLUMNS}

    if eval_json:
        row["experiment_id"] = eval_json.get("experiment_id", exp_dir.parent.name)
        row["raw_micro"] = eval_json.get("best_val_acc", "")
        row["raw_macro"] = eval_json.get("post_eval_macro_accuracy", "")
        row["best_epoch"] = eval_json.get("dev_best_epoch", "")
        row["split_seed"] = eval_json.get("split_seed", "")
        row["train_seed"] = eval_json.get("train_seed", "")

    if manifest_json:
        row["commit_sha"] = manifest_json.get("commit_sha", "")
        row["checkpoint_sha256"] = manifest_json.get("checkpoint_sha256", "")
        row["train_split_sha256"] = manifest_json.get("train_csv_sha256", "")
        row["val_split_sha256"] = manifest_json.get("val_csv_sha256", "")
        row["parent_experiment"] = manifest_json.get("parent_experiment", "")

    # CLI overrides
    row["output_dir"] = str(exp_dir)
    row["status"] = args.status
    row["wave"] = args.wave
    row["priority"] = args.priority
    row["notes"] = args.notes
    if args.parent_experiment:
        row["parent_experiment"] = args.parent_experiment
    if args.platform_score is not None:
        row["platform_score"] = args.platform_score
    if args.platform_delta_vs_ref is not None:
        row["platform_delta_vs_ref"] = args.platform_delta_vs_ref
    if args.platform_delta_vs_parent is not None:
        row["platform_delta_vs_parent"] = args.platform_delta_vs_parent

    # Write CSV
    registry = Path(REGISTRY_PATH)
    registry.parent.mkdir(parents=True, exist_ok=True)
    write_header = not registry.exists()

    with open(registry, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.info("Registered %s → %s [%s]", row["experiment_id"], REGISTRY_PATH, args.status)


if __name__ == "__main__":
    main()
