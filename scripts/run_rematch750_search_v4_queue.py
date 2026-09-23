#!/usr/bin/env python3
"""Serial local-first runner for REMATCH750_SEARCH_V4.

The runner is deliberately conservative: it executes one full training point at
a time, records selected metrics, skips points whose run directory already has
a selected report, and deletes disposable large artifacts from failed points.
It never calls the platform submission interface.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))
CONFIG_DIR = ROOT / "configs/rematch750_search_v4"
RESULT_DIR = ROOT / "results/rematch750_search_v4"
QUEUE_DIR = ROOT / "outputs/rematch750_search_v4/queue"

# First wave from the approved plan, restricted to mechanisms that are already
# implemented and fail-closed by the V4 protocol.
V3_MACRO = 0.7234723567962646
V3_MICRO = 0.7341398000717163
SIGNIFICANT_MACRO_PP = 2.0
SIGNIFICANT_MICRO_PP = 1.0

DEFAULT_TRIALS = [
    "A01", "A02", "A04", "A07",
    "B02", "B08",
    "D02", "D08",
    "E01", "E03",
]


def load_yaml(path: Path) -> dict:
    from aegis_clip.config import load_config

    return load_config(path)


def run_directory(config: dict) -> Path:
    return (
        Path(config["output"]["root"])
        / str(config["project"]["experiment_id"])
        / f"seed{int(config['project'].get('seed', 42))}"
    )


def read_selected_metrics(run_dir: Path) -> dict:
    report_path = run_dir / "checkpoints/selected_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "selected_epoch": report.get("selected_epoch"),
        "raw_macro": report.get("raw_macro"),
        "raw_micro": report.get("raw_micro"),
        "schedule_epochs": report.get("schedule_epochs"),
        "validation_covered_classes": report.get("validation_covered_classes"),
    }


def append_result(trial_id: str, status: str, metrics: dict | None, log_path: Path) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "trial_id": trial_id,
        "family": trial_id[0],
        "status": status,
        "completed_at_unix": int(time.time()),
        "selected_epoch": None,
        "raw_macro": None,
        "raw_micro": None,
        "delta_macro_vs_v3_pp": None,
        "delta_micro_vs_v3_pp": None,
        "log_path": str(log_path.relative_to(ROOT)) if log_path else None,
    }
    if metrics:
        row.update(metrics)
        v3_macro = 0.7234723567962646
        v3_micro = 0.7341398000717163
        if metrics.get("raw_macro") is not None:
            row["delta_macro_vs_v3_pp"] = (
                float(metrics["raw_macro"]) - v3_macro
            ) * 100.0
        if metrics.get("raw_micro") is not None:
            row["delta_micro_vs_v3_pp"] = (
                float(metrics["raw_micro"]) - v3_micro
            ) * 100.0
    path = RESULT_DIR / f"{trial_id}.json"
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive_failure(run_dir: Path, trial_id: str, log_path: Path) -> None:
    """Keep small textual diagnostics and remove multi-GB checkpoints."""
    failure_dir = RESULT_DIR / "failed"
    failure_dir.mkdir(parents=True, exist_ok=True)
    snippets = []
    for rel in ("logs/train.log", "training_code_manifest.json"):
        candidate = run_dir / rel
        if candidate.is_file():
            if candidate.stat().st_size > 2_000_000:
                snippets.append(f"--- {rel}: {candidate.stat().st_size} bytes omitted ---\n")
            else:
                snippets.append(f"--- {rel} ---\n" + candidate.read_text(errors="replace") + "\n")
    if log_path.is_file():
        tail = log_path.read_text(errors="replace").splitlines()[-200:]
        snippets.append("--- queue log tail ---\n" + "\n".join(tail) + "\n")
    (failure_dir / f"{trial_id}.txt").write_text("\n".join(snippets), encoding="utf-8")
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)


def significant(metrics: dict) -> bool:
    macro = metrics.get("raw_macro")
    micro = metrics.get("raw_micro")
    if macro is None or micro is None:
        return False
    return (
        (float(macro) - V3_MACRO) * 100.0 >= SIGNIFICANT_MACRO_PP
        and (float(micro) - V3_MICRO) * 100.0 >= SIGNIFICANT_MICRO_PP
    )


def cleanup_success(run_dir: Path, metrics: dict, trial_id: str) -> None:
    """Keep best.pt only for a significant local gain; keep small evidence always."""
    checkpoints = run_dir / "checkpoints"
    for path in checkpoints.glob("last.pt"):
        path.unlink(missing_ok=True)
    for path in checkpoints.glob("last.binding.json"):
        path.unlink(missing_ok=True)
    for path in checkpoints.glob("epoch_*.pt"):
        path.unlink(missing_ok=True)
    if significant(metrics):
        return
    removed = []
    for path in sorted(checkpoints.glob("*.pt")):
        path.unlink(missing_ok=True)
        removed.append(str(path.relative_to(run_dir)))
    for name in ("best.binding.json",):
        path = checkpoints / name
        if path.exists():
            path.unlink(missing_ok=True)
            removed.append(str(path.relative_to(run_dir)))
    (checkpoints / "PRUNED_NOT_SIGNIFICANT.json").write_text(
        json.dumps(
            {
                "trial_id": trial_id,
                "reason": "below_v4_local_significance_threshold",
                "macro_delta_pp": (float(metrics["raw_macro"]) - V3_MACRO) * 100.0,
                "micro_delta_pp": (float(metrics["raw_micro"]) - V3_MICRO) * 100.0,
                "required_macro_delta_pp": SIGNIFICANT_MACRO_PP,
                "required_micro_delta_pp": SIGNIFICANT_MICRO_PP,
                "removed": removed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def execute_trial(trial_id: str) -> int:
    config_path = CONFIG_DIR / f"{trial_id}.yaml"
    if not config_path.is_file():
        print(f"MISSING CONFIG {config_path}", flush=True)
        return 2
    config = load_yaml(config_path)
    run_dir = run_directory(config).resolve()
    selected = run_dir / "checkpoints/selected_report.json"
    queue_log = QUEUE_DIR / f"{trial_id}.log"
    queue_log.parent.mkdir(parents=True, exist_ok=True)
    if selected.is_file():
        metrics = read_selected_metrics(run_dir)
        append_result(trial_id, "already_complete", metrics, queue_log)
        cleanup_success(run_dir, metrics, trial_id)
        print(f"SKIP {trial_id}: selected report exists", flush=True)
        return 0

    if run_dir.exists():
        # A directory without a selected report is a failed/aborted attempt.
        print(f"Removing incomplete run directory for {trial_id}: {run_dir}", flush=True)
        shutil.rmtree(run_dir, ignore_errors=True)

    command = [
        sys.executable,
        "-m",
        "aegis_clip.cli.rematch",
        "train",
        "--config",
        str(config_path),
    ]
    environment = dict(os.environ)
    environment.setdefault("PYTHONPATH", str(ROOT / "reproducibility/aegis_f1"))
    print(f"START {trial_id}: {' '.join(command)}", flush=True)
    started = time.time()
    with queue_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
    elapsed = time.time() - started
    print(f"END {trial_id}: rc={return_code} elapsed={elapsed:.1f}s", flush=True)
    if return_code == 0 and selected.is_file():
        metrics = read_selected_metrics(run_dir)
        append_result(trial_id, "passed_training", metrics, queue_log)
        cleanup_success(run_dir, metrics, trial_id)
        return 0
    append_result(trial_id, f"failed_rc_{return_code}", None, queue_log)
    archive_failure(run_dir, trial_id, queue_log)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", nargs="+", default=DEFAULT_TRIALS)
    parser.add_argument("--lock", default=str(QUEUE_DIR / "queue.lock"))
    args = parser.parse_args()
    lock_path = Path(args.lock)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        else:
            raise SystemExit(f"queue already running with pid={pid}")
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        failures = 0
        for trial_id in args.trials:
            failures += int(execute_trial(trial_id) != 0)
        return 1 if failures else 0
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
