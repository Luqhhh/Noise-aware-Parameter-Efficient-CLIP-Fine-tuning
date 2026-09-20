#!/usr/bin/env python3
"""Run the fixed PRELIM75 v9 D0, smoke, diagnostic, replay, and delivery queue."""
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

from aegis_clip.prelim75_source_recrop import (  # noqa: E402
    CANDIDATES,
    PLAN_ID,
    load_v9_plan,
    preflight_v9,
)
from aegis_clip.runtime import atomic_json_dump, sha256_file  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/prelim75_v9.yaml"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = load_v9_plan(args.config)
    output = Path(plan["output"])
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != "main":
        raise ValueError("The fixed v9 segment must run directly on main")
    if output.exists():
        raise FileExistsError(f"Existing v9 output is never overwritten: {output}")
    checked = preflight_v9(plan)
    registration = {
        "status": "audit_passed",
        "plan_id": PLAN_ID,
        "execute": args.execute,
        "candidates": list(CANDIDATES),
        "d0_train_groups": 2048,
        "maximum_platform_candidates": 2,
        "optimizer_updates": 0,
        "gpu_action_seconds_budget": plan["budget"]["gpu_action_seconds"],
        "preflight": checked,
        "automatic_upload": False,
        "automatic_push": False,
        "parameter_scan": False,
        "test_batch_statistics_for_selection": False,
    }
    print(json.dumps(registration, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return

    subprocess.run(["git", "fetch", "--prune", "origin"], cwd=ROOT, check=True)
    origin_main = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], cwd=ROOT
    ).returncode:
        raise RuntimeError("origin/main changed; synchronize before executing v9")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ):
        raise RuntimeError("Commit the verified v9 implementation before starting the queue")

    output.mkdir(parents=True, exist_ok=False)
    atomic_json_dump(checked, output / "asset_audit.json")
    snapshot = output / "execution_source"
    package_source = ROOT / "reproducibility/aegis_f1/aegis_clip"
    shutil.copytree(package_source, snapshot / "aegis_clip")
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    shutil.copy2(Path(args.config), snapshot / "prelim75_v9.yaml")
    source_files = {
        str(path.relative_to(snapshot)): sha256_file(path)
        for path in sorted(snapshot.rglob("*.py"))
    }
    execution = {
        **registration,
        "git_head": head,
        "origin_main": origin_main,
        "git_status": "clean",
        "config_sha256": plan["_config_sha256"],
        "execution_source_files_sha256": source_files,
        "execution_source_snapshot": str(snapshot),
    }
    atomic_json_dump(execution, output / "plan.json")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(snapshot)
    queue_started = time.monotonic()
    gpu_action_seconds = 0.0
    history = []

    def run(action: str, log_name: str, *, gpu: bool) -> None:
        nonlocal gpu_action_seconds
        remaining = int(plan["budget"]["gpu_action_seconds"] - gpu_action_seconds)
        if gpu and remaining <= 0:
            raise RuntimeError("PRELIM75 v9 GPU action budget exhausted")
        command = [
            sys.executable,
            "-u",
            "-m",
            "aegis_clip.cli.infer_prelim75_v9",
            "--config",
            str(Path(args.config).resolve()),
            "--action",
            action,
        ]
        atomic_json_dump(
            {
                "status": "running",
                "action": action,
                "completed_steps": history,
                "gpu_action_seconds": gpu_action_seconds,
            },
            output / "queue_status.json",
        )
        started = time.monotonic()
        try:
            with (output / log_name).open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=remaining if gpu else None,
                )
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
        elapsed = time.monotonic() - started
        if gpu:
            gpu_action_seconds += elapsed
        row = {
            "action": action,
            "command": command,
            "exit_code": exit_code,
            "elapsed_seconds": elapsed,
            "counts_against_gpu_budget": gpu,
            "log": log_name,
        }
        history.append(row)
        if exit_code:
            atomic_json_dump(
                {
                    "status": "failed_or_budget_exhausted",
                    "failed_step": row,
                    "completed_steps": history,
                    "gpu_action_seconds": gpu_action_seconds,
                    "online_scores": {"R0": None, "R1": None},
                },
                output / "final.json",
            )
            raise RuntimeError(f"Fixed v9 queue stopped; inspect {log_name}")

    run("d0", "d0.log", gpu=False)
    d0 = _read_json(output / "d0/result.json")
    if not d0["passed"]:
        final = {
            "status": "closed_insufficient_source_support",
            "plan_id": PLAN_ID,
            "d0": d0,
            "completed_steps": history,
            "gpu_action_seconds": 0.0,
            "platform_packages_created": 0,
            "online_scores": {"R0": None, "R1": None},
            "decision": "keep_L1_no_prior_69.2794",
            "automatic_upload": False,
            "automatic_push": False,
        }
        atomic_json_dump(final, output / "final.json")
        print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
        return
    run("smoke", "smoke.log", gpu=True)
    run("diagnostic", "diagnostic.log", gpu=True)
    run("infer", "inference.log", gpu=True)
    inference = _read_json(output / "inference/result.json")
    final = {
        "status": inference["status"],
        "plan_id": PLAN_ID,
        "d0": d0,
        "smoke": _read_json(output / "smoke/result.json"),
        "diagnostic": _read_json(output / "diagnostic/result.json"),
        "inference": inference,
        "completed_steps": history,
        "gpu_action_seconds": gpu_action_seconds,
        "gpu_action_seconds_budget": plan["budget"]["gpu_action_seconds"],
        "platform_packages_created": inference["platform_packages_created"],
        "online_scores": {"R0": None, "R1": None},
        "online_exact_correct": {"R0": None, "R1": None},
        "actual_platform_upload_time": {"R0": None, "R1": None},
        "optimizer_updates": 0,
        "automatic_upload": False,
        "automatic_push": False,
        "elapsed_seconds": time.monotonic() - queue_started,
    }
    atomic_json_dump(final, output / "final.json")
    atomic_json_dump(final, output / "queue_status.json")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
