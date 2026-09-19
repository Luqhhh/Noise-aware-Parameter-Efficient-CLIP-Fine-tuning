#!/usr/bin/env python3
"""Run the fixed PRELIM75 v8 training-source calibration queue."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))

from aegis_clip.prelim75_source_bias import (  # noqa: E402
    CANDIDATES,
    PLAN_ID,
    load_v8_plan,
    preflight_v8,
)
from aegis_clip.runtime import atomic_json_dump, sha256_file  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/prelim75_v8.yaml"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = load_v8_plan(args.config)
    output = Path(plan["output"])
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != "main":
        raise ValueError("The fixed v8 segment must run directly on main")
    if output.exists():
        raise FileExistsError(f"Existing v8 output is never overwritten: {output}")
    checked = preflight_v8(plan)
    registration = {
        "status": "audit_passed",
        "plan_id": PLAN_ID,
        "execute": args.execute,
        "candidates": list(CANDIDATES),
        "maximum_train_forwards": 2,
        "maximum_cpu_fits": 2,
        "maximum_test_candidates": 2,
        "backbone_optimizer_updates": 0,
        "gpu_action_seconds_budget": plan["budget"]["gpu_action_seconds"],
        "cpu_fit_seconds_budget": plan["budget"]["cpu_fit_seconds"],
        "preflight": checked,
        "automatic_upload": False,
        "automatic_push": False,
        "parameter_sweep": False,
        "test_batch_prior_fit": False,
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
        raise RuntimeError("origin/main changed; synchronize before executing v8")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ):
        raise RuntimeError("Commit the verified v8 implementation before starting the queue")

    output.mkdir(parents=True, exist_ok=False)
    atomic_json_dump(checked, output / "preflight.json")
    snapshot = output / "execution_source"
    snapshot.mkdir(exist_ok=False)
    package = ROOT / "reproducibility/aegis_f1/aegis_clip"
    for source in package.rglob("*.py"):
        destination = snapshot / "aegis_clip" / source.relative_to(package)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    (snapshot / "run_prelim75_v8_queue.py").write_bytes(Path(__file__).read_bytes())
    registration.update(
        {
            "git_head": head,
            "origin_main": origin_main,
            "git_status": "clean",
            "branch_refs": subprocess.check_output(
                [
                    "git",
                    "for-each-ref",
                    "--format=%(refname:short) %(objectname) %(subject)",
                    "refs/heads",
                    "refs/remotes",
                ],
                cwd=ROOT,
                text=True,
            ),
            "recent_history": subprocess.check_output(
                ["git", "log", "-12", "--oneline"], cwd=ROOT, text=True
            ),
            "execution_source_sha256": {
                str(path.relative_to(snapshot)): sha256_file(path)
                for path in snapshot.rglob("*.py")
            },
        }
    )
    atomic_json_dump(registration, output / "queue_registration.json")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(snapshot)
    queue_start = time.monotonic()
    history: list[dict] = []
    gpu_action_seconds = 0.0
    cpu_fit_seconds = 0.0

    def run(
        action: str,
        log_name: str,
        candidate: str | None = None,
        budget_kind: str | None = None,
    ) -> dict:
        nonlocal gpu_action_seconds, cpu_fit_seconds
        if budget_kind == "gpu":
            remaining = float(plan["budget"]["gpu_action_seconds"]) - gpu_action_seconds
        elif budget_kind == "cpu_fit":
            remaining = float(plan["budget"]["cpu_fit_seconds"]) - cpu_fit_seconds
        else:
            remaining = None
        if remaining is not None and remaining <= 0:
            return {
                "action": action,
                "candidate": candidate,
                "exit_code": 124,
                "elapsed_seconds": 0.0,
                "log": log_name,
                "budget_exhausted_before_start": True,
            }
        command = [
            sys.executable,
            "-u",
            "-m",
            "aegis_clip.cli.calibrate_prelim75_v8",
            "--config",
            plan["_config_path"],
            "--action",
            action,
        ]
        if candidate is not None:
            command.extend(("--candidate", candidate))
        atomic_json_dump(
            {
                "status": "running",
                "action": action,
                "candidate": candidate,
                "completed_steps": history,
                "gpu_action_seconds": gpu_action_seconds,
                "cpu_fit_seconds": cpu_fit_seconds,
                "elapsed_seconds": time.monotonic() - queue_start,
            },
            output / "queue_status.json",
        )
        started = time.monotonic()
        try:
            with (output / log_name).open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=remaining,
                )
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
        elapsed = time.monotonic() - started
        if budget_kind == "gpu":
            gpu_action_seconds += elapsed
        elif budget_kind == "cpu_fit":
            cpu_fit_seconds += elapsed
        row = {
            "command": command,
            "action": action,
            "candidate": candidate,
            "exit_code": exit_code,
            "elapsed_seconds": elapsed,
            "budget_kind": budget_kind,
            "log": log_name,
        }
        history.append(row)
        return row

    prepare = run("prepare", "source_preparation.log")
    if prepare["exit_code"]:
        atomic_json_dump(
            {
                "status": "failed_source_preparation",
                "failed_step": prepare,
                "completed_steps": history,
            },
            output / "final_status.json",
        )
        raise RuntimeError("Fixed v8 source preparation failed")

    delivered = []
    stopped = []
    for candidate in CANDIDATES:
        candidate_steps = []
        cache = run(
            "cache", f"{candidate}_training_scores.log", candidate, "gpu"
        )
        candidate_steps.append(cache)
        if cache["exit_code"]:
            stopped.append(
                {"candidate": candidate, "failed_action": "cache", "step": cache}
            )
            continue
        fit = run("fit", f"{candidate}_bias_fit.log", candidate, "cpu_fit")
        candidate_steps.append(fit)
        if fit["exit_code"]:
            stopped.append(
                {"candidate": candidate, "failed_action": "fit", "step": fit}
            )
            continue
        delivery = run(
            "deliver", f"{candidate}_test_delivery.log", candidate, "gpu"
        )
        candidate_steps.append(delivery)
        if delivery["exit_code"]:
            stopped.append(
                {"candidate": candidate, "failed_action": "deliver", "step": delivery}
            )
            continue
        delivered.append(candidate)

    reports = {
        candidate: _read_json(output / candidate / "candidate_report.json")
        for candidate in delivered
    }
    final = {
        "status": "complete_pending_real_platform_feedback",
        "plan_id": PLAN_ID,
        "source_identity_audit": _read_json(
            output / "source/source_identity_audit.json"
        ),
        "delivered_candidates": delivered,
        "stopped_candidates": stopped,
        "candidate_reports": reports,
        "completed_steps": history,
        "platform_packages_created": len(delivered),
        "gpu_action_seconds": gpu_action_seconds,
        "gpu_action_seconds_budget": plan["budget"]["gpu_action_seconds"],
        "cpu_fit_seconds": cpu_fit_seconds,
        "cpu_fit_seconds_budget": plan["budget"]["cpu_fit_seconds"],
        "backbone_optimizer_updates": 0,
        "online_scores": {"B0": None, "B1": None},
        "online_exact_correct": {"B0": None, "B1": None},
        "actual_platform_upload_time": {"B0": None, "B1": None},
        "reference_platform_percent": plan["reference_platform_percent"],
        "decision_rules": {
            "both_not_above_L1": "keep_L1_close_training_source_calibration",
            "new_training_source_best": "keep_real_winner_with_its_own_bias",
            "investment_gate_pp": 0.30,
            "tie_with_L1": "keep_existing_L1",
            "B0_B1_tie_above_L1": "keep_B1",
            "target_reached": "freeze_artifacts_stop_experiments",
        },
        "automatic_upload": False,
        "automatic_push": False,
        "parameter_sweep": False,
        "test_batch_prior_fit": False,
        "elapsed_seconds": time.monotonic() - queue_start,
    }
    atomic_json_dump(final, output / "final_status.json")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    if len(delivered) != len(CANDIDATES):
        raise RuntimeError("One or more fixed v8 candidates stopped; inspect final_status.json")


if __name__ == "__main__":
    main()
