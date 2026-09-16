"""Run the single fixed PRELIM75 v3 queue; default mode is read-only.

An explicit ``--execute`` performs D0 and stops if its fixed feasibility gate
is closed.  If it passes, S0 and S1 are independently trained, diagnosed, and
packaged.  This script never uploads a package or pushes Git state.
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

from aegis_clip.prelim75_recovery import load_recovery_plan, preflight
from aegis_clip.runtime import atomic_json_dump, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/prelim75_v3.yaml"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = load_recovery_plan(args.config)
    output = Path(plan["output"])
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "main":
        raise ValueError("The fixed v3 segment must run directly on main")
    if output.exists():
        raise FileExistsError(f"Existing v3 output is never overwritten or resumed: {output}")
    checked = preflight(plan)
    registration = {
        "status": "audit_passed",
        "plan_id": plan["plan_id"],
        "execute": args.execute,
        "candidates": ["S0", "S1"],
        "maximum_new_platform_candidates": 2,
        "gradient_training_epoch_budget": 6,
        "gpu_budget_seconds": plan["protocol"]["gpu_budget_seconds"],
        "D0_gate": {"minimum_groups": 512, "minimum_classes": 100},
        "preflight": checked,
        "automatic_upload": False,
        "automatic_push": False,
        "automatic_v4": False,
    }
    print(json.dumps(registration, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return

    subprocess.run(["git", "fetch", "--prune", "origin"], cwd=ROOT, check=True)
    origin_main = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError("origin/main changed; synchronize before executing v3")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status:
        raise RuntimeError("Commit the verified v3 implementation before starting the fixed queue")

    output.mkdir(parents=True, exist_ok=False)
    atomic_json_dump(checked, output / "preflight.json")
    snapshot = output / "training_source"
    snapshot.mkdir(exist_ok=False)
    package = ROOT / "reproducibility/aegis_f1/aegis_clip"
    for source in package.rglob("*.py"):
        destination = snapshot / "aegis_clip" / source.relative_to(package)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(__file__, snapshot / "run_prelim75_v3_queue.py")
    registration.update({
        "git_head": head,
        "origin_main": origin_main,
        "git_status": "clean",
        "branch_refs": subprocess.check_output(
            ["git", "for-each-ref", "--format=%(refname:short) %(objectname) %(subject)",
             "refs/heads", "refs/remotes"], cwd=ROOT, text=True,
        ),
        "recent_history": subprocess.check_output(["git", "log", "-12", "--oneline"], cwd=ROOT, text=True),
        "training_source_sha256": {
            str(path.relative_to(snapshot)): sha256_file(path) for path in snapshot.rglob("*.py")
        },
    })
    atomic_json_dump(registration, output / "queue_registration.json")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(snapshot)
    queue_start = time.monotonic()
    history = []

    def run(module, log_name, candidate=None):
        elapsed = time.monotonic() - queue_start
        remaining = plan["protocol"]["gpu_budget_seconds"] - elapsed
        if remaining <= 0:
            raise TimeoutError("Fixed eight-hour v3 GPU budget exhausted")
        command = [sys.executable, "-u", "-m", module, "--config", plan["_config_path"]]
        if candidate is not None:
            command.extend(("--candidate", candidate))
        step_start = time.monotonic()
        atomic_json_dump({
            "status": "running", "module": module, "candidate": candidate,
            "completed_steps": history, "elapsed_seconds": elapsed,
        }, output / "queue_status.json")
        try:
            with (output / log_name).open("w") as log:
                result = subprocess.run(
                    command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    timeout=remaining,
                )
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
        row = {
            "command": command, "candidate": candidate, "exit_code": exit_code,
            "elapsed_seconds": time.monotonic() - step_start, "log": log_name,
        }
        history.append(row)
        if exit_code:
            atomic_json_dump({
                "status": "failed_or_budget_exhausted", "failed_step": row,
                "completed_steps": history, "online_accuracy": None,
            }, output / "final_status.json")
            raise RuntimeError(f"Fixed v3 queue stopped; inspect {log_name}")

    run("aegis_clip.cli.cache_prelim75_recovery", "D0.log")
    cohort = json.loads((output / "cohort/report.json").read_text())
    if not cohort["gate_passed"]:
        final = {
            "status": "closed_insufficient_recovery_cohort",
            "D0": cohort,
            "completed_steps": history,
            "training_started": False,
            "platform_packages_created": 0,
            "online_scores": {"S0": None, "S1": None},
            "automatic_v4_started": False,
        }
        atomic_json_dump(final, output / "final_status.json")
        print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
        return

    delivered = []
    stopped = []
    for candidate in ("S0", "S1"):
        run("aegis_clip.cli.train_prelim75_recovery", f"{candidate}_training.log", candidate)
        run("aegis_clip.cli.evaluate_prelim75_recovery", f"{candidate}_diagnostic.log", candidate)
        diagnostic = json.loads((output / candidate / "diagnostic/result.json").read_text())
        if diagnostic["engineering_stop"]:
            stopped.append(candidate)
            atomic_json_dump({
                "status": "engineering_stopped_no_submission", "diagnostic": diagnostic,
                "online_accuracy": None,
            }, output / candidate / "status.json")
        else:
            run("aegis_clip.cli.infer_prelim75_recovery", f"{candidate}_delivery.log", candidate)
            delivered.append(candidate)
    final = {
        "status": "complete_pending_real_platform_feedback",
        "D0": cohort,
        "delivered_candidates": delivered,
        "engineering_stopped_candidates": stopped,
        "completed_steps": history,
        "platform_packages_created": len(delivered),
        "online_scores": {"S0": None, "S1": None},
        "online_exact_correct": {"S0": None, "S1": None},
        "actual_platform_upload_time": {"S0": None, "S1": None},
        "reference_platform_percent": {"V1": 67.5932, "historical_prior_0.90": 70.352866},
        "automatic_upload": False,
        "automatic_push": False,
        "automatic_v4_started": False,
        "elapsed_seconds": time.monotonic() - queue_start,
    }
    atomic_json_dump(final, output / "final_status.json")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
