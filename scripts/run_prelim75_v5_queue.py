"""Run the one fixed PRELIM75 v5 queue; default mode is read-only."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))

from aegis_clip.prelim75_fusion import load_fusion_plan, preflight
from aegis_clip.runtime import atomic_json_dump, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/prelim75_v5.yaml"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = load_fusion_plan(args.config)
    output = Path(plan["output"])
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "main":
        raise ValueError("The fixed v5 segment must run directly on main")
    if output.exists():
        raise FileExistsError(f"Existing v5 output is never overwritten or resumed: {output}")
    checked = preflight(plan)
    candidates = ["F0", "F1"]
    registration = {
        "status": "audit_passed", "plan_id": plan["plan_id"], "execute": args.execute,
        "candidates": candidates, "maximum_new_platform_candidates": 2,
        "gradient_training_epoch_budget": 6,
        "gpu_budget_seconds": plan["protocol"]["gpu_budget_seconds"], "preflight": checked,
        "automatic_upload": False, "automatic_push": False, "automatic_F2": False,
        "prior_scan": False,
    }
    print(json.dumps(registration, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return

    subprocess.run(["git", "fetch", "--prune", "origin"], cwd=ROOT, check=True)
    origin_main = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError("origin/main changed; synchronize before executing v5")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise RuntimeError("Commit the verified v5 implementation before starting the fixed queue")

    output.mkdir(parents=True, exist_ok=False); atomic_json_dump(checked, output / "preflight.json")
    snapshot = output / "training_source"; snapshot.mkdir(exist_ok=False)
    package = ROOT / "reproducibility/aegis_f1/aegis_clip"
    for source in package.rglob("*.py"):
        destination = snapshot / "aegis_clip" / source.relative_to(package)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
    shutil.copy2(__file__, snapshot / "run_prelim75_v5_queue.py")
    registration.update({
        "git_head": head, "origin_main": origin_main, "git_status": "clean",
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
    environment = dict(os.environ); environment["PYTHONPATH"] = str(snapshot)
    queue_start = time.monotonic(); history = []

    def run(action, log_name, candidate=None):
        elapsed = time.monotonic() - queue_start
        remaining = plan["protocol"]["gpu_budget_seconds"] - elapsed
        if remaining <= 0:
            raise TimeoutError("Fixed eight-hour v5 budget exhausted")
        command = [
            sys.executable, "-u", "-m", "aegis_clip.cli.train_prelim75_v5",
            "--config", plan["_config_path"], "--action", action,
        ]
        if candidate is not None:
            command.extend(("--candidate", candidate))
        started = time.monotonic()
        atomic_json_dump({
            "status": "running", "action": action, "candidate": candidate,
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
            "command": command, "action": action, "candidate": candidate, "exit_code": exit_code,
            "elapsed_seconds": time.monotonic() - started, "log": log_name,
        }
        history.append(row)
        if exit_code:
            atomic_json_dump({
                "status": "failed_or_budget_exhausted", "failed_step": row,
                "completed_steps": history, "online_accuracy": None,
            }, output / "final_status.json")
            raise RuntimeError(f"Fixed v5 queue stopped; inspect {log_name}")

    run("prepare", "preparation.log")
    run("smoke", "smoke.log")
    delivered = []; stopped = []
    for candidate in candidates:
        run("train", f"{candidate}_training.log", candidate)
        if candidate == "F1":
            first_f0 = json.loads((output / "F0/first_step_gradient_audit.json").read_text())
            first_f1 = json.loads((output / "F1/first_step_gradient_audit.json").read_text())
            if first_f0["batch_indices_sha256"] != first_f1["batch_indices_sha256"]:
                raise RuntimeError("F0/F1 first effective batch identity differs")
            if first_f0["blend"] != 0.0 or first_f1["blend"] != 0.5:
                raise RuntimeError("F0/F1 fixed fusion blends changed")
            for key in ("global_gce", "local_gce", "separate_gce", "fusion_gce", "anchor"):
                if not math.isclose(first_f0[key], first_f1[key], rel_tol=1e-6, abs_tol=1e-7):
                    raise RuntimeError(f"F0/F1 common first-step component differs: {key}")
        run("evaluate", f"{candidate}_diagnostic.log", candidate)
        diagnostic = json.loads((output / candidate / "diagnostic/result.json").read_text())
        if diagnostic["engineering_stop"]:
            stopped.append(candidate)
            atomic_json_dump({
                "status": "engineering_stopped_no_submission", "diagnostic": diagnostic,
                "online_accuracy": None,
            }, output / candidate / "status.json")
        else:
            run("deliver", f"{candidate}_delivery.log", candidate); delivered.append(candidate)
    final = {
        "status": "complete_pending_real_platform_feedback",
        "preparation": json.loads((output / "preparation.json").read_text()),
        "smoke": json.loads((output / "smoke.json").read_text()),
        "delivered_candidates": delivered, "engineering_stopped_candidates": stopped,
        "completed_steps": history, "platform_packages_created": len(delivered),
        "online_scores": {"F0": None, "F1": None},
        "online_exact_correct": {"F0": None, "F1": None},
        "actual_platform_upload_time": {"F0": None, "F1": None},
        "reference_platform_percent": {
            "C0": 68.4544, "S0": 68.2982,
            "historical_prior_0.90": 70.352866, "target": 75.0,
        },
        "automatic_upload": False, "automatic_push": False, "automatic_F2": False,
        "prior_scan": False, "elapsed_seconds": time.monotonic() - queue_start,
    }
    atomic_json_dump(final, output / "final_status.json")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
