"""Run the single fixed PRELIM75 v4 queue; default mode is read-only.

An explicit ``--execute`` freezes the original-supervision trusted mask,
performs an isolated common-microbatch smoke, and then runs C0 plus C1 when
the support gate passes.  It never uploads packages, starts C2, or pushes Git.
"""
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

from aegis_clip.prelim75_trusted_ce import load_trusted_ce_plan, preflight
from aegis_clip.runtime import atomic_json_dump, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/prelim75_v4.yaml"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = load_trusted_ce_plan(args.config)
    output = Path(plan["output"])
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "main":
        raise ValueError("The fixed v4 segment must run directly on main")
    if output.exists():
        raise FileExistsError(f"Existing v4 output is never overwritten or resumed: {output}")
    checked = preflight(plan)
    support = checked["trusted_support"]
    candidates = ["C0"] + (["C1"] if support["gate_passed"] else [])
    registration = {
        "status": "audit_passed", "plan_id": plan["plan_id"], "execute": args.execute,
        "candidates": candidates, "C1_support_gate": support,
        "maximum_new_platform_candidates": 2, "gradient_training_epoch_budget": 3 * len(candidates),
        "gpu_budget_seconds": plan["protocol"]["gpu_budget_seconds"], "preflight": checked,
        "automatic_upload": False, "automatic_push": False, "automatic_C2": False,
    }
    print(json.dumps(registration, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return

    subprocess.run(["git", "fetch", "--prune", "origin"], cwd=ROOT, check=True)
    origin_main = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError("origin/main changed; synchronize before executing v4")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise RuntimeError("Commit the verified v4 implementation before starting the fixed queue")

    output.mkdir(parents=True, exist_ok=False); atomic_json_dump(checked, output / "preflight.json")
    snapshot = output / "training_source"; snapshot.mkdir(exist_ok=False)
    package = ROOT / "reproducibility/aegis_f1/aegis_clip"
    for source in package.rglob("*.py"):
        destination = snapshot / "aegis_clip" / source.relative_to(package)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
    shutil.copy2(__file__, snapshot / "run_prelim75_v4_queue.py")
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
            raise TimeoutError("Fixed eight-hour v4 budget exhausted")
        command = [
            sys.executable, "-u", "-m", "aegis_clip.cli.train_prelim75_v4",
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
            raise RuntimeError(f"Fixed v4 queue stopped; inspect {log_name}")

    run("prepare", "trusted_support.log")
    frozen_support = json.loads((output / "trusted_support/report.json").read_text())
    if frozen_support["gate_passed"] != support["gate_passed"]:
        raise RuntimeError("C1 support decision changed after output creation")
    run("smoke", "smoke.log")
    delivered = []; stopped = []
    for candidate in candidates:
        run("train", f"{candidate}_training.log", candidate)
        if candidate == "C1":
            first_c0 = json.loads((output / "C0/first_step_gradient_audit.json").read_text())
            first_c1 = json.loads((output / "C1/first_step_gradient_audit.json").read_text())
            if first_c0["batch_indices_sha256"] != first_c1["batch_indices_sha256"]:
                raise RuntimeError("C0/C1 first effective batch identity differs")
            if first_c0["beta"] != 0 or first_c1["beta"] != 0:
                raise RuntimeError("C0/C1 first step is not the required beta=0 equivalence point")
            for key in (
                "global_gce", "global_mixed", "local_gce", "anchor", "loss",
                "visual_gradient_norm", "head_gradient_norm", "o3_gradient_norm", "pta_gradient_norm",
            ):
                if not math.isclose(first_c0[key], first_c1[key], rel_tol=1e-6, abs_tol=1e-7):
                    raise RuntimeError(f"C0/C1 beta=0 first-step value differs: {key}")
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
        "trusted_support": frozen_support, "smoke": json.loads((output / "smoke.json").read_text()),
        "delivered_candidates": delivered, "engineering_stopped_candidates": stopped,
        "completed_steps": history, "platform_packages_created": len(delivered),
        "online_scores": {"C0": None, "C1": None},
        "online_exact_correct": {"C0": None, "C1": None},
        "actual_platform_upload_time": {"C0": None, "C1": None},
        "reference_platform_percent": {
            "S0": 68.2982, "V1": 67.5932, "historical_prior_0.90": 70.352866,
        },
        "automatic_upload": False, "automatic_push": False, "automatic_C2": False,
        "elapsed_seconds": time.monotonic() - queue_start,
    }
    atomic_json_dump(final, output / "final_status.json")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
