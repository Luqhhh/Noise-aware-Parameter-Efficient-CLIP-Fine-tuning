#!/usr/bin/env python3
"""Run the single fixed PRELIM75 v10 AdamW/SAM queue; default is read-only."""
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

from aegis_clip.prelim75_sam import (  # noqa: E402
    CANDIDATES,
    DEFAULT_STEPS_PER_EPOCH,
    PLAN_ID,
    TOTAL_UPDATES,
    load_v10_plan,
    preflight_v10,
    scheduler_reference,
)
from aegis_clip.runtime import atomic_json_dump, sha256_file  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _common_control(output: Path) -> dict:
    left = _read_json(output / "S0/first_step_gradient_audit.json")
    right = _read_json(output / "S1/first_step_gradient_audit.json")
    exact_fields = (
        "batch_indices_sha256", "batch_paths_sha256", "batch_images_sha256",
        "batch_flip_scale_sha256", "batch_targets_sha256", "batch_weights_sha256",
        "batch_reference_sha256", "global_logits_sha256", "global_features_sha256",
        "local_logits_sha256", "fixed_boxes_sha256", "reference_sha256",
    )
    result = {
        "exact_fields": {key: left[key] == right[key] for key in exact_fields},
        "first_pass_loss_abs_difference": abs(
            float(left["full_effective_batch_loss"]) - float(right["full_effective_batch_loss"])
        ),
        "first_used_lrs_equal": left["first_used_lrs"] == right["first_used_lrs"],
        "S0_standard_sam": False,
        "S1_standard_sam": True,
        "post_update_parameters_expected_to_differ": True,
    }
    result["passed"] = (
        all(result["exact_fields"].values())
        and result["first_pass_loss_abs_difference"] <= 1e-7
        and result["first_used_lrs_equal"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/prelim75_v10_sam.yaml"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = load_v10_plan(args.config)
    output = Path(plan["output"])
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "main":
        raise ValueError("The fixed PRELIM75 v10 segment must run directly on main")
    if output.exists():
        raise FileExistsError(f"Existing PRELIM75 v10 output is never overwritten or auto-resumed: {output}")
    checked = preflight_v10(plan)
    refs = subprocess.check_output(
        ["git", "for-each-ref", "--format=%(refname:short) %(objectname) %(subject)",
         "refs/heads", "refs/remotes"], cwd=ROOT, text=True,
    )
    processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    competing = [
        line for line in processes.splitlines()
        if "prelim75_v10_sam" in line.lower() and "--execute" in line
        and "python" in line.lower()
        and int(line.split(None, 1)[0]) not in (os.getpid(), os.getppid())
    ]
    if competing:
        raise RuntimeError(f"Another PRELIM75 v10 SAM process is running: {competing}")
    registration = {
        "status": "audit_passed",
        "plan_id": PLAN_ID,
        "execute": args.execute,
        "candidates": list(CANDIDATES),
        "maximum_platform_candidates": 2,
        "formal_optimizer_updates_per_candidate": TOTAL_UPDATES,
        "formal_optimizer_updates_total": 2 * TOTAL_UPDATES,
        "steps_per_epoch": DEFAULT_STEPS_PER_EPOCH,
        "gpu_action_seconds_budget": plan["budget"]["gpu_action_seconds"],
        "preflight": checked,
        "scheduler_reference": scheduler_reference(),
        "branch_refs": refs,
        "competing_sam_processes": competing,
        "automatic_upload": False,
        "automatic_push": False,
        "parameter_sweep": False,
        "prior_scan": False,
        "automatic_SANER_or_NCSAM": False,
    }
    print(json.dumps(registration, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return

    subprocess.run(["git", "fetch", "--prune", "origin"], cwd=ROOT, check=True)
    origin_main = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != origin_main:
        raise RuntimeError("PRELIM75 v10 execution requires HEAD exactly equal to origin/main")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise RuntimeError("Commit and push the verified PRELIM75 v10 implementation before execution")

    output.mkdir(parents=True, exist_ok=False)
    atomic_json_dump(checked, output / "asset_audit.json")
    snapshot = output / "execution_source"
    package_source = ROOT / "reproducibility/aegis_f1/aegis_clip"
    shutil.copytree(package_source, snapshot / "aegis_clip")
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    shutil.copy2(Path(args.config), snapshot / "prelim75_v10_sam.yaml")
    execution = {
        **registration,
        "git_head": head,
        "origin_main": origin_main,
        "git_status": "clean",
        "config_sha256": plan["_config_sha256"],
        "execution_source_snapshot": str(snapshot),
        "execution_source_files_sha256": {
            str(path.relative_to(snapshot)): sha256_file(path)
            for path in sorted(snapshot.rglob("*.py"))
        },
        "recent_history": subprocess.check_output(["git", "log", "-15", "--oneline"], cwd=ROOT, text=True),
    }
    atomic_json_dump(execution, output / "plan.json")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(snapshot)
    queue_started = time.monotonic()
    gpu_action_seconds = 0.0
    history = []

    def fail_status(status: str, **extra) -> None:
        atomic_json_dump({
            "status": status,
            "plan_id": PLAN_ID,
            "completed_steps": history,
            "gpu_action_seconds": gpu_action_seconds,
            "online_scores": {"S0": None, "S1": None},
            "platform_packages_created": 0,
            "automatic_upload": False,
            "automatic_push": False,
            **extra,
        }, output / "final_status.json")

    def run(action: str, log_name: str, candidate: str | None = None, *, gpu: bool) -> None:
        nonlocal gpu_action_seconds
        remaining = float(plan["budget"]["gpu_action_seconds"]) - gpu_action_seconds
        if gpu and remaining <= 0:
            fail_status("failed_gpu_budget_exhausted_before_action", action=action, candidate=candidate)
            raise RuntimeError("PRELIM75 v10 GPU budget exhausted")
        command = [
            sys.executable, "-u", "-m", "aegis_clip.cli.train_prelim75_v10_sam",
            "--config", str(Path(args.config).resolve()), "--action", action,
        ]
        if candidate is not None:
            command.extend(("--candidate", candidate))
        atomic_json_dump({
            "status": "running", "action": action, "candidate": candidate,
            "completed_steps": history, "gpu_action_seconds": gpu_action_seconds,
            "remaining_gpu_action_seconds": remaining,
        }, output / "queue_status.json")
        started = time.monotonic()
        try:
            with (output / log_name).open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, cwd=ROOT, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, timeout=remaining if gpu else None,
                )
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
        elapsed = time.monotonic() - started
        if gpu:
            gpu_action_seconds += elapsed
        row = {
            "action": action, "candidate": candidate, "command": command,
            "exit_code": exit_code, "elapsed_seconds": elapsed,
            "counts_against_gpu_budget": gpu, "log": log_name,
        }
        history.append(row)
        if exit_code:
            fail_status("failed_or_budget_exhausted", failed_step=row)
            raise RuntimeError(f"Fixed PRELIM75 v10 queue stopped; inspect {log_name}")

    run("prepare", "preparation.log", gpu=False)
    run("a0", "a0.log", gpu=True)
    run("smoke", "smoke.log", gpu=True)
    smoke = _read_json(output / "smoke.json")
    estimate = smoke["resource_estimate"]
    projected_with_gates = gpu_action_seconds + estimate["estimated_total_after_gates_seconds"]
    if projected_with_gates >= plan["budget"]["gpu_action_seconds"]:
        fail_status(
            "closed_projected_total_exceeds_budget",
            resource_estimate=estimate,
            projected_gpu_action_seconds=projected_with_gates,
        )
        print(json.dumps(_read_json(output / "final_status.json"), indent=2), flush=True)
        return

    run("train", "S0_training.log", "S0", gpu=True)
    remaining = plan["budget"]["gpu_action_seconds"] - gpu_action_seconds
    required = estimate["estimated_S1_training_seconds"] + estimate["diagnostic_and_delivery_reserve_seconds"]
    if remaining <= required:
        fail_status(
            "closed_insufficient_remaining_budget_before_S1",
            remaining_gpu_action_seconds=remaining,
            required_estimated_seconds=required,
        )
        print(json.dumps(_read_json(output / "final_status.json"), indent=2), flush=True)
        return
    run("train", "S1_training.log", "S1", gpu=True)
    common = _common_control(output)
    atomic_json_dump(common, output / "common_control.json")
    if not common["passed"]:
        fail_status("failed_common_control_alignment", common_control=common)
        raise RuntimeError("PRELIM75 v10 S0/S1 common-control alignment failed")

    delivered, stopped = [], []
    for candidate in CANDIDATES:
        run("evaluate", f"{candidate}_diagnostic.log", candidate, gpu=True)
        diagnostic = _read_json(output / candidate / "diagnostic/result.json")
        if diagnostic.get("engineering_stop"):
            stopped.append(candidate)
            atomic_json_dump({
                "status": "engineering_stopped_no_submission",
                "diagnostic": diagnostic,
                "online_accuracy": None,
            }, output / candidate / "status.json")
        else:
            run("deliver", f"{candidate}_delivery.log", candidate, gpu=True)
            delivered.append(candidate)
    final = {
        "status": "complete_pending_real_platform_feedback",
        "plan_id": PLAN_ID,
        "a0": _read_json(output / "a0.json"),
        "smoke": smoke,
        "common_control": common,
        "delivered_candidates": delivered,
        "engineering_stopped_candidates": stopped,
        "completed_steps": history,
        "platform_packages_created": len(delivered),
        "online_scores": {"S0": None, "S1": None},
        "online_exact_correct": {"S0": None, "S1": None},
        "actual_platform_upload_time": {"S0": None, "S1": None},
        "reference_platform_percent": plan["reference_platform_percent"],
        "decision_rules": {
            "S1_not_above_S0": "close_SAM_and_forbid_automatic_SANER_NCSAM",
            "S1_above_S0_not_above_L1": "retain_L1",
            "S1_above_both_but_gain_below_0.30pp": "retain_S1_small_gain_and_close",
            "S1_gain_at_least_0.30pp": "retain_S1_and_allow_separate_future_preregistration_only",
            "ties_prefer_L1_then_S0": True,
        },
        "formal_optimizer_updates": {"S0": TOTAL_UPDATES, "S1": TOTAL_UPDATES},
        "gpu_action_seconds": gpu_action_seconds,
        "gpu_action_seconds_budget": plan["budget"]["gpu_action_seconds"],
        "automatic_upload": False,
        "automatic_push": False,
        "parameter_sweep": False,
        "prior_used": False,
        "elapsed_seconds": time.monotonic() - queue_started,
    }
    atomic_json_dump(final, output / "final_status.json")
    atomic_json_dump(final, output / "queue_status.json")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
