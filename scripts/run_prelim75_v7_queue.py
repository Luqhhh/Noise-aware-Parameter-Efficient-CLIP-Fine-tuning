"""Run the one fixed PRELIM75 v7 queue; default mode is read-only."""
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

from aegis_clip.prelim75_cooldown import (  # noqa: E402
    CANDIDATES,
    DEFAULT_STEPS_PER_EPOCH,
    PLAN_ID,
    load_v7_plan,
    preflight_v7,
    scheduler_alignment_check,
)
from aegis_clip.runtime import atomic_json_dump, sha256_file  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_common_control(output: Path) -> dict:
    import torch

    l0 = _read_json(output / "L0/first_step_gradient_audit.json")
    l1 = _read_json(output / "L1/first_step_gradient_audit.json")
    result = {
        "batch_indices_equal": l0["batch_indices_sha256"] == l1["batch_indices_sha256"],
        "batch_flip_scale_equal": l0["batch_flip_scale_sha256"] == l1["batch_flip_scale_sha256"],
        "global_logits_equal": l0["global_logits_sha256"] == l1["global_logits_sha256"],
        "global_features_equal": l0["global_features_sha256"] == l1["global_features_sha256"],
        "first_used_lrs_close": all(
            abs(float(l0["first_used_lrs"][key]) - float(l1["first_used_lrs"][key])) <= 1e-15
            for key in ("backbone", "head", "local_adapters")
        ),
        "horizon_fields_differ": int(l0["cosine_horizon_epochs"]) != int(l1["cosine_horizon_epochs"]),
    }
    try:
        probes0 = torch.load(output / "L0/first_update_probes.pt", map_location="cpu", weights_only=False)
        probes1 = torch.load(output / "L1/first_update_probes.pt", map_location="cpu", weights_only=False)
        def compare(left, right):
            if isinstance(left, dict):
                return all(compare(left[key], right[key]) for key in left)
            return bool(torch.allclose(left, right, atol=1e-6, rtol=1e-5))

        result["first_update_probes_close"] = compare(probes0, probes1)
    except Exception as exc:  # pragma: no cover - fail closed in execution
        result["first_update_probes_error"] = repr(exc)
        close = False
    result["passed"] = all(
        value for key, value in result.items()
        if key not in ("horizon_fields_differ", "first_update_probes_error")
    ) and result.get("first_update_probes_close", False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/prelim75_v7.yaml"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = load_v7_plan(args.config)
    output = Path(plan["output"])
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "main":
        raise ValueError("The fixed v7 segment must run directly on main")
    if output.exists():
        raise FileExistsError(f"Existing v7 output is never overwritten or resumed: {output}")
    checked = preflight_v7(plan)
    registration = {
        "status": "audit_passed",
        "plan_id": PLAN_ID,
        "execute": args.execute,
        "candidates": list(CANDIDATES),
        "maximum_new_platform_candidates": 2,
        "gradient_training_epoch_budget": 6,
        "gpu_budget_seconds": plan["protocol"]["gpu_budget_seconds"],
        "steps_per_epoch_expected": DEFAULT_STEPS_PER_EPOCH,
        "scheduler": plan["scheduler"],
        "preflight": checked,
        "scheduler_check": scheduler_alignment_check(DEFAULT_STEPS_PER_EPOCH),
        "automatic_upload": False,
        "automatic_push": False,
        "automatic_g2": False,
        "parameter_sweep": False,
        "prior_scan": False,
    }
    print(json.dumps(registration, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return

    subprocess.run(["git", "fetch", "--prune", "origin"], cwd=ROOT, check=True)
    origin_main = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError("origin/main changed; synchronize before executing v7")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise RuntimeError("Commit the verified v7 implementation before starting the fixed queue")

    output.mkdir(parents=True, exist_ok=False)
    atomic_json_dump(checked, output / "preflight.json")
    snapshot = output / "training_source"
    snapshot.mkdir(exist_ok=False)
    package = ROOT / "reproducibility/aegis_f1/aegis_clip"
    for source in package.rglob("*.py"):
        destination = snapshot / "aegis_clip" / source.relative_to(package)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    (snapshot / "run_prelim75_v7_queue.py").write_bytes(Path(__file__).read_bytes())
    registration.update({
        "git_head": head,
        "origin_main": origin_main,
        "git_status": "clean",
        "branch_refs": subprocess.check_output(
            ["git", "for-each-ref", "--format=%(refname:short) %(objectname) %(subject)",
             "refs/heads", "refs/remotes"],
            cwd=ROOT, text=True,
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

    def run(action: str, log_name: str, candidate: str | None = None) -> None:
        elapsed = time.monotonic() - queue_start
        remaining = float(plan["protocol"]["gpu_budget_seconds"]) - elapsed
        if remaining <= 0:
            raise TimeoutError("Fixed eight-hour v7 budget exhausted")
        command = [
            sys.executable, "-u", "-m", "aegis_clip.cli.train_prelim75_v7",
            "--config", plan["_config_path"], "--action", action,
        ]
        if candidate is not None:
            command.extend(("--candidate", candidate))
        atomic_json_dump({
            "status": "running",
            "action": action,
            "candidate": candidate,
            "completed_steps": history,
            "elapsed_seconds": elapsed,
        }, output / "queue_status.json")
        started = time.monotonic()
        try:
            with (output / log_name).open("w") as log:
                result = subprocess.run(
                    command, cwd=ROOT, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, timeout=remaining,
                )
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
        row = {
            "command": command,
            "action": action,
            "candidate": candidate,
            "exit_code": exit_code,
            "elapsed_seconds": time.monotonic() - started,
            "log": log_name,
        }
        history.append(row)
        if exit_code:
            atomic_json_dump({
                "status": "failed_or_budget_exhausted",
                "failed_step": row,
                "completed_steps": history,
                "online_accuracy": None,
            }, output / "final_status.json")
            raise RuntimeError(f"Fixed v7 queue stopped; inspect {log_name}")

    run("prepare", "preparation.log")
    run("smoke", "smoke.log")
    stopped = []
    delivered = []
    for candidate in CANDIDATES:
        run("train", f"{candidate}_training.log", candidate)
        run("evaluate", f"{candidate}_diagnostic.log", candidate)
        diagnostic = _read_json(output / candidate / "diagnostic/result.json")
        if diagnostic.get("engineering_stop"):
            stopped.append(candidate)
            atomic_json_dump({
                "status": "engineering_stopped_no_submission",
                "diagnostic": diagnostic,
                "online_accuracy": None,
            }, output / candidate / "status.json")
        else:
            run("deliver", f"{candidate}_delivery.log", candidate)
            delivered.append(candidate)
    common_control = _compare_common_control(output)
    if not common_control["passed"]:
        final = {
            "status": "failed_common_control_alignment",
            "plan_id": PLAN_ID,
            "common_control": common_control,
            "delivered_candidates": delivered,
            "engineering_stopped_candidates": stopped,
            "completed_steps": history,
            "platform_packages_created": len(delivered),
            "online_scores": {"L0": None, "L1": None},
            "automatic_upload": False,
            "automatic_push": False,
            "elapsed_seconds": time.monotonic() - queue_start,
        }
        atomic_json_dump(final, output / "final_status.json")
        raise RuntimeError("L0/L1 common first-batch/first-update control alignment failed")
    final = {
        "status": "complete_pending_real_platform_feedback",
        "plan_id": PLAN_ID,
        "smoke": _read_json(output / "smoke.json"),
        "common_control": common_control,
        "delivered_candidates": delivered,
        "engineering_stopped_candidates": stopped,
        "completed_steps": history,
        "platform_packages_created": len(delivered),
        "online_scores": {"L0": None, "L1": None},
        "online_exact_correct": {"L0": None, "L1": None},
        "actual_platform_upload_time": {"L0": None, "L1": None},
        "reference_platform_percent": {
            "G0": 69.2274,
            "v5_F1": 69.1393,
            "historical_prior_0.90": 70.352866,
            "target": 75.0,
        },
        "decision_rules": {
            "both_not_above_G0": "keep_G0_close_cooldown_recipe",
            "L0_best": "keep_L0_no_cooldown_claim",
            "L1_best": "keep_L1_report_L1_minus_L0",
            "tie_keep_G0": True,
            "L1_investment_gate_pp": 0.30,
            "no_automatic_G2": True,
        },
        "automatic_upload": False,
        "automatic_push": False,
        "automatic_g2": False,
        "parameter_sweep": False,
        "prior_scan": False,
        "elapsed_seconds": time.monotonic() - queue_start,
    }
    atomic_json_dump(final, output / "final_status.json")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
