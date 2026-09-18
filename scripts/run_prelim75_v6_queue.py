"""Run the one fixed PRELIM75 v6 queue; default mode is read-only."""
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

from aegis_clip.prelim75_online_geometry import (  # noqa: E402
    EXPECTED_D0,
    EXPECTED_PROTOCOL,
    PLAN_ID,
    load_v6_plan,
    preflight_v6,
)
from aegis_clip.runtime import atomic_json_dump, sha256_file  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))




def _write_result_records(final: dict, root: Path) -> None:
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(final, results_dir / "prelim75_v6_final_20260918.json")
    (results_dir / "prelim75_v6_execution_20260918.md").write_text(
        "# PRELIM75 v6 execution status\n\n"
        f"- status: `{final.get('status')}`\n"
        f"- delivered candidates: {final.get('delivered_candidates', [])}\n"
        f"- engineering stopped: {final.get('engineering_stopped_candidates', [])}\n"
        f"- D0 gate passed: {final.get('d0', {}).get('gate_passed')}\n"
        "- platform scores: pending real user feedback; no online score is fabricated.\n",
        encoding="utf-8",
    )


def _compare_common_control(output: Path) -> dict:
    import torch

    g0_audit = _read_json(output / "G0/first_step_gradient_audit.json")
    g1_audit = _read_json(output / "G1/first_step_gradient_audit.json")
    result = {
        "batch_indices_equal": g0_audit["batch_indices_sha256"] == g1_audit["batch_indices_sha256"],
        "batch_flip_scale_equal": g0_audit["batch_flip_scale_sha256"] == g1_audit["batch_flip_scale_sha256"],
        "blend_equal": g0_audit["blend"] == g1_audit["blend"] == 0.5,
        "anchor_close": bool(
            abs(float(g0_audit["anchor"]) - float(g1_audit["anchor"])) <= 1e-6
        ),
    }
    try:
        g0_tensors = torch.load(output / "G0/first_step_global.pt", map_location="cpu", weights_only=False)
        g1_tensors = torch.load(output / "G1/first_step_global.pt", map_location="cpu", weights_only=False)
        for key in ("global_logits", "features", "reference", "targets", "weights"):
            result[f"{key}_close"] = bool(
                torch.allclose(g0_tensors[key], g1_tensors[key], atol=1e-6, rtol=1e-5)
            )
        result["batch_indices_tensor_equal"] = bool(
            torch.equal(g0_tensors["batch_indices"], g1_tensors["batch_indices"])
        )
        result["flips_equal"] = bool(
            torch.equal(g0_tensors["selected_flips"], g1_tensors["selected_flips"])
        )
        result["scales_equal"] = bool(
            torch.equal(g0_tensors["selected_scales"], g1_tensors["selected_scales"])
        )
    except Exception as exc:  # pragma: no cover - fail closed in real execution
        result["tensor_alignment_error"] = repr(exc)
    result["passed"] = all(
        value for key, value in result.items() if key != "tensor_alignment_error"
    ) and "tensor_alignment_error" not in result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/prelim75_v6.yaml"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = load_v6_plan(args.config)
    output = Path(plan["output"])
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "main":
        raise ValueError("The fixed v6 segment must run directly on main")
    if output.exists():
        raise FileExistsError(f"Existing v6 output is never overwritten or resumed: {output}")
    checked = preflight_v6(plan)
    candidates = ["G0", "G1"]
    registration = {
        "status": "audit_passed",
        "plan_id": PLAN_ID,
        "execute": args.execute,
        "candidates": candidates,
        "maximum_new_platform_candidates": 2,
        "gradient_training_epoch_budget": 6,
        "gpu_budget_seconds": EXPECTED_PROTOCOL["gpu_budget_seconds"],
        "d0": dict(EXPECTED_D0),
        "preflight": checked,
        "automatic_upload": False,
        "automatic_push": False,
        "automatic_g2": False,
        "prior_scan": False,
        "parameter_sweep": False,
    }
    print(json.dumps(registration, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return

    subprocess.run(["git", "fetch", "--prune", "origin"], cwd=ROOT, check=True)
    origin_main = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError("origin/main changed; synchronize before executing v6")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise RuntimeError("Commit the verified v6 implementation before starting the fixed queue")

    output.mkdir(parents=True, exist_ok=False)
    atomic_json_dump(checked, output / "preflight.json")
    snapshot = output / "training_source"
    snapshot.mkdir(exist_ok=False)
    package = ROOT / "reproducibility/aegis_f1/aegis_clip"
    for source in package.rglob("*.py"):
        destination = snapshot / "aegis_clip" / source.relative_to(package)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    (snapshot / "run_prelim75_v6_queue.py").write_bytes(Path(__file__).read_bytes())
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
        remaining = float(EXPECTED_PROTOCOL["gpu_budget_seconds"]) - elapsed
        if remaining <= 0:
            raise TimeoutError("Fixed eight-hour v6 budget exhausted")
        command = [
            sys.executable, "-u", "-m", "aegis_clip.cli.train_prelim75_v6",
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
            raise RuntimeError(f"Fixed v6 queue stopped; inspect {log_name}")

    run("prepare", "preparation.log")
    run("d0", "D0_run.log")
    d0 = _read_json(output / "D0/result.json")
    if not d0.get("gate_passed"):
        final = {
            "status": "d0_gate_not_met_no_training",
            "plan_id": PLAN_ID,
            "d0": d0,
            "completed_steps": history,
            "platform_packages_created": 0,
            "online_scores": {"G0": None, "G1": None},
            "automatic_upload": False,
            "automatic_push": False,
            "automatic_g2": False,
            "elapsed_seconds": time.monotonic() - queue_start,
        }
        atomic_json_dump(final, output / "final_status.json")
        _write_result_records(final, ROOT)
        print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
        return

    run("smoke", "smoke.log")
    stopped = []
    delivered = []
    for candidate in candidates:
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
            "online_scores": {"G0": None, "G1": None},
            "automatic_upload": False,
            "automatic_push": False,
            "elapsed_seconds": time.monotonic() - queue_start,
        }
        atomic_json_dump(final, output / "final_status.json")
        _write_result_records(final, ROOT)
        raise RuntimeError("G0/G1 common first-batch control alignment failed")
    final = {
        "status": "complete_pending_real_platform_feedback",
        "plan_id": PLAN_ID,
        "d0": d0,
        "smoke": _read_json(output / "smoke.json"),
        "common_control": common_control,
        "delivered_candidates": delivered,
        "engineering_stopped_candidates": stopped,
        "completed_steps": history,
        "platform_packages_created": len(delivered),
        "online_scores": {"G0": None, "G1": None},
        "online_exact_correct": {"G0": None, "G1": None},
        "actual_platform_upload_time": {"G0": None, "G1": None},
        "reference_platform_percent": {
            "F1": 69.1393,
            "F0": 69.0752,
            "C0": 68.4544,
            "historical_prior_0.90": 70.352866,
            "target": 75.0,
        },
        "decision_rules": {
            "d0_failed": "no_training_keep_F1",
            "both_not_above_F1": "keep_F1_close_geometry_recipe",
            "g0_best": "keep_G0_no_online_gain_claim",
            "g1_best": "keep_G1_report_g1_minus_g0",
            "g1_investment_gate_pp": 0.30,
        },
        "automatic_upload": False,
        "automatic_push": False,
        "automatic_g2": False,
        "parameter_sweep": False,
        "prior_scan": False,
        "elapsed_seconds": time.monotonic() - queue_start,
    }
    atomic_json_dump(final, output / "final_status.json")
    _write_result_records(final, ROOT)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
