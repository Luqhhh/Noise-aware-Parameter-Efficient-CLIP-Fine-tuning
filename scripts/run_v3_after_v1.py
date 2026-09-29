"""Wait for checked v1 delivery, then run the fixed bounded v3 pair locally."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def check_v1(path):
    status = read(path)
    if status["status"] == "failed":
        raise RuntimeError(f"v1 failed; v3 will not start: {status.get('error')}")
    if status["status"] != "completed":
        return None
    root = Path(status["output"])
    for name in ("training/selected.pt", "validation_report.json", "submission/pred_results.csv",
                 "submission/submission.zip", "submission/submission_check.log"):
        if not (root / name).is_file():
            raise RuntimeError(f"v1 completion lacks {name}")
    if "All checks passed!" not in (root / "submission/submission_check.log").read_text():
        raise RuntimeError("v1 submission did not pass its recorded checks")
    return status


def gpu_busy():
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return bool(result.stdout.strip())


def train_estimate(plan, cost):
    if cost["status"] != "measured_probe" or cost["plan_sha256"] != plan["sha256"]:
        raise ValueError("Invalid v3 cost report")
    values = [cost[k] for k in ("seconds_per_update", "validation_seconds_per_arm_epoch", "overhead_seconds")]
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in values):
        raise ValueError("Invalid v3 cost report")
    return plan["total_updates"] * values[0] + 2 * plan["epochs"] * values[1] + (plan["epochs"] + 1) * values[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-status", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-hours", type=float, default=8)
    parser.add_argument("--probe-hours", type=float, default=2)
    parser.add_argument("--train-hours", type=float, default=8)
    parser.add_argument("--infer-hours", type=float, default=2)
    args = parser.parse_args()
    if any(not math.isfinite(x) or x <= 0 for x in
           (args.wait_hours, args.probe_hours, args.train_hours, args.infer_hours)):
        parser.error("All time limits must be finite and positive")
    from v2.plan import sha
    from v3.plan import verify

    v1_status, plan_path, output = [Path(x).resolve() for x in (args.v1_status, args.plan, args.output)]
    if output.exists():
        raise FileExistsError("v3 handoff requires a fresh output directory")
    output.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "reproducibility/aegis_f1"),
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="0")
    status_path = output / "status.json"
    status = dict(status="waiting_for_v1", stage="wait", v1_status=str(v1_status), plan=str(plan_path),
                  started_at=utc_now(), runner_pid=os.getpid(), train_budget_seconds=args.train_hours * 3600,
                  automatic_retry=False, platform_score=None)
    dump(status_path, status)
    child = None
    started = time.monotonic()

    def update(**fields):
        status.update(fields, heartbeat_at=utc_now())
        dump(status_path, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    def run(stage, command):
        nonlocal child
        log = output / f"{stage}.log"
        with log.open("x") as handle:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            update(status="running", stage=stage, child_pid=child.pid, log=str(log))
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    update()
        child = None
        if code:
            raise RuntimeError(f"v3 {stage} exited {code}; see {log}")

    try:
        while True:
            if time.monotonic() - started > args.wait_hours * 3600:
                raise TimeoutError("v1 did not finish within the handoff wait budget")
            completed = check_v1(v1_status)
            if completed and not gpu_busy():
                break
            update(stage="wait_for_gpu" if completed else "wait_for_v1",
                   v1_completed=bool(completed), gpu_busy=bool(completed))
            time.sleep(30)
        plan = verify(plan_path)
        plan_sha = sha(plan_path)
        plan_info = dict(sha256=plan_sha, total_updates=plan["total_updates"], epochs=plan["recipe"]["epochs"])
        update(stage="v1_checked", v1_completed=True, v1_submission=completed["zip"], plan_sha256=plan_sha)
        probe_budget = args.probe_hours * 3600
        probe_auth = output / "probe_authorization.json"
        dump(probe_auth, dict(authorized=True, operation="probe", plan_sha256=plan_sha,
                              max_seconds=probe_budget, estimated_seconds=probe_budget * .6,
                              source="user_requested_v1_then_v3_20260930"))
        run("probe", [sys.executable, "-u", "-m", "v3.runtime", "probe", "--plan", str(plan_path),
                      "--authorization", str(probe_auth), "--steps", "5"])
        cost_path = plan_path.parent / "runs/probe/cost_report.json"
        cost = read(cost_path)
        estimate = train_estimate(plan_info, cost)
        train_budget = args.train_hours * 3600
        update(stage="cost_gate", measured_train_estimate_seconds=estimate,
               cost_report=str(cost_path), cost_report_sha256=sha(cost_path))
        if estimate > .8 * train_budget:
            update(status="cost_gate_stopped", stage="cost_gate", reason="Measured pair exceeds 80% of finite training budget",
                   training_started=False, completed_at=utc_now())
            return
        if gpu_busy():
            raise RuntimeError("Another CUDA process began after the v3 probe; preserving its GPU slot")
        train_auth = output / "train_authorization.json"
        dump(train_auth, dict(authorized=True, operation="train", plan_sha256=plan_sha,
                              max_seconds=train_budget, estimated_seconds=estimate,
                              cost_report=str(cost_path), cost_report_sha256=sha(cost_path),
                              source="user_requested_v1_then_v3_20260930"))
        run("train", [sys.executable, "-u", "-m", "v3.runtime", "train", "--plan", str(plan_path),
                      "--authorization", str(train_auth)])
        submission = plan_path.parent / "submission"
        run("infer", [sys.executable, "-u", "-m", "v3.delivery", "--plan", str(plan_path),
                      "--output", str(submission), "--max-seconds", str(args.infer_hours * 3600),
                      "--device", "cuda", "--execute"])
        package = read(submission / "report.json")
        update(status="completed", stage="delivered", csv=str(submission / "pred_results.csv"),
               zip=str(submission / "submission.zip"), delivery=package, completed_at=utc_now())
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        update(status="failed", error=str(error), failed_at=utc_now())
        raise


if __name__ == "__main__":
    main()
