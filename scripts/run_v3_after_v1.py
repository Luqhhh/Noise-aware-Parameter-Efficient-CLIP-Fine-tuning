"""Wait for checked v1 delivery, then run the fixed v3 pair locally."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
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
    # User services do not inherit the interactive WSL PATH.
    binary = Path("/usr/lib/wsl/lib/nvidia-smi")
    if not binary.is_file():
        found = shutil.which("nvidia-smi")
        if found is None:
            raise FileNotFoundError("nvidia-smi is unavailable for the GPU-idle check")
        binary = Path(found)
    command = [str(binary), "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return bool(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-status", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
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
                  started_at=utc_now(), runner_pid=os.getpid(), automatic_retry=False, platform_score=None)
    dump(status_path, status)
    child = None

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
            completed = check_v1(v1_status)
            if completed and not gpu_busy():
                break
            update(stage="wait_for_gpu" if completed else "wait_for_v1",
                   v1_completed=bool(completed), gpu_busy=bool(completed))
            time.sleep(30)
        verify(plan_path)
        plan_sha = sha(plan_path)
        update(stage="v1_checked", v1_completed=True, v1_submission=completed["zip"], plan_sha256=plan_sha)
        train_auth = output / "train_authorization.json"
        dump(train_auth, dict(authorized=True, operation="train", plan_sha256=plan_sha,
                              source="user_requested_v1_then_v3_20260930"))
        run("train", [sys.executable, "-u", "-m", "v3.runtime", "train", "--plan", str(plan_path),
                      "--authorization", str(train_auth)])
        submission = plan_path.parent / "submission"
        run("infer", [sys.executable, "-u", "-m", "v3.delivery", "--plan", str(plan_path),
                      "--output", str(submission), "--device", "cuda", "--execute"])
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
