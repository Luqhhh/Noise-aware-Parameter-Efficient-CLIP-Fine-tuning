"""Durable local controller for explicitly requested v3 checkpoint recovery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_v3_after_v1 import dump, gpu_busy, read, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--frozen-code-root", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_path, frozen, output = [Path(p).resolve() for p in (args.plan, args.frozen_code_root, args.output)]
    sys.path.insert(0, str(frozen / "reproducibility/aegis_f1"))
    from v2.plan import sha
    from v3.plan import verify
    verify(plan_path)
    if gpu_busy():
        raise RuntimeError("Local GPU already has a compute task; refusing to compete")
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(frozen / "reproducibility/aegis_f1"), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="0")
    status = dict(status="starting", stage="checkpoint_recovery", plan=str(plan_path),
                  plan_sha256=sha(plan_path), frozen_code_root=str(frozen), runner_pid=os.getpid(),
                  checkpoint_sha256=args.checkpoint_sha256, original_retrained=False,
                  automatic_retry=False, platform_score=None, started_at=utc_now())
    child = None

    def update(**fields):
        status.update(fields, heartbeat_at=utc_now())
        dump(output / "status.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    def run(stage, command):
        nonlocal child
        log = output / f"{stage}.log"
        with log.open("x") as handle:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
            update(status="running", stage=stage, child_pid=child.pid, log=str(log))
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    update()
        child = None
        if code:
            raise RuntimeError(f"{stage} exited {code}; see {log}")

    try:
        authorization = output / "train_authorization.json"
        dump(authorization, dict(authorized=True, operation="train", plan_sha256=sha(plan_path),
                                 source="user_continue_20260930_after_epoch2_pause"))
        run("recover_train", [sys.executable, "-u", str(root / "scripts/recover_v3_checkpoint.py"),
                              "--plan", str(plan_path), "--authorization", str(authorization),
                              "--checkpoint-sha256", args.checkpoint_sha256, "--execute"])
        submission = plan_path.parent / "submission"
        run("infer", [sys.executable, "-u", "-m", "v3.delivery", "--plan", str(plan_path),
                      "--output", str(submission), "--device", "cuda", "--execute"])
        update(status="completed", stage="delivered", csv=str(submission / "pred_results.csv"),
               zip=str(submission / "submission.zip"), delivery=read(submission / "report.json"), completed_at=utc_now())
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
