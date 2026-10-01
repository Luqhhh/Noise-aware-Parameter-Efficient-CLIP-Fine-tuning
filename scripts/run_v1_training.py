"""Run one authorized local v1 trajectory through checked CSV/ZIP delivery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--desktop-zip", type=Path, help="Copy the checked final ZIP to a new desktop path")
    args = parser.parse_args()
    if args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    root = Path(__file__).resolve().parents[1]
    config = Path(args.config).resolve()
    recipe = yaml.safe_load(config.read_text())
    if args.desktop_zip and args.desktop_zip.exists():
        raise FileExistsError(f"Desktop delivery already exists: {args.desktop_zip}")
    output = Path(recipe["output"]["root"])
    if not output.is_absolute():
        output = (config.parent / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.exists() or (output / "targets.pt").exists() or (output / "training").exists():
        raise FileExistsError("Fresh v1 execution requires a new output directory")
    checkpoint = output / "training/selected.pt"
    env = dict(os.environ, PYTHONPATH=str(root / "reproducibility/aegis_f1"),
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="0")
    started = time.monotonic()
    status = dict(experiment_id=recipe["project"]["experiment_id"], strategy="v1", status="starting",
        worktree=str(root), config=str(config), output=str(output), runner_pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(), max_hours=args.max_hours,
        training_partition=recipe["source"]["partition"], swa_enabled=recipe["train"]["swa_enabled"],
        swa_epochs=list(range(recipe["train"]["swa_start"], recipe["train"]["epochs"] + 1))
            if recipe["train"]["swa_enabled"] else [],
        training_started=False, local_score=None, platform_score=None, automatic_retry=False)
    commands = []
    stages = ("targets", "train", "infer") if recipe["decode"]["bias_source"] == "test_uniform_experimental" else (
        "targets", "train", "calibrate", "infer")
    for stage in stages:
        command = [sys.executable, "-u", "-m", "aegis_clip.cli.v1", stage,
                   "--config", str(config), "--device", "cuda", "--execute"]
        if stage in ("calibrate", "infer"):
            command += ["--checkpoint", str(checkpoint)]
        commands.append((stage, command))
    write_json(output / "run_protocol.json", dict(config=str(config), recipe=recipe,
        max_hours=args.max_hours, commands=[dict(stage=s, argv=c) for s, c in commands],
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()))
    child = None
    try:
        for stage, command in commands:
            log_path = output / f"{stage}.log"
            with log_path.open("x") as log:
                child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                status.update(status="running", stage=stage, child_pid=child.pid, log=str(log_path))
                write_json(status_path, status)
                print(json.dumps(status, ensure_ascii=False), flush=True)
                while True:
                    remaining = args.max_hours * 3600 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("The fixed v1 execution time cap was reached")
                    try:
                        code = child.wait(timeout=min(30, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        status.update(elapsed_seconds=time.monotonic() - started,
                            heartbeat_at=datetime.now(timezone.utc).isoformat())
                        progress = output / "training/progress.json"
                        if progress.exists():
                            status["training_progress"] = json.loads(progress.read_text())
                            status["training_started"] = True
                        write_json(status_path, status)
                status["elapsed_seconds"] = time.monotonic() - started
                if code != 0:
                    raise RuntimeError(f"v1 {stage} exited with code {code}; see {log_path}")
                if stage == "train":
                    status["training_started"] = True
                if stage == "calibrate":
                    status["local_validation"] = json.loads((output / "validation_report.json").read_text())
            child = None
        submission = output / "submission"
        for name in ("pred_results.csv", "submission.zip", "submission_check.log"):
            if not (submission / name).is_file():
                raise FileNotFoundError(submission / name)
        if "All checks passed" not in (submission / "submission_check.log").read_text():
            raise RuntimeError("Submission checker did not report success")
        manifest = json.loads((submission / "manifest.json").read_text())
        if recipe["train"]["swa_enabled"] and manifest["selected_policy"] != "swa_ema":
            raise ValueError("The requested EMA SWA checkpoint was not selected")
        zip_hash = hashlib.sha256((submission / "submission.zip").read_bytes()).hexdigest()
        status.update(zip_sha256=zip_hash, selected_policy=manifest["selected_policy"])
        if args.desktop_zip:
            # Exclusive creation protects previous desktop submissions.
            with (submission / "submission.zip").open("rb") as source, args.desktop_zip.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            desktop_hash = hashlib.sha256(args.desktop_zip.read_bytes()).hexdigest()
            if desktop_hash != zip_hash:
                raise ValueError("Desktop submission hash differs")
            status.update(desktop_zip=str(args.desktop_zip), desktop_zip_sha256=desktop_hash)
        status.update(status="completed", stage="delivered", checkpoint=str(checkpoint),
            csv=str(submission / "pred_results.csv"), zip=str(submission / "submission.zip"),
            submission_check=str(submission / "submission_check.log"),
            elapsed_seconds=time.monotonic() - started,
            completed_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        status.update(status="failed", error=str(error), elapsed_seconds=time.monotonic() - started,
                      failed_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
