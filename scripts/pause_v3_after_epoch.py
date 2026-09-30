"""Suspend an existing v3 process tree after a durable, validated epoch boundary."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pause-watcher.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def process(pid):
    root = Path(f"/proc/{pid}")
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, state=fields[0], ppid=int(fields[1]), start_ticks=int(fields[19]),
                argv=(root / "cmdline").read_bytes().decode().strip("\0").split("\0"))


def descendants(parent):
    processes = {}
    for root in Path("/proc").iterdir():
        if root.name.isdecimal():
            try:
                processes[int(root.name)] = process(int(root.name))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
    found = {parent}
    while True:
        expanded = found | {pid for pid, p in processes.items() if p["ppid"] in found}
        if expanded == found:
            return {pid: processes[pid] for pid in found if pid in processes}
        found = expanded


def boundary(arm_dir, epoch, plan_hash, updates):
    history_path = arm_dir / "history.json"
    checkpoint = arm_dir / f"epoch{epoch:02d}.pt"
    meta_path = checkpoint.with_suffix(".binding.json")
    if not all(p.is_file() for p in (history_path, checkpoint, meta_path)):
        return None
    history, meta = [json.loads(p.read_text()) for p in (history_path, meta_path)]
    if not history or history[-1]["epoch"] < epoch:
        return None
    if history[-1]["epoch"] != epoch or history[-1]["updates"] != updates:
        raise ValueError("Epoch history differs from the fixed pause boundary")
    expected = dict(plan_sha256=plan_hash, arm=arm_dir.name, probe=False, complete=True)
    if meta["binding"] != expected:
        raise ValueError("Checkpoint binding differs from the requested completed arm")
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=meta["sha256"], history=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads(args.status.read_text())
    plan_path = Path(status["plan"])
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan = json.loads(plan_path.read_text())
    if plan_hash != status["plan_sha256"] or args.epoch != plan["recipe"]["epochs"]:
        raise ValueError("Pause must bind to the final epoch of the existing fixed arm")
    if args.arm_dir.resolve() != plan_path.parent / "runs/train/original":
        raise ValueError("Only the existing original arm may be paused by this watcher")
    identities = {pid: process(pid) for pid in (status["runner_pid"], status["child_pid"])}
    child_pid, runner_pid = status["child_pid"], status["runner_pid"]
    if ("v3.runtime" not in identities[child_pid]["argv"] or
            str(plan_path) not in identities[child_pid]["argv"] or
            not any(arg.endswith("run_v3_after_v1.py") for arg in identities[runner_pid]["argv"])):
        raise ValueError("Process identities do not match the intended v3 task")
    report = dict(status="armed_waiting_epoch", requested_at=now(), epoch=args.epoch,
                  arm="original", plan_sha256=plan_hash, watched_pids=list(identities),
                  automatic_resume=False, watcher_pid=os.getpid())
    dump(args.report, report)
    print(json.dumps(report), flush=True)
    try:
        last_heartbeat = 0.0
        while True:
            for pid, identity in identities.items():
                if process(pid)["start_ticks"] != identity["start_ticks"]:
                    raise RuntimeError("A watched PID was replaced; refusing to signal it")
            finished = boundary(args.arm_dir, args.epoch, plan_hash, plan["updates_per_arm"])
            if finished:
                # The epoch checkpoint and history are already durable. Stop the
                # training process before the runner or any next-arm work.
                os.kill(child_pid, signal.SIGSTOP)
                os.kill(runner_pid, signal.SIGSTOP)
                tree = descendants(runner_pid)
                paused = []
                for pid, p in tree.items():
                    if p["state"] != "Z":
                        os.kill(pid, signal.SIGSTOP)
                        paused.append(pid)
                for _ in range(100):
                    states = {pid: process(pid)["state"] for pid in paused}
                    if all(state == "T" for state in states.values()):
                        break
                    time.sleep(.02)
                else:
                    raise RuntimeError("Process suspension did not complete")
                next_arm = args.arm_dir.parent / "v1_supervision"
                report.update(status="paused_by_user", paused_at=now(), paused_pids=paused,
                              process_states=states, next_arm_started=next_arm.exists(), **finished)
                checkpoint = Path(finished["checkpoint"])
                digest = hashlib.sha256()
                with checkpoint.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                actual = digest.hexdigest()
                if actual != finished["checkpoint_sha256"]:
                    raise ValueError("Paused checkpoint checksum mismatch")
                status = json.loads(args.status.read_text())
                status.update(status="paused_by_user", pause_reason="User requested pause after original arm epoch 2",
                              paused_at=report["paused_at"], paused_pids=paused, automatic_resume=False,
                              pause_boundary="original_epoch_02_checkpoint_and_history",
                              epoch_checkpoint=str(checkpoint))
                dump(args.status, status)
                dump(args.report, report)
                print(json.dumps(report), flush=True)
                return
            if time.monotonic() - last_heartbeat >= 30:
                report.update(heartbeat_at=now())
                dump(args.report, report)
                last_heartbeat = time.monotonic()
            time.sleep(.05)
    except BaseException as error:
        report.update(status="watcher_failed", error=str(error), failed_at=now())
        dump(args.report, report)
        raise


if __name__ == "__main__":
    main()
