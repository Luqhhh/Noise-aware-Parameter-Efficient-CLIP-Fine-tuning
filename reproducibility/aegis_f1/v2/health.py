"""Read real controller/log/GPU progress; keep an hourly record on the server."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .plan import dump, json_read


def process_matches(pid, plan_path):
    try:
        args = (Path('/proc') / str(int(pid)) / 'cmdline').read_bytes().split(b'\0')
        return b'v2.controller' in args and str(Path(plan_path).resolve()).encode() in args
    except (OSError, ValueError, TypeError):
        return False


def log_progress(path):
    if not path.exists():
        return None, []
    with path.open('rb') as stream:
        stream.seek(max(path.stat().st_size - 65536, 0))
        lines = stream.read().decode(errors='replace').splitlines()
    progress = None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and 'updates' in value and 'loss' in value:
            progress = value
            break
    return progress, lines[-12:]


def classify(state, alive, age, free_bytes):
    if state.get('status') == 'completed_delivered':
        return 'completed_delivered'
    if state.get('status') == 'failed_needs_diagnosis' or not alive:
        return 'needs_diagnosis'
    if free_bytes < 2 * 1024**3:
        return 'low_disk_needs_diagnosis'
    if age is not None and age > 3600:
        return 'no_recent_log_needs_diagnosis'
    return 'running'


def snapshot(plan_path):
    plan_path = Path(plan_path).resolve()
    workspace = plan_path.parent
    state_path = workspace / 'controller.json'
    state = json_read(state_path) if state_path.exists() else {}
    alive = process_matches(state.get('pid'), plan_path)
    log = workspace / 'controller.log'
    progress, tail = log_progress(log)
    stage = state.get('stage')
    if progress and progress.get('stage') != stage:
        progress = None
    history_path = workspace / 'runs' / str(stage) / 'history.json'
    history = json_read(history_path) if history_path.exists() else []
    usage = shutil.disk_usage(workspace)
    age = time.time() - log.stat().st_mtime if log.exists() else None
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    health = classify(state, alive, age, usage.free)
    if health == 'running' and (gpu.returncode != 0 or apps.returncode != 0):
        health = 'gpu_query_needs_diagnosis'
    return dict(recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                health=health, controller=state,
                controller_alive=alive, real_training_progress=progress,
                completed_stage_epochs=len(history), last_epoch_metrics=history[-1] if history else None,
                log_age_seconds=age, disk_free_bytes=usage.free,
                gpu_query_exit_code=gpu.returncode, gpu=gpu.stdout.strip(),
                compute_processes=apps.stdout.strip(), log_tail=tail, keep_server_running=True)


def record(plan_path):
    report = snapshot(plan_path)
    output = Path(plan_path).resolve().parent / 'monitoring'
    output.mkdir(exist_ok=True)
    name = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'
    dump(output / name, report)
    dump(output / 'latest.tmp', report)
    (output / 'latest.tmp').replace(output / 'latest.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=3600)
    args = parser.parse_args()
    if args.watch:
        import fcntl
        lock = (Path(args.plan).resolve().parent / 'health.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dump(Path(args.plan).resolve().parent / 'health_watcher.json',
             dict(pid=os.getpid(), interval_seconds=args.interval, keep_server_running=True))
    while True:
        report = record(args.plan)
        print(json.dumps(report), flush=True)
        if not args.watch:
            return
        if report['health'] == 'completed_delivered':
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
