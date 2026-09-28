#!/usr/bin/env python3
"""SAM train/evaluate pipeline with hourly health observations and no restarts."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from run_p75_full_sam import OUT, ROOT, RUN, phase_log


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    smoke = json.loads((OUT/'sam_gpu_smoke.json').read_text())
    assert smoke['status'] == 'passed' and len(smoke['updates']) == 2
    status_path = OUT/'campaign_status.json'
    if status_path.exists():
        raise FileExistsError(status_path)
    done = threading.Event()
    lock = threading.Lock()
    state = {'status': 'starting', 'supervisor_pid': os.getpid(),
             'started_at': now(), 'monitor_interval_seconds': 3600,
             'next_health_check_unix': time.time()+3600}

    def update(**values):
        with lock:
            state.update(values, updated_at=now())
            temporary = status_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2)+'\n')
            temporary.replace(status_path)

    def monitor():
        while not done.wait(3600):
            with lock:
                snapshot = dict(state)
            try:
                health = {'observed_at': now(), 'state': snapshot}
                health['processes'] = subprocess.run(
                    ['ps', '-eo', 'pid,ppid,comm,etime'], capture_output=True,
                    text=True, timeout=20).stdout
                health['gpu'] = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu',
                     '--format=csv,noheader'], capture_output=True, text=True,
                    timeout=20).stdout
                log = phase_log(RUN/'p75_train.log')
                if log.exists():
                    with log.open('rb') as stream:
                        stream.seek(max(0, log.stat().st_size-6000))
                        health['training_log_tail'] = stream.read().decode(errors='replace')
                with (OUT/'hourly_health.jsonl').open('a') as stream:
                    stream.write(json.dumps(health)+'\n')
                update(last_health_check_at=health['observed_at'],
                       next_health_check_unix=time.time()+3600)
            except Exception as error:
                update(monitor_error=repr(error), next_health_check_unix=time.time()+3600)

    def phase(name):
        command = [sys.executable, str(ROOT/'scripts/run_p75_full_sam.py'),
                   '--phase', name, '--execute-gpu']
        child = subprocess.Popen(command, cwd=ROOT)
        update(status='running', phase=name, child_pid=child.pid, command=command)
        # Wait on the child handle; do not poll logs or process state.
        result = child.wait()
        if result:
            raise RuntimeError(f'{name} exited {result}')

    update()
    threading.Thread(target=monitor, daemon=True).start()
    try:
        for name in ('train', 'cache', 'evaluate'):
            phase(name)
        evaluation = json.loads((RUN/'evaluation.json').read_text())
        promoted = (evaluation['decode']['macro'] >= .7582651238982523+.003
                    and evaluation['decode']['micro'] >= .7665322580645161)
        if promoted:
            phase('deliver')
        update(status='completed_pending_audit', promoted=promoted,
               evaluation=str(RUN/'evaluation.json'), child_pid=None,
               next_health_check_unix=None)
    except BaseException as error:
        update(status='failed', error=repr(error), child_pid=None,
               next_health_check_unix=None)
        raise
    finally:
        done.set()


if __name__ == '__main__':
    main()
