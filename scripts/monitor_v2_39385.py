"""Hourly SSH health collection and verified local delivery for the fixed V2 run.

Credentials and transport command are private configuration, never repository
content. This monitor never powers off, uploads to the competition or repeats a
partial training stage.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import time


def invoke(config, arguments, timeout=120):
    result = None
    for attempt in range(3):
        try:
            result = subprocess.run(config['transport'] + arguments, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return result.stdout
        except subprocess.TimeoutExpired:
            pass
        if attempt < 2:
            time.sleep(5)
    raise RuntimeError('Remote command unavailable after three attempts: ' +
                       (result.stderr[-1000:] if result is not None else 'timed out'))


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    output = Path(config['local_output'])
    output.mkdir(parents=True, exist_ok=True)
    command = 'cd ' + shlex.quote(config['remote_worktree']) + ' && ' + shlex.join([
        'env', 'PYTHONPATH=' + config['remote_worktree'] + '/reproducibility/aegis_f1',
        config['remote_python'], '-m', 'v2.health', '--plan', config['remote_plan']])
    try:
        report = json.loads(invoke(config, ['remote', command]))
        if report['health'] == 'completed_delivered':
            delivery = output.parent / 'submission'
            delivery.mkdir(exist_ok=True)
            for name in ('pred_results.csv', 'submission.zip', 'report.json'):
                invoke(config, ['fetch', str(Path(config['remote_plan']).parent / 'submission' / name),
                                str(delivery / name)], timeout=180)
            invoke(config, ['fetch', str(Path(config['remote_plan']).parent / 'submission_check.log'),
                            str(delivery / 'server_submission_check.log')])
            import hashlib
            artifact = json.loads((delivery / 'report.json').read_text())
            for name, field in [('submission.zip', 'zip_sha256'), ('pred_results.csv', 'csv_sha256')]:
                assert hashlib.sha256((delivery / name).read_bytes()).hexdigest() == artifact[field], 'Delivery digest mismatch'
            check = subprocess.run([config['local_python'], str(Path(__file__).resolve().parent / 'check_submission.py'),
                                    '--test_dir', config['local_test_root'], '--num-classes', '750',
                                    '--csv', str(delivery / 'pred_results.csv'), '--zip', str(delivery / 'submission.zip')],
                                   capture_output=True, text=True, check=True)
            (delivery / 'local_submission_check.log').write_text(check.stdout + check.stderr)
            report['local_verified_delivery'] = str(delivery)
            report['local_submission_check_passed'] = True
            subprocess.run(['systemctl', '--user', 'stop', config['timer_unit']], check=True)
    except Exception as error:
        report = dict(health='monitor_or_delivery_needs_diagnosis', error=str(error), keep_server_running=True)
    report['local_recorded_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    name = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'
    content = json.dumps(report, indent=2) + '\n'
    (output / name).write_text(content)
    temporary = output / 'latest.tmp'
    temporary.write_text(content)
    temporary.replace(output / 'latest.json')
    if report['health'] not in ('running', 'completed_delivered'):
        (output / 'ALERT.json').write_text(content)
    elif (output / 'ALERT.json').exists():
        (output / 'ALERT.json').unlink()
    print(content, flush=True)
    return 0 if report['health'] in ('running', 'completed_delivered') else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.config))
