"""Explicit V2 continuation or full 768 route, bounded cost checks and final package."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .plan import ROOT, STAGES, cost_estimate, dump, json_read, require, sha, verify_prepared
from .runtime import infer, read_checkpoint, train


def auth_from_cost(plan_path, stage, cost):
    plan = verify_prepared(plan_path)
    require(cost['plan_sha256'] == sha(plan_path), 'Cost measurement belongs to another plan')
    full = stage == 'full_576'
    validation = 0.0 if full else cost['measured_validation_seconds']
    overhead = cost['measured_overhead_seconds'] * (1.25 if full else 1.0)
    estimate = cost_estimate(plan['stages'][stage], cost['measured_seconds_per_update'], validation, overhead)
    return dict(authorized=True, operation='train', stage=stage, plan_sha256=sha(plan_path),
                measured_seconds_per_update=cost['measured_seconds_per_update'],
                measured_validation_seconds=validation, measured_overhead_seconds=overhead,
                estimated_seconds=estimate, max_seconds=math.ceil(estimate * 1.4),
                cost_source='s3_576 measured same-resolution path, without holdout' if full else stage + ' real CUDA probe',
                user_instruction='Provided S1 bundle and explicitly requested remaining V2 continuation and hourly health monitoring; keep server running.')


def run(plan_path, from_official=False):
    plan_path = Path(plan_path).resolve()
    workspace = plan_path.parent
    lock = (workspace / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = verify_prepared(plan_path)
    if from_official:
        require(plan['recipe'].get('feature_dim') == 768 and not plan.get('imported_stages')
                and plan['recipe']['initial_parent'] == 'official_openai',
                'Full 768 route starts at official initialization without imported task weights')
        stages = STAGES
    else:
        require('s1_384' in plan.get('imported_stages', {}), 'This controller requires the validated completed S1 import')
        read_checkpoint(workspace / 'runs/s1_384/best.pt', plan, 's1_384')
        stages = STAGES[1:]
    state = dict(pid=os.getpid(), plan=str(plan_path), plan_sha256=sha(plan_path), keep_server_running=not from_official,
                 from_official=from_official, architecture=plan.get('architecture'),
                 started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 training_completed=False, platform_metrics=None)

    def status(operation, stage=None, **extra):
        state.update(status=operation, stage=stage,
                     updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
        temporary = workspace / 'controller.json.tmp'
        dump(temporary, state)
        temporary.replace(workspace / 'controller.json')
        print(json.dumps(state), flush=True)

    def wait_gpu(stage):
        while True:
            query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
            foreign = [int(v.strip()) for v in query.splitlines() if v.strip().isdigit() and int(v.strip()) != os.getpid()]
            if not foreign:
                return
            status('waiting_for_existing_gpu_task', stage, foreign_gpu_pids=foreign)
            time.sleep(60)

    try:
        for stage in stages:
            destination = workspace / 'runs' / stage
            if destination.exists():
                previous = json_read(destination / 'status.json') if (destination / 'status.json').exists() else {}
                require(previous.get('status') == 'complete', 'Existing partial stage needs diagnosis; no automatic repeat')
                checkpoint = destination / ('last.pt' if stage == 'full_576' else 'best.pt')
                read_checkpoint(checkpoint, plan, stage)
                continue
            wait_gpu(stage)
            cost_stage = 's3_576' if stage == 'full_576' else stage
            cost_path = workspace / 'probes' / cost_stage / 'cost.json'
            if not cost_path.exists():
                require(stage != 'full_576', 'Final stage uses the measured S3 resolution; no final-data probe')
                require(not cost_path.parent.exists(), 'A failed cost check needs diagnosis; do not repeat it implicitly')
                probe_auth = workspace / 'authorization' / (stage + '.probe.json')
                dump(probe_auth, dict(authorized=True, operation='probe', stage=stage, plan_sha256=sha(plan_path),
                                     estimated_seconds=900, max_seconds=1200,
                                     note='User-authorized bounded continuation cost check: eight logical updates and full raw/EMA holdout.'))
                status('cost_probe', stage, probe_steps=8)
                train(plan_path, probe_auth, stage, probe_steps=8)
            cost = json_read(cost_path)
            require(cost['status'] == 'cost_probe_only' and cost['no_candidate'] is True, 'Invalid cost result')
            authorization = auth_from_cost(plan_path, stage, cost)
            if from_official:
                authorization['user_instruction'] = 'Explicit local full V2 768 execution via --from-official; start from official CLIP.'
            auth_path = workspace / 'authorization' / (stage + '.train.json')
            dump(auth_path, authorization)
            status('training', stage, estimated_stage_seconds=authorization['estimated_seconds'],
                   max_stage_seconds=authorization['max_seconds'], measured_peak_bytes=cost['peak_allocated_bytes'])
            train(plan_path, auth_path, stage)
            read_checkpoint(destination / ('last.pt' if stage == 'full_576' else 'best.pt'), plan, stage)
            status('stage_complete', stage)
        output = workspace / 'submission'
        if not output.exists():
            cost = json_read(workspace / 'probes/s3_576/cost.json')
            estimate = max(600.0, cost['measured_validation_seconds'] * 2 * plan['recipe']['test_rows']
                           / plan['split']['val_rows'] * 2 + cost['measured_overhead_seconds'])
            auth_path = workspace / 'authorization/infer.json'
            dump(auth_path, dict(authorized=True, operation='infer', stage=None, plan_sha256=sha(plan_path),
                                 estimated_seconds=estimate, max_seconds=math.ceil(estimate * 1.4),
                                 cost_source='Same 576 resolution measured raw/EMA holdout, scaled to fixed four-view test inference with 2x margin',
                                 note='Original final raw-last checkpoint and original four-view policy; no platform upload.'))
            wait_gpu('inference')
            status('inference', 'full_576', training_completed=True)
            check_log = workspace / 'submission_check.log'
            # runtime.infer executes the official checker; also retain a direct original record.
            infer(plan_path, auth_path, output)
            with check_log.open('w') as stream:
                subprocess.run([sys.executable, str(ROOT / 'scripts/check_submission.py'),
                                '--test_dir', plan['recipe']['test_root'], '--num-classes', '750',
                                '--csv', str(output / 'pred_results.csv'), '--zip', str(output / 'submission.zip')],
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        report = json_read(output / 'report.json')
        require(report['status'] == 'package_ready' and report['rows'] == 37444 and report['weights'] == 'raw', 'Final package is not ready')
        require(sha(output / 'submission.zip') == report['zip_sha256'] and
                sha(output / 'pred_results.csv') == report['csv_sha256'], 'Final package bytes changed')
        status('completed_delivered', 'full_576', training_completed=True, submission=str(output),
               zip_sha256=report['zip_sha256'], csv_sha256=report['csv_sha256'], rows=report['rows'])
    except Exception as error:
        status('failed_needs_diagnosis', state.get('stage'), error=str(error), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--from-official', action='store_true', help='Explicitly execute all four stages of the fresh 768 route')
    args = parser.parse_args()
    run(args.plan, args.from_official)


if __name__ == '__main__':
    main()
