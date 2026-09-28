#!/usr/bin/env python3
"""Explicit phases; default CPU preflight never launches P75 GPU work."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from p75_supported_ce import (CONFIG, FIXED, FRAMEWORK, OUT, ROOT, RUN, environment,
                              preflight, prepare, read_json, read_rows, select, sha, verify_files)

GPU_PHASES = frozenset(('snapshot', 'train', 'cache', 'deliver'))


def ensure_idle_device():
    """Check the fixed L05 CUDA device at launch; never reserve or stop jobs."""
    original = read_json(OUT/'configs/l05_original.json')
    device = original['train']['device']
    if not device.startswith('cuda:'):
        raise ValueError('This pinned L05 runner expects its original CUDA device')
    gpu_index = device.split(':', 1)[1]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        devices = visible.split(',')
        if int(gpu_index) >= len(devices) or not devices[int(gpu_index)].strip():
            raise ValueError('Original L05 device is not visible')
        gpu_index = devices[int(gpu_index)].strip()
    uuid = subprocess.check_output(['nvidia-smi', '-i', gpu_index, '--query-gpu=uuid',
        '--format=csv,noheader'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    if any(line.split(',')[0].strip() == uuid for line in processes.splitlines()):
        raise RuntimeError(f'{device} is busy; wait for the existing task to release it')


def phase_log(name):
    return RUN.parent/f'p75_supported_ce_{name}.log'


def execute(command, phase):
    log = phase_log(phase)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x') as stream:
        subprocess.run([sys.executable, *command], cwd=FRAMEWORK, env=environment(),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(log)


def verify_delivery_gate():
    report_path = RUN/'paired_validation.json'
    result = read_json(report_path)
    binding = result['binding']
    checks = {'checkpoint_sha256': RUN/'checkpoints/best.pt', 'config_sha256': CONFIG,
              'candidate_cache_sha256': RUN/'val_branch_logits.pt',
              'evaluation_sha256': RUN/'evaluation.json',
              'reference_cache_sha256': read_json(FIXED)['reference_validation_cache'],
              'selection_manifest_sha256': OUT/'selection/manifest.json'}
    if any(sha(path) != binding[key] for key, path in checks.items()):
        raise ValueError('Candidate changed after its fixed-decode paired report')
    verify_files('/', result['report_files'])
    from p75_supported_ce_report import grouped_report
    rows = read_rows(RUN/'paired_validation_rows.csv')
    classes = result['by_class']
    support_labels = [row['class_id'] for row in classes
                      for _ in range(row['restored_training_samples'])]
    recomputed = grouped_report([int(row['label']) for row in rows],
        [int(row['baseline_prediction']) for row in rows],
        [int(row['candidate_prediction']) for row in rows], support_labels, len(classes))
    if recomputed['baseline'] != result['baseline'] or recomputed['candidate'] != result['candidate']:
        raise ValueError('Paired report metrics changed')
    if not recomputed['package_gate_passed']:
        raise RuntimeError('Candidate failed macro +0.30pp / micro non-decreasing package gate')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'preflight', 'snapshot', 'select',
                                           'train', 'cache', 'evaluate', 'deliver'), default='preflight')
    parser.add_argument('--execute-gpu', action='store_true')
    parser.add_argument('--require-support', action='store_true',
                        help='CPU preflight also verifies the completed support binding')
    args = parser.parse_args(argv)
    # Guard before any preparation, model construction, or subprocess dispatch.
    if args.phase in GPU_PHASES and not args.execute_gpu:
        raise ValueError('GPU phases require explicit --execute-gpu; no automatic launch')
    if args.phase == 'prepare':
        print(prepare()['status'])
        return
    if args.phase == 'preflight':
        print(preflight(require_support=args.require_support)['status'])
        return
    if args.phase == 'select':
        result = select()
        print({key: result[key] for key in ('status', 'weak_eligible_count', 'support_count')})
        return
    preflight(require_support=args.phase != 'snapshot')
    if args.phase in GPU_PHASES:
        ensure_idle_device()
    checkpoint, cache = RUN/'checkpoints/best.pt', RUN/'val_branch_logits.pt'
    if args.phase == 'snapshot':
        execute([str(ROOT/'scripts/p75_supported_ce_snapshot.py'), '--execute-gpu'], 'snapshot')
    elif args.phase == 'train':
        # Fresh full training only. No resume option for L05/SAM checkpoints.
        execute(['-m', 'aegis_clip.cli.train', '--config', str(CONFIG)], 'train')
    elif args.phase == 'cache':
        if not checkpoint.exists() or cache.exists():
            raise ValueError('Missing candidate checkpoint or validation cache already exists')
        execute(['scripts/cache_validation_tta_logits.py', '--checkpoint', str(checkpoint),
            '--config', str(CONFIG), '--output', str(cache), '--device', 'cuda:0',
            '--batch-size', '32', '--num-workers', '2'], 'cache')
    elif args.phase == 'evaluate':
        if not checkpoint.exists() or not cache.exists():
            raise FileNotFoundError('Candidate checkpoint/validation cache is missing')
        execute(['scripts/evaluate_l05_candidate.py', '--checkpoint', str(checkpoint),
            '--config', str(CONFIG), '--val-branch-cache', str(cache), '--temperature', '1.4',
            '--prior-strength', '0.6', '--fusion', 'mean_probabilities',
            '--output', str(RUN/'evaluation.json')], 'evaluate')
        execute([str(ROOT/'scripts/p75_supported_ce_report.py'), '--candidate-cache', str(cache),
            '--evaluation', str(RUN/'evaluation.json'), '--output', str(RUN/'paired_validation.json')],
            'paired')
    else:
        verify_delivery_gate()
        execute(['scripts/build_l05_tta_prior_submission_final.py', '--checkpoint', str(checkpoint),
            '--config', str(CONFIG), '--val-branch-cache', str(cache), '--temperature', '1.4',
            '--prior-strength', '0.6', '--fusion', 'mean_probabilities', '--tta', 'horizontal_flip',
            '--tag', 'P75_SUPPORTED_CE', '--output-root', str(OUT/'deliveries'), '--skip-desktop-copy',
            '--device', 'cuda:0', '--batch-size', '32', '--num-workers', '2'], 'deliver')


if __name__ == '__main__':
    main()
