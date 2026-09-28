#!/usr/bin/env python3
"""Pinned P75 patch readout queue; GPU phases require an explicit flag."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FOCUS = Path('/home/lux1/noise/worktrees/rematch750_f05_focus')
OUT = ROOT/'outputs/codex/p75_mechanisms_20260927'
FRAMEWORK = OUT/'framework_patch'
CONFIG = OUT/'configs/P75_PATCH_READOUT.yaml'
RUN = OUT/'runs/P75_PATCH_READOUT/seed42'
COMMIT = 'f050ecb59e0a885da0b43cdc6c8c316953fb5e7d'
PINNED = (
    'reproducibility/aegis_f1/aegis_clip/model.py',
    'reproducibility/aegis_f1/aegis_clip/trainer.py',
    'reproducibility/aegis_f1/aegis_clip/checkpoint.py',
    'reproducibility/aegis_f1/aegis_clip/local_inference.py',
    'reproducibility/aegis_f1/aegis_clip/cli/train.py',
    'scripts/cache_validation_tta_logits.py',
    'scripts/evaluate_l05_candidate.py',
    'scripts/build_l05_tta_prior_submission_final.py',
)


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT/'scripts/p75_runtime'),
        str(FRAMEWORK/'reproducibility/aegis_f1'), env.get('PYTHONPATH', '')))
    return env


def preflight() -> None:
    from p75_prepare import preflight as stage_preflight, sha
    from p75_framework import materialize
    import yaml
    identity = stage_preflight()
    if materialize('patch') != FRAMEWORK:
        raise RuntimeError('Private patch framework path changed')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=FOCUS, text=True).strip() != COMMIT:
        raise RuntimeError('Focus framework HEAD changed')
    for path in PINNED:
        expected = subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'], cwd=FOCUS)
        if (FOCUS/path).read_bytes() != expected:
            raise RuntimeError(f'Pinned focus code changed: {path}')
    saved = json.loads((OUT/'preparation.json').read_text())
    if saved['identity'] != identity:
        raise RuntimeError('P75 preparation identity changed')
    implementation = yaml.safe_load(CONFIG.read_text())['project']['p75_implementation_files']
    if any(sha(ROOT/name) != expected for name, expected in implementation.items()):
        raise RuntimeError('P75 patch implementation changed after config freeze')
    script = ('from aegis_clip.config import load_config;'
              'from aegis_clip.rematch_protocol import validate_checkpoint;'
              f'c=load_config({str(CONFIG)!r});'
              'assert c["model"]["patch_readout"]=={"enabled":True,"version":1};'
              'validate_checkpoint(c["train"]["init_checkpoint"],c,parent=True)')
    subprocess.run([sys.executable, '-c', script], cwd=FRAMEWORK,
                   env=environment(), check=True)
    print('Pinned source, stage assets, config and parent binding verified')


def execute(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x') as stream:
        subprocess.run([sys.executable, *command], cwd=FRAMEWORK,
                       env=environment(), stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'train', 'cache', 'evaluate', 'deliver'),
                        required=True)
    parser.add_argument('--execute-gpu', action='store_true')
    parser.add_argument('--resume')
    args = parser.parse_args()
    preflight()
    if args.phase == 'preflight':
        return
    if not args.execute_gpu:
        raise ValueError('GPU phases require --execute-gpu; preparation does not launch them')
    checkpoint = RUN/'checkpoints/best.pt'
    cache = RUN/'val_branch_logits.pt'
    if args.phase == 'train':
        command = ['-m', 'aegis_clip.cli.train', '--config', str(CONFIG)]
        if args.resume:
            command.extend(['--resume', args.resume])
        execute(command, RUN/'p75_train.log')
    elif args.phase == 'cache':
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        execute(['scripts/cache_validation_tta_logits.py', '--checkpoint', str(checkpoint),
                 '--config', str(CONFIG), '--output', str(cache), '--device', 'cuda:0',
                 '--batch-size', '32', '--num-workers', '2'], RUN/'p75_cache.log')
    elif args.phase == 'evaluate':
        if not checkpoint.exists() or not cache.exists():
            raise FileNotFoundError('Candidate checkpoint or validation cache is missing')
        execute(['scripts/evaluate_l05_candidate.py', '--checkpoint', str(checkpoint),
                 '--config', str(CONFIG), '--val-branch-cache', str(cache),
                 '--temperature', '1.4', '--prior-strength', '0.6',
                 '--fusion', 'mean_probabilities', '--output', str(RUN/'evaluation.json')],
                RUN/'p75_evaluate.log')
        execute([str(ROOT/'scripts/p75_compare.py'), '--candidate-cache', str(cache),
                 '--candidate-evaluation', str(RUN/'evaluation.json'),
                 '--output', str(RUN/'paired_validation.json')],
                RUN/'p75_paired.log')
    else:
        result = json.loads((RUN/'evaluation.json').read_text())
        if result['decode']['macro'] < .7582651238982523 + .003 or \
           result['decode']['micro'] < .7665322580645161:
            raise RuntimeError('Candidate has not passed the fixed local delivery gate')
        execute(['scripts/build_l05_tta_prior_submission_final.py',
                 '--checkpoint', str(checkpoint), '--config', str(CONFIG),
                 '--val-branch-cache', str(cache), '--temperature', '1.4',
                 '--prior-strength', '0.6', '--fusion', 'mean_probabilities',
                 '--tta', 'horizontal_flip', '--tag', 'P75_PATCH_READOUT',
                 '--output-root', str(OUT/'deliveries'), '--skip-desktop-copy',
                 '--device', 'cuda:0', '--batch-size', '32', '--num-workers', '2'],
                RUN/'p75_deliver.log')


if __name__ == '__main__':
    main()
