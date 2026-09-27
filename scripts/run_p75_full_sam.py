#!/usr/bin/env python3
"""Full L05 SAM queue using a private patched trainer; explicit GPU phases."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from p75_framework import materialize
from p75_prepare import preflight as stage_preflight, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/codex/p75_mechanisms_20260927'
FRAMEWORK = OUT/'framework_sam'
CONFIG = OUT/'configs/P75_FULL_SAM.yaml'
RUN = OUT/'runs/P75_FULL_SAM/seed42'


def environment():
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT/'scripts/p75_runtime'),
        str(FRAMEWORK/'reproducibility/aegis_f1'), env.get('PYTHONPATH', '')))
    return env


def preflight():
    import yaml
    identity = stage_preflight()
    if materialize('sam') != FRAMEWORK:
        raise RuntimeError('Private SAM framework path changed')
    saved = json.loads((OUT/'preparation.json').read_text())
    if saved['identity'] != identity:
        raise RuntimeError('P75 preparation identity changed')
    implementation = yaml.safe_load(CONFIG.read_text())['project']['p75_implementation_files']
    if any(sha(ROOT/name) != expected for name, expected in implementation.items()):
        raise RuntimeError('P75 SAM implementation changed after config freeze')
    script = ('from aegis_clip.config import load_config;'
              'from aegis_clip.rematch_protocol import validate_checkpoint;'
              f'c=load_config({str(CONFIG)!r});'
              'assert c["train"]["sam"]=={"enabled":True,"rho":0.05,"mode":"standard_global_l2"};'
              'assert c["train"]["amp"] is False;'
              'assert c["loss"]["attention_local_training"]["enabled"] is True;'
              'assert c["loss"]["attention_local_training"]["consistency_weight"]==0;'
              'assert c["loss"]["mixup_probability"]==0;'
              'validate_checkpoint(c["train"]["init_checkpoint"],c,parent=True)')
    subprocess.run([sys.executable, '-c', script], cwd=FRAMEWORK,
                   env=environment(), check=True)
    print('Pinned full-loss SAM runtime, stage, config and parent verified')


def execute(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x') as stream:
        subprocess.run([sys.executable, *command], cwd=FRAMEWORK, env=environment(),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
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
                 '--tta', 'horizontal_flip', '--tag', 'P75_FULL_SAM',
                 '--output-root', str(OUT/'deliveries'), '--skip-desktop-copy',
                 '--device', 'cuda:0', '--batch-size', '32', '--num-workers', '2'],
                RUN/'p75_deliver.log')


if __name__ == '__main__':
    main()
