#!/usr/bin/env python3
"""Fixed L05 logit-penalty continuation, reusing the audited MS00 control."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import yaml
from l05_logit_penalty_support import LogitPenalty
from run_l05_mixstyle import verify_framework, digest, dump, StopLoss

ROOT = Path(__file__).resolve().parents[1]


def verify_control(cfg):
    assert digest(cfg['control_config']) == cfg['control_config_sha256']
    result = json.loads(Path(cfg['control_result']).read_text())
    assert digest(result['checkpoint']) == cfg['control_checkpoint_sha256'] == result['checkpoint_sha256']
    audit = json.loads(Path(cfg['control_result']).with_name('decode_audit.json').read_text())
    assert audit['evaluation_sha256'] == digest(cfg['control_result'])
    assert audit['identical_decoded_predictions'] == 14880
    return result


def calibrate(cfg, out):
    from run_l05_nonlinear_head import preflight
    from run_l05_covariance_head import load_shards
    torch.set_num_threads(2)
    source = json.loads(Path(cfg['source_feature_config']).read_text())
    rows, _, identity = preflight(source)
    directory = Path(cfg['source_features'])
    assert json.loads((directory/'complete.json').read_text()) == identity
    train, shards = load_shards(directory, 'train', rows['train'], identity)
    parent = torch.load(cfg['parent_checkpoint'], map_location='cpu', weights_only=False)
    w = parent['model_state_dict']['classifier.weight']
    b = parent['model_state_dict']['classifier.bias']
    gce_sum = squared_sum = 0.
    with torch.no_grad():
        for offset in range(0, len(train['labels']), 2048):
            x = torch.nn.functional.normalize(train['center'][offset:offset+2048], dim=-1)
            labels = train['labels'][offset:offset+len(x)]
            logits = torch.nn.functional.linear(x, w, b).double()
            target = logits.softmax(1).gather(1, labels[:, None]).squeeze(1)
            gce_sum += float(((1-target.clamp_min(1e-7).sqrt())/.5).sum())
            squared_sum += float(logits.square().mean(1).sum())
    coefficient = 2 * cfg['initial_penalty_to_gce_ratio'] * gce_sum / squared_sum
    assert 0 < coefficient < 1 and cfg['initial_penalty_to_gce_ratio'] == .1
    record = {'coefficient': coefficient, 'training_samples': len(train['labels']),
              'mean_gce': gce_sum/len(train['labels']),
              'mean_squared_logit': squared_sum/len(train['labels']),
              'initial_penalty_to_gce_ratio': .1, 'source_identity': identity,
              'feature_shards': shards, 'validation_used_for_coefficient': False,
              'test_used': False}
    target = out/'coefficient.json'
    if target.exists():
        assert json.loads(target.read_text()) == record
    else:
        dump(target, record)
    return record


def runtime_config(cfg, out, smoke=False):
    original = yaml.safe_load(Path(cfg['control_config']).read_text())
    c = copy.deepcopy(original)
    name = 'SD01_SMOKE' if smoke else 'SD01'
    coefficient = json.loads((out/'coefficient.json').read_text())['coefficient']
    c['project'].update(experiment_id=name, trial_id='SD01', mechanism='gce_plus_logit_l2')
    c['project'].pop('mixstyle', None)
    c['project']['logit_penalty'] = {'coefficient': coefficient,
        'coefficient_file_sha256': digest(out/'coefficient.json'),
        'source_sha256': digest(ROOT/'scripts/l05_logit_penalty_support.py')}
    c['output']['root'] = str(out/'runs')
    if smoke:
        c['train'].update(epochs=1, schedule_epochs=1, max_steps=2, grad_accum_steps=1, effective_batch_size=4)
        c['evaluation'].update(evaluate_initial_checkpoint=False, record_initial=False)
    else:
        for section in ('model', 'data', 'train', 'loss', 'evaluation'):
            assert c[section] == original[section], section
    path = out/'configs'/f'{name}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_dump(c, sort_keys=False)
    if path.exists():
        assert path.read_text() == data, 'Runtime config drift'
    else:
        path.write_text(data)
    return path, out/'runs'/name/'seed42'


def train(cfg, out, smoke, resume):
    from aegis_clip.config import load_config
    from aegis_clip.rematch_protocol import validate_checkpoint
    from aegis_clip import trainer
    path, run = runtime_config(cfg, out, smoke)
    config = load_config(path)
    validate_checkpoint(cfg['parent_checkpoint'], config, parent=True)
    options = config['project']['logit_penalty']
    penalty = LogitPenalty(options['coefficient'])
    if resume:
        sidecar = json.loads(Path(resume+'.logit_penalty.json').read_text())
        assert sidecar['checkpoint_sha256'] == digest(resume)
        assert sidecar['options'] == options
        penalty.load_state_dict(sidecar['state'])
    original_loss, original_save = trainer._per_sample_loss, trainer.save_checkpoint
    def loss(logits, *args, **kwargs):
        return penalty.apply(original_loss(logits, *args, **kwargs), logits)
    def save(path, **kwargs):
        measured = kwargs['metrics']
        if kwargs['epoch'] == 0 and not smoke:
            assert abs(measured['raw_macro']-.7524971006956281) < 1e-6
            assert abs(measured['raw_micro']-.7629704301075269) < 1e-6
        original_save(path, **kwargs)
        dump(str(path)+'.logit_penalty.json', {'checkpoint_sha256': digest(path),
             'state': penalty.state_dict(), 'options': options})
        if Path(path).name == 'last.pt' and kwargs['epoch'] > 0 and not smoke:
            if measured['raw_macro'] < .7324971006956281 or measured['raw_micro'] < .7429704301075269:
                raise StopLoss('Predeclared -2pp stop loss')
    trainer._per_sample_loss, trainer.save_checkpoint = loss, save
    stopped = False
    try:
        best = trainer.train(config, resume=resume)
    except StopLoss:
        best, stopped = run/'checkpoints/best.pt', True
    finally:
        trainer._per_sample_loss, trainer.save_checkpoint = original_loss, original_save
    epoch0 = run/'checkpoints/epoch0.pt'
    if epoch0.exists() and digest(best) == digest(epoch0):
        for source, dest in ((epoch0.with_suffix('.binding.json'), Path(best).with_suffix('.binding.json')),
                             (Path(str(epoch0)+'.logit_penalty.json'), Path(str(best)+'.logit_penalty.json'))):
            if dest.exists():
                assert dest.read_bytes() == source.read_bytes()
            else:
                shutil.copy2(source, dest)
    assert penalty.calls > 0 and penalty.penalty_sum > 0, 'Penalty never active'
    validate_checkpoint(best, config, parent=False)
    # A stop-loss interrupts the trainer before its final reload evaluation;
    # the queue's separate native image/cache evaluation still runs afterwards.
    reload_metrics = None
    if not stopped:
        measured = json.loads((run/'checkpoints/best_evaluation.json').read_text())
        assert measured['samples'] == 14880
        reload_metrics = {k: measured[k] for k in ('raw_macro', 'raw_micro', 'samples')}
    dump(run/'training_complete.json', {'checkpoint': str(best), 'checkpoint_sha256': digest(best),
        'config_sha256': digest(path), 'smoke': smoke, 'stopped': stopped,
        'state': penalty.state_dict(), 'reload_center': reload_metrics})


def queue(cfg, out, framework, config_path):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(framework/'reproducibility/aegis_f1')
    for smoke in (True, False):
        name = 'SD01_SMOKE' if smoke else 'SD01'
        dump(out/'queue_state.json', {'phase': 'smoke' if smoke else 'training', 'driver_pid': os.getpid()})
        command = [sys.executable, __file__, '--config', str(config_path), '--phase', 'train']
        if smoke:
            command.append('--smoke')
        with (out/f'{name}_train.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        path, run = runtime_config(cfg, out, smoke)
        complete = json.loads((run/'training_complete.json').read_text())
        assert complete['smoke'] == smoke and complete['state']['calls'] > 0
    dump(out/'queue_state.json', {'phase': 'evaluation', 'driver_pid': os.getpid()})
    best, cache = run/'checkpoints/best.pt', run/'val_branch_logits.pt'
    commands = [
        [str(framework/'scripts/cache_validation_tta_logits.py'), '--checkpoint', str(best), '--config', str(path),
         '--output', str(cache), '--tta-fusion', 'mean_probabilities', '--tta-temperature', '1.0',
         '--device', 'cuda:0', '--batch-size', '32', '--num-workers', '2'],
        [str(framework/'scripts/evaluate_l05_candidate.py'), '--checkpoint', str(best), '--config', str(path),
         '--val-branch-cache', str(cache), '--output', str(run/'evaluation.json')]]
    with (out/'SD01_evaluate.log').open('x') as log:
        for command in commands:
            subprocess.run([sys.executable, *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    candidate = json.loads((run/'evaluation.json').read_text())
    control = verify_control(cfg)
    baseline = {'macro': .7582651238982523, 'micro': .7665322580645161}
    def gate(ref):
        return candidate['decode']['macro'] >= ref['macro'] + .003 and candidate['decode']['micro'] >= ref['micro']
    dump(out/'result.json', {'candidate': candidate, 'control': control, 'baseline': baseline,
        'pass': not complete['stopped'] and gate(baseline) and gate(control['decode']),
        'training_complete': complete, 'coefficient': json.loads((out/'coefficient.json').read_text()),
        'test_used': False})
    dump(out/'queue_state.json', {'phase': 'complete', 'driver_pid': os.getpid()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--phase', choices=('prepare', 'train', 'queue'), required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume')
    args = p.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text())
    out = (ROOT/cfg['output']).resolve()
    framework = out/'framework'
    receipt = verify_framework(cfg, framework)
    verify_control(cfg)
    sys.path.insert(0, str(framework/'reproducibility/aegis_f1'))
    if args.phase == 'prepare':
        calibrate(cfg, out)
        runtime_config(cfg, out)
        dump(out/'preflight.json', receipt)
    elif args.phase == 'train':
        train(cfg, out, args.smoke, args.resume)
    else:
        queue(cfg, out, framework, config_path)
