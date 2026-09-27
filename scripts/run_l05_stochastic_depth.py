#!/usr/bin/env python3
"""Fixed L05 full sixteen-epoch stochastic-depth training from the original LP parent."""
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
from l05_stochastic_depth_support import StochasticDepth
from run_l05_mixstyle import verify_framework, digest, dump, StopLoss

ROOT = Path(__file__).resolve().parents[1]


def verify_control(cfg):
    assert digest(cfg['control_checkpoint']) == cfg['control_checkpoint_sha256']
    control = torch.load(cfg['control_checkpoint'], map_location='cpu', weights_only=False)
    assert control['epoch'] == 16
    assert control['config']['train']['init_checkpoint'] == cfg['parent_checkpoint']
    for item in cfg['control_curve'].values():
        assert digest(item['path']) == item['sha256']
    return control['config']


def runtime_config(cfg, out, smoke=False):
    original = verify_control(cfg)
    c = copy.deepcopy(original)
    c.pop('_config_path', None)
    name = 'DP01_SMOKE' if smoke else 'DP01'
    c['project'].update(experiment_id=name, trial_id='DP01', mechanism='whole_block_stochastic_depth_from_lp')
    c['project']['stochastic_depth'] = {'max_drop':cfg['max_drop'], 'seed':cfg['seed'], 'depth':12,
        'source_sha256':digest(ROOT/'scripts/l05_stochastic_depth_support.py')}
    c['output']['root'] = str(out/'runs')
    if smoke:
        c['train'].update(epochs=1,schedule_epochs=1,max_steps=4,grad_accum_steps=1,effective_batch_size=4,log_every_steps=1)
        c['loss']['attention_local_training']['start_epoch']=1
        c['evaluation']['interval_epochs']=1
    else:
        for section in ('model','data','train','loss','evaluation','trust','longtail'):
            assert c[section] == original[section], section
    path=out/'configs'/f'{name}.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    data=yaml.safe_dump(c,sort_keys=False)
    if path.exists():assert path.read_text()==data,'Runtime config drift'
    else:path.write_text(data)
    return path,out/'runs'/name/'seed42'


def train(cfg, out, smoke, resume):
    from aegis_clip.config import load_config
    from aegis_clip.rematch_protocol import validate_checkpoint
    from aegis_clip import trainer
    path, run = runtime_config(cfg, out, smoke)
    config = load_config(path)
    validate_checkpoint(cfg['parent_checkpoint'], config, parent=True)
    options = config['project']['stochastic_depth']
    mixer = StochasticDepth(options['max_drop'], options['seed'], options['depth'])
    assert config['longtail']['balanced_softmax_tau'] == 0
    assert config['loss']['mixup_probability'] == 0
    assert config['loss']['attention_local_training']['consistency_weight'] == 0
    if resume:
        sidecar = json.loads(Path(resume+'.stochastic_depth.json').read_text())
        assert sidecar['checkpoint_sha256'] == digest(resume)
        assert sidecar['options'] == options
        mixer.load_state_dict(sidecar['state'])
    original_save = trainer.save_checkpoint
    original_builder = trainer.build_model
    handles = []
    def builder(c, device):
        model, preprocess = original_builder(c, device)
        handles.extend(mixer.install(model))
        return model, preprocess
    def save(path, **kwargs):
        measured = kwargs['metrics']
        original_save(path, **kwargs)
        dump(str(path)+'.stochastic_depth.json', {'checkpoint_sha256': digest(path),
             'state': mixer.state_dict(), 'options': options})
        if Path(path).name == 'last.pt' and 'raw_macro' in measured and not smoke:
            reference=cfg['control_curve'][str(kwargs['epoch'])]
            if measured['raw_macro'] <= reference['macro']-.02 or measured['raw_micro'] <= reference['micro']-.02:
                raise StopLoss('Predeclared -2pp stop vs matched original L05 epoch')
    trainer.save_checkpoint = save
    trainer.build_model = builder
    stopped = False
    try:
        best = trainer.train(config, resume=resume)
    except StopLoss:
        best=run/'checkpoints/best.pt'
        if not best.exists():best=run/'checkpoints/last.pt'
        stopped=True
    finally:
        trainer.save_checkpoint = original_save
        trainer.build_model = original_builder
        for handle in handles:
            handle.remove()
    assert mixer.calls>0 and sum(mixer.dropped)>0 and all(mixer.block_calls==mixer.calls), 'Inactive or bypassed block hooks'
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
        'state': mixer.state_dict(), 'reload_center': reload_metrics})


def queue(cfg, out, framework, config_path):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(framework/'reproducibility/aegis_f1')
    for smoke in (True, False):
        name = 'DP01_SMOKE' if smoke else 'DP01'
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
    best, cache = Path(complete['checkpoint']), run/'val_branch_logits.pt'
    commands = [
        [str(framework/'scripts/cache_validation_tta_logits.py'), '--checkpoint', str(best), '--config', str(path),
         '--output', str(cache), '--tta-fusion', 'mean_probabilities', '--tta-temperature', '1.0',
         '--device', 'cuda:0', '--batch-size', '32', '--num-workers', '2'],
        [str(framework/'scripts/evaluate_l05_candidate.py'), '--checkpoint', str(best), '--config', str(path),
         '--val-branch-cache', str(cache), '--output', str(run/'evaluation.json')]]
    with (out/'DP01_evaluate.log').open('x') as log:
        for command in commands:
            subprocess.run([sys.executable, *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    candidate = json.loads((run/'evaluation.json').read_text())
    verify_control(cfg)
    baseline = {'macro': .7582651238982523, 'micro': .7665322580645161}
    def gate(ref):
        return candidate['decode']['macro'] >= ref['macro'] + .003 and candidate['decode']['micro'] >= ref['micro']
    dump(out/'result.json', {'candidate': candidate, 'historical_control_checkpoint':cfg['control_checkpoint'], 'baseline': baseline,
        'pass': not complete['stopped'] and gate(baseline),
        'training_complete': complete, 'recipe': {k:cfg[k] for k in ('max_drop','seed')},
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
        runtime_config(cfg, out)
        dump(out/'preflight.json', receipt)
    elif args.phase == 'train':
        train(cfg, out, args.smoke, args.resume)
    else:
        queue(cfg, out, framework, config_path)
