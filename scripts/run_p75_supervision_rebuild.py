#!/usr/bin/env python3
"""Explicit P0/R1/R2 phases. Prepare never starts training; no automatic route queue."""
from __future__ import annotations
import argparse
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import yaml
from p75_framework import ARCHIVE_PATHS
from p75_supervision_evidence import context, ROOT, OUT, read_json, sha, write_json
from p75_supervision_overlay import transform

FRAMEWORK=OUT/'framework'
CONFIGS=OUT/'configs'


def materialize(source):
    marker=FRAMEWORK/'source_manifest.json'
    if FRAMEWORK.exists():
        from p75_supported_ce import verify_files
        saved=read_json(marker)
        if saved['commit']!=source['framework_commit']:
            raise ValueError('Framework revision changed')
        verify_files(FRAMEWORK,saved['files'])
        if saved['adapter_sha256']!=sha(ROOT/'scripts/p75_supervision_runtime.py') or saved['overlay_sha256']!=sha(ROOT/'scripts/p75_supervision_overlay.py'):
            raise ValueError('Evidence adapter changed after materialization')
        return
    raw=subprocess.check_output(['git','archive',source['framework_commit'],*ARCHIVE_PATHS],cwd=ROOT)
    FRAMEWORK.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        if any(Path(m.name).is_absolute() or '..' in Path(m.name).parts or not (m.isfile() or m.isdir()) for m in archive.getmembers()):
            raise ValueError('Unsafe archive')
        archive.extractall(FRAMEWORK)
    trainer=FRAMEWORK/'reproducibility/aegis_f1/aegis_clip/trainer.py'
    trainer.write_text(transform(trainer.read_text()))
    (FRAMEWORK/'p75_supervision_runtime.py').write_bytes((ROOT/'scripts/p75_supervision_runtime.py').read_bytes())
    write_json(marker,dict(commit=source['framework_commit'],
        adapter_sha256=sha(ROOT/'scripts/p75_supervision_runtime.py'),
        overlay_sha256=sha(ROOT/'scripts/p75_supervision_overlay.py'),
        files={str(p.relative_to(FRAMEWORK)):sha(p) for p in FRAMEWORK.rglob('*') if p.is_file()}))


def environment(stop_epoch=None):
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join((str(FRAMEWORK),str(FRAMEWORK/'reproducibility/aegis_f1')))
    env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if stop_epoch is not None:
        env['P75_STOP_AFTER_EPOCH']=str(stop_epoch)
    return env


def prepare(route):
    _,source,identity,_,_,_=context()
    diagnosis=OUT/'diagnosis'
    summary=read_json(diagnosis/'summary.json')
    if summary['identity']!=identity or not summary['snapshot_manifest_sha256']:
        raise ValueError('Completed current-stage P0 snapshot diagnosis required')
    if route=='R1' and not summary['coverage_ok']:
        raise ValueError('Fixed supervision construction failed class coverage; no training config')
    if route=='R2' and summary['slices']['stable_trusted_confusion']['errors']<1000:
        raise ValueError('R2 requires at least 1,000 associated trusted-confusion validation errors')
    if route=='R1' and summary['decision']=='R2_bounded_pilot_first':
        raise ValueError('P0 selected R2; do not automatically stack R1')
    evidence=diagnosis/'sample_evidence.csv'
    slices=diagnosis/'frozen_slices.json'
    from p75_supported_ce import verify_files
    verify_files(diagnosis,summary['files'])
    materialize(source)
    # Read resolved original metadata only; no student weights come from L05.
    original=read_json(ROOT/'outputs/codex/p75_supported_ce_20260929/configs/l05_original.json')
    base=copy.deepcopy(original)
    base.pop('_config_path',None)
    base['project'].update(experiment_id=f'P75_{route}',trial_id=f'P75_{route}',
        mechanism='evidence_channels' if route=='R1' else 'trusted_confusion_representation',
        preparation_identity=identity)
    base['output']['root']=str(OUT/'runs')
    base['loss'].pop('hard_support',None)
    base['loss']['supervision_evidence']=dict(enabled=True,route=route,table=str(evidence),
        sha256=sha(evidence),slices=str(slices),slices_sha256=sha(slices))
    base['evaluation']['evaluate_initial_checkpoint']=False
    base['train'].update(epochs=16,schedule_epochs=16,save_epoch_checkpoints=True)
    if route=='R1':
        base['project'].update(parent_kind='same_split_continue',parent_experiment_id='P75_R1_HEAD')
        base['train']['init_checkpoint']=str(OUT/'runs/P75_R1_HEAD/seed42/checkpoints/last.pt')
        base['loss'].update(name='cross_entropy',ce_warmup_epochs=0)
        head=copy.deepcopy(base)
        head['project'].update(experiment_id='P75_R1_HEAD',trial_id='P75_R1_HEAD',parent_kind='official_clip_head')
        head['project'].pop('parent_experiment_id',None)
        head['model'].update(peft_mode='frozen',use_cached_training=True,input_resolution=224)
        head['project']['search']['resolution']=224
        head['loss']['attention_local_training']['enabled']=False
        head['loss']['feature_distillation_weight']=0.
        head['train'].update(device='cpu',epochs=20,schedule_epochs=20,head_lr=.01,
            batch_size=1024,grad_accum_steps=1,effective_batch_size=1024,
            num_workers=0,loader_timeout=0,pin_memory=False,amp=False,
            init_checkpoint=None,optimizer_impl='default')
        head['evaluation'].update(selection_policy='last_epoch',interval_epochs=20)
        configurations={'R1_HEAD':head,'R1':base}
    else:
        # Independent L05-like training with its existing official RM-LP parent;
        # R1 channels are never applied to R2 classification targets.
        base['project']['parent_kind']='shared_lp'
        base['train']['init_checkpoint']=source['parent_checkpoint']
        configurations={'R2':base}
    CONFIGS.mkdir(parents=True,exist_ok=True)
    for name,config in configurations.items():
        path=CONFIGS/f'{name}.yaml'
        text=yaml.safe_dump(config,sort_keys=False)
        if path.exists():
            if path.read_text()!=text:
                raise ValueError('Prepared config changed')
        else:
            path.write_text(text)
    return configurations


def execute(args,name,timeout,stop_epoch=None):
    log=OUT/'logs'/f'{name}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as stream:
        subprocess.run([sys.executable,*args],cwd=FRAMEWORK,env=environment(stop_epoch),
            stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    print(log)


def evaluate(route,epoch):
    run=OUT/f'runs/P75_{route}/seed42'
    checkpoint=run/f'checkpoints/epoch_{epoch}.pt'
    cache=run/f'val_epoch_{epoch}.pt'
    config=CONFIGS/f'{route}.yaml'
    execute(['scripts/cache_validation_tta_logits.py','--checkpoint',str(checkpoint),
        '--config',str(config),'--output',str(cache),'--device','cuda:0',
        '--batch-size','32','--num-workers','2'],f'{route}_cache_{epoch}',3600)
    execute(['scripts/evaluate_l05_candidate.py','--checkpoint',str(checkpoint),'--config',str(config),
        '--val-branch-cache',str(cache),'--temperature','1.4','--prior-strength','.6',
        '--fusion','mean_probabilities','--output',str(run/f'evaluation_{epoch}.json')],
        f'{route}_evaluate_{epoch}',600)
    # CPU child uses frozen decoder matching the training/inference runtime.
    report_args=[str(ROOT/'scripts/p75_supervision_report.py'),'--cache',str(cache),
        '--checkpoint',str(checkpoint),'--config',str(config),
        '--output',str(run/f'paired_{epoch}.json')]
    if epoch==6:
        report_args.extend(['--earlier-cache',str(run/'val_epoch_4.pt')])
    execute(report_args,f'{route}_paired_{epoch}',600)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['prepare','head','pilot','evaluate-pilot','full','evaluate-full','deliver'],required=True)
    p.add_argument('--route',choices=['R1','R2'],default='R1')
    p.add_argument('--execute-gpu',action='store_true')
    a=p.parse_args()
    prepare(a.route)
    if a.phase=='prepare':
        print(CONFIGS)
        return
    run=OUT/f'runs/P75_{a.route}/seed42'
    config=CONFIGS/f'{a.route}.yaml'
    if a.phase=='head':
        if a.route!='R1':
            raise ValueError('R2 reuses the original RM-LP head')
        execute(['-m','aegis_clip.cli.train','--config',str(CONFIGS/'R1_HEAD.yaml')], 'R1_head',1800,20)
        return
    if not a.execute_gpu:
        raise ValueError('--execute-gpu required')
    from run_p75_supported_ce import ensure_idle_device
    ensure_idle_device()
    if a.phase=='pilot':
        execute(['-m','aegis_clip.cli.train','--config',str(config)],f'{a.route}_pilot',21600,6)
    elif a.phase=='evaluate-pilot':
        evaluate(a.route,4)
        evaluate(a.route,6)
    elif a.phase in ('full','deliver'):
        summary=read_json(run/('paired_6.json' if a.phase=='full' else 'paired_16.json'))
        from p75_supervision_report import verify_report
        verify_report(summary)
        if not summary['full_training_signal' if a.phase=='full' else 'priority_submission']:
            raise ValueError('Frozen paired mechanism/full-validation gate failed')
        if a.phase=='full':
            execute(['-m','aegis_clip.cli.train','--config',str(config),'--resume',str(run/'checkpoints/last.pt')],
                f'{a.route}_full',57600,16)
        else:
            execute(['scripts/build_l05_tta_prior_submission_final.py','--checkpoint',str(run/'checkpoints/epoch_16.pt'),
                '--config',str(config),'--val-branch-cache',str(run/'val_epoch_16.pt'),
                '--temperature','1.4','--prior-strength','.6','--fusion','mean_probabilities',
                '--tta','horizontal_flip','--tag',f'P75_{a.route}','--output-root',str(OUT/'deliveries'),
                '--skip-desktop-copy','--device','cuda:0','--batch-size','32','--num-workers','2'],f'{a.route}_deliver',7200)
    else:
        evaluate(a.route,16)

if __name__=='__main__':
    main()
