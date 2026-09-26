#!/usr/bin/env python3
"""Pinned Aegis paired L05 continuation with training-only patch-stem MixStyle."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import yaml
from l05_mixstyle_support import PatchStemMixStyle, install_mixstyle

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); tmp.replace(path)


def verify_framework(cfg, framework):
    commit=subprocess.check_output(['git','rev-parse',cfg['framework_commit']],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(['git','ls-tree','-r',commit,'reproducibility/aegis_f1',
                                 'scripts/cache_validation_tta_logits.py','scripts/evaluate_l05_candidate.py'],cwd=ROOT,text=True)
    checked=0
    for line in tree.splitlines():
        meta,name=line.split('\t'); mode,kind,blob=meta.split()
        if kind != 'blob': continue
        content=(framework/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==blob,name
        checked+=1
    assert digest(cfg['parent_checkpoint'])==cfg['parent_sha256']
    assets=Path(cfg['assets']); manifest=json.loads((assets/'dataset_manifest.json').read_text())
    assert manifest['stage']=='repechage' and not manifest['validation_overlap_with_training']
    for name,sha in manifest['files'].items(): assert digest(assets/name)==sha,name
    return {'framework_commit':commit,'verified_source_files':checked,'parent_sha256':cfg['parent_sha256'],
            'dataset_manifest_sha256':digest(assets/'dataset_manifest.json')}


def prepare(cfg, arm, smoke):
    out=(ROOT/cfg['output']).resolve()
    parent=torch.load(cfg['parent_checkpoint'],map_location='cpu',weights_only=False)
    c=copy.deepcopy(parent['config'])
    name=arm+('_SMOKE_V2' if smoke else '')
    c['project'].update(experiment_id=name,trial_id=arm,parent_kind='same_split_continue',
                        parent_experiment_id='RM_V5_L05_CUDA_LOCAL',seed=cfg['seed'],
                        mechanism='paired_patch_stem_mixstyle' if arm=='MS01' else 'paired_continuation_control')
    c['project']['mixstyle']={'probability':cfg['probability'] if arm=='MS01' else 0.,'alpha':cfg['alpha'],
                             'seed':cfg['seed'],'placement':'visual.conv1 output',
                             'framework_commit':cfg['framework_commit']}
    c['project']['search'].update(micro_batch_candidates=[cfg['microbatch']],smoke_required=False,local_start_epoch=1)
    c['train'].update(epochs=1 if smoke else cfg['epochs'],schedule_epochs=1 if smoke else cfg['epochs'],
                      batch_size=cfg['microbatch'],grad_accum_steps=1 if smoke else cfg['accumulation'],
                      effective_batch_size=cfg['microbatch']*(1 if smoke else cfg['accumulation']),
                      backbone_lr=cfg['backbone_lr'],head_lr=cfg['head_lr'],num_workers=cfg['workers'],
                      device='cuda:0',save_epoch_checkpoints=True,init_checkpoint=cfg['parent_checkpoint'],
                      early_stop_patience=0,lr_warmup_epochs=0,log_every_steps=1)
    if smoke: c['train']['max_steps']=2
    c['loss']['ce_warmup_epochs']=0
    c['loss']['attention_local_training']['start_epoch']=1
    c['evaluation'].update(interval_epochs=1,evaluate_initial_checkpoint=not smoke,record_initial=not smoke,
                           batch_size=32,inference_batch_size=32)
    c['output']['root']=str(out/'runs')
    c.pop('_config_path',None)
    path=out/'configs'/f'{name}.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    data=yaml.safe_dump(c,sort_keys=False)
    if path.exists() and path.read_text()!=data: raise FileExistsError('Runtime config drift')
    path.write_text(data)
    return path,out/'runs'/name/f'seed{cfg["seed"]}'


class StopLoss(Exception):
    pass


def training(cfg, arm, smoke, resume, framework):
    from aegis_clip.config import load_config
    from aegis_clip import trainer
    from aegis_clip.rematch_protocol import validate_checkpoint
    path,run=prepare(cfg,arm,smoke); config=load_config(path)
    validate_checkpoint(cfg['parent_checkpoint'],config,parent=True)
    options=config['project']['mixstyle']
    mixer=PatchStemMixStyle(options['probability'],options['alpha'],options['seed'])
    if resume:
        record=json.loads(Path(resume+'.mixstyle.json').read_text())
        assert record['checkpoint_sha256']==digest(resume)
        assert record['hook_source_sha256']==digest(ROOT/'scripts/l05_mixstyle_support.py')
        mixer.load_state_dict(record['state'])
    original_builder,original_save=trainer.build_model,trainer.save_checkpoint
    def builder(c,device):
        model,preprocess=original_builder(c,device)
        install_mixstyle(model,mixer)
        return model,preprocess
    def save(path,**kwargs):
        original_save(path,**kwargs)
        dump(str(path)+'.mixstyle.json',{'checkpoint_sha256':digest(path),'state':mixer.state_dict(),
               'hook_source_sha256':digest(ROOT/'scripts/l05_mixstyle_support.py')})
        m=kwargs['metrics']
        if Path(path).name=='last.pt' and kwargs['epoch']>0 and not smoke:
            if m.get('raw_macro',1)<.7524971006956281-.02 or m.get('raw_micro',1)<.7629704301075269-.02:
                raise StopLoss('Predeclared -2pp epoch stop')
    trainer.build_model,trainer.save_checkpoint=builder,save
    stopped=False
    try:
        best=trainer.train(config,resume=resume)
    except StopLoss:
        best=run/'checkpoints/best.pt';stopped=True
    finally:
        trainer.build_model,trainer.save_checkpoint=original_builder,original_save
    # The pinned trainer copies epoch0.pt to best.pt but not its sidecars.
    epoch0=run/'checkpoints/epoch0.pt'
    if epoch0.exists() and digest(best)==digest(epoch0):
        for source,dest in ((epoch0.with_suffix('.binding.json'),Path(best).with_suffix('.binding.json')),
                            (Path(str(epoch0)+'.mixstyle.json'),Path(str(best)+'.mixstyle.json'))):
            if dest.exists() and dest.read_bytes()!=source.read_bytes(): raise ValueError('Best sidecar mismatch')
            if not dest.exists(): shutil.copy2(source,dest)
    validate_checkpoint(best,config,parent=False)
    if arm=='MS01' and (mixer.calls==0 or mixer.applied==0):
        raise RuntimeError('MixStyle did not actually activate; run is invalid')
    dump(run/'training_complete.json',{'checkpoint':str(best),'stopped':stopped,'smoke':smoke,
         'state':mixer.state_dict(),'checkpoint_sha256':digest(best),'config_sha256':digest(path)})
    print(json.dumps({'checkpoint':str(best),'stopped':stopped,'smoke':smoke,'state':mixer.state_dict()}),flush=True)


def queue(cfg, framework):
    out=(ROOT/cfg['output']).resolve()
    environment=dict(os.environ);environment['PYTHONPATH']=str(framework/'reproducibility/aegis_f1')
    for arm in ('MS00','MS01'):
        path,run=prepare(cfg,arm,False)
        log=out/f'{arm}_train.log'
        dump(out/'queue_state.json',{'arm':arm,'phase':'training','driver_pid':os.getpid()})
        with log.open('x') as handle:
            subprocess.run([sys.executable,__file__,'--config',str(args_config),'--phase','train','--arm',arm],
                           cwd=ROOT,env=environment,stdout=handle,stderr=subprocess.STDOUT,check=True)
        best=run/'checkpoints/best.pt';cache=run/'val_branch_logits.pt'
        dump(out/'queue_state.json',{'arm':arm,'phase':'evaluation','driver_pid':os.getpid()})
        commands=[
          [str(framework/'scripts/cache_validation_tta_logits.py'),'--checkpoint',str(best),'--config',str(path),
           '--output',str(cache),'--tta-fusion','mean_probabilities','--tta-temperature','1.0',
           '--device','cuda:0','--batch-size','32','--num-workers',str(cfg['workers'])],
          [str(framework/'scripts/evaluate_l05_candidate.py'),'--checkpoint',str(best),'--config',str(path),
           '--val-branch-cache',str(cache),'--output',str(run/'evaluation.json')]]
        with (out/f'{arm}_evaluate.log').open('x') as handle:
            for command in commands:
                subprocess.run([sys.executable,*command],cwd=ROOT,env=environment,stdout=handle,stderr=subprocess.STDOUT,check=True)
    results={arm:json.loads((out/'runs'/arm/f'seed{cfg["seed"]}'/'evaluation.json').read_text()) for arm in ('MS00','MS01')}
    base={'macro':.7582651238982523,'micro':.7665322580645161}
    def gate(value,reference):
        return value['macro']>=reference['macro']+.003 and value['micro']>=reference['micro']
    summary={'arms':results,'baseline_decode':base,'test_used':False}
    for arm in ('MS00','MS01'):
        complete=json.loads((out/'runs'/arm/f'seed{cfg["seed"]}'/'training_complete.json').read_text())
        summary[arm+'_pass']=not complete['stopped'] and gate(results[arm]['decode'],base)
    summary['MS01_pass'] &= gate(results['MS01']['decode'],results['MS00']['decode'])
    dump(out/'result.json',summary)
    dump(out/'queue_state.json',{'phase':'complete','driver_pid':os.getpid()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--phase',choices=['prepare','train','queue'],required=True)
    p.add_argument('--arm',choices=['MS00','MS01'],default='MS01');p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume')
    args=p.parse_args();args_config=Path(args.config).resolve()
    torch.set_num_threads(4)
    cfg=json.loads(args_config.read_text());out=(ROOT/cfg['output']).resolve();framework=out/'framework'
    receipt=verify_framework(cfg,framework)
    sys.path.insert(0,str(framework/'reproducibility/aegis_f1'))
    dump(out/'source_receipt.json',receipt)
    if args.phase=='prepare':
        for arm in ('MS00','MS01'): print(prepare(cfg,arm,args.smoke)[0])
    elif args.phase=='train': training(cfg,args.arm,args.smoke,args.resume,framework)
    else: queue(cfg,framework)
