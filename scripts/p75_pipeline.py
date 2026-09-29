#!/usr/bin/env python3
"""Proposal-only supervision pipeline: prepare and audit, never starts training."""
from __future__ import annotations
import argparse
import copy
import csv
import gzip
import io
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tarfile
from collections import Counter
import torch
from run_p75_semantic_pair import sha, write, SELECTION_SHA, PARENT_SHA
from p75_framework import COMMIT, ARCHIVE_PATHS
from p75_pipeline_overlay import transform

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/home/lux1/noise-worktrees/p75_semantic_mask_pair_20260929/outputs/codex/p75_semantic_mask_pair_20260929')
DEFAULT = ROOT/'outputs/codex/p75_supervision_pipeline_20260929_final'
STATES = ('proposal','engineering_ready','running','incomplete','local_result','package_ready','platform_measured')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def equal_state(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    if hasattr(a,'shape'):
        import numpy as np
        return np.array_equal(a,b)
    return a==b


def inspect_resume(payload, arm, epoch=4):
    required = ('model_state_dict','optimizer_state_dict','scheduler_state_dict',
                'scaler_state_dict','rng_state','data_generator_state','config')
    require(all(payload.get(k) is not None for k in required), 'Incomplete resume state')
    require(payload['epoch']==epoch and payload['global_step']==131*epoch, 'Unexpected epoch/update endpoint')
    c=payload['config']; t=c['train']; loss=c['loss']; local=loss['attention_local_training']
    require(c['project']['data_version']=='20260921' and c['model']['num_classes']==750, 'Wrong stage')
    require(c['model']['input_resolution']==384 and t['batch_size']==4 and t['grad_accum_steps']==256
            and t['effective_batch_size']==1024 and t['epochs']==t['schedule_epochs']==16, 'Changed FT schedule')
    require(loss['gce_q']==.5 and loss['feature_distillation_weight']==2 and local['start_epoch']==5
            and local['confidence_gate']==.7, 'Changed FT objective')
    require(loss['text_page_mask']['enabled']==(arm=='masked') and
            loss['text_page_mask']['sha256']==SELECTION_SHA, 'Wrong arm/mask')
    require(payload['scheduler_state_dict']['last_epoch']==payload['global_step'], 'Scheduler endpoint mismatch')
    require(payload['optimizer_state_dict']['state'] and payload['scaler_state_dict'], 'Empty optimizer/AMP state')
    require(all(payload['rng_state'].get(k) is not None for k in ('python','numpy','torch','cuda')), 'Incomplete RNG')
    require(payload['rng_state'].get('npu') is None, 'NPU resume forbidden')
    return dict(epoch=epoch, updates=payload['global_step'],remaining_updates_to_e6=131*(6-epoch),
                scheduler=payload['scheduler_state_dict'], worker_rng_restored=False)


def cost_gate(seconds_per_local_update, validation_seconds, overhead_seconds, budget=21600):
    require(all(isinstance(v,(int,float)) and 0<v<float('inf') for v in
                (seconds_per_local_update,validation_seconds,overhead_seconds,budget)), 'Measured positive costs required')
    estimate=524*seconds_per_local_update+validation_seconds+overhead_seconds
    return dict(estimated_seconds=estimate,budget_seconds=budget,reserve_fraction=.2,
                fits=estimate<=.8*budget)


def frozen_groups(rows):
    train=[r for r in rows if r['split']=='train']; val=[r for r in rows if r['split']=='val']
    require(len(train)==133815 and len(val)==14880, 'Wrong frozen split sizes')
    require(len({r['image_path'] for r in rows})==len(rows), 'Duplicate frozen paths')
    total=Counter(r['label'] for r in train); selected=Counter(r['label'] for r in train if r['selected'])
    require(sum(selected.values())==1246 and len(selected)==423, 'Changed train mask')
    require(len(total)==750 and all(total[c]>selected[c] for c in total), 'Empty remaining class')
    hot={c for c in total if selected[c]/total[c]>=.05}
    require(len(hot)==23, 'Changed 23-class slice')
    result=[]
    for r in val:
        bio=r['sB']>=max(r['sN'],r['sA'])
        result.append(dict(image_path=r['image_path'],label=r['label'],selected_proxy=r['selected'],
            high_hit_classes=r['label'] in hot,bio_dominant=bio,
            bio_high_hit=bio and r['label'] in hot,bio_other_classes=bio and r['label'] not in hot))
    require(sum(r['selected_proxy'] for r in result)==132 and sum(r['high_hit_classes'] for r in result)==419
            and sum(r['bio_high_hit'] for r in result)==332, 'Changed validation slices')
    remaining_groups={c:set() for c in total}
    for r in train:
        if not r['selected']: remaining_groups[r['label']].add(r['content_group'])
    coverage=[dict(label=c,train_rows=total[c],masked_rows=selected[c],remaining_rows=total[c]-selected[c],
                   remaining_content_groups=len(remaining_groups[c])) for c in sorted(total)]
    return result,coverage


def prepare(destination):
    destination=destination.resolve()
    require(not destination.exists(), 'Refusing to overwrite existing experiment/budget')
    destination.mkdir(parents=True)
    framework=destination/'framework'; framework.mkdir()
    raw=subprocess.check_output(['git','archive',COMMIT,*ARCHIVE_PATHS],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        require(all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and
                    (m.isfile() or m.isdir()) for m in archive.getmembers()), 'Unsafe archive')
        archive.extractall(framework)
    trainer=framework/'reproducibility/aegis_f1/aegis_clip/trainer.py'
    trainer.write_text(transform(trainer.read_text()))
    (framework/'p75_text_page_runtime.py').write_bytes((ROOT/'scripts/p75_text_page_runtime.py').read_bytes())
    selection=destination/'selection.jsonl'
    with gzip.open(ROOT/'results/p75_semantic_content_probe_20260929/frozen_selection.jsonl.gz','rb') as f:
        selection.write_bytes(f.read())
    require(sha(selection)==SELECTION_SHA, 'Frozen selection changed')
    rows=[json.loads(s) for s in selection.read_text().splitlines()]
    groups,coverage=frozen_groups(rows)
    write(destination/'groups.json',groups); write(destination/'class_coverage.json',coverage)
    inputs={str(selection):sha(selection)}
    for key in ('groups.json','class_coverage.json'):
        inputs[str(destination/key)]=sha(destination/key)
    a=dict(experiment_id='P75_MASK_E6_EXTENSION',status='proposal',authorized=False,
           budget_seconds=21600,arms={},blockers=['New training authorization and measured local-branch cost estimate required'],
           worker_limitation='Main RNG and generator restored; DataLoader workers restart in both arms at identical boundaries. No uninterrupted-view equivalence claimed.')
    oldreport=json.loads((ROOT/'results/p75_semantic_mask_pair_20260929/report.json').read_text())
    saved=[]
    for arm in ('control','masked'):
        p=OLD/f'runs/P75_SEMANTIC_{arm.upper()}/seed42/checkpoints/last.pt'
        config=json.loads((ROOT/f'configs/p75_semantic_mask_pair_20260929/{arm}.json').read_text())
        entry=dict(resume=str(p),exists=p.is_file())
        if p.is_file():
            digest=sha(p)
            expected=oldreport['epochs']['4']['binding'][arm]['checkpoint_sha256']
            evaluated=p.with_name('epoch_4.pt')
            require(sha(evaluated)==expected, 'Archived E4 checkpoint identity changed')
            payload=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
            reference=torch.load(evaluated,map_location='cpu',weights_only=False,mmap=True)
            require(equal_state(payload,reference), 'E4 last state differs from evaluated E4')
            inputs[str(evaluated)]=expected
            entry['evaluated_checkpoint_sha256']=expected
            entry.update(sha256=digest,state=inspect_resume(payload,arm))
            saved.append(payload['data_generator_state'])
            inputs[str(p)]=digest
        else:
            a['blockers'].append(f'Missing {arm} E4 full state; no from-scratch fallback')
        name=f'P75_MASK_E6_{arm.upper()}'
        config['project'].update(experiment_id=name,trial_id=name)
        config['output']['root']=str(destination/'A/runs')
        config['loss']['text_page_mask']['selection']=str(selection)
        write(destination/f'A/{arm}.json',config)
        entry['config']=str(destination/f'A/{arm}.json')
        if p.is_file():
            # Byte-identical private input + explicit continuation binding; historical sidecar untouched.
            old_binding=p.with_suffix('.binding.json')
            meta=json.loads(old_binding.read_text())
            require(meta['checkpoint_sha256']==digest, 'E4 original binding changed')
            old_config=ROOT/f'configs/p75_semantic_mask_pair_20260929/{arm}.json'
            require(meta['training_config_sha256']==sha(old_config), 'E4 original config identity changed')
            inputs[str(old_binding)]=sha(old_binding); inputs[str(old_config)]=sha(old_config)
            # Only segment identity, output directory and the byte-identical selection path may change.
            before=copy.deepcopy(payload['config']); after=copy.deepcopy(config)
            for value in (before,after):
                value.pop('_config_path',None)
                for key in ('experiment_id','trial_id'): value['project'].pop(key,None)
                value['output'].pop('root',None); value['loss']['text_page_mask'].pop('selection',None)
            require(before==after, 'Continuation changes the frozen E4 recipe')
            copied=destination/f'A/{arm}_resume_E4.pt'
            shutil.copyfile(p,copied)
            require(sha(copied)==digest, 'Private E4 input copy mismatch')
            meta.update(training_config_sha256=sha(entry['config']),
                original_training_config_sha256=sha(old_config),original_checkpoint_path=str(p),
                continuation_note='Byte-identical E4 state; new segment configuration binding only. Original experiment identity retained.')
            write(copied.with_suffix('.binding.json'),meta)
            inputs[str(copied)]=digest; inputs[str(copied.with_suffix('.binding.json'))]=sha(copied.with_suffix('.binding.json'))
            entry['original_resume']=str(p); entry['resume']=str(copied)
        if p.is_file():
            best=p.with_name('best.pt'); best_meta=json.loads(best.with_suffix('.binding.json').read_text())
            require(sha(best)==best_meta['checkpoint_sha256'], 'Original selected checkpoint changed')
            best_state=torch.load(best,map_location='cpu',weights_only=False,mmap=True)
            require(best_state['epoch']<=4 and best_state['best_selector']==payload['best_selector'], 'Original selection state mismatch')
            best_copy=destination/f'A/runs/{name}/seed42/checkpoints/best.pt'
            best_copy.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(best,best_copy)
            best_meta.update(original_training_config_sha256=best_meta['training_config_sha256'],
                training_config_sha256=sha(entry['config']),original_checkpoint_path=str(best),
                continuation_note='Historical selected weights, byte-identical copy in new segment')
            write(best_copy.with_suffix('.binding.json'),best_meta)
            # Working best.pt may later be replaced by E6; immutable source is the input binding.
            inputs[str(best)]=sha(best); inputs[str(best.with_suffix('.binding.json'))]=sha(best.with_suffix('.binding.json'))
            entry['historical_best']=str(best); entry['historical_best_sha256']=sha(best)
        a['arms'][arm]=entry
    if len(saved)==2:
        require(torch.equal(*saved), 'E4 generator endpoints differ')
    write(destination/'A/manifest.json',a)
    # Read the actual RM-LP parent configuration using the existing CPU storage compatibility shim.
    # No NPU device, service, or remote host is accessed.
    parent=Path(config['train']['init_checkpoint'])
    b=dict(experiment_id='P75_MASK_LP',status='proposal',authorized=False,budget_seconds=3600,
           blockers=['New CPU training authorization required'],source_parent=str(parent))
    if parent.is_file():
        require(sha(parent)==PARENT_SHA, 'RM-LP parent identity changed')
        sys.path.insert(0,str(framework/'reproducibility/aegis_f1'))
        from aegis_clip.npu_checkpoint_compat import ensure_npu_checkpoint_stubs
        ensure_npu_checkpoint_stubs()
        lp=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)['config']
        require(lp['project']['experiment_id']=='RM_LP' and lp['project']['data_version']=='20260921', 'Wrong LP configuration')
        write(destination/'B/original_config.json',lp)
        b['actual_config']=str(destination/'B/original_config.json'); inputs[str(parent)]=PARENT_SHA
        # Rebase only asset locations; retain the actual head objective, optimizer and schedule.
        lp=copy.deepcopy(lp)
        lp['data']=copy.deepcopy(config['data']); lp['data']['train_augmentation']='clip_center_crop'
        lp['features']=copy.deepcopy(config['features'])
        lp['train'].update(device='cpu',amp=False,num_workers=0,pin_memory=False)
        lp['train'].pop('init_checkpoint',None)
        lp['output']['root']=str(destination/'B')
        lp['project']['experiment_id']='P75_MASK_LP'
        lp['model']['official_checkpoint']=config['model']['official_checkpoint']
        write(destination/'B/config.json',lp)
        b['config']=str(destination/'B/config.json')
        for section,keys in [('data',('train_csv','val_csv','class_mapping','dataset_manifest')),
                             ('features',('tensor_path','paths_path','manifest_path'))]:
            for key in keys:
                path=Path(lp[section][key])
                if path.is_file(): inputs[str(path)]=sha(path)
                else: b['blockers'].append('Missing existing asset: '+str(path))
        b['blockers'].append('Fixed center+flip report requires existing bound flip feature cache; no image re-encoding allowed')
    else:
        b['blockers'].append('Actual RM-LP parent/config unavailable')
    write(destination/'B/manifest.json',b)
    files={str(p):sha(p) for p in framework.rglob('*') if p.is_file()}
    files.update(inputs)
    for p in destination.glob('[AB]/*.json'): files[str(p)]=sha(p)
    official=Path(config['model']['official_checkpoint'])
    if official.is_file(): files[str(official)]=sha(official)
    for name in ('p75_pipeline.py','p75_pipeline_overlay.py','p75_pipeline_report.py','p75_pipeline_lp.py','run_p75_pipeline_a.py','run_p75_semantic_pair.py','p75_semantic_pair_report.py','p75_framework.py',
                 'p75_hard_overlay.py','p75_text_page_overlay.py','p75_text_page_runtime.py'):
        p=ROOT/'scripts'/name
        files[str(p)]=sha(p)
    write(destination/'manifest.json',dict(experiment_id='P75_SUPERVISION_PIPELINE',status='proposal',
        implementation_status='engineering_ready',training_started=False,framework_commit=COMMIT,files=files,
        decode=dict(tta='horizontal_flip',fusion='mean_probabilities',temperature=1.4,prior_strength=.6),
        groups_limitation='Overlapping content proxies, not clean truth or an independent validation set; 23 classes are post hoc for E4.',
        historical_budget_untouched=True))
    for task in ('A','B'):
        write(destination/f'{task}/conclusion.json',dict(status='proposal',local_result=None,platform_result=None,
            decision='Await separate authorization and all preflight gates; no automatic C1/C2 or platform upload'))
        with (destination/f'{task}/paired.csv').open('w') as f:
            f.write('group,rows,control_macro,masked_macro,control_micro,masked_micro,F,D,net\n')
            sizes=dict(all=len(groups),selected_proxy=132,remaining_proxy=14748)
            sizes.update({k:sum(r[k] for r in groups) for k in ('high_hit_classes','bio_dominant','bio_high_hit','bio_other_classes')})
            for group,size in sizes.items(): f.write(f'{group},{size},,,,,,,\n')
    return destination


def verify(destination):
    m=json.loads((destination/'manifest.json').read_text())
    for path,digest in m['files'].items(): require(sha(path)==digest,'Frozen input/source changed: '+path)
    return m


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','verify']); p.add_argument('--out',type=Path,default=DEFAULT)
    args=p.parse_args()
    print(prepare(args.out) if args.phase=='prepare' else verify(args.out)['status'])
