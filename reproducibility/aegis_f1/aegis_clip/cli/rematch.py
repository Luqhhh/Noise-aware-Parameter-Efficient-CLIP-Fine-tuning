"""Small staged entry point using the existing Aegis cache/train/infer runner."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import torch

from aegis_clip.device import resolve_device
from aegis_clip.config import load_config
from aegis_clip.rematch_assets import validate_dataset, validate_cache, validate_checkpoint
from aegis_clip.rematch_data import prepare, write_csv
from aegis_clip.runtime import atomic_json_dump, sha256_file

ROOT=Path(__file__).resolve().parents[4]
REGISTRY=ROOT/'results/rematch_submission_registry.csv'
FIELDS=['candidate','stage','data_version','code_commit','config_sha256','checkpoint_sha256',
        'class_mapping_sha256','dataset_manifest_sha256','inference_config','csv_sha256','zip_sha256',
        'submission_path','platform_submission_id','submitted_at','platform_period','platform_score',
        'status','is_highest_score']


def read_registry():
    if not REGISTRY.exists():return []
    with REGISTRY.open() as f:return list(csv.DictReader(f))


def register(config, checkpoint, submission):
    m=json.loads((submission/'manifest.json').read_text())
    validate_checkpoint(checkpoint,config)
    assert m['prediction_count']==config['data']['expected_test_samples']
    assert m['corrupt_images']==0
    rows=read_registry()
    candidate=config['project']['experiment_id']
    previous=[r for r in rows if r['candidate']==candidate]
    if previous:
        if previous[0]['zip_sha256']!=m['submission_zip_sha256']:
            raise ValueError('Candidate ID already registered with different predictions')
        return
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    # Code hash also binds uncommitted implementation while a segment is running.
    code_files=[*sorted((ROOT/'reproducibility/aegis_f1/aegis_clip').rglob('*.py'))]
    atomic_json_dump({str(p.relative_to(ROOT)):sha256_file(p) for p in code_files},submission/'code_hashes.json')
    row=dict.fromkeys(FIELDS,'')
    row.update(candidate=candidate,stage='repechage',data_version=config['project']['data_version'],
        code_commit=commit,config_sha256=sha256_file(config['_config_path']),checkpoint_sha256=m['checkpoint_sha256'],
        class_mapping_sha256=sha256_file(config['data']['class_mapping']),dataset_manifest_sha256=sha256_file(config['data']['dataset_manifest']),
        inference_config='global_224_center_crop;tta=none;prior=none;single_checkpoint',
        csv_sha256=m['prediction_csv_sha256'],zip_sha256=m['submission_zip_sha256'],
        submission_path=str(submission.relative_to(ROOT)),status='ready',is_highest_score='false')
    rows.append(row)
    write_csv(REGISTRY,rows,FIELDS)
    atomic_json_dump(row,submission/'registry_binding.json')


def per_class_report(config, run):
    best=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    metrics=best['metrics']
    with (Path(config['data']['dataset_manifest']).parent/'class_support.csv').open() as f:
        support=list(csv.DictReader(f))
    if config['project'].get('full_training'):
        with (Path(config['data']['dataset_manifest']).parent/'full_train.csv').open() as f:
            groups=[set() for _ in support]
            for row in csv.DictReader(f):groups[int(row['label'])].add(row['content_group'])
        for i,row in enumerate(support):
            row['train_groups']=len(groups[i])
            row['validation_status']='overlapping_diagnosis'
    draws=[0]*config['model']['num_classes']
    for epoch in range(1,best['epoch']+1):
        ledger=json.loads((run/f'logs/longtail_epoch_{epoch}.json').read_text())
        for c in ledger['classes']:draws[c['class_id']]+=c['actual_draws']
    rows=[]
    for s,m in zip(support,metrics['per_class']):
        row={**s,**m,'actual_training_exposures_to_selected_epoch':draws[m['label']]}
        row['segment']='tail' if m['train_samples']<20 else 'middle' if m['train_samples']<100 else 'head'
        rows.append(row)
    write_csv(run/'checkpoints/selected_per_class.csv',rows)
    atomic_json_dump(dict(selected_epoch=best['epoch'],schedule_epochs=config['train']['schedule_epochs'],
        raw_macro=metrics['raw_macro'],raw_micro=metrics['raw_micro'],
        validation_covered_classes=metrics['validation_covered_classes'],
        support_segments={k:v for k,v in metrics.items() if k.startswith('support_')},
        validation_interpretation='overlapping diagnosis' if config['project'].get('full_training') else 'independent content groups, noisy labels'),run/'checkpoints/selected_report.json')


def execute(action, config_path):
    config=load_config(config_path)
    run=Path(config['output']['root'])/config['project']['experiment_id']/f"seed{config['project']['seed']}"
    if action=='prepare':return prepare(config)
    if action=='verify':
        validate_dataset(config)
        return print('Current-stage dataset bindings verified')
    device=resolve_device(config['train'].get('device','cuda'))
    if device.type=='cpu':raise RuntimeError('Rematch requires an accelerator')
    if action=='cache':
        from aegis_clip.cli.cache_features import cache_stage_features
        return cache_stage_features(config,device=device,batch_size=128,workers=4)
    if action=='train':
        from aegis_clip.trainer import train
        code_files=sorted((ROOT/'reproducibility/aegis_f1/aegis_clip').rglob('*.py'))
        # Keep the snapshot outside run_dir: trainer refuses pre-existing run dirs.
        atomic_json_dump(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            files={str(p.relative_to(ROOT)):sha256_file(p) for p in code_files}),
            run.parent/'training_code_manifest.json')
        checkpoint=train(config)
        per_class_report(config,run)
        return checkpoint
    if action=='infer':
        checkpoint=run/'checkpoints/best.pt'
        validate_dataset(config)
        validate_checkpoint(checkpoint,config)
        submission=run/'submission'
        subprocess.run([sys.executable,'-m','aegis_clip.cli.infer','--checkpoint',str(checkpoint),
                        '--config',str(config_path),'--device',str(device),'--output-dir',str(submission),'--tta','none'],check=True,cwd=ROOT)
        subprocess.run([sys.executable,'scripts/check_submission.py','--test_dir',config['data']['test_root'],
                        '--num-classes',str(config['model']['num_classes']),'--class-mapping',config['data']['class_mapping'],
                        '--csv',str(submission/'pred_results.csv'),'--zip',str(submission/'submission.zip')],check=True,cwd=ROOT)
        register(config,checkpoint,submission)
        return submission
    if action=='run':
        if device.type=='npu':
            raise ValueError('Use explicit NPU train/infer configs; run is the legacy CUDA LP/FT/LT queue')
        for candidate in ('lp','ft','lt'):
            p=ROOT/f'configs/rematch750_{candidate}.yaml'
            execute('train',p)
            execute('infer',p)
        return
    raise ValueError(action)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','verify','cache','train','infer','run','record','prepare-full'])
    p.add_argument('--config',default=str(ROOT/'configs/rematch750_lp.yaml'))
    p.add_argument('--candidate')
    p.add_argument('--submission-id')
    p.add_argument('--submitted-at')
    p.add_argument('--platform-period')
    p.add_argument('--status',choices=['pending','valid','failed'])
    p.add_argument('--score',type=float)
    args=p.parse_args()
    if args.action=='record':
        from aegis_clip.rematch_registry import record_receipt
        rows=read_registry()
        matching=[r for r in rows if r['candidate']==args.candidate]
        if len(matching)!=1:
            raise ValueError('Candidate has no unique locally validated package')
        row=matching[0]
        package=ROOT/row['submission_path']
        manifest=json.loads((package/'manifest.json').read_text())
        for path,digest in [(package/'submission.zip',row['zip_sha256']),
                            (package/'pred_results.csv',row['csv_sha256']),
                            (Path(manifest['checkpoint']),row['checkpoint_sha256'])]:
            if sha256_file(path)!=digest:
                raise ValueError(f'Platform package binding changed: {path}')
        rows=record_receipt(rows,args.candidate,submission_id=args.submission_id,
            submitted_at=args.submitted_at,platform_period=args.platform_period,status=args.status,score=args.score)
        write_csv(REGISTRY,rows,FIELDS)
        return
    if args.action=='prepare-full':
        from aegis_clip.rematch_registry import full_configs
        print(full_configs(ROOT,args.candidate,read_registry()))
        return
    execute(args.action,Path(args.config).resolve())


if __name__=='__main__':main()
