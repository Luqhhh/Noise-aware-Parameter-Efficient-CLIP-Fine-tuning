"""Execute this run's fixed recovery queue using the existing frozen trainer.
No scans, overwrite, resume, publication, or changes to registered configs.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import copy
import time
from pathlib import Path
import yaml

ROOT = PLAN = STATE = None


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def save(state):
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATE)


def verify_completion(node, config):
    directory=Path(config['output']['root'])/node['node_id']/'seed42'/'checkpoints'
    manifest=json.loads((directory/'artifact_manifest.json').read_text())
    for field in ('train_csv','val_csv'):
        if manifest[field+'_sha256'] != sha(config['data'][field]):
            raise ValueError('Completed training data binding mismatch')
    checkpoint=Path(manifest['best_checkpoint'])
    if checkpoint.resolve().parent != directory.resolve() or sha(checkpoint)!=manifest['best_checkpoint_sha256']:
        raise ValueError('Completed model binding mismatch')
    if manifest['experiment_id']!=node['node_id']:
        raise ValueError('Wrong completed experiment')
    return manifest


def audit():
    plan=json.loads(PLAN.read_text())
    if len(plan['nodes'])!=7 or plan['maximum_registered_epochs']!=73 or sum(n['epochs'] for n in plan['nodes']) != 73:
        raise ValueError('This script is restricted to the registered seven-node recovery')
    source=json.loads((ROOT/'training_source_hashes.json').read_text())
    for relative,digest in source.items():
        if sha(ROOT/'training_source'/relative)!=digest:raise ValueError('Frozen training code changed')
    sys.path.insert(0,str(ROOT/'training_source'))
    from aegis_clip.config import load_config
    for index, node in enumerate(plan['nodes']):
        if sha(node['config_path'])!=node['config_sha256']:raise ValueError('Registered config changed')
        config=yaml.safe_load(Path(node['config_path']).read_text())
        if config['train']['epochs']!=node['epochs']:raise ValueError('Epoch budget changed')
        original=load_config(node['original_config'])
        def method_only(value):
            value=copy.deepcopy(value)
            for key in ('features','output','diagnostics','_config_path'):value.pop(key,None)
            value['train'].pop('init_checkpoint',None)
            for key in ('artifact_scope','dataset_id'):value['project'].pop(key,None)
            return value
        if method_only(original)!=method_only(config):
            raise ValueError('Recovery changes the registered historical training method')

        parent = config['train'].get('init_checkpoint')
        if parent != node['parent']:
            raise ValueError('Parent differs from registered dependency')
        if index:
            expected = ROOT/'recovery_models'/plan['nodes'][index-1]['node_id']/'seed42/checkpoints'/Path(original['train']['init_checkpoint']).name
            if Path(parent).resolve() != expected:
                raise ValueError('Parent does not follow the historical checkpoint selection')
        elif parent:
            raise ValueError('Root recovery must use official initialization')
        if config['project']['experiment_id']!=node['node_id']:raise ValueError('Experiment mismatch')
        if Path(config['output']['root']).resolve()!=ROOT/'recovery_models':raise ValueError('Output outside recovery root')
        if node['argv']!=['python3','-m','aegis_clip.cli.train','--config',str(Path(node['config_path']).resolve())]:
            raise ValueError('Unexpected training command')
    return plan


def main():
    global ROOT,PLAN,STATE
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--run-dir',required=True);args=parser.parse_args()
    ROOT=Path(args.run_dir).resolve();PLAN=ROOT/'recovery_plan.json';STATE=ROOT/'recovery_queue_status.json'
    plan=audit()
    if not args.execute:print('Fixed recovery queue audit passed; no execution');return
    lock=ROOT/'.recovery_queue.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    state={'status':'running','plan_sha256':sha(PLAN),'nodes':{},'scope':'historical recipe recovery; owner acceptance pending'}
    try:
        # E2 was explicitly launched before this queue. Its trainer publishes the
        # final artifact manifest only after all training and final evaluation.
        first=plan['nodes'][0];cfg=yaml.safe_load(Path(first['config_path']).read_text())
        marker=Path(cfg['output']['root'])/first['node_id']/'seed42/checkpoints/artifact_manifest.json'
        state['nodes'][first['node_id']]={'status':'waiting_for_final_training_manifest'};save(state)
        deadline=time.monotonic()+24*3600
        while not marker.exists():
            if time.monotonic()>deadline:raise TimeoutError('E2 completion was not verified within 24h; queue stopped')
            if 'Traceback (most recent call last)' in (ROOT/'e2_recovery.log').read_text():
                raise RuntimeError('E2 training error; queue stopped')
            time.sleep(10)
        if sha(PLAN)!=state['plan_sha256']:raise ValueError('Recovery plan changed while waiting')
        audit()
        result=verify_completion(first,cfg)
        state['nodes'][first['node_id']]={'status':'verified_final_artifact','process_exit_code':None,'artifact':result};save(state)
        for node in plan['nodes'][1:]:
            if sha(PLAN)!=state['plan_sha256']:raise ValueError('Recovery plan changed during execution')
            audit();cfg=yaml.safe_load(Path(node['config_path']).read_text());directory=Path(cfg['output']['root'])/node['node_id']/'seed42'
            if directory.exists():raise FileExistsError('Recovery output already exists; no implicit overwrite/resume')
            if not Path(cfg['train']['init_checkpoint']).is_file():raise FileNotFoundError('Required parent checkpoint missing')
            parent_hash=sha(cfg['train']['init_checkpoint'])
            state['nodes'][node['node_id']]={'status':'running','argv':node['argv'],'parent_sha256':parent_hash,'started_unix':time.time()};save(state)
            env={**os.environ,'PYTHONPATH':str(ROOT/'training_source')}
            with (ROOT/(node['node_id']+'.log')).open('x') as log:
                result=subprocess.run(node['argv'],cwd=ROOT/'training_source',env=env,stdout=log,stderr=subprocess.STDOUT)
            audit()
            if result.returncode:raise RuntimeError(f"{node['node_id']} exited {result.returncode}")
            if sha(cfg['train']['init_checkpoint']) != parent_hash:raise ValueError('Parent changed during training')
            manifest=verify_completion(node,cfg)
            state['nodes'][node['node_id']].update(status='checks_passed',exit_code=0,artifact=manifest);save(state)
        state['status']='checks_passed';save(state)
    except BaseException as error:
        state['status']='failed';state['error']=repr(error);save(state);raise
    finally:lock.unlink()


if __name__=='__main__':main()
