#!/usr/bin/env python3
"""Explicitly authorized new E4-to-E6 segment; never resumes the historical budget."""
import argparse
import json
import shutil
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from p75_pipeline import DEFAULT, verify, require, inspect_resume, cost_gate
from p75_pipeline_report import decode, emit
from run_p75_semantic_pair import write, idle_gpu, sha


def authorize(root, authorization, task):
    record=json.loads(authorization.read_text())
    manifest=root/task/'manifest.json'; m=json.loads(manifest.read_text())
    require(record.get('approved') is True and record.get('experiment_id')==m['experiment_id'] and
            record.get('manifest_sha256')==sha(root/'manifest.json') and
            record.get('budget_seconds')==m['budget_seconds'] and bool(record.get('user_instruction')),
            'Separate explicit user authorization bound to this manifest/budget is required')
    return record


def audit_epoch(root,epoch):
    records=[]
    for arm in ('control','masked'):
        run=root/f'A/runs/P75_MASK_E6_{arm.upper()}/seed42'
        order=json.loads((run/f'logs/pair_order_{epoch}.json').read_text())
        groups=json.loads((run/f'logs/text_page_epoch_{epoch}.json').read_text())
        require(order['rows']==133815 and order['successful_updates']==order['attempted_updates']==epoch*131,'Update/coverage mismatch')
        require(groups['enabled']==(arm=='masked') and groups['text_page_selected_rows']==1246 and
                groups['text_page_remaining_rows']==132569 and groups['text_page_local_selected_rows']==1246 and
                groups['text_page_local_remaining_rows']==132569,'Actual mask coverage mismatch')
        payload=torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False,mmap=True)
        inspect_resume(payload,arm,epoch)
        records.append(order)
    require(records[0]==records[1],'Pair order/update/LR mismatch')
    return records[0]


def continuation_selection(run_dir, payload, config_path):
    """Apply original center macro/micro selection without changing the evaluated E6 identity."""
    from p75_semantic_pair_report import paired
    import numpy as np
    from aegis_clip.checkpoint import _atomic_torch_save
    labels=np.asarray(payload['labels']); pred=payload['original_logits'].argmax(1).numpy()
    metrics=paired(labels,pred,pred)
    previous=torch.load(run_dir/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    endpoint_path=run_dir/'checkpoints/epoch_6.pt'
    endpoint=torch.load(endpoint_path,map_location='cpu',weights_only=False)
    selector=metrics['control_macro']; micro=metrics['control_micro']
    improved=(selector>previous['best_selector'] or (selector==previous['best_selector'] and micro>previous['metrics']['raw_micro']))
    endpoint['metrics'].update(raw_macro=selector,raw_micro=micro,selector=selector)
    if improved: endpoint['best_selector']=selector
    continuation=run_dir/'checkpoints/continuation_E6.pt'
    _atomic_torch_save(endpoint,continuation)
    meta=json.loads(endpoint_path.with_suffix('.binding.json').read_text())
    meta.update(checkpoint_sha256=sha(continuation),evaluated_checkpoint_sha256=sha(endpoint_path),
                continuation_note='Identical E6 weights/optimizer/RNG; original center selection metadata applied from bound cache')
    write(continuation.with_suffix('.binding.json'),meta)
    if improved:
        shutil.copyfile(continuation,run_dir/'checkpoints/best.pt')
        write(run_dir/'checkpoints/best.binding.json',meta)
    return dict(checkpoint=str(continuation),sha256=sha(continuation),selected_epoch=6 if improved else previous['epoch'],
                raw_macro=selector,raw_micro=micro,config_sha256=sha(config_path))


def execute(root, argv, name, state, cap, stop=None):
    idle_gpu()
    left=cap-state['used_seconds']
    require(left>120,'Insufficient budget to launch child')
    start=time.monotonic(); entry=dict(argv=argv,name=name,started_unix=time.time())
    env=os.environ.copy(); framework=root/'framework'
    env.update(PYTHONPATH=os.pathsep.join([str(framework),str(framework/'reproducibility/aegis_f1')]),
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    if stop:
        env.update(P75_PIPELINE_STOP=str(stop),P75_PIPELINE_DEADLINE=str(time.time()+left-90))
    logs=root/'A/logs'; logs.mkdir(exist_ok=True)
    state['current']=entry; write(root/'A/status.json',state)
    with (logs/f'{name}.log').open('x') as f:
        child=subprocess.Popen([sys.executable,*argv],cwd=framework,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        entry['pid']=child.pid; write(root/'A/status.json',state)
        try:
            code=child.wait(timeout=left-10)
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait()
            raise
        finally:
            entry.update(seconds=time.monotonic()-start,returncode=child.returncode)
            state['used_seconds']+=entry['seconds']; state['history'].append(entry); state['current']=None
            write(root/'A/status.json',state)
    require(code==0,'Child incomplete/failed; no automatic replay: '+name)


def run(root,authorization):
    verify(root); record=authorize(root,authorization,'A')
    require(set(record['measured_cost'])=={'seconds_per_local_update','validation_seconds','overhead_seconds'},'Unexpected cost/budget override')
    probe=record.get('preflight_probe')
    spent=0.
    if probe:
        require(sha(probe['path'])==probe['sha256'],'Preflight timing record changed')
        measured=json.loads(Path(probe['path']).read_text())
        spent=float(measured['seconds'])
        require(measured['returncode']==0 and 0<spent<21600,'Invalid measured preflight charge')
    costs=dict(record['measured_cost']); costs['overhead_seconds']+=spent
    gate=cost_gate(**costs)
    require(gate['fits'],'Cannot fit both E6 endpoints and evaluation with 20% reserve')
    require(record.get('local_step_measurement_source') and record.get('validation_measurement_source'), 'Cost evidence required')
    # Do not benchmark by silently starting a training step under proposal authorization.
    require(not (root/'A/status.json').exists(),'Segment already started; no replay/budget reset')
    idle_gpu()
    manifest=json.loads((root/'A/manifest.json').read_text())
    for arm in ('control','masked'):
        best=root/f'A/runs/P75_MASK_E6_{arm.upper()}/seed42/checkpoints/best.pt'
        require(sha(best)==manifest['arms'][arm]['historical_best_sha256'],'Historical selection input changed')
    state=dict(status='running',used_seconds=spent,preflight_seconds=spent,history=[],current=None,cost_gate=gate,authorization_sha256=sha(authorization),
        local_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip())
    write(root/'A/status.json',state)
    audit=[]; started=None
    try:
        reserve=record['measured_cost']['validation_seconds']+record['measured_cost']['overhead_seconds']
        for epoch in (5,6):
            for arm in ('control','masked'):
                checkpoint=Path(manifest['arms'][arm]['resume']) if epoch==5 else root/f'A/runs/P75_MASK_E6_{arm.upper()}/seed42/checkpoints/last.pt'
                inspect_resume(torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True),arm,epoch-1)
                execute(root,['-m','aegis_clip.cli.train','--config',manifest['arms'][arm]['config'],'--resume',str(checkpoint)],
                        f'{arm}_E{epoch}',state,21600-reserve,epoch)
            audit.append(audit_epoch(root,epoch))
        for arm in ('control','masked'):
            run_dir=root/f'A/runs/P75_MASK_E6_{arm.upper()}/seed42'
            execute(root,['scripts/cache_validation_tta_logits.py','--checkpoint',str(run_dir/'checkpoints/epoch_6.pt'),
                '--config',manifest['arms'][arm]['config'],'--output',str(run_dir/'val_epoch_6.pt'),
                '--tta-temperature','1.4','--device','cuda:0','--batch-size','32','--num-workers','2'],f'{arm}_decode',state,21600)
        torch.set_num_threads(4)
        started=time.monotonic(); decoded=[]; bindings={}
        def expired(signum, frame):
            raise TimeoutError('Evaluation wall-clock cap reached')
        signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,max(.1,21600-state['used_seconds']-5))
        rows=json.loads((root/'groups.json').read_text())
        for arm in ('control','masked'):
            run_dir=root/f'A/runs/P75_MASK_E6_{arm.upper()}/seed42'
            payload,pred,binding=decode(run_dir/'checkpoints/epoch_6.pt',run_dir/'val_epoch_6.pt',Path(manifest['arms'][arm]['config']),root/'framework')
            require([p.removeprefix('train/') for p in payload['paths']]==[r['image_path'].removeprefix('train/') for r in rows], 'Validation path mismatch')
            require(list(payload['labels'])==[r['label'] for r in rows], 'Validation label mismatch')
            binding['continuation']=continuation_selection(run_dir,payload,Path(manifest['arms'][arm]['config']))
            decoded.append(pred); bindings[arm]=binding
        state['used_seconds']+=time.monotonic()-started
        started=None
        require(state['used_seconds']<21600,'Evaluation exceeded segment budget')
        emit(root/'A',rows,decoded,bindings,manifest['experiment_id'],audit)
        state['status']='local_result'
    except BaseException as exc:
        state.update(status='incomplete',error=repr(exc))
        write(root/'A/conclusion.json',dict(status='incomplete',reason=repr(exc),mechanism_disproved=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if started is not None: state['used_seconds']+=time.monotonic()-started
        write(root/'A/status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--authorization',type=Path,required=True)
    a=p.parse_args(); run(a.out.resolve(),a.authorization)
