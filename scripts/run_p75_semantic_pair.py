#!/usr/bin/env python3
"""One frozen semantic masking pair; bounded GPU execution, no recipe search."""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

from p75_framework import ARCHIVE_PATHS, COMMIT
from p75_hard_overlay import once
from p75_text_page_overlay import transform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/codex/p75_semantic_mask_pair_20260929'
FRAMEWORK = OUT/'framework'
CONFIGS = ROOT/'configs/p75_semantic_mask_pair_20260929'
ORIGINAL = Path('/home/lux1/noise/worktrees/p75_supported_ce_20260929/outputs/codex/p75_supported_ce_20260929/configs/l05_original.json')
SELECTION_SHA = '39b2cd9e8d94d1fd114c7f0a7581297528ecae49a82d8d4ddf6f5e3d06bbf5de'
PARENT_SHA = 'd5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b'
BUDGET = 21600
RESERVE = 1800  # validation + one diagnostic single-student delivery, inside total cap


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def run_dir(arm):
    return OUT/f'runs/P75_SEMANTIC_{arm.upper()}/seed42'


def prepare():
    if OUT.exists():
        raise FileExistsError('Use verify/run for an already prepared experiment')
    OUT.mkdir(parents=True)
    selection = OUT/'selection.jsonl'
    with gzip.open(ROOT/'results/p75_semantic_content_probe_20260929/frozen_selection.jsonl.gz','rb') as f:
        selection.write_bytes(f.read())
    if sha(selection) != SELECTION_SHA:
        raise ValueError('Frozen automatic selection changed')
    original = json.loads(ORIGINAL.read_text())
    parent = Path(original['train']['init_checkpoint'])
    if sha(parent) != PARENT_SHA:
        raise ValueError('Legal RM-LP parent changed')
    raw = subprocess.check_output(['git','archive',COMMIT,*ARCHIVE_PATHS],cwd=ROOT)
    FRAMEWORK.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        if any(Path(m.name).is_absolute() or '..' in Path(m.name).parts or not (m.isfile() or m.isdir())
               for m in archive.getmembers()):
            raise ValueError('Unsafe source archive')
        archive.extractall(FRAMEWORK)
    trainer = FRAMEWORK/'reproducibility/aegis_f1/aegis_clip/trainer.py'
    source = transform(trainer.read_text())
    # Identical process boundaries at epoch 4 in both arms. Scheduler remains 16 epochs.
    source = once(source, '    for epoch in range(start_epoch, min(epochs, 6) + 1):\n',
        '    import os, hashlib\n'
        '    pair_stop = int(os.environ.get("P75_PAIR_STOP", "6"))\n'
        '    if pair_stop not in (4, 6):\n'
        '        raise ValueError("Only fixed epoch 4/6 pair boundaries allowed")\n'
        '    for epoch in range(start_epoch, min(epochs, pair_stop) + 1):\n')
    source = once(source, '        for batch_index, batch in enumerate(train_loader):\n',
        '        pair_order = hashlib.sha256()\n'
        '        for batch_index, batch in enumerate(train_loader):\n'
        '            pair_order.update(batch["index"].numpy().astype("<i8").tobytes())\n')
    source = once(source, '        train_metrics = {\n',
        '        atomic_json_dump(dict(epoch=epoch, order_sha256=pair_order.hexdigest(),\n'
        '            rows=int(totals["examples"]), successful_updates=successful_updates,\n'
        '            attempted_updates=global_step), log_dir / f"pair_order_{epoch}.json")\n'
        '        train_metrics = {\n')
    compile(source, str(trainer), 'exec')
    trainer.write_text(source)
    (FRAMEWORK/'p75_text_page_runtime.py').write_bytes((ROOT/'scripts/p75_text_page_runtime.py').read_bytes())
    CONFIGS.mkdir(parents=True, exist_ok=False)
    for arm in ('control','masked'):
        config = copy.deepcopy(original)
        name = f'P75_SEMANTIC_{arm.upper()}'
        config['project'].update(experiment_id=name, trial_id=name,
            mechanism='frozen_semantic_content_classification_mask')
        config['output']['root'] = str(OUT/'runs')
        config['train']['save_epoch_checkpoints'] = True
        config['loss']['text_page_mask'] = dict(enabled=arm=='masked',selection=str(selection),sha256=SELECTION_SHA)
        write(CONFIGS/f'{arm}.json',config)
    files = {str(p):sha(p) for p in sorted(FRAMEWORK.rglob('*')) if p.is_file()}
    files.update({str(p):sha(p) for p in CONFIGS.glob('*.json')})
    files.update({str(p):sha(p) for p in (selection, parent, ORIGINAL,
        ROOT/'scripts/p75_text_page_runtime.py',ROOT/'scripts/p75_text_page_overlay.py',
        ROOT/'scripts/run_p75_semantic_pair.py')})
    write(OUT/'manifest.json',dict(framework_commit=COMMIT,files=files,
        budget_seconds=BUDGET,evaluation_delivery_reserve_seconds=RESERVE,
        primary_epoch=6,observation_epoch=4,selected_train_rows=1246,
        semantics='automatic content proxy; not confirmed noisy/clean labels',
        process_boundaries='both arms stop after 4 and resume to 6; worker RNG is restarted identically',
        parent_limitation='RM-LP head already learned original supervision',
        delivery='masked last matched epoch; diagnostic package only, no automatic platform promotion'))
    verify()
    print(OUT/'manifest.json',flush=True)


def verify():
    saved = json.loads((OUT/'manifest.json').read_text())
    for path, expected in saved['files'].items():
        if sha(path) != expected:
            raise ValueError(f'Frozen source/config/input changed: {path}')
    a,b = [json.loads((CONFIGS/f'{arm}.json').read_text()) for arm in ('control','masked')]
    for c in (a,b):
        c['project'].pop('experiment_id')
        c['project'].pop('trial_id')
        c['loss']['text_page_mask'].pop('enabled')
    if a != b:
        raise ValueError('Pair differs beyond classification mask and identifiers')
    return saved


def idle_gpu():
    result = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'],text=True).strip()
    if result:
        raise RuntimeError(f'GPU has an existing compute process; do not preempt: {result}')


def environment(stop=None):
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join((str(FRAMEWORK),str(FRAMEWORK/'reproducibility/aegis_f1'))),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    if stop is not None:
        env['P75_PAIR_STOP'] = str(stop)
    return env


def execute(args, name, state, allowance, stop=None):
    idle_gpu()
    log = OUT/'logs'/f'{name}.log'
    log.parent.mkdir(exist_ok=True)
    started = time.monotonic()
    entry = dict(name=name,argv=args,start_unix=time.time(),allowance_seconds=allowance,log=str(log))
    state['current'] = entry
    write(OUT/'status.json',state)
    with log.open('x') as stream:
        proc = subprocess.Popen([sys.executable,*args],cwd=FRAMEWORK,env=environment(stop),
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        entry['pid']=proc.pid
        write(OUT/'status.json',state)
        try:
            code = proc.wait(timeout=max(1,allowance))
            entry['returncode']=code
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            entry.update(returncode=proc.returncode,budget_timeout=True)
    entry['seconds']=time.monotonic()-started
    state['used_seconds'] += entry['seconds']
    state['history'].append(entry)
    state['current']=None
    write(OUT/'status.json',state)
    if entry.get('budget_timeout'):
        return False
    if entry['returncode']:
        raise RuntimeError(f'{name} failed; see {log}')
    return True


def remaining(state):
    return BUDGET-state['used_seconds']


def run():
    verify()
    if (OUT/'status.json').exists():
        raise FileExistsError('Campaign already started; no automatic replay or budget reset')
    state=dict(status='running',started_unix=time.time(),used_seconds=0.,history=[],current=None)
    write(OUT/'status.json',state)
    try:
        completed=0
        for epoch in (4,6):
            passed=True
            for i,arm in enumerate(('control','masked')):
                allowance=(remaining(state)-RESERVE)/(2-i)
                if allowance<=0:
                    passed=False
                    break
                args=['-m','aegis_clip.cli.train','--config',str(CONFIGS/f'{arm}.json')]
                if epoch==6: args += ['--resume',str(run_dir(arm)/'checkpoints/last.pt')]
                if not execute(args,f'{arm}_train_to_{epoch}',state,allowance,epoch):
                    passed=False
                    break
            if not passed: break
            completed=epoch
        state['matched_epoch']=completed
        if not completed:
            state['status']='budget_incomplete_no_matched_epoch'
            return
        # Primary comparison first; epoch 4 is observation only and can be omitted if capped.
        for epoch in ([6,4] if completed==6 else [4]):
            for arm in ('control','masked'):
                checkpoint=run_dir(arm)/f'checkpoints/epoch_{epoch}.pt'
                cache=run_dir(arm)/f'val_epoch_{epoch}.pt'
                if not execute(['scripts/cache_validation_tta_logits.py','--checkpoint',str(checkpoint),
                    '--config',str(CONFIGS/f'{arm}.json'),'--output',str(cache),'--device','cuda:0',
                    '--batch-size','32','--num-workers','2'],f'{arm}_cache_{epoch}',state,
                    remaining(state)):
                    state['status']='budget_incomplete_validation'
                    return
        checkpoint=run_dir('masked')/f'checkpoints/epoch_{completed}.pt'
        delivered=execute(['scripts/build_l05_tta_prior_submission_final.py','--checkpoint',str(checkpoint),
            '--config',str(CONFIGS/'masked.json'),'--val-branch-cache',str(run_dir('masked')/f'val_epoch_{completed}.pt'),
            '--temperature','1.4','--prior-strength','.6','--fusion','mean_probabilities',
            '--tta','horizontal_flip','--tag',f'P75_SEMANTIC_MASKED_E{completed}',
            '--output-root',str(OUT/'deliveries'),'--skip-desktop-copy','--device','cuda:0',
            '--batch-size','32','--num-workers','2'],f'masked_deliver_{completed}',state,remaining(state))
        state['status']=('paired_training_validation_delivery_complete' if completed==6 else
            'epoch4_only_primary_incomplete') if delivered else 'budget_incomplete_delivery'
    except Exception as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:
        state['ended_unix']=time.time()
        write(OUT/'status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','verify','run'])
    args=p.parse_args()
    globals()[args.phase]()
