"""Manual platform receipts and frozen full-data recipe; no network uploads."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import math

import yaml

from aegis_clip.rematch_data import write_csv


def record_receipt(rows, candidate, *, submission_id, submitted_at, platform_period, status, score=None):
    if status not in ('pending','valid','failed'):
        raise ValueError('Receipt status must be pending, valid or failed')
    if not submission_id or not platform_period or not submitted_at:
        raise ValueError('Use an actual platform submission ID and reset-period identifier')
    if datetime.fromisoformat(submitted_at).tzinfo is None:
        raise ValueError('Submission timestamp requires explicit timezone')
    selected=[r for r in rows if r['candidate']==candidate]
    if len(selected)!=1:raise ValueError('Candidate must have one validated ready package')
    row=selected[0]
    if any(r is not row and r['platform_submission_id']==submission_id for r in rows):
        raise ValueError('Platform submission ID already belongs to another candidate')
    if row['platform_submission_id'] and row['platform_submission_id']!=submission_id:
        raise ValueError('Do not replace an existing submission with a duplicate upload')
    duplicate=[r for r in rows if r is not row and r['csv_sha256']==row['csv_sha256'] and r['platform_submission_id']]
    if duplicate:raise ValueError('Identical prediction CSV has already been submitted')
    if not row['platform_submission_id']:
        used=sum(bool(r['platform_submission_id']) and r['platform_period']==platform_period for r in rows)
        if used>=2:raise ValueError('Two submissions already recorded for this platform reset period')
    if status=='valid' and (score is None or not math.isfinite(score)):
        raise ValueError('A valid result needs a finite measured platform score')
    row.update(platform_submission_id=submission_id,submitted_at=submitted_at,platform_period=platform_period,
               status=status,platform_score=str(score) if status=='valid' else '')
    valid=[r for r in rows if r['status']=='valid']
    winner=max(valid,key=lambda r:float(r['platform_score'])) if valid else None
    for r in rows:r['is_highest_score']='true' if r is winner else 'false'
    return rows


def full_configs(root, winner, registry):
    """Freeze selected epochs while preserving the original cosine horizons."""
    if winner not in ('RM_FT','RM_LT'):raise ValueError('Winner must be RM_FT or RM_LT')
    for name in ('RM_FT','RM_LT'):
        if not any(r['candidate']==name and r['status']=='valid' for r in registry):
            raise ValueError('RM-FULL waits for both FT and LT platform measurements')
    selected={}
    for name in ('RM_LP',winner):
        p=root/'outputs/rematch750'/name/'seed42/checkpoints/selected_report.json'
        selected[name]=json.loads(p.read_text())
    paths=[]
    for src,name,parent in [('lp','RM_FULL_LP',None),(winner.removeprefix('RM_').lower(),'RM_FULL','RM_FULL_LP')]:
        cfg=yaml.safe_load((root/f'configs/rematch750_{src}.yaml').read_text())
        base_name='RM_LP' if src=='lp' else winner
        cfg['project'].update(experiment_id=name,full_training=True,selected_recipe=winner)
        cfg['data'].update(train_csv='../artifacts/stages/repechage/20260921/full_train.csv',validation_overlap_with_training=True)
        cfg['train']['epochs']=selected[base_name]['selected_epoch']
        cfg['evaluation'].update(selection_policy='last_epoch',tiebreak_metric=None)
        if parent:cfg['train']['init_checkpoint']=f'../outputs/rematch750/{parent}/seed42/checkpoints/best.pt'
        path=root/f'configs/rematch750_{name.lower()}.yaml'
        if path.exists():raise FileExistsError(path)
        path.write_text(yaml.safe_dump(cfg,sort_keys=False));paths.append(path)
    return paths
