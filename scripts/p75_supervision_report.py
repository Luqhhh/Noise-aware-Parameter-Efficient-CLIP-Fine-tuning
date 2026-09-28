#!/usr/bin/env python3
"""Paired full-val and frozen-slice reports; disputed labels never become truth."""
import argparse
from pathlib import Path
import numpy as np
import torch
from p75_supervision_evidence import OUT, ROOT, context, paired_counts, canonical
from p75_supported_ce import read_json, read_rows, sha, write_json, write_rows
from p75_supported_ce_report import predictions


def verify_report(result):
    for path,expected in result['binding'].items():
        if sha(path)!=expected:
            raise ValueError(f'Report source changed: {path}')
    for path,expected in result['report_files'].items():
        if sha(path)!=expected:
            raise ValueError(f'Paired rows changed: {path}')
    rows=read_rows(next(iter(result['report_files'])))
    labels=np.array([int(r['original_label']) for r in rows])
    base=np.array([int(r['l05_prediction']) for r in rows])
    candidate=np.array([int(r['candidate_prediction']) for r in rows])
    overall=paired_counts(labels,base,candidate)
    if overall!=result['all_validation']:
        raise ValueError('Paired full-validation metrics changed')
    target='hard_supervision_associated' if result['route']=='R1' else 'stable_trusted_confusion'
    mask=np.array([r[target]=='True' for r in rows])
    for key,value in result['slices'].items():
        if value!=paired_counts(labels,base,candidate,np.array([r[key]=='True' for r in rows])):
            raise ValueError('Paired slice metrics changed')
    priority=(overall['candidate_macro']-overall['baseline_macro']>=.015 and
        overall['candidate_micro']-overall['baseline_micro']>=.015 and result['slices'][target]['net']>0)
    signal=False
    if result['earlier_cache']:
        for key in ('earlier_cache','candidate_cache','progress_control_log'):
            if result[key] not in result['binding']:
                raise ValueError('Unbound progress source')
        previous=predictions(torch.load(result['earlier_cache'],map_location='cpu',weights_only=False))
        payload=torch.load(result['candidate_cache'],map_location='cpu',weights_only=False)
        raw=payload['original_logits'].argmax(1).numpy()
        center=paired_counts(labels,raw,raw)
        control=read_json(result['progress_control_log'])
        signal=(int(((base!=labels)&mask).sum())>=1000 and
            paired_counts(labels,previous,candidate,mask)['net']>=50 and
            paired_counts(labels,previous,candidate,~mask)['net']>=-50 and
            paired_counts(labels,previous,candidate)['net']>0 and
            center['candidate_macro']>=control['raw_macro']-.02)
    if bool(priority)!=result['priority_submission'] or bool(signal)!=result['full_training_signal']:
        raise ValueError('Promotion flags disagree with paired evidence')


def report(cache,checkpoint,config_path,output,earlier_cache=None):
    from aegis_clip.config import load_config
    from aegis_clip.rematch_protocol import validate_checkpoint
    torch.set_num_threads(4)
    _,source,identity,_,validation,_=context()
    config=load_config(config_path)
    meta=validate_checkpoint(checkpoint,config,parent=False)
    frozen_path=OUT/'diagnosis/frozen_slices.json'
    frozen=read_json(frozen_path)
    evidence=OUT/'diagnosis/sample_evidence.csv'
    if frozen['identity']!=identity or frozen['evidence_sha256']!=sha(evidence):
        raise ValueError('Frozen slices or evidence changed')
    baseline_cache=torch.load(source['reference_validation_cache'],map_location='cpu',weights_only=False)
    baseline=predictions(baseline_cache)
    if not np.array_equal(baseline,np.asarray(frozen['baseline_predictions'])):
        raise ValueError('Pinned L05 decoder changed')
    labels=np.asarray(frozen['labels'])
    payload=torch.load(cache,map_location='cpu',weights_only=False)
    def check_rows(value):
        if (value['paths']!=[canonical(r['image_path']) for r in validation] or
            not np.array_equal(np.asarray(value['labels']),labels) or
            value['validation_csv_sha256']!=identity['split_hashes']['val_dev.csv']):
            raise ValueError('Full validation row identity mismatch')
    check_rows(payload)
    if payload['checkpoint_sha256']!=sha(checkpoint):
        raise ValueError('Candidate checkpoint/cache mismatch')
    candidate=predictions(payload)
    overall=paired_counts(labels,baseline,candidate)
    masks={k:np.isin(np.arange(len(labels)),v) for k,v in frozen['slices'].items()}
    slices={k:paired_counts(labels,baseline,candidate,mask) for k,mask in masks.items()}
    route=config['loss']['supervision_evidence']['route']
    target='hard_supervision_associated' if route=='R1' else 'stable_trusted_confusion'
    delta_micro=overall['candidate_micro']-overall['baseline_micro']
    delta_macro=overall['candidate_macro']-overall['baseline_macro']
    # Full candidate target is a resource gate, not a prediction of platform gain.
    priority=delta_macro>=.015 and delta_micro>=.015 and slices[target]['net']>0
    early=None
    control_log=None
    signal=False
    bindings={str(p):sha(p) for p in (Path(cache),Path(checkpoint),Path(config_path),
        frozen_path,evidence,Path(source['reference_validation_cache']))}
    if earlier_cache:
        previous_checkpoint=Path(checkpoint).with_name('epoch_4.pt')
        previous_meta=validate_checkpoint(previous_checkpoint,config,parent=False)
        if previous_meta['epoch']!=4 or meta['epoch']!=6:
            raise ValueError('Pilot progress must compare epoch4 to epoch6')
        earlier=torch.load(earlier_cache,map_location='cpu',weights_only=False)
        check_rows(earlier)
        if earlier['checkpoint_sha256']!=sha(previous_checkpoint):
            raise ValueError('Earlier progress cache has the wrong checkpoint')
        prior=predictions(earlier)
        early=dict(all=paired_counts(labels,prior,candidate),
            target=paired_counts(labels,prior,candidate,masks[target]),
            outside=paired_counts(labels,prior,candidate,~masks[target]))
        control_log=Path(source['control_checkpoint']).parents[1]/'logs/evaluation_epoch_6.json'
        control=read_json(control_log)
        center=payload['original_logits'].argmax(1).numpy()
        center_metrics=paired_counts(labels,center,center)
        early.update(l05_epoch6_raw_macro=control['raw_macro'],
            candidate_epoch6_raw_macro=center_metrics['candidate_macro'],
            comparison='center raw macro at equal visual epochs; different head curriculum')
        signal=(int(((baseline!=labels)&masks[target]).sum())>=1000 and
            early['target']['net']>=50 and early['outside']['net']>=-50 and
            early['all']['net']>0 and center_metrics['candidate_macro']>=control['raw_macro']-.02)
        bindings.update({str(earlier_cache):sha(earlier_cache),str(control_log):sha(control_log),
            str(previous_checkpoint):sha(previous_checkpoint)})
    row_path=Path(output).with_suffix('.csv')
    rows=[dict(image_path=r['image_path'],original_label=int(labels[i]),
        l05_prediction=int(baseline[i]),candidate_prediction=int(candidate[i]),
        corrected=bool(candidate[i]==labels[i]!=baseline[i]),
        regressed=bool(baseline[i]==labels[i]!=candidate[i]),
        **{k:bool(mask[i]) for k,mask in masks.items()}) for i,r in enumerate(validation)]
    write_rows(row_path,list(rows[0]),rows)
    result=dict(identity=identity,route=route,all_validation=overall,slices=slices,
        delta_macro=delta_macro,delta_micro=delta_micro,priority_submission=bool(priority),
        early_progress=early,full_training_signal=bool(signal),binding=bindings,
        epoch=meta['epoch'],candidate_cache=str(cache),earlier_cache=str(earlier_cache) if earlier_cache else None,
        progress_control_log=str(control_log) if control_log else None,
        report_files={str(row_path):sha(row_path)},platform_gain=None,
        label_dispute_scope='changes against original labels only; not clean accuracy',
        slices_are_independent_test=False)
    write_json(output,result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','checkpoint','config','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--earlier-cache',type=Path)
    a=p.parse_args()
    result=report(a.cache,a.checkpoint,a.config,a.output,a.earlier_cache)
    print(result['all_validation'])
