"""Replay archived heads with CPU Torch float64; recount risk with Python counters."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_v1_frozen_teacher_recovery import inputs
from diagnose_v1_candidate_errors import sha, require, read_json

ROOT=Path(__file__).resolve().parents[1]


def replay_torch(features,state):
    w=F.normalize(state['weight'].to(torch.float64),dim=1)
    scale=state['logit_scale'].to(torch.float64).exp().clamp(1,100)
    result={k:[] for k in ['prediction','confidence','margin']}
    for start in range(0,len(features),1024):
        x=F.normalize(torch.from_numpy(features[start:start+1024].copy()).to(torch.float64),dim=1)
        scores=scale*(x@w.T)
        p=scores.softmax(1); top=p.topk(2,dim=1)
        result['prediction'].append(scores.argmax(1).numpy())
        result['confidence'].append(top.values[:,0].numpy())
        result['margin'].append((top.values[:,0]-top.values[:,1]).numpy())
    return {k:np.concatenate(v) for k,v in result.items()}


def counts(labels,prediction,mask):
    population=Counter(int(y) for y,m in zip(labels,mask) if m)
    correct=Counter(int(y) for y,p,m in zip(labels,prediction,mask) if m and y==p)
    n=sum(population.values()); c=sum(correct.values())
    return dict(rows=n,correct=c,errors=n-c,classes=len(population),micro=c/n if n else None,
                macro=sum(correct[y]/v for y,v in population.items())/len(population) if population else None)


def compare_pair(labels,before,after,mask):
    selected=[(int(y),int(a),int(b)) for y,a,b,m in zip(labels,before,after,mask) if m]
    corr=sum(a!=y and b==y for y,a,b in selected); reg=sum(a==y and b!=y for y,a,b in selected)
    return dict(rows=len(selected),corrections=corr,regressions=reg,net=corr-reg,
                changed=sum(a!=b for y,a,b in selected),both_wrong=sum(a!=y and b!=y for y,a,b in selected))


def equal_counts(expected,actual,name):
    require(set(expected)==set(actual),f'{name}: fields differ')
    for key,value in actual.items():
        if isinstance(value,float): require(abs(expected[key]-value)<1e-12,f'{name}/{key}: metric differs')
        else: require(expected[key]==value,f'{name}/{key}: count differs')


def verify(config,report_path):
    report=read_json(report_path)
    require(report['config_sha256']==sha(config) and report['script_sha256']==sha(ROOT/'scripts/diagnose_v1_frozen_teacher_recovery.py'),
            'Frozen config/implementation changed')
    require(report['status']=='completed_cpu_diagnostic' and report['training_started'] is False and
            report['new_candidate'] is False and report['no_fusion_predictions_created'] is True and
            report['test_data_used'] is False and report['automatic_training'] is False and
            report['native_decoder_comparison_is_causal'] is False and report['catastrophic_forgetting_proven'] is False and
            report['platform_gain_known'] is False, 'Scope claim differs')
    for path,digest in report['source_files'].items(): require(sha(path)==digest,f'Source changed: {path}')
    cfg,labels,paths,classes,arrays,heads,archived,student,groups,sources,incumbent=inputs(config)
    require(report['experiment_id']==cfg['experiment_id'] and report['validation_rows']==len(labels) and
            report['classes']==len(classes), 'Population differs')
    require(sha(report['score_file'])==report['score_sha256'], 'Private score checksum differs')
    cache=np.load(report['score_file'],allow_pickle=False)
    expected_keys={'labels','image_paths','student'}|set(groups)|{
        f'{prefix}{d}' for prefix in ['teacher','confidence','margin','supported'] for d in cfg['teacher_dimensions']}
    require(set(cache.files)==expected_keys, 'Unexpected/fusion score columns')
    require(np.array_equal(labels,cache['labels']) and np.array_equal(paths,cache['image_paths']) and
            np.array_equal(student,cache['student']), 'Score rows/labels differ')
    for group,mask in groups.items(): require(np.array_equal(mask,cache[group]),f'{group}: membership differs')
    numerical_errors={}; comparisons={}; verified_metrics=verified_pairs=0
    for d in cfg['teacher_dimensions']:
        replay=replay_torch(arrays[d],heads[d]); prediction=replay['prediction']
        require(np.array_equal(prediction,archived[f'head{d}']) and np.array_equal(prediction,cache[f'teacher{d}']),
                f'Teacher{d}: independent native prediction replay differs')
        numerical_errors[str(d)]={}
        for field in ['confidence','margin']:
            error=float(np.max(np.abs(replay[field]-cache[f'{field}{d}'])))
            require(error<1e-11,f'Teacher{d}: numerical {field} differs'); numerical_errors[str(d)][field]=error
        mask_support=(replay['confidence']>=.7)&(replay['margin']>=.2)
        require(np.array_equal(mask_support,cache[f'supported{d}']), 'Independent supported membership differs')
        recorded=report['teacher_reports'][str(d)]
        require(recorded['native_predictions_replayed']==len(labels) and recorded['support_rows']==int(mask_support.sum()) and
                recorded['unsupported_rows']==int((~mask_support).sum()), 'Support population differs')
        comparisons[str(d)]={}
        for group,mask in groups.items():
            expected=recorded['groups'][group]
            for name,pred in [('teacher',prediction),('student',student)]:
                equal_counts(expected[name],counts(labels,pred,mask),f'{d}/{group}/{name}');verified_metrics+=1
            require(expected['supported_rows']==int((mask&mask_support).sum()), 'Supported group count differs')
            actual=compare_pair(labels,student,prediction,mask)
            supported_actual=compare_pair(labels,student,prediction,mask&mask_support)
            equal_counts(expected['comparison'],actual,f'{d}/{group}/all_pair')
            equal_counts(expected['supported_comparison'],supported_actual,f'{d}/{group}/supported_pair');verified_pairs+=2
            comparisons[str(d)][group]=supported_actual
    primary=comparisons[str(cfg['primary_teacher_dimension'])]
    overall=primary['all']; common=primary['common_candidate_wrong']['corrections']; gate=cfg['feasibility_gate']
    passed=bool(overall['net']>=gate['supported_net_min'] and overall['corrections']>=gate['supported_corrections_min'] and
        common>=gate['common_error_supported_recoveries_min'] and
        overall['corrections']>=gate['corrections_to_regressions_min']*overall['regressions'])
    decision='supports_transfer_review' if passed else 'close_fixed_confident_teacher_logit_transfer'
    require(report['primary_teacher_dimension']==cfg['primary_teacher_dimension'] and report['gate_passed'] is passed and
            report['decision']==decision, 'Frozen feasibility gate differs')
    return dict(status='passed_independent_torch64_and_counter_replay',experiment_id=cfg['experiment_id'],
        teacher_predictions_replayed=2*len(labels),native_head_predictions_identical=True,
        teacher_support_memberships_identical=True,numerical_max_errors=numerical_errors,
        metric_groups_checked=verified_metrics,paired_groups_checked=verified_pairs,
        source_files_checked=len(report['source_files']),scores_sha256=sha(report['score_file']),report_sha256=sha(report_path),
        supported_comparisons=comparisons,decision=decision,new_candidate=False,platform_gain_known=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--report',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    require(not Path(args.output).exists(),'Use a fresh independent verification path')
    torch.set_num_threads(2); result=verify(args.config,args.report)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)
