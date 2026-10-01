"""Independently replay view scores in NumPy64 and error groups with counters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from diagnose_v1_candidate_errors import require, read_json, read_rows, sha
from verify_v1_frozen_teacher_recovery import counts, compare_pair, equal_counts

ROOT=Path(__file__).resolve().parents[1]


def replay_views(a,b,labels,threshold):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    require(a.ndim==2 and a.shape==b.shape and len(a)==len(labels) and np.isfinite(a).all() and np.isfinite(b).all(),
            'Invalid view scores')
    def probability(x):
        e=np.exp(x-x.max(1,keepdims=True));return e/e.sum(1,keepdims=True)
    p,q=probability(a),probability(b);m=(p+q)/2
    floor=np.finfo(np.float64).tiny
    js=.5*((p*(np.log(np.maximum(p,floor))-np.log(np.maximum(m,floor)))).sum(1)+
           (q*(np.log(np.maximum(q,floor))-np.log(np.maximum(m,floor)))).sum(1))
    pa,pb=a.argmax(1),b.argmax(1)
    class_indices=np.arange(a.shape[1])[None,:];label_indices=np.asarray(labels)[:,None]
    def label_is_top5(x):
        value=x[np.arange(len(labels)),labels][:,None]
        rank=(x>value).sum(1)+((x==value)&(class_indices<label_indices)).sum(1)
        return rank<5
    return dict(original=pa,flipped=pb,js_nats=js,disagreement=pa!=pb,
        high_instability=(pa!=pb)&(js>=threshold),either_view_top1=(pa==labels)|(pb==labels),
        either_view_top5=label_is_top5(a)|label_is_top5(b))


def verify(config,run_root):
    cfg=read_json(config);run_root=Path(run_root).resolve();report=read_json(run_root/'report.json')
    pre=read_json(run_root/'preflight.json')
    require(report['status']=='completed_verified_native_diagnostic' and
        report['config_sha256']==pre['config_sha256']==sha(config) and
        report['implementation_sha256']==pre['implementation_sha256']==sha(ROOT/'scripts/probe_v1_reflection_consistency.py') and
        report['preflight_sha256']==sha(run_root/'preflight.json'),'Frozen report/config changed')
    require(report['experiment_id']==cfg['experiment_id'] and report['primary_model']==cfg['primary_model'] and
        report['training_updates']==0 and report['original_labels_noisy'] is True and
        all(report[k] is False for k in ['test_data_used','new_candidate','automatic_training','parameter_search',
            'oracle_view_selection_is_deliverable','reflection_is_label_preserving_proven',
            'training_recoverable_gain_known','platform_gain_known']),'Scope claim changed')
    require(report['source_files']==pre['source_files'],'Sources changed after preflight')
    for p,d in report['source_files'].items():require(sha(p)==d,f'Source changed: {p}')
    require(sha(ROOT/cfg['source_config'])==cfg['source_config_sha256'] and
        sha(ROOT/'scripts/probe_v1_resolution_degradation.py')==cfg['source_helper_sha256'],'Source helper differs')
    source_cfg=read_json(ROOT/cfg['source_config']);source=read_json(ROOT/source_cfg['source_report'])
    plan=read_json(source['plan'])
    for p,d in plan['inputs'].items():require(sha(p)==d,f'Original input changed: {p}')
    parent_cfg=yaml.safe_load(Path(plan['config']['parent_config']).read_text())
    source_root=Path(parent_cfg['source']['root'])
    reference_path=source_root/parent_cfg['source']['reference_config']
    reference=yaml.safe_load(reference_path.read_text())
    manifest_path=(reference_path.parent/reference['data']['dataset_manifest']).resolve()
    manifest=read_json(manifest_path);stage=manifest_path.parent
    require(manifest['data_version']=='20260921' and manifest['stage']=='repechage' and
        sha(manifest_path)==plan['source_binding']['dataset_manifest_sha256'],'Wrong stage')
    require(sha(stage/'val_dev.csv')==manifest['files']['val_dev.csv'],'Official validation changed')
    val=read_rows(stage/'val_dev.csv');labels=np.array([int(r['label']) for r in val]);paths=np.array([r['image_path'] for r in val])
    require(len(val)==manifest['val_dev_samples']==report['validation_rows']==pre['validation_rows'] and
        report['classes']==len(plan['classes'])==manifest['num_classes'],'Population differs')
    errors=read_json(ROOT/cfg['error_report']);aligned=read_rows(errors['aligned_csv'])
    require(sha(errors['aligned_csv'])==errors['aligned_csv_sha256'] and
        [r['image_path'] for r in aligned]==paths.tolist() and [int(r['label']) for r in aligned]==labels.tolist(),'Error rows differ')
    common=np.array([all(int(row[c])!=int(row['label']) for c in errors['candidate_names']) for row in aligned])
    require(int(common.sum())==errors['candidate_error_overlap']['all']['all_candidates_wrong'],'Common error count differs')
    base=dict(all=np.ones(len(labels),bool),common_candidate_wrong=common,
        tail75=np.array([int(y) in set(plan['groups']['tail_classes']) for y in labels]),
        small=np.array([min(int(r['width']),int(r['height']))<224 for r in val]))
    require(sha(run_root/'frozen_groups.npz')==pre['frozen_groups_sha256'],'Frozen groups differ')
    frozen=np.load(run_root/'frozen_groups.npz',allow_pickle=False)
    for k,a in dict(labels=labels,image_paths=paths,**base).items():require(np.array_equal(a,frozen[k]),f'Frozen {k} differs')
    require(pre['groups']=={k:int(m.sum()) for k,m in base.items()},'Preflight group counts differ')
    numerical_errors={};decisions={};metrics_checked=paired_checked=0
    for name in cfg['models']:
        file=run_root/(name+'.npz');require(report['artifacts'][str(file)]==sha(file),f'{name} scores changed')
        z=np.load(file,allow_pickle=False)
        require(np.array_equal(z['labels'],labels) and np.array_equal(z['image_paths'],paths),'View population differs')
        a,b=z['original_logits'],z['flipped_logits'];require(a.shape==(len(labels),len(plan['classes'])),'View class axis differs')
        native=((a+b)/2).argmax(1)
        native_file='baseline.npz' if name=='native_parent' else 'val_epoch04_ema_swa_2_4.npz'
        original_path=next(p for p in source['artifacts'] if Path(p).name==native_file)
        require(sha(original_path)==source['artifacts'][original_path],'Archived native predictions changed')
        archived=np.load(original_path,allow_pickle=False)
        require(np.array_equal(archived['image_paths'],paths) and np.array_equal(archived['labels'],labels) and
            np.array_equal(archived['predictions'],native) and np.array_equal(native,z['native']),'Full native replay differs')
        s=replay_views(a,b,labels,cfg['js_threshold']);masks=dict(base,disagreement=s['disagreement'],
            high_instability=s['high_instability'],low_instability=~s['high_instability'],
            common_wrong_high_instability=base['common_candidate_wrong']&s['high_instability'])
        numerical_errors[name]=float(np.max(np.abs(s['js_nats']-z['js_nats'])))
        require(numerical_errors[name]<1e-12,'Independent JS differs')
        for k,array in dict(**{k:v for k,v in s.items() if k!='js_nats'},**{k:v for k,v in masks.items() if k not in s}).items():
            require(np.array_equal(array,z[k]),f'{name}/{k}: score/group differs')
        expected_keys={'labels','image_paths','original_logits','flipped_logits','native'}|set(s)|set(masks)
        require(set(z.files)==expected_keys,'Unexpected view columns')
        validation={};recorded=report['models'][name]
        for group,mask in masks.items():
            actual=dict(native=counts(labels,native,mask),original=counts(labels,s['original'],mask),
                flipped=counts(labels,s['flipped'],mask),original_vs_native=compare_pair(labels,native,s['original'],mask),
                flipped_vs_native=compare_pair(labels,native,s['flipped'],mask))
            for k,v in actual.items():equal_counts(recorded['validation'][group][k],v,f'{name}/{group}/{k}')
            metrics_checked+=3;paired_checked+=2;validation[group]=actual
        wrong=native!=labels;high=s['high_instability']
        aggregate=dict(disagreement_rows=int(s['disagreement'].sum()),high_instability_rows=int(high.sum()),
            js_mean_nats=float(s['js_nats'].mean()),js_max_nats=float(s['js_nats'].max()),
            native_errors_either_view_top1=int((wrong&s['either_view_top1']).sum()),
            native_errors_either_view_top5=int((wrong&s['either_view_top5']).sum()),
            high_instability_native_errors_either_view_top1=int((wrong&high&s['either_view_top1']).sum()),
            high_instability_native_errors_either_view_top5=int((wrong&high&s['either_view_top5']).sum()),
            common_wrong_high_instability_either_view_top5=int((common&high&s['either_view_top5']).sum()))
        equal_counts(recorded['aggregate'],aggregate,f'{name}/aggregate')
        target=validation['high_instability']['native'];g=cfg['mechanism_gate']
        excess=validation['all']['native']['micro']-target['micro'] if target['rows'] else None
        passed=bool(aggregate['high_instability_rows']>=g['high_instability_rows_min'] and
            target['errors']>=g['high_instability_native_errors_min'] and excess is not None and
            excess>=g['error_rate_excess_over_all_min'] and
            aggregate['high_instability_native_errors_either_view_top5']>=g['high_instability_native_errors_either_view_top5_min'])
        decision='supports_one_bounded_reflection_consistency_review' if passed else 'close_fixed_reflection_instability_entry'
        require(recorded['mechanism_gate_passed'] is passed and recorded['decision']==decision,'Model mechanism gate differs')
        value=recorded['high_instability_error_rate_excess_over_all']
        require(value is None if excess is None else abs(value-excess)<1e-12,'Error excess differs')
        require(report['native_parity'][name]==dict(rows=cfg['cold_rows'],matches=cfg['cold_rows']),'Cold record differs')
        decisions[name]=(passed,decision)
    require(report['full_native_predictions_replayed']==2*len(labels) and
        (report['mechanism_gate_passed'],report['decision'])==decisions[cfg['primary_model']],'Primary gate differs')
    incumbent=read_json(ROOT/cfg['incumbent_reference']);require(report['incumbent']==incumbent['artifacts'],'Incumbent identity differs')
    return dict(status='passed_independent_numpy64_counter_replay',experiment_id=cfg['experiment_id'],
        native_predictions_replayed_from_saved_views=2*len(labels),orientation_predictions_replayed=4*len(labels),
        js_rows_replayed=2*len(labels),max_js_errors=numerical_errors,cold_native_match_records_checked=2*cfg['cold_rows'],
        metric_groups_checked=metrics_checked,paired_groups_checked=paired_checked,source_files_checked=len(report['source_files']),
        group_and_top5_membership_identical=True,mechanism_gate_independently_verified=True,
        report_sha256=sha(run_root/'report.json'),verifier_sha256=sha(__file__),decision=report['decision'],
        training_updates=0,new_candidate=False,platform_gain_known=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-root',required=True)
    p.add_argument('--output',required=True);args=p.parse_args();require(not Path(args.output).exists(),'Use a fresh verification path')
    result=verify(args.config,args.run_root);Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
