"""Measure reflection instability of two archived DEV models, without training."""
from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_pipeline import Images, image_transform, seed_training
from aegis_clip.v1_strategy import load_trainable_state
from v1_continuation.runtime import initial_model, require_idle_cuda
from probe_v1_resolution_degradation import inspect as inspect_source
from diagnose_v1_candidate_errors import require, read_json, read_rows, metrics, paired

ROOT=Path(__file__).resolve().parents[1]


def reflection_statistics(original,flipped,labels,threshold=.1):
    a=torch.as_tensor(original,dtype=torch.float64)
    b=torch.as_tensor(flipped,dtype=torch.float64)
    require(a.ndim==2 and a.shape==b.shape and len(a)==len(labels) and a.shape[1]>=5 and
            torch.isfinite(a).all() and torch.isfinite(b).all(),'Invalid paired view logits')
    log_a,log_b=a.log_softmax(1),b.log_softmax(1)
    log_m=torch.logaddexp(log_a,log_b)-math.log(2.)
    js=(.5*((log_a.exp()*(log_a-log_m)).sum(1)+(log_b.exp()*(log_b-log_m)).sum(1))).numpy()
    pred_a,pred_b=a.argmax(1).numpy(),b.argmax(1).numpy()
    different=pred_a!=pred_b
    labels=np.asarray(labels)
    top5=((np.argsort(-a.numpy(),axis=1,kind='stable')[:,:5]==labels[:,None]).any(1)|
          (np.argsort(-b.numpy(),axis=1,kind='stable')[:,:5]==labels[:,None]).any(1))
    return dict(js_nats=js,original=pred_a,flipped=pred_b,disagreement=different,
        high_instability=different&(js>=threshold),either_view_top1=(pred_a==labels)|(pred_b==labels),
        either_view_top5=top5)


def gate_decision(cfg,aggregate,validation):
    g=cfg['mechanism_gate']; target=validation['high_instability']['native']
    excess=validation['all']['native']['micro']-target['micro'] if target['rows'] else None
    passed=bool(aggregate['high_instability_rows']>=g['high_instability_rows_min'] and
        target['errors']>=g['high_instability_native_errors_min'] and excess is not None and
        excess>=g['error_rate_excess_over_all_min'] and
        aggregate['high_instability_native_errors_either_view_top5']>=g['high_instability_native_errors_either_view_top5_min'])
    return passed,excess,'supports_one_bounded_reflection_consistency_review' if passed else 'close_fixed_reflection_instability_entry'


def inspect(config):
    cfg=read_json(config)
    require(cfg['models']==['native_parent','lr512_swa'] and cfg['primary_model']=='lr512_swa' and
        cfg['decoder']==dict(input_size=512,scales=[512],flip=True,bias=None) and
        cfg['js_threshold']==.1 and cfg['high_instability_rule']=='different_top1_and_jensen_shannon_nats_at_least_0.1' and
        cfg['batch_size']==32 and cfg['workers']==2 and cfg['cold_rows']==64 and cfg['seed']==42 and
        cfg['training_updates']==0 and all(cfg[k] is False for k in ['test_data_used','automatic_training','parameter_search','new_candidate']),
        'Fixed diagnostic protocol changed')
    require(sha256_file(ROOT/cfg['source_config'])==cfg['source_config_sha256'] and
        sha256_file(ROOT/'scripts/probe_v1_resolution_degradation.py')==cfg['source_helper_sha256'],'Archived source helper changed')
    _,plan,context,payload,native,labels,paths,_,files=inspect_source(ROOT/cfg['source_config'])
    for p,d in plan['inputs'].items():require(sha256_file(p)==d,f'Original input changed: {p}'); files[p]=d
    files[str(ROOT/cfg['source_config'])]=cfg['source_config_sha256']
    files[str(ROOT/'scripts/probe_v1_resolution_degradation.py')]=cfg['source_helper_sha256']
    error_path=ROOT/cfg['error_report'];errors=read_json(error_path);files[str(error_path)]=sha256_file(error_path)
    aligned_path=Path(errors['aligned_csv']);require(sha256_file(aligned_path)==errors['aligned_csv_sha256'],'Error alignment changed')
    files[str(aligned_path)]=errors['aligned_csv_sha256'];aligned=read_rows(aligned_path)
    require([r['image_path'] for r in aligned]==paths.tolist() and [int(r['label']) for r in aligned]==labels.tolist(),
            'Error population differs')
    common=np.array([all(int(r[k])!=int(r['label']) for k in errors['candidate_names']) for r in aligned])
    require(int(common.sum())==errors['candidate_error_overlap']['all']['all_candidates_wrong'],'Common-error budget differs')
    groups=dict(all=np.ones(len(labels),bool),common_candidate_wrong=common,
        tail75=np.isin(labels,plan['groups']['tail_classes']),
        small=np.array([min(int(r['width']),int(r['height']))<224 for r in context.val]))
    incumbent_path=ROOT/cfg['incumbent_reference'];incumbent=read_json(incumbent_path)
    files[str(incumbent_path)]=sha256_file(incumbent_path)
    for k in ['csv','zip']:
        p=incumbent['artifacts'][k]['path'];d=incumbent['artifacts'][k]['sha256']
        require(sha256_file(p)==d,'Incumbent artifact differs');files[p]=d
    with zipfile.ZipFile(incumbent['artifacts']['zip']['path']) as z:
        require(z.namelist()==['pred_results.csv'] and z.read('pred_results.csv')==Path(incumbent['artifacts']['csv']['path']).read_bytes(),
                'Incumbent ZIP/CSV differ')
    check_log=ROOT/incumbent['submission_check_log'];files[str(check_log)]=sha256_file(check_log)
    require(incumbent['submission_rows']==37444 and incumbent['submission_checks_passed']==9,'Incumbent evidence incomplete')
    return cfg,plan,context,payload,native,labels,paths,groups,files,incumbent


def prepare(config,output):
    cfg,plan,context,payload,native,labels,paths,groups,files,incumbent=inspect(config)
    model,zero=initial_model(plan,context);load_trainable_state(model,payload['selected_state']);model.eval()
    images=torch.stack([Images(context.train_root,context.val[:2],image_transform(512))[i][0] for i in range(2)])
    require(torch.equal(images,images.flip(3).flip(3)),'Reflection does not preserve exact image tensors')
    with torch.no_grad():a,b=model(images),model(images.flip(3))
    require(torch.isfinite(a).all() and torch.isfinite(b).all(),'Nonfinite CPU forward')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'frozen_groups.npz',labels=labels,image_paths=paths,**groups)
    pre=dict(status='prepared_verified_cpu',config_sha256=sha256_file(config),implementation_sha256=sha256_file(__file__),
        source_files=files,frozen_groups_sha256=sha256_file(out/'frozen_groups.npz'),validation_rows=len(labels),
        groups={g:int(m.sum()) for g,m in groups.items()},zero_update=zero,cpu_images=2,cpu_forward_finite=True,
        reflection_tensor_involution=True,frozen_before_gpu_output=True,training_updates=0)
    atomic_json_dump(pre,out/'preflight.json');print(json.dumps(pre),flush=True)


@torch.no_grad()
def predict_views(model,context,rows,cfg,out,phase):
    data=DataLoader(Images(context.train_root,rows,image_transform(512)),batch_size=cfg['batch_size'],
                    num_workers=cfg['workers'],shuffle=False)
    a,b=torch.empty(len(rows),len(context.classes)),torch.empty(len(rows),len(context.classes))
    model.eval();started=time.monotonic();last=0
    for batch,(images,indices) in enumerate(data,1):
        images=images.to('cuda');x,y=model(images),model(images.flip(3))
        require(torch.isfinite(x).all() and torch.isfinite(y).all(),'Nonfinite view logits')
        a[indices],b[indices]=x.cpu(),y.cpu();now=time.monotonic()
        if batch==1 or batch==len(data) or now-last>=30:
            progress=dict(status='evaluating',phase=phase,batch=batch,batches=len(data),rows=len(rows),elapsed_seconds=now-started)
            atomic_json_dump(progress,out/'progress.json');print(json.dumps(progress),flush=True);last=now
    require(all(p.grad is None for p in model.parameters()),'Diagnostic computed gradients')
    return a.numpy(),b.numpy()


def summarize(a,b,native,labels,classes,groups,cfg):
    stats=reflection_statistics(a,b,labels,cfg['js_threshold'])
    combined=((a+b)/2).argmax(1)
    require(np.array_equal(combined,native),'Full native checkpoint replay differs')
    masks=dict(groups,disagreement=stats['disagreement'],high_instability=stats['high_instability'],
        low_instability=~stats['high_instability'],common_wrong_high_instability=groups['common_candidate_wrong']&stats['high_instability'])
    validation={g:dict(native=metrics(labels,native,classes,m),original=metrics(labels,stats['original'],classes,m),
        flipped=metrics(labels,stats['flipped'],classes,m),original_vs_native=paired(labels,native,stats['original'],m),
        flipped_vs_native=paired(labels,native,stats['flipped'],m)) for g,m in masks.items()}
    wrong=native!=labels;high=stats['high_instability']
    aggregate=dict(disagreement_rows=int(stats['disagreement'].sum()),high_instability_rows=int(high.sum()),
        js_mean_nats=float(stats['js_nats'].mean()),js_max_nats=float(stats['js_nats'].max()),
        native_errors_either_view_top1=int((wrong&stats['either_view_top1']).sum()),
        native_errors_either_view_top5=int((wrong&stats['either_view_top5']).sum()),
        high_instability_native_errors_either_view_top1=int((wrong&high&stats['either_view_top1']).sum()),
        high_instability_native_errors_either_view_top5=int((wrong&high&stats['either_view_top5']).sum()),
        common_wrong_high_instability_either_view_top5=int((groups['common_candidate_wrong']&high&stats['either_view_top5']).sum()))
    passed,excess,decision=gate_decision(cfg,aggregate,validation)
    return dict(aggregate=aggregate,validation=validation,mechanism_gate_passed=passed,
                high_instability_error_rate_excess_over_all=excess,decision=decision),stats,masks


def run(config,output):
    cfg,plan,context,payload,native,labels,paths,groups,files,incumbent=inspect(config);out=Path(output).resolve()
    pre=read_json(out/'preflight.json')
    require(pre['config_sha256']==sha256_file(config) and pre['implementation_sha256']==sha256_file(__file__) and
        pre['source_files']==files and pre['frozen_groups_sha256']==sha256_file(out/'frozen_groups.npz'),'Preflight changed')
    require(not (out/'status.json').exists(),'No implicit restart or overwrite');require_idle_cuda();seed_training(cfg['seed'])
    started=time.monotonic();results={};parity={};artifacts={}
    probe=np.sort(np.random.default_rng(cfg['seed']).choice(len(labels),cfg['cold_rows'],replace=False))
    atomic_json_dump(dict(status='running',models=cfg['models'],training_updates=0),out/'status.json')
    try:
        for name in cfg['models']:
            model,_=initial_model(plan,context)
            if name=='lr512_swa':load_trainable_state(model,payload['selected_state'])
            model.to('cuda').eval()
            cold_a,cold_b=predict_views(model,context,[context.val[i] for i in probe],cfg,out,name+'_cold')
            matches=int((((cold_a+cold_b)/2).argmax(1)==native[name][probe]).sum())
            require(matches==len(probe),'Native cold replay differs');parity[name]=dict(rows=len(probe),matches=matches)
            a,b=predict_views(model,context,context.val,cfg,out,name+'_full_views')
            results[name],stats,masks=summarize(a,b,native[name],labels,len(context.classes),groups,cfg)
            np.savez_compressed(out/(name+'.npz'),labels=labels,image_paths=paths,original_logits=a,flipped_logits=b,
                native=native[name],**{k:v for k,v in stats.items() if k not in masks},**masks)
            artifacts[str(out/(name+'.npz'))]=sha256_file(out/(name+'.npz'));del model;torch.cuda.empty_cache()
        primary=results[cfg['primary_model']]
        report=dict(status='completed_verified_native_diagnostic',experiment_id=cfg['experiment_id'],validation_rows=len(labels),
            classes=len(context.classes),models=results,native_parity=parity,full_native_predictions_replayed=2*len(labels),
            config_sha256=sha256_file(config),implementation_sha256=sha256_file(__file__),source_files=files,artifacts=artifacts,
            preflight_sha256=sha256_file(out/'preflight.json'),mechanism_gate_passed=primary['mechanism_gate_passed'],
            decision=primary['decision'],primary_model=cfg['primary_model'],training_updates=0,test_data_used=False,
            new_candidate=False,automatic_training=False,parameter_search=False,original_labels_noisy=True,
            oracle_view_selection_is_deliverable=False,reflection_is_label_preserving_proven=False,
            training_recoverable_gain_known=False,platform_gain_known=False,incumbent=incumbent['artifacts'],
            elapsed_seconds=time.monotonic()-started,peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated())
        atomic_json_dump(report,out/'report.json');atomic_json_dump(dict(status=report['status'],report=str(out/'report.json')),out/'status.json')
        print(json.dumps(dict(status=report['status'],decision=report['decision'],elapsed_seconds=report['elapsed_seconds'],
            models={k:v['aggregate'] for k,v in results.items()})),flush=True)
    except Exception as error:
        atomic_json_dump(dict(status='failed',error=str(error),automatically_restarted=False),out/'status.json');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.action=='run' and not args.execute:parser.error('CUDA diagnostic requires --execute')
    torch.set_num_threads(2);(prepare if args.action=='prepare' else run)(args.config,args.output)
