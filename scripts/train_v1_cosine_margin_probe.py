"""One fixed 1024-update cosine-margin candidate against a delivered control."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

import train_v1_detail_aug_pair as base
import v1_cosine_margin_loss as margin_loss
from aegis_clip.runtime import atomic_json_dump,sha256_file
from aegis_clip.v1_pipeline import save_artifact,seed_training
from aegis_clip.v1_pipeline import load_artifact
from aegis_clip.v1_strategy import load_trainable_state
from aegis_clip.submission import create_submission
from v1_continuation.runtime import initial_model
from aegis_clip.v1_strategy import trainable_state
from v1_continuation.runtime import evaluate,require_idle_cuda
from diagnose_v1_candidate_errors import require,read_json,metrics,paired,load_aligned_npz

ROOT=Path(__file__).resolve().parents[1]


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def inspect(config):
    cfg=read_json(config)
    require(cfg['cosine_margin']==.05 and cfg['steps']==1024 and cfg['evaluation_steps']==[512,1024] and
        cfg['cost_steps']==8 and cfg['batch_size']==32 and cfg['micro_batch_size']==8 and cfg['workers']==2 and
        cfg['seed']==42 and cfg['primary_checkpoint']=='last_raw_step1024' and cfg['accuracy_early_stop'] is False and
        cfg['target_rule']=='lowest_quartile_native_top1_minus_top2_logit_gap;ties_image_path_ascending' and
        cfg['test_decoder']==dict(input_size=512,scales=[512],flip=True,bias=None) and
        all(cfg[k] is False for k in ['automatic_full_training','parameter_search','platform_upload']), 'Fixed finite protocol changed')
    files={}
    def checked(path,digest=None):
        p=Path(path);d=sha256_file(p);require(digest is None or d==digest,f'Source changed: {p}')
        files[str(p.resolve())]=d;return p
    src=Path(cfg['source_control_root']);pre=read_json(checked(src/'preflight.json'))
    old=read_json(checked(src/'report.json'));public=read_json(checked(ROOT/cfg['source_control_report']))
    require(old==public and old['status']=='completed_packaged_pair' and pre['binding']==old['binding'],'Control is not delivered')
    checked(cfg['source_control_config'],pre['binding']['config_sha256'])
    checked(base.__file__,pre['binding']['implementation_sha256'])
    old_cfg,plan,ctx,parent,targets,native,labels,paths,_,sources=base.inspect(cfg['source_control_config'])
    for p,d in sources.items():checked(p,d)
    for p,d in plan['inputs'].items():checked(p,d)
    checked(src/'schedule.npz',cfg['source_schedule_sha256'])
    require(cfg['source_schedule_sha256']==pre['binding']['schedule_sha256'],'Original schedule differs')
    with np.load(src/'schedule.npz',allow_pickle=False) as z:frozen={k:z[k].copy() for k in z.files}
    expected=base.schedule(np.flatnonzero(targets['weights']>0),old_cfg)
    require(all(np.array_equal(frozen[k],v) for k,v in expected.items()),'Original schedule is not reproducible')
    idx=frozen['train_indices'];require(bool((targets['original_alpha'][idx]==0).all()),'Margin components require original hard v1 targets')
    pkg=old['arms']['control']['package']
    checked(pkg['checkpoint'],cfg['source_control_checkpoint_sha256'])
    require(pkg['checkpoint_sha256']==cfg['source_control_checkpoint_sha256'] and pkg['rows']==ctx.manifest['test_samples'] and
        pkg['checks_passed']==9 and pkg['cold_validation_matches']==64,'Control package is incomplete')
    for k in ['csv','zip']:checked(pkg[k],pkg[k+'_sha256'])
    old_check=read_json(checked(ROOT/'results/v1_detail_aug_pair_20261001/independent_verification.json'))
    val_file=src/'control/val_step1024.npz'
    checked(val_file,old_check['source_and_artifact_sha256'][str(val_file)])
    control=load_aligned_npz(val_file,paths,labels)['predictions']
    require(metrics(labels,control,len(ctx.classes),np.ones(len(labels),bool))==old['arms']['control']['history'][-1]['metrics']['all'],
            'Delivered control validation differs')
    reflection=read_json(checked(ROOT/cfg['source_reflection_report']))
    view_path=next(p for p in reflection['artifacts'] if Path(p).name=='lr512_swa.npz')
    checked(view_path,reflection['artifacts'][view_path])
    with np.load(view_path,allow_pickle=False) as z:
        require(np.array_equal(z['labels'],labels) and np.array_equal(z['image_paths'],paths) and
            np.array_equal(z['native'],native),'Margin diagnostic population differs')
        logits=(z['original_logits']+z['flipped_logits'])/2;common=z['common_candidate_wrong'].copy()
    require(np.array_equal(logits.argmax(1),native),'Native gap-source replay differs')
    top=np.partition(logits,-2,axis=1)[:,-2:];gap=top[:,1]-top[:,0]
    order=np.lexsort((paths,gap));low=np.zeros(len(labels),bool);low[order[:len(labels)//4]]=True
    groups=dict(all=np.ones(len(labels),bool),low_gap=low,other_gap=~low,
        tail75=np.isin(labels,plan['groups']['tail_classes']),
        small=np.array([min(int(r['width']),int(r['height']))<224 for r in ctx.val]),common_candidate_wrong=common)
    require(int((low&(native!=labels)).sum())>=cfg['target_parent_errors_min'],'Insufficient fixed target error budget')
    incumbent=read_json(checked(ROOT/cfg['incumbent_reference']))
    for k in ['csv','zip']:checked(incumbent['artifacts'][k]['path'],incumbent['artifacts'][k]['sha256'])
    require(incumbent['submission_checks_passed']==9,'Incumbent check reference incomplete')
    checked(ROOT/incumbent['submission_check_log'])
    return cfg,old_cfg,plan,ctx,parent,targets,native,control,labels,paths,groups,frozen,files,gap,incumbent


def binding(config,schedule,files):
    return dict(config_sha256=sha256_file(config),implementation_sha256=sha256_file(__file__),
        loss_sha256=sha256_file(margin_loss.__file__),schedule_sha256=sha256_file(schedule),parent_sources=files,
        stage='repechage',data_version='20260921',partition='train_dev')


def prepare(config,output):
    cfg,old_cfg,plan,ctx,parent,targets,native,control,labels,paths,groups,frozen,files,gap,incumbent=inspect(config)
    seed_training(cfg['seed']);model,zero=base.model_from_parent(plan,ctx,parent)
    first_images,_=next(iter(base.loader(ctx,old_cfg,frozen,'control',cfg['cost_steps'])))
    require(first_images.shape==(cfg['batch_size'],3,512,512),'Original augmentation shape differs')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(cfg['source_control_root'])/'schedule.npz',out/'schedule.npz')
    np.savez_compressed(out/'frozen_groups.npz',labels=labels,image_paths=paths,native_gap=gap,**groups)
    bind=binding(config,out/'schedule.npz',files)
    pre=dict(status='prepared_verified_cpu',binding=bind,source_plan=next(p for p in files if Path(p).name=='plan.json'),
        training_rows=len(frozen['train_indices']),training_target_classes_absent=np.flatnonzero(np.bincount(
            targets['targets'][frozen['train_indices']],minlength=len(ctx.classes))==0).tolist(),
        zero_update=zero,source_first_batch_pixel_sha256=tensor_sha(first_images),
        frozen_groups_sha256=sha256_file(out/'frozen_groups.npz'),validation_is_independent=True,
        target_group_rule=cfg['target_rule'],original_targets_weights_and_sampler_unchanged=True,
        reused_control_checkpoint_sha256=cfg['source_control_checkpoint_sha256'],
        parent={g:metrics(labels,native,len(ctx.classes),m) for g,m in groups.items()},
        reused_control={g:metrics(labels,control,len(ctx.classes),m) for g,m in groups.items()},
        primary_checkpoint=cfg['primary_checkpoint'],wall_clock_time_limit=None)
    atomic_json_dump(pre,out/'preflight.json');print(json.dumps(dict(status=pre['status'],zero_update=zero,
        target=pre['parent']['low_gap'],training_rows=pre['training_rows'])),flush=True)


def frozen_context(config,output):
    values=inspect(config);out=Path(output).resolve();pre=read_json(out/'preflight.json')
    require(pre['binding']==binding(config,out/'schedule.npz',values[12]) and
        pre['frozen_groups_sha256']==sha256_file(out/'frozen_groups.npz') and
        sha256_file(out/'schedule.npz')==values[0]['source_schedule_sha256'],'Frozen inputs changed')
    return values,pre


def update(model,opt,scheduler,probabilities,weights,scaler,images,indices,step,frozen,cfg):
    selected=indices.to('cuda');perm=torch.tensor(frozen['mixup_permutations'][step-1],device='cuda')
    loss,numeric=margin_loss.logical_update(model,opt,images.to('cuda'),probabilities[selected],weights[selected],
        perm,float(frozen['mixup_lambdas'][step-1]),cfg['micro_batch_size'],scaler,cfg['cosine_margin'],1.)
    scheduler.step();return loss,numeric


def cost(config,output):
    values,pre=frozen_context(config,output)
    cfg,old_cfg,plan,ctx,parent,targets,native,control,labels,paths,groups,frozen,files,gap,incumbent=values
    out=Path(output).resolve();require(not (out/'cost.json').exists(),'No repeated cost probe');require_idle_cuda()
    seed_training(cfg['seed']);torch.cuda.reset_peak_memory_stats();model,_=base.model_from_parent(plan,ctx,parent)
    model.to('cuda').train();opt,scheduler,probabilities,weights,scaler=base.learning_state(model,old_cfg,targets,frozen)
    elapsed=[];retries=0
    for step,(images,indices) in enumerate(base.loader(ctx,old_cfg,frozen,'control',cfg['cost_steps']),1):
        if step==1:require(tensor_sha(images)==pre['source_first_batch_pixel_sha256'],'Common augmentation pixels changed')
        torch.cuda.synchronize();start=time.monotonic()
        loss,numeric=update(model,opt,scheduler,probabilities,weights,scaler,images,indices,step,frozen,cfg)
        torch.cuda.synchronize();elapsed.append(time.monotonic()-start);retries+=len(numeric['amp_retries'])
        print(json.dumps(dict(status='cost',update=step,loss=loss,seconds=elapsed[-1])),flush=True)
    record=dict(status='passed_discarded_probe_weights',real_updates=len(elapsed),seconds=elapsed,
        steady_seconds_per_update=float(np.mean(elapsed[4:])),amp_retries=retries,
        peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(),common_augmentation_pixel_hash_matches=True,
        formal_initialization='original LR512 EMA2-4; cost states discarded',wall_clock_time_limit=None)
    atomic_json_dump(record,out/'cost.json');print(json.dumps(record),flush=True)


def summary(labels,native,control,prediction,groups,classes):
    return dict(metrics={g:metrics(labels,prediction,classes,m) for g,m in groups.items()},
        paired_to_parent={g:paired(labels,native,prediction,m) for g,m in groups.items()},
        paired_to_control={g:paired(labels,control,prediction,m) for g,m in groups.items()})


def review(cfg,item,control_macro):
    g=cfg['review_gate'];all_pair=item['paired_to_control']['all']
    return bool(all_pair['net']>=g['net_vs_control_min'] and
        item['paired_to_parent']['all']['net']>=g['net_vs_parent_min'] and
        item['paired_to_control']['low_gap']['net']>=g['target_net_vs_control_min'] and
        100*(item['metrics']['all']['macro']-control_macro)>=g['macro_delta_pp_vs_control_min'] and
        all_pair['corrections']>=g['corrections_to_regressions_min']*all_pair['regressions'])


def package(plan,ctx,model,checkpoint,directory,prediction,bind):
    selected=load_artifact(checkpoint,bind)
    require(selected['complete'] is True and selected['optimizer_updates']==1024 and selected['primary']=='last_raw' and
        selected['cosine_margin']==.05 and selected['classes']==ctx.classes and selected['bias'] is None,'Incomplete primary checkpoint')
    cold,_=initial_model(plan,ctx);load_trainable_state(cold,selected['selected_state']);cold.to('cuda').eval()
    probe=np.sort(np.random.default_rng(42).choice(len(ctx.val),64,replace=False))
    parity=evaluate(plan,ctx,cold,[ctx.val[i] for i in probe],'cuda').argmax(1).numpy()
    require(np.array_equal(parity,prediction[probe]),'Cold single-checkpoint replay differs')
    del cold;torch.cuda.empty_cache();model.eval()
    rows=base.read_rows(Path(ctx.reference['data']['dataset_manifest']).parent/'test_manifest.csv')
    names=[Path(r['image_path']).name for r in rows]
    require(len(names)==ctx.manifest['test_samples'] and set(names)=={p.name for p in ctx.test_root.iterdir() if p.is_file()},
        'Official test coverage differs')
    logits=evaluate(plan,ctx,model,rows,'cuda',ctx.test_root,progress_path=directory/'inference_progress.json',phase='native512_flip_no_margin')
    np.savez_compressed(directory/'test_predictions.npz',names=names,logits=logits.numpy())
    submit=directory/'submission'
    create_submission([(name,ctx.classes[i]) for name,i in zip(names,logits.argmax(1).tolist())],names,submit,checkpoint,
        inference_mode='v1_cosine_margin_probe_native512_flip',tta_risk_acknowledged=True,valid_labels=set(ctx.classes),
        space_after_comma=True,extra_manifest=dict(binding=bind,decoder=dict(input_size=512,scales=[512],flip=True,bias=None),
            training_updates=1024,primary='last_raw',cosine_margin_training_only=.05,margin_applied_at_inference=False))
    checked=subprocess.run([sys.executable,str(ROOT/'scripts/check_submission.py'),'--test_dir',str(ctx.test_root),
        '--class-mapping',ctx.reference['data']['class_mapping'],'--csv',str(submit/'pred_results.csv'),
        '--zip',str(submit/'submission.zip')],text=True,capture_output=True)
    (submit/'submission_check.log').write_text(checked.stdout+checked.stderr);checked.check_returncode()
    with zipfile.ZipFile(submit/'submission.zip') as archive:
        require(archive.namelist()==['pred_results.csv'] and archive.read('pred_results.csv')==(submit/'pred_results.csv').read_bytes(),
            'ZIP/external CSV differ')
    return dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256_file(checkpoint),csv=str(submit/'pred_results.csv'),
        zip=str(submit/'submission.zip'),csv_sha256=sha256_file(submit/'pred_results.csv'),zip_sha256=sha256_file(submit/'submission.zip'),
        rows=len(names),checks_passed=9,cold_validation_matches=len(probe),zip_csv_identical=True,margin_applied_at_inference=False)


def run(config,output):
    values,pre=frozen_context(config,output)
    cfg,old_cfg,plan,ctx,parent,targets,native,control,labels,paths,groups,frozen,files,gap,incumbent=values
    out=Path(output).resolve();cost_report=read_json(out/'cost.json')
    require(cost_report['status']=='passed_discarded_probe_weights' and cost_report['real_updates']==8,'Real cost check missing')
    require(not (out/'status.json').exists(),'No implicit formal restart');require_idle_cuda()
    bind=pre['binding'];started=time.monotonic();seed_training(cfg['seed'])
    model,_=base.model_from_parent(plan,ctx,parent);model.to('cuda').train()
    opt,scheduler,probabilities,weights,scaler=base.learning_state(model,old_cfg,targets,frozen)
    directory=out/'candidate';directory.mkdir(exist_ok=False);history=[];retries=[];last_report=0.;losses=[]
    atomic_json_dump(dict(status='training',updates=cfg['steps'],reused_control=True),out/'status.json')
    try:
        for step,(images,indices) in enumerate(base.loader(ctx,old_cfg,frozen,'control',cfg['steps']),1):
            model.train()
            if step==1:require(tensor_sha(images)==pre['source_first_batch_pixel_sha256'],'Formal common augmentation pixels differ')
            loss,numeric=update(model,opt,scheduler,probabilities,weights,scaler,images,indices,step,frozen,cfg);losses.append(loss)
            if numeric['amp_retries']:retries.append(dict(step=step,**numeric))
            now=time.monotonic()
            if step==1 or now-last_report>=30 or step in cfg['evaluation_steps']:
                progress=dict(status='training',update=step,updates=cfg['steps'],loss=float(np.mean(losses[-32:])),elapsed_seconds=now-started)
                atomic_json_dump(progress,out/'progress.json');print(json.dumps(progress),flush=True);last_report=now
            if step in cfg['evaluation_steps']:
                checkpoint=directory/('selected.pt' if step==cfg['steps'] else f'step_{step:04d}.pt')
                save_artifact(checkpoint,dict(selected_state=trainable_state(model),classes=ctx.classes,primary='last_raw',
                    complete=step==cfg['steps'],optimizer_updates=step,image_size=512,bias=None,arm='cosine_margin',cosine_margin=.05,
                    margin_applied_at_inference=False,parent_checkpoint_sha256=sha256_file(next(p for p in files if Path(p).name=='ema_swa.pt'))),bind)
                logits=evaluate(plan,ctx,model,ctx.val,'cuda',progress_path=out/'evaluation_progress.json',phase=f'candidate_step{step}')
                prediction=logits.argmax(1).numpy()
                np.savez_compressed(directory/f'val_step{step:04d}.npz',labels=labels,image_paths=paths,predictions=prediction,logits=logits.numpy())
                item=dict(step=step,**summary(labels,native,control,prediction,groups,len(ctx.classes)))
                history.append(item);atomic_json_dump(history,directory/'history.json');print(json.dumps(dict(status='evaluated',step=step,
                    all=item['metrics']['all'],control_pair=item['paired_to_control']['all'],target_pair=item['paired_to_control']['low_gap'])),flush=True)
                model.train()
        require(step==cfg['steps'],'Incomplete fixed candidate schedule')
        atomic_json_dump(retries,directory/'amp_retries.json');del opt,scheduler,probabilities,weights,scaler
        atomic_json_dump(dict(status='packaging',completed_updates=step),out/'status.json')
        delivered=package(plan,ctx,model,checkpoint,directory,prediction,bind)
        passed=review(cfg,history[-1],pre['reused_control']['all']['macro'])
        report=dict(status='completed_packaged_single_candidate',experiment_id=cfg['experiment_id'],binding=bind,
            optimizer_updates=step,read_only_control=True,control_training_repeated=False,cosine_margin=.05,
            training_population='32768 original frozen active train_dev rows without replacement',validation_rows=len(labels),
            primary_checkpoint=cfg['primary_checkpoint'],history=history,package=delivered,reused_control=pre['reused_control'],
            reused_control_package=read_json(ROOT/cfg['source_control_report'])['arms']['control']['package'],
            source_first_batch_pixel_sha256=pre['source_first_batch_pixel_sha256'],common_augmentation_pixel_hash_matches=True,
            amp_retry_batches=len(retries),supports_review=passed,
            decision='supports_bounded_margin_review' if passed else 'close_fixed_cosine_margin_recipe',
            automatic_full=False,parameter_search=False,platform_upload=False,platform_gain_known=False,original_labels_noisy=True,
            accuracy_early_stop=False,margin_applied_at_inference=False,incumbent=incumbent['artifacts'],
            elapsed_seconds=time.monotonic()-started)
        atomic_json_dump(report,out/'report.json');atomic_json_dump(dict(status=report['status'],report=str(out/'report.json')),out/'status.json')
        print(json.dumps(dict(status=report['status'],decision=report['decision'],elapsed_seconds=report['elapsed_seconds'])),flush=True)
    except Exception as error:
        atomic_json_dump(dict(status='failed',error=str(error),automatically_restarted=False),out/'status.json');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','cost','run']);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
    if args.action!='prepare' and not args.execute:p.error('CUDA requires --execute')
    torch.set_num_threads(2);dict(prepare=prepare,cost=cost,run=run)[args.action](args.config,args.output)
