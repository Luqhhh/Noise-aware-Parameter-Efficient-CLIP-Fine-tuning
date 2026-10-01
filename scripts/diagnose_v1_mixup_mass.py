"""Quantify fixed weighted Mixup exposure, without changing weights or targets."""
from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from diagnose_v1_candidate_errors import require, sha, read_json, read_rows, load_aligned_npz, metrics, paired

ROOT=Path(__file__).resolve().parents[1]


def effective_fraction(lam, weight_a, weight_b):
    a,b=np.asarray(weight_a,dtype=np.float64),np.asarray(weight_b,dtype=np.float64)
    require(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(a>0) and np.all(b>0), 'Active reliability must be finite and positive')
    return lam*a/(lam*a+(1-lam)*b)


def select_classes(owner, event, classes):
    exposure=np.bincount(owner,minlength=classes)
    events=np.bincount(owner[event],minlength=classes)
    rate=np.divide(events,exposure,out=np.zeros(classes,dtype=float),where=exposure>0)
    present=np.flatnonzero(exposure>0)
    order=present[np.lexsort((present,-rate[present]))]
    selected=order[:max(1,len(present)//4)]
    return exposure,events,rate,selected


def source(config):
    cfg=read_json(config); files={}
    require(cfg['steps']==1024 and cfg['batch_size']==32 and cfg['mixup_alpha']==.2 and cfg['label_smoothing']==.1 and
            cfg['displacement_event_min']==.1 and all(cfg[k] is False for k in ['new_training','gpu_used','new_submission',
                'test_data_used','automatic_training','parameter_search']), 'Fixed CPU protocol changed')
    def check(path,digest=None):
        p=Path(path); actual=sha(p);require(digest is None or actual==digest,f'Source checksum changed: {p}')
        files[str(p.resolve())]=actual;return p
    check(config);check(__file__)
    control=Path(cfg['source_control_root']); pre=read_json(check(control/'preflight.json'))
    report=read_json(check(control/'report.json')); public=read_json(check(ROOT/cfg['source_control_report']))
    require(report==public and report['status']=='completed_packaged_pair' and report['arms']['control']['optimizer_updates']==1024,
            'Only the delivered complete control may be reused')
    require(pre['binding']==report['binding'] and pre['binding']['partition']=='train_dev' and pre['binding']['data_version']=='20260921', 'Control binding differs')
    check(cfg['source_control_config'],pre['binding']['config_sha256'])
    check(cfg['source_control_implementation'],pre['binding']['implementation_sha256'])
    for p,d in pre['binding']['parent_sources'].items(): check(p,d)
    plan=read_json(pre['source_plan'])
    for p,d in plan['inputs'].items(): check(p,d)
    stage=Path(cfg['stage_root']);manifest=read_json(check(stage/'dataset_manifest.json',plan['source_binding']['dataset_manifest_sha256']))
    require(manifest['stage']=='repechage' and manifest['data_version']=='20260921', 'Wrong stage')
    for filename in ['train_dev.csv','val_dev.csv','class_to_idx.json']:check(stage/filename,manifest['files'][filename])
    mapping=read_json(stage/'class_to_idx.json');names=[name for name,i in sorted(mapping.items(),key=lambda pair:pair[1])]
    require(sorted(mapping.values())==list(range(len(names))), 'Class indices differ')
    train,val=read_rows(stage/'train_dev.csv'),read_rows(stage/'val_dev.csv')
    labels=np.array([int(r['label']) for r in val]);paths=np.array([r['image_path'] for r in val])
    require(len(val)==manifest['val_dev_samples'] and not {r['content_group'] for r in train}&{r['content_group'] for r in val}, 'Invalid independent holdout')
    target_path=Path(plan['config']['targets']);target_meta=read_json(target_path.with_suffix('.sha256.json'))
    targets=torch.load(target_path,map_location='cpu',weights_only=False)
    require(target_meta['binding']==targets['binding'] and target_meta['binding']['dataset_manifest_sha256']==sha(stage/'dataset_manifest.json') and
            targets['class_names']==names and targets['image_paths']==[r['image_path'] for r in train], 'Target population/binding differs')
    schedule_file=check(control/'schedule.npz',cfg['schedule_sha256'])
    require(cfg['schedule_sha256']==pre['binding']['schedule_sha256'], 'Control schedule changed')
    with np.load(schedule_file,allow_pickle=False) as a: frozen={key:a[key].copy() for key in a.files}
    idx=frozen['train_indices'];active=np.flatnonzero(targets['weights'].numpy()>0)
    require(np.array_equal(idx,np.random.default_rng(42).permutation(active)[:32768]) and len(np.unique(idx))==32768, 'Sampling differs')
    require(np.array_equal(frozen['mixup_lambdas'],np.random.default_rng(44).beta(.2,.2,1024)), 'Mixup coefficients differ')
    rng=np.random.default_rng(45)
    require(np.array_equal(frozen['mixup_permutations'],np.stack([rng.permutation(32) for _ in range(1024)])), 'Mixup pairs differ')
    require(bool((targets['original_alpha'][idx]==0).all()), 'This component analysis requires the original hard v1 targets')
    parent_path=next(p for p in pre['binding']['parent_sources'] if Path(p).name=='val_epoch04_ema_swa_2_4.npz')
    parent=load_aligned_npz(parent_path,paths,labels)['predictions']
    original_verification=read_json(check(ROOT/'results/v1_detail_aug_pair_20261001/independent_verification.json'))
    control_val=control/'control/val_step1024.npz'
    raw=load_aligned_npz(check(control_val,original_verification['source_and_artifact_sha256'][str(control_val)]),paths,labels)['predictions']
    require(metrics(labels,raw,len(names),np.ones(len(labels),bool))==report['arms']['control']['history'][-1]['metrics']['all'],
            'Original delivered control validation metrics differ')
    package=report['arms']['control']['package']
    check(package['checkpoint'],cfg['source_control_checkpoint_sha256'])
    require(package['checkpoint_sha256']==cfg['source_control_checkpoint_sha256'], 'Wrong fixed control checkpoint')
    for key in ['csv','zip']:check(package[key],package[key+'_sha256'])
    require(package['rows']==manifest['test_samples'] and package['checks_passed']==9 and package['cold_validation_matches']==64, 'Control package is incomplete')
    incumbent=read_json(check(ROOT/cfg['incumbent_reference']))
    for key in ['csv','zip']:check(incumbent['artifacts'][key]['path'],incumbent['artifacts'][key]['sha256'])
    with zipfile.ZipFile(incumbent['artifacts']['zip']['path']) as archive:
        require(archive.namelist()==['pred_results.csv'] and archive.read('pred_results.csv')==
                Path(incumbent['artifacts']['csv']['path']).read_bytes(),'Incumbent ZIP/CSV differs')
    check(ROOT/incumbent['submission_check_log'])
    require(incumbent['submission_checks_passed']==9 and incumbent['submission_rows']==manifest['test_samples'], 'Incumbent evidence incomplete')
    return cfg,files,train,val,names,targets,frozen,labels,paths,parent,raw,incumbent


def analyze(config,output):
    started=time.monotonic();output=Path(output).resolve();require(not output.exists(),'Preserve existing output')
    cfg,files,train,val,names,targets,frozen,labels,paths,parent,raw,incumbent=source(config)
    idx=frozen['train_indices'].reshape(1024,32); permutations=frozen['mixup_permutations']
    weight=targets['weights'][idx].numpy().astype(np.float64); target=targets['targets'][idx].numpy()
    partner_weight=np.take_along_axis(weight,permutations,axis=1)
    partner_target=np.take_along_axis(target,permutations,axis=1)
    lam=frozen['mixup_lambdas'][:,None];effective=effective_fraction(lam,weight,partner_weight)
    displacement=np.abs(effective-lam);different=target!=partner_target
    event=different&(displacement>=cfg['displacement_event_min'])
    flip=different&((lam-.5)*(effective-.5)<0)
    owner=np.where(lam>=.5,target,partner_target).ravel()
    exposure,events,rates,cohort=select_classes(owner,event.ravel(),len(names))
    flip_counts=np.bincount(owner[flip.ravel()],minlength=len(names))
    cohort_mask=np.isin(labels,cohort)
    tail=np.lexsort((np.arange(len(names)),np.bincount([int(r['label']) for r in train],minlength=len(names))))[:len(names)//10]
    groups=dict(all=np.ones(len(labels),bool),high_displacement_classes=cohort_mask,
                other_classes=~cohort_mask,tail75=np.isin(labels,tail),small=np.array([min(int(r['width']),int(r['height']))<224 for r in val]))
    validation={g:dict(parent=metrics(labels,parent,len(names),m),control=metrics(labels,raw,len(names),m),
                          control_vs_parent=paired(labels,parent,raw,m)) for g,m in groups.items()}
    aggregate=dict(mixed_rows=weight.size,different_target_rows=int(different.sum()),displacement_events=int(event.sum()),
        displacement_fraction=float(event.mean()),different_target_dominance_flips=int(flip.sum()),
        mean_absolute_component_displacement=float(displacement.mean()),max_absolute_component_displacement=float(displacement.max()),
        mean_class_distribution_l1_vs_unweighted_mix=float((2*(1-cfg['label_smoothing'])*displacement*different).mean()),
        reliability_min=float(weight.min()),reliability_max=float(weight.max()),
        scheduled_target_classes_present=int((exposure>0).sum()),scheduled_target_classes_absent=np.flatnonzero(exposure==0).tolist(),
        cohort_class_indices=cohort.tolist())
    gate=cfg['mechanism_gate'];target_errors=validation['high_displacement_classes']['parent']['errors']
    excess=(1-validation['high_displacement_classes']['parent']['micro'])-(1-validation['all']['parent']['micro'])
    passed=bool(aggregate['displacement_fraction']>=gate['displacement_fraction_min'] and
        aggregate['different_target_dominance_flips']>=gate['different_target_dominance_flips_min'] and
        target_errors>=gate['cohort_parent_errors_min'] and excess>=gate['cohort_error_rate_excess_over_all_min'])
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'exposure.npz',train_indices=idx,image_fraction=np.broadcast_to(lam,weight.shape),
        supervision_fraction=effective,weight_a=weight,weight_b=partner_weight,target_a=target,target_b=partner_target,
        displacement_event=event,dominance_flip=flip,image_owner=owner.reshape(weight.shape),labels=labels,image_paths=paths,
        **groups)
    with (output/'class_exposure.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['class_name','class_index','scheduled_rows','displacement_events','dominance_flips','event_rate','in_cohort'])
        for i,name in enumerate(names):writer.writerow([name,i,int(exposure[i]),int(events[i]),int(flip_counts[i]),float(rates[i]),int(i in cohort)])
    report=dict(experiment_id=cfg['experiment_id'],status='completed_cpu_mechanism_diagnostic',source_files=files,
        config_sha256=sha(config),script_sha256=sha(__file__),aggregate=aggregate,validation=validation,
        cohort_error_rate_excess_over_all=float(excess),mechanism_gate_passed=passed,
        decision='supports_one_bounded_image_mass_alignment_probe' if passed else 'close_small_or_unlinked_mixup_mass_effect',
        prospective_image_fraction='lambda_i = lambda*w_i/(lambda*w_i+(1-lambda)*w_partner); original supervised mass unchanged',
        weighted_mixup_implementation_is_correct_for_current_objective=True,current_objective_declared_bug=False,
        training_started=False,new_candidate=False,automatic_training=False,parameter_search=False,test_data_used=False,
        causal_effect_on_validation_known=False,platform_gain_known=False,
        exposure_file=str(output/'exposure.npz'),exposure_sha256=sha(output/'exposure.npz'),
        class_file=str(output/'class_exposure.csv'),class_sha256=sha(output/'class_exposure.csv'),
        incumbent=incumbent['artifacts'],seconds=time.monotonic()-started)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(decision=report['decision'],aggregate={k:v for k,v in aggregate.items() if k!='cohort_class_indices'},
        cohort=validation['high_displacement_classes'],error_rate_excess=excess,seconds=report['seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(2);analyze(args.config,args.output)
