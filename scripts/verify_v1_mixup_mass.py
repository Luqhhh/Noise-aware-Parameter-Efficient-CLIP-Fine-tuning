"""Independently replay weighted fractions in Torch64 and cohorts with counters."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from diagnose_v1_mixup_mass import source
from diagnose_v1_candidate_errors import sha, require, read_json, read_rows
from verify_v1_frozen_teacher_recovery import counts, compare_pair, equal_counts

ROOT=Path(__file__).resolve().parents[1]


def verify(config,report_path):
    report=read_json(report_path)
    require(report['config_sha256']==sha(config) and report['script_sha256']==sha(ROOT/'scripts/diagnose_v1_mixup_mass.py'),
            'Frozen config/implementation changed')
    require(report['status']=='completed_cpu_mechanism_diagnostic' and
            report['weighted_mixup_implementation_is_correct_for_current_objective'] is True and
            all(report[key] is False for key in ['current_objective_declared_bug','training_started','new_candidate',
                'automatic_training','parameter_search','test_data_used','causal_effect_on_validation_known','platform_gain_known']),
            'Scope claim changed')
    for path,digest in report['source_files'].items(): require(sha(path)==digest,f'Source changed: {path}')
    cfg,files,train,val,names,targets,schedule,labels,paths,parent,raw,incumbent=source(config)
    require(report['experiment_id']==cfg['experiment_id'] and report['incumbent']==incumbent['artifacts'],'Identity differs')
    for key in ['exposure','class']:require(sha(report[key+'_file'])==report[key+'_sha256'],f'{key} checksum differs')
    cache=np.load(report['exposure_file'],allow_pickle=False)
    indices=torch.tensor(schedule['train_indices'].reshape(cfg['steps'],cfg['batch_size']))
    perm=torch.tensor(schedule['mixup_permutations']); lam=torch.tensor(schedule['mixup_lambdas'],dtype=torch.float64)[:,None]
    wa=targets['weights'][indices].to(torch.float64);wb=wa.gather(1,perm)
    ta=targets['targets'][indices];tb=ta.gather(1,perm)
    effective=1/(1+((1-lam)*wb)/(lam*wa))
    difference=(effective-lam).abs();different=ta!=tb
    event=different&(difference>=cfg['displacement_event_min'])
    flip=different&((lam>.5)!=(effective>.5))&(lam!=.5)&(effective!=.5)
    owner=torch.where(lam>=.5,ta,tb).flatten().tolist()
    exposure=Counter(owner); events=Counter(o for o,e in zip(owner,event.flatten().tolist()) if e)
    flips=Counter(o for o,f in zip(owner,flip.flatten().tolist()) if f)
    cohort=sorted(exposure,key=lambda c:(-Fraction(events[c],exposure[c]),c))[:max(1,len(exposure)//4)]
    numerical_error=float(np.max(np.abs(cache['supervision_fraction']-effective.numpy())))
    require(numerical_error<1e-12,'Independent coefficient replay differs')
    expected_arrays=dict(train_indices=indices.numpy(),image_fraction=lam.expand_as(wa).numpy(),
        weight_a=wa.numpy(),weight_b=wb.numpy(),target_a=ta.numpy(),target_b=tb.numpy(),
        displacement_event=event.numpy(),dominance_flip=flip.numpy(),image_owner=np.array(owner).reshape(wa.shape),
        labels=labels,image_paths=paths)
    train_counts=Counter(int(r['label']) for r in train)
    tail=sorted(range(len(names)),key=lambda c:(train_counts[c],c))[:len(names)//10]
    selected=set(cohort);tail_set=set(tail)
    group_lists=dict(all=[True]*len(val),high_displacement_classes=[int(y) in selected for y in labels],
        other_classes=[int(y) not in selected for y in labels],tail75=[int(y) in tail_set for y in labels],
        small=[min(int(r['width']),int(r['height']))<224 for r in val])
    expected_arrays.update({g:np.array(m) for g,m in group_lists.items()})
    require(set(cache.files)==set(expected_arrays)|{'supervision_fraction'},'Unexpected private columns')
    for key,array in expected_arrays.items():require(np.array_equal(array,cache[key]),f'{key} differs')
    aggregate=dict(mixed_rows=wa.numel(),different_target_rows=int(different.sum()),displacement_events=int(event.sum()),
        displacement_fraction=float(event.double().mean()),different_target_dominance_flips=int(flip.sum()),
        mean_absolute_component_displacement=float(difference.mean()),max_absolute_component_displacement=float(difference.max()),
        mean_class_distribution_l1_vs_unweighted_mix=float((2*(1-cfg['label_smoothing'])*difference*different).mean()),
        reliability_min=float(wa.min()),reliability_max=float(wa.max()),scheduled_target_classes_present=len(exposure),
        scheduled_target_classes_absent=[c for c in range(len(names)) if c not in exposure],cohort_class_indices=cohort)
    equal_counts(report['aggregate'],aggregate,'aggregate')
    rows=read_rows(report['class_file']);require(len(rows)==len(names),'Class table population differs')
    for c,row in enumerate(rows):
        actual=dict(class_name=names[c],class_index=c,scheduled_rows=exposure[c],displacement_events=events[c],
            dominance_flips=flips[c],event_rate=events[c]/exposure[c] if exposure[c] else 0.,in_cohort=int(c in selected))
        require(set(row)==set(actual),'Class table columns differ')
        for key,value in actual.items():
            require(row[key]==value if isinstance(value,str) else abs(float(row[key])-value)<1e-12,f'Class {c}/{key} differs')
    validation={}
    for group,mask in group_lists.items():
        actual=dict(parent=counts(labels,parent,mask),control=counts(labels,raw,mask),
                    control_vs_parent=compare_pair(labels,parent,raw,mask))
        for key,value in actual.items():equal_counts(report['validation'][group][key],value,f'{group}/{key}')
        validation[group]=actual
    excess=validation['all']['parent']['micro']-validation['high_displacement_classes']['parent']['micro']
    require(abs(report['cohort_error_rate_excess_over_all']-excess)<1e-12,'Error-rate excess differs')
    gate=cfg['mechanism_gate']
    passed=bool(aggregate['displacement_fraction']>=gate['displacement_fraction_min'] and
        aggregate['different_target_dominance_flips']>=gate['different_target_dominance_flips_min'] and
        validation['high_displacement_classes']['parent']['errors']>=gate['cohort_parent_errors_min'] and
        excess>=gate['cohort_error_rate_excess_over_all_min'])
    decision='supports_one_bounded_image_mass_alignment_probe' if passed else 'close_small_or_unlinked_mixup_mass_effect'
    require(report['mechanism_gate_passed'] is passed and report['decision']==decision,'Frozen gate differs')
    return dict(status='passed_independent_torch64_counter_fraction_replay',experiment_id=cfg['experiment_id'],
        mixed_rows_replayed=wa.numel(),max_fraction_error=numerical_error,event_and_flip_membership_identical=True,
        class_rows_checked=len(rows),cohort_membership_identical=True,validation_rows=len(labels),metric_groups_checked=10,
        paired_groups_checked=5,source_files_checked=len(report['source_files']),report_sha256=sha(report_path),
        verifier_sha256=sha(__file__),decision=decision,training_started=False,new_candidate=False,platform_gain_known=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--report',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    require(not Path(args.output).exists(),'Use a fresh independent verification path')
    torch.set_num_threads(2);result=verify(args.config,args.report)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
