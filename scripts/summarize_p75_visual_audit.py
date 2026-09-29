#!/usr/bin/env python3
"""Summarize development observations with design weights; never produces accuracy or labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

STATES=('original_supported_model_wrong','original_label_problem','cannot_determine')


def readl(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def weighted(cases, observations, adjudications):
    ids=[r['case_id'] for r in cases]
    if len(ids)!=200 or len(set(ids))!=200: raise ValueError('Expected 200 distinct frozen audit cases')
    if ids!=[r['case_id'] for r in observations] or ids!=[r['case_id'] for r in adjudications]:
        raise ValueError('Observation/adjudication IDs must match frozen order')
    if any(r['status'] not in STATES or r['training_use_allowed'] is not False for r in adjudications):
        raise ValueError('Invalid audit-only adjudication')
    groups={}
    flags=sorted({flag for r in observations for flag in r['flags']})
    for group in sorted({r['stratum'] for r in cases}):
        triples=[(c,o,a) for c,o,a in zip(cases,observations,adjudications) if c['stratum']==group]
        n=len(triples); N=triples[0][0]['population']
        if any(c['sample_size']!=n or c['population']!=N or abs(c['weight']-N/n)>1e-9 for c,_,_ in triples):
            raise ValueError('Invalid stratification weights')
        groups[group]=dict(population=N,sampled=n,status_counts={s:sum(a['status']==s for _,_,a in triples) for s in STATES},
            phenomena={f:dict(sample_count=sum(f in o['flags'] for _,o,_ in triples),
                              estimated_population_count=N*sum(f in o['flags'] for _,o,_ in triples)/n) for f in flags})
    totals={}
    for cohort in ('all_validation','original_label_errors','agreement_controls'):
        selected=[v for g,v in groups.items() if cohort=='all_validation' or (g=='control_agreement')==(cohort=='agreement_controls')]
        denominator=sum(v['population'] for v in selected)
        totals[cohort]=dict(population=denominator,
            status_estimated_counts={s:sum(v['population']*v['status_counts'][s]/v['sampled'] for v in selected) for s in STATES},
            phenomena={f:dict(sample_count=sum(v['phenomena'][f]['sample_count'] for v in selected),
                              estimated_population_count=sum(v['phenomena'][f]['estimated_population_count'] for v in selected),
                              weighted_fraction=sum(v['phenomena'][f]['estimated_population_count'] for v in selected)/denominator) for f in flags})
    return dict(strata=groups,weighted=totals,limitations=[
        'Observations are from one assistant reviewer, not human/expert ground truth.',
        'Flags overlap and describe visible content only, not causal errors, noise rates or recoverable gain.',
        'Weighted estimates account for the stratified sample design; small strata samples and subjective flags remain uncertain.',
        'All original-label status uncertainty is retained; no clean accuracy or corrected labels are produced.',
        'References are random current-stage training groups, with unverified labels; not extra adjudicated query cases.',
        'Full-size crop followups qualify blind wording without changing frozen sample or labels.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--recheck-report',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError('Output already exists')
    root=args.audit
    obsfile=root/'blind_observations.jsonl'
    if hashlib.sha256(obsfile.read_bytes()).hexdigest()!=json.loads((root/'blind_lock.json').read_text())['sha256']:
        raise ValueError('Blind notes changed after label reveal')
    result=weighted(json.loads((root/'cases.json').read_text()),readl(obsfile),readl(root/'adjudications.jsonl'))
    r=json.loads(args.recheck_report.read_text()); v=r['validation_neighbor_partitions']
    corrections=v['unique_original_not_model']['errors']
    regressions=v['unique_neither']['rows']-v['unique_neither']['errors']
    count=r['counts']['val']; original_correct=count-r['counts']['val_errors']
    result['cached_counterfactual']=dict(rule='Replace with unique neighbor plurality; preserve L05 on ties',
        corrections=corrections,regressions=regressions,net=corrections-regressions,
        baseline_micro=original_correct/count,hypothetical_micro=(original_correct+corrections-regressions)/count,
        candidate=False,training=False)
    inputs=('adjudications.jsonl','blind_lock.json','blind_observations.jsonl','cases.json',
            'detail_followup.json','protocol.json','reference_manifest.json','reference_observations.tsv')
    result['hashes']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in inputs}
    result['recheck_sha256']=hashlib.sha256(args.recheck_report.read_bytes()).hexdigest()
    with args.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result['weighted'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
