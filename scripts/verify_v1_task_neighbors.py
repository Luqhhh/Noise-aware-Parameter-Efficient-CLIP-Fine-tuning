"""Recompute all votes/similarities and 64 complete neighbor searches on CPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def unit(features):
    x = np.asarray(features, dtype=np.float64)
    return x/np.sqrt(np.einsum('ij,ij->i', x, x))[:, None]


def counts(labels, prediction, mask):
    totals, correct = Counter(), Counter()
    for y, p, included in zip(labels, prediction, mask):
        if included:
            totals[int(y)] += 1
            correct[int(y)] += int(y == p)
    n, c = sum(totals.values()), sum(correct.values())
    return dict(rows=n, correct=c, errors=n-c, classes=len(totals), micro=c/n if n else None,
        macro=sum(correct[k]/totals[k] for k in sorted(totals))/len(totals) if totals else None)


def compare(labels, before, after, mask):
    correction = regression = changed = wrong = n = 0
    for y, a, b, included in zip(labels, before, after, mask):
        if included:
            n += 1; correction += int(a != y and b == y); regression += int(a == y and b != y)
            changed += int(a != b); wrong += int(a != y and b != y)
    return dict(rows=n, corrections=correction, regressions=regression, net=correction-regression,
                changed=changed, both_wrong=wrong)


def equal(expected, actual, label):
    require(expected.keys() == actual.keys(), f'{label}: keys differ')
    for k, v in actual.items():
        require(abs(expected[k]-v) < 1e-12 if isinstance(v, float) else expected[k] == v, f'{label}/{k}: differs')


def verify(config, output):
    started = time.monotonic(); cfg=read(config); out=Path(output).resolve()
    report, pre, cost = [read(out/f'{name}.json') for name in ['report','preflight','cost']]
    bind = pre['binding']
    require(bind == report['binding'] and bind['config_sha256'] == sha(config) and
        bind['script_sha256'] == sha(ROOT/'scripts/probe_v1_task_neighbors.py'), 'Frozen implementation differs')
    require(report['status'] == 'completed_fixed_neighbor_diagnostic' == read(out/'status.json')['status'] and
        report['training_updates'] == 0 and report['new_candidate'] is False and
        all(report[k] is False for k in ['test_predictions_used','fused_predictions_generated',
            'parameter_search','automatic_full_training','platform_gain_known','clean_labels_known']), 'Scope differs')
    sources = {}
    def checked(path, expected=None):
        path=Path(path).resolve(); digest=sha(path)
        require(expected is None or digest == expected, f'Checksum differs: {path}')
        sources[str(path)] = digest
    for path, digest in bind['source_files'].items(): checked(path, digest)
    for path, digest in report['artifacts'].items(): checked(path, digest)
    checked(out/'frozen.npz', pre['frozen_sha256'])
    require(cost['status']=='passed_discarded_cost_outputs' and cost['query_rows']==64 and
        cost['training_updates']==0 and cost['dtype']=='float64' and cost['wall_clock_time_limit'] is None, 'Real cost check differs')
    require(cfg['gallery_rule']=='one_lexicographically_first_train_dev_image_per_single_original_label_content_group;exclude_cross_label_groups'
        and cfg['neighbors']==16 and cfg['consensus_votes_min']==12 and cfg['cosine_round_decimals']==9,
        'Fixed gallery/vote recipe differs')
    stage=Path(cfg['stage_root']); train, val, full = [rows(stage/name) for name in ['train_dev.csv','val_dev.csv','full_train.csv']]
    require(not set(r['content_group'] for r in train)&set(r['content_group'] for r in val), 'Train/val content overlap')
    mapping=read(stage/'class_to_idx.json'); classes=[k for k,v in sorted(mapping.items(),key=lambda item:item[1])]
    labels=np.array([int(r['label']) for r in val]); paths=np.array([r['image_path'] for r in val])
    grouped=defaultdict(list)
    for i,r in enumerate(train): grouped[r['content_group']].append(i)
    canonical=[]; conflicts=[]
    for group, members in grouped.items():
        if len(Counter(train[i]['label'] for i in members))>1: conflicts.append(members)
        else: canonical.append(sorted(members,key=lambda i:train[i]['image_path'])[0])
    canonical=np.array(sorted(canonical,key=lambda i:train[i]['image_path']))
    gallery_paths=np.array([train[i]['image_path'] for i in canonical]); glabel=np.array([int(train[i]['label']) for i in canonical])
    require(len(set(train[i]['content_group'] for i in canonical))==len(canonical), 'Repeated content group in gallery')
    info=dict(original_train_rows=len(train),original_content_groups=len(grouped),gallery_rows=len(canonical),
        excluded_conflict_groups=len(conflicts),excluded_conflict_rows=sum(map(len,conflicts)),
        excluded_redundant_same_label_rows=len(train)-sum(map(len,conflicts))-len(canonical),
        classes=len(set(glabel.tolist())),classes_absent=sorted(set(range(len(classes)))-set(glabel.tolist())))
    require(info==pre['gallery']==report['gallery'] and cost['gallery_rows']==len(canonical), 'Gallery counts differ')
    with np.load(out/'frozen.npz',allow_pickle=False) as z:
        for name, value in dict(labels=labels,image_paths=paths,gallery_train_indices=canonical,
                                gallery_image_paths=gallery_paths,gallery_labels=glabel).items():
            require(np.array_equal(z[name],value), f'Frozen {name} differs')
        native=z['native'].copy()
    source=Path(cfg['source_root']); cache=torch.load(source/'frozen_features768.pt',map_location='cpu',weights_only=False)
    require(cache['image_paths']==[r['image_path'] for r in full] and cache['feature_path']=='ln_post_pre_projection' and
        cache['image_size']==448 and cache['visual_parameter_updates'] is False, 'Feature source scope differs')
    by_path={p:i for i,p in enumerate(cache['image_paths'])}
    gx=np.array([by_path[p] for p in gallery_paths]); qx=np.array([by_path[p] for p in paths])
    require(not set(gx)&set(qx), 'Validation feature entered gallery')
    gallery=unit(cache['features'][gx].numpy()); query=unit(cache['features'][qx].numpy())
    with np.load(source/'validation_predictions.npz',allow_pickle=False) as z:
        require(np.array_equal(z['unbalanced768'],native), 'Native predictor changed')
        tail=z['tail'].copy()
    head=torch.load(source/'unbalanced/selected.pt',map_location='cpu',weights_only=False)['selected_state']['head.weight'].numpy()
    x=cache['features'][qx].numpy(); x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    w=head/np.maximum(np.linalg.norm(head,axis=1,keepdims=True),1e-12)
    require(np.array_equal((x@w.T).argmax(1),native), 'All source head predictions must replay')
    reflection=read(ROOT/cfg['source_reflection'])
    common_path=next(p for p in reflection['artifacts'] if Path(p).name=='lr512_swa.npz')
    with np.load(common_path,allow_pickle=False) as z:
        require(np.array_equal(z['labels'],labels) and np.array_equal(z['image_paths'],paths), 'Common-error source alignment differs')
        common=z['common_candidate_wrong'].copy()
    z=np.load(out/'neighbors.npz',allow_pickle=False)
    require(np.array_equal(z['labels'],labels) and np.array_equal(z['image_paths'],paths) and np.array_equal(z['native'],native), 'Neighbor rows differ')
    idx, score=z['neighbor_gallery_indices'], z['cosine_keys']
    require(idx.shape==score.shape==(len(val),16) and np.min(idx)>=0 and np.max(idx)<len(canonical) and
        all(len(set(r))==16 for r in idx), 'Invalid neighbor indices')
    for start in range(0,len(val),128):
        chosen=gallery[idx[start:start+128]]
        keys=np.rint(np.einsum('ij,ikj->ik',query[start:start+128],chosen)*1e9).astype(np.int64)
        require(np.array_equal(keys,score[start:start+128]), 'Returned cosine keys differ from independent dot products')
    require(np.all(score[:,:-1]>=score[:,1:]) and np.all((score[:,:-1]!=score[:,1:])|(idx[:,:-1]<idx[:,1:])), 'Neighbor tie ordering differs')
    sample=np.sort(np.random.default_rng(42).choice(len(val),64,replace=False))
    direct_keys=np.rint((query[sample]@gallery.T)*1e9).astype(np.int64)
    direct_indices=np.argsort(-direct_keys,axis=1,kind='stable')[:,:16]
    require(np.array_equal(direct_indices,idx[sample]), 'Independent full-gallery CPU64 nearest-neighbor search differs')
    require(np.array_equal(np.take_along_axis(direct_keys,direct_indices,axis=1),score[sample]), 'Independent nearest scores differ')
    del direct_keys
    pred=[]; strength=[]; votes=np.zeros((len(val),len(classes)),np.int64)
    for i,row in enumerate(idx):
        counter=Counter(glabel[row].tolist()); order=sorted(counter,key=lambda c:(-counter[c],c))
        pred.append(order[0]); strength.append(counter[order[0]])
        for c,n in counter.items(): votes[i,c]=n
    pred=np.array(pred); strength=np.array(strength)
    for key,array in dict(prediction=pred,max_votes=strength,class_votes=votes).items():
        require(np.array_equal(z[key],array), f'Independent {key} differs')
    support=strength>=12
    masks=dict(all=np.ones(len(val),bool),common_candidate_wrong=common,tail75=np.isin(labels,tail),
        small=np.array([min(int(r['width']),int(r['height']))<224 for r in val]),
        supported=support,unsupported=~support,supported_disagreement=support&(pred!=native),common_wrong_supported=common&support)
    for group,mask in masks.items():
        require(np.array_equal(z[group],mask), f'{group}: membership differs')
        for name,value in dict(native=counts(labels,native,mask),neighbors=counts(labels,pred,mask),paired_to_native=compare(labels,native,pred,mask)).items():
            equal(report['validation'][group][name],value,f'{group}/{name}')
        if group in pre['native']: equal(pre['native'][group],counts(labels,native,mask),f'preflight/{group}')
    target=masks['supported_disagreement']; pair=compare(labels,native,pred,target); gate=cfg['mechanism_gate']
    criteria=dict(target_errors=counts(labels,native,target)['errors']>=gate['supported_disagreement_parent_errors_min'],
        corrections=pair['corrections']>=gate['supported_disagreement_corrections_min'],net=pair['net']>=gate['supported_disagreement_net_min'],
        correction_regression_ratio=pair['corrections']>=gate['supported_corrections_to_regressions_min']*pair['regressions'],
        common_wrong_corrections=compare(labels,native,pred,masks['common_wrong_supported'])['corrections']>=gate['common_wrong_supported_corrections_min'])
    passed=all(criteria.values())
    require(criteria==report['gate_criteria'] and report['mechanism_gate_passed'] is passed and report['decision']==
        ('supports_bounded_task_geometry_review' if passed else 'close_fixed_task_neighbor_entry'), 'Frozen decision differs')
    for path in [out/'report.json',out/'preflight.json',out/'cost.json',out/'status.json']: checked(path)
    return dict(status='passed_independent_numpy64_counter_replay',experiment_id=cfg['experiment_id'],
        verifier_sha256=sha(__file__),report_sha256=sha(out/'report.json'),source_files_checked=len(bind['source_files']),
        source_native_predictions_replayed=len(val),returned_cosines_recomputed=len(val)*16,votes_and_groups_replayed=len(val),
        independent_full_gallery_query_rows=len(sample),gallery_rows=len(canonical),metric_groups_checked=2*len(masks)+len(pre['native']),
        paired_groups_checked=len(masks),training_updates=0,new_candidate=False,test_predictions_used=False,
        mechanism_gate_passed=passed,decision=report['decision'],source_and_artifact_sha256=sources,elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--result',required=True);args=parser.parse_args()
    require(not Path(args.result).exists(), 'Use a fresh independent result path')
    torch.set_num_threads(2);result=verify(args.config,args.output)
    Path(args.result).parent.mkdir(parents=True,exist_ok=True);Path(args.result).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_and_artifact_sha256'}))
