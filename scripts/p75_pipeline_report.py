"""Bound fixed decoding, overlapping slices and conservative investment decisions."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
from p75_semantic_pair_report import paired
from run_p75_semantic_pair import sha, write


def decide(groups):
    all_, remaining, bio = [groups[k] for k in ('all','remaining_proxy','bio_dominant')]
    gain=remaining['masked_macro']-remaining['control_macro']
    if remaining['net']>=45 and gain>0:
        if all_['net']>=0 and all_['masked_macro']>=all_['control_macro']:
            return 'candidate_review_required'
        if all_['net']<0 and bio['net']>0 and bio['masked_macro']>bio['control_macro']:
            return 'proxy_supported_review_required'
    if remaining['net']>0:
        return 'bounded_signal_no_automatic_extension'
    return 'close_recipe_no_full_training'


def summarize(rows, labels, before, after):
    labels,before,after=[np.asarray(v) for v in (labels,before,after)]
    if any(v.ndim!=1 or len(v)!=len(rows) for v in (labels,before,after)):
        raise ValueError('Pair length mismatch')
    if not np.array_equal(labels,[r['label'] for r in rows]): raise ValueError('Label identity mismatch')
    for values in (labels,before,after):
        if values.dtype.kind not in 'iu' or np.any((values<0)|(values>=750)): raise ValueError('Invalid class indices')
    selected=np.array([r['selected_proxy'] for r in rows],dtype=bool)
    masks={'all':np.ones(len(rows),dtype=bool),'selected_proxy':selected,'remaining_proxy':~selected}
    masks.update({k:np.array([r[k] for r in rows],dtype=bool) for k in
                  ('high_hit_classes','bio_dominant','bio_high_hit','bio_other_classes')})
    report={}
    for name,mask in masks.items():
        item=paired(labels,before,after,mask)
        item['per_class']={str(int(c)):paired(labels,before,after,mask&(labels==c)) for c in np.unique(labels[mask])}
        report[name]=item
    return report


def decode(checkpoint, cache, config_path, framework):
    sys.path.insert(0,str(framework/'reproducibility/aegis_f1'))
    from aegis_clip.config import load_config
    from aegis_clip.tta_prior_binding import checkpoint_identity, validation_cache_identity, fit_bound_prior, resolved_recipe
    from aegis_clip.tta import fuse_paired_logits
    from aegis_clip.prior_alignment import apply_prior_bias
    config=load_config(config_path)
    identity=checkpoint_identity(checkpoint,config)
    binding=validation_cache_identity(cache,config,identity)
    payload=torch.load(cache,map_location='cpu',weights_only=False)
    recipe=resolved_recipe(tta='horizontal_flip',fusion='mean_probabilities',temperature=1.4,prior_strength=.6,enforce_fixed=True)
    bias,fit,prior=fit_bound_prior(payload,binding,recipe=recipe,max_iterations=50)
    logits=fuse_paired_logits(payload['original_logits'],payload['flip_logits'],mode='mean_probabilities',temperature=1.4)
    prediction=apply_prior_bias(logits,bias,strength=.6).argmax(1).numpy()
    return payload,prediction,dict(checkpoint_sha256=sha(checkpoint),cache_sha256=sha(cache),prior=prior,fit=fit)


def emit(destination, rows, predictions, bindings, experiment, training_audit):
    labels=np.array([r['label'] for r in rows])
    report=summarize(rows,labels,*predictions)
    write(destination/'report.json',dict(status='local_result',experiment_id=experiment,groups=report,
        bindings=bindings,training_audit=training_audit,platform_measured=False,
        warning='Overlapping content proxies, not clean truth; reused validation, not independent holdout.'))
    fields=['group','rows','control_macro','masked_macro','control_micro','masked_micro','F','D','net']
    with (destination/'paired.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for name,r in report.items():
            w.writerow(dict(group=name,**{k:r[k] for k in fields[1:6]},F=r['corrected'],D=r['regressed'],net=r['net']))
    with (destination/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['image_path','label','control','masked'])
        w.writerows((r['image_path'],r['label'],int(predictions[0][i]),int(predictions[1][i])) for i,r in enumerate(rows))
    decision=decide(report) if experiment=='P75_MASK_E6_EXTENSION' else 'LP_evidence_review_required_do_not_add_A_and_B_gains'
    write(destination/'conclusion.json',dict(status='local_result',decision=decision,
        automatic_training_allowed=False,automatic_platform_upload=False,
        concentration_review_required=True,independent_holdout_evidence=False,
        caveat='23-class-only gains do not promote; review all per-class changes. B negative cannot disprove FT masking.'))
    return report
