#!/usr/bin/env python3
"""Same-epoch pair and frozen content proxies, never a clean-label accuracy."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import torch
from run_p75_semantic_pair import OUT, FRAMEWORK, ROOT, CONFIGS, run_dir, sha, write


def paired(labels, before, after, mask=None):
    if mask is not None:
        labels,before,after = [v[mask] for v in (labels,before,after)]
    a,b = before==labels,after==labels
    classes = np.unique(labels)
    corrected,regressed = int((~a & b).sum()),int((a & ~b).sum())
    return dict(rows=len(labels),classes_present=len(classes),
        control_micro=float(a.mean()) if len(a) else None,
        masked_micro=float(b.mean()) if len(b) else None,
        control_macro=float(np.mean([a[labels==c].mean() for c in classes])) if len(classes) else None,
        masked_macro=float(np.mean([b[labels==c].mean() for c in classes])) if len(classes) else None,
        corrected=corrected,regressed=regressed,net=corrected-regressed,
        changed=int((before!=after).sum()))


def decode(arm, epoch):
    from aegis_clip.config import load_config
    from aegis_clip.tta_prior_binding import checkpoint_identity, validation_cache_identity, fit_bound_prior, resolved_recipe
    from aegis_clip.tta import fuse_paired_logits
    from aegis_clip.prior_alignment import apply_prior_bias
    checkpoint=run_dir(arm)/f'checkpoints/epoch_{epoch}.pt'
    cache=run_dir(arm)/f'val_epoch_{epoch}.pt'
    config=load_config(CONFIGS/f'{arm}.json')
    identity=checkpoint_identity(checkpoint,config)
    bound=validation_cache_identity(cache,config,identity)
    payload=torch.load(cache,map_location='cpu',weights_only=False)
    recipe=resolved_recipe(tta='horizontal_flip',fusion='mean_probabilities',temperature=1.4,prior_strength=.6,enforce_fixed=True)
    bias,fit,prior=fit_bound_prior(payload,bound,recipe=recipe,max_iterations=50)
    logits=fuse_paired_logits(payload['original_logits'],payload['flip_logits'],mode='mean_probabilities',temperature=1.4)
    prediction=apply_prior_bias(logits,bias,strength=.6).argmax(1).numpy()
    py=payload['original_logits'].float().softmax(1).gather(1,torch.as_tensor(payload['labels']).long()[:,None]).squeeze(1).numpy()
    return payload,prediction,py,dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        cache=str(cache),cache_sha256=sha(cache),prior_record=prior,fit_report=fit)


def training_audit(epoch):
    orders,group_means=[],[]
    for e in range(1,epoch+1):
        a,b=[json.loads((run_dir(arm)/f'logs/pair_order_{e}.json').read_text()) for arm in ('control','masked')]
        if a!=b or a['rows']!=133815 or a['successful_updates']!=131*e or a['attempted_updates']!=131*e:
            raise ValueError(f'Pair order/update mismatch at epoch {e}: {a}, {b}')
        orders.append(a)
        for arm in ('control','masked'):
            value=json.loads((run_dir(arm)/f'logs/text_page_epoch_{e}.json').read_text())
            if value['enabled']!=(arm=='masked'):
                raise ValueError('Intervention state mismatch')
            for group,count in [('selected',1246),('remaining',132569)]:
                prefix=f'text_page_{group}_'
                if value[prefix+'rows']!=count:
                    raise ValueError('Training group coverage mismatch')
                group_means.append(dict(arm=arm,epoch=e,group=group,rows=count,loss_phase=value['loss_phase'],
                    **{key: value[prefix+key+'_sum']/count for key in ('py','sqrt_py','pmax','raw_classification')}))
    progress=[]
    for arm in ('control','masked'):
        rows=[json.loads(s) for s in (run_dir(arm)/'logs/progress.jsonl').read_text().splitlines()]
        progress.append([{k:r[k] for k in ('epoch','global_step','successful_optimizer_updates',
            'attempted_optimizer_updates','head_lr','visual_lr','amp_scale','loss_phase')} for r in rows if r['epoch']<=epoch])
    if progress[0]!=progress[1]:
        raise ValueError('Pair learning rate, AMP scale or update trajectory differs')
    return dict(orders=orders,progress_equal=True,group_means=group_means,
        limitation='Order digest checks all indices; worker RNG restart is identical but image views were not exhaustively hashed.')


def report(destination):
    if destination.exists(): raise FileExistsError(destination)
    sys.path.insert(0,str(FRAMEWORK/'reproducibility/aegis_f1'))
    torch.set_num_threads(4)
    with (OUT/'selection.jsonl').open() as f:
        rows=[r for s in f if (r:=json.loads(s))['split']=='val']
    labels=np.array([r['label'] for r in rows])
    mask=np.array([r['selected'] for r in rows])
    paths=[r['image_path'].removeprefix('train/') for r in rows]
    with gzip.open(ROOT/'results/p75_supervision_rebuild_20260929/sample_evidence.csv.gz','rt') as f:
        baseline_rows={r['image_path'].removeprefix('train/'):r for r in csv.DictReader(f) if r['split']=='val'}
    baseline=np.array([int(baseline_rows[p]['l05_fixed_decode_prediction']) for p in paths])
    if len(rows)!=14880 or int(mask.sum())!=132 or int((baseline!=labels).sum())!=3474:
        raise ValueError('Frozen validation identity changed')
    result=dict(experiment='P75_SEMANTIC_MASK_PAIR',epochs={},platform_gain=None,
        limitations=['Content groups are automatic proxies, not clean labels.',
            'Same used validation split, not an independent test.',
            'Shared RM-LP parent head already saw original supervision.',
            'No automatic full-training or platform promotion.'])
    destination.mkdir(parents=True)
    for epoch in (4,6):
        if not all((run_dir(arm)/f'val_epoch_{epoch}.pt').exists() for arm in ('control','masked')):
            continue
        decoded=[decode(arm,epoch) for arm in ('control','masked')]
        for payload,_,_,_ in decoded:
            if [p.removeprefix('train/') for p in payload['paths']]!=paths or not np.array_equal(payload['labels'],labels):
                raise ValueError('Validation ordering differs')
        a,b=[d[1] for d in decoded]
        groups={'all':np.ones(len(rows),dtype=bool),'selected_proxy':mask,'remaining_proxy':~mask}
        entry=dict(same_epoch_pair={k:paired(labels,a,b,m) for k,m in groups.items()},
            versus_l05={arm:{k:paired(labels,baseline,d[1],m) for k,m in groups.items()}
                for arm,d in zip(('control','masked'),decoded)},
            original_label_probability={arm:{k:float(d[2][m].mean()) for k,m in groups.items()}
                for arm,d in zip(('control','masked'),decoded)},
            binding={arm:d[3] for arm,d in zip(('control','masked'),decoded)},
            training_audit=training_audit(epoch))
        table=destination/f'paired_epoch_{epoch}.csv.gz'
        with gzip.open(table,'wt',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['image_path','original_label','selected_proxy','l05','control','masked','corrected','regressed'],lineterminator='\n')
            writer.writeheader()
            for i,r in enumerate(rows):
                writer.writerow(dict(image_path=r['image_path'],original_label=int(labels[i]),selected_proxy=bool(mask[i]),
                    l05=int(baseline[i]),control=int(a[i]),masked=int(b[i]),
                    corrected=bool(a[i]!=labels[i]==b[i]),regressed=bool(a[i]==labels[i]!=b[i])))
        entry['paired_csv_sha256']=sha(table)
        result['epochs'][str(epoch)]=entry
    if not result['epochs']: raise ValueError('No completed matched validation caches')
    result['primary_complete']='6' in result['epochs']
    write(destination/'report.json',result)
    lines=['# P75 semantic masking pair','',
        'Original-label accuracy; selected and remaining groups are content proxies, not clean truth.','',
        '| Epoch | Proxy | Rows | Control macro | Masked macro | Control micro | Masked micro | Fixed | Damaged | Net |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for epoch,e in result['epochs'].items():
        for name,v in e['same_epoch_pair'].items():
            lines.append(f"| {epoch} | {name} | {v['rows']} | {v['control_macro']:.6%} | {v['masked_macro']:.6%} | {v['control_micro']:.6%} | {v['masked_micro']:.6%} | {v['corrected']} | {v['regressed']} | {v['net']} |")
    lines+=['','Slice macro averages only classes present in that slice.','',*result['limitations']]
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({e:r['same_epoch_pair'] for e,r in result['epochs'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    report(p.parse_args().out)
