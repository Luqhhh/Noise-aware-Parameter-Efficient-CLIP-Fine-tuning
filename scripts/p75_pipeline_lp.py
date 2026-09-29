#!/usr/bin/env python3
"""CPU-only cached RM-LP pair. No image encoding, no full-model initialization."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F
from p75_pipeline import DEFAULT, require, verify
from run_p75_pipeline_a import authorize
from run_p75_semantic_pair import write, sha
from p75_text_page_runtime import mask_classification


def make_heads(dimension,classes,seed):
    torch.manual_seed(seed)
    head=torch.nn.Linear(dimension,classes)
    torch.nn.init.xavier_uniform_(head.weight); torch.nn.init.zeros_(head.bias)
    return head,copy.deepcopy(head)


def head_loss(head,features,labels,mask,enabled):
    return mask_classification(F.cross_entropy(head(F.normalize(features.float(),dim=1)),labels,reduction='none'),mask,enabled).mean()


def train_pair(features, labels, selected, config, destination, deadline):
    """Same cached minibatches and initialization; CE denominator and AdamW unchanged."""
    t=config['train']; m=config['model']; loss=config['loss']
    require(m['use_cached_training'] and m['peft_mode']=='frozen' and m.get('classifier_mode','linear')=='linear', 'LP must be frozen linear head')
    require(loss['name']=='cross_entropy' and not loss.get('mixup_probability',0) and
            not loss.get('feature_distillation_weight',0) and not config['trust']['enabled'], 'Unsupported actual LP objective')
    require(config['longtail']['sampler_mode']=='none' and config['longtail']['loss_reweighting']=='none'
            and config['longtail']['balanced_softmax_tau']==0, 'Unsupported LP weighting')
    require(t.get('grad_accum_steps',1)==1 and t.get('lr_warmup_epochs',0)==0, 'Unexpected actual LP update scheme')
    require(t.get('head_weight_decay_filter','all')=='all','Unsupported decay filter')
    heads=make_heads(m['feature_dim'],m['num_classes'],config['project']['seed'])
    initial=copy.deepcopy(heads[0].state_dict())
    optimizers=[torch.optim.AdamW(h.parameters(),lr=t['head_lr'],weight_decay=t['head_weight_decay']) for h in heads]
    count=math.ceil(len(labels)/t['batch_size']); total=t['schedule_epochs']*count
    schedulers=[torch.optim.lr_scheduler.LambdaLR(o,lambda step: .01+.99*.5*(1+math.cos(math.pi*min(step,total)/total))) for o in optimizers]
    generator=torch.Generator().manual_seed(config['project']['seed']); records=[]; updates=0
    def save(status,epoch):
        for arm,h,o,s in zip(('control','masked'),heads,optimizers,schedulers):
            torch.save(dict(status=status,epoch=epoch,global_step=updates,model_state_dict=h.state_dict(),
                optimizer_state_dict=o.state_dict(),scheduler_state_dict=s.state_dict(),initial_state_dict=initial,
                generator_state=generator.get_state(),config=config,
                resumable=status=='complete',kind='cached_classifier_only'),destination/f'{arm}.pt')
    epoch=0
    try:
        for epoch in range(1,t['epochs']+1):
            # DataLoader consumes a base seed even with zero workers; mirror that before RandomSampler.
            torch.empty((),dtype=torch.int64).random_(generator=generator)
            order=torch.randperm(len(labels),generator=generator)
            digest=hashlib.sha256(order.numpy().astype('<i8').tobytes()).hexdigest()
            hits=0
            for batch in order.split(t['batch_size']):
                if time.monotonic()>=deadline:
                    save('incomplete',epoch-1)
                    write(destination/'training_audit.json',records)
                    raise TimeoutError('CPU budget expired; partial epoch pair saved but not resumable')
                for enabled,h,o,s in zip((False,True),heads,optimizers,schedulers):
                    o.zero_grad(set_to_none=True)
                    value=head_loss(h,features[batch],labels[batch],selected[batch],enabled)
                    require(bool(torch.isfinite(value)), 'Nonfinite LP loss')
                    value.backward(); norm=torch.nn.utils.clip_grad_norm_(h.parameters(),t['max_grad_norm'])
                    require(bool(torch.isfinite(norm)), 'Nonfinite LP gradient')
                    o.step(); s.step()
                updates+=1; hits+=int(selected[batch].sum())
            # Match RandomSampler's zero-length remainder randperm RNG consumption.
            torch.randperm(len(labels),generator=generator)
            records.append(dict(epoch=epoch,order_sha256=digest,rows=len(labels),mask_hits=hits,
                                updates=updates,lr=[o.param_groups[0]['lr'] for o in optimizers]))
    except BaseException:
        save('incomplete',max(0,epoch-1))
        write(destination/'training_audit.json',records)
        raise
    save('complete',t['epochs']); write(destination/'training_audit.json',records)
    return heads,records


def run(root,authorization):
    verify(root); approval=authorize(root,authorization,'B')
    require(not (root/'B/status.json').exists(),'No automatic replay/budget reset')
    # A center-only feature cache cannot represent the required fixed final decoder.
    flip_spec=approval.get('existing_flip_features')
    require(isinstance(flip_spec,dict),'Existing validation flip cache with binding required; no re-encoding fallback')
    cache_path=Path(flip_spec['path'])
    require(sha(cache_path)==flip_spec['sha256'],'Flip cache digest mismatch')
    sys.path.insert(0,str(root/'framework/reproducibility/aegis_f1'))
    from aegis_clip.features import FrozenFeatureStore, canonical_sample_path
    from aegis_clip.data import _load_split as read_split_csv
    from aegis_clip.prior_alignment import fit_prior_bias, apply_prior_bias
    from aegis_clip.tta import fuse_paired_logits
    from p75_pipeline_report import emit
    c=json.loads((root/'B/config.json').read_text()); torch.set_num_threads(4)
    torch.use_deterministic_algorithms(bool(c['train'].get('deterministic',True)))
    start=time.monotonic(); state=dict(status='running',authorization_sha256=sha(authorization))
    write(root/'B/status.json',state)
    def expired(signum,frame):
        raise TimeoutError('CPU task wall-clock cap reached')
    old_handler=signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,3595)
    try:
        store=FrozenFeatureStore(**{k:c['features'][k] for k in ('tensor_path','paths_path','manifest_path')})
        train=read_split_csv(c['data']['train_csv']); val=read_split_csv(c['data']['val_csv'])
        rows=json.loads((root/'groups.json').read_text())
        require([canonical_sample_path(p) for p in val['image_path']]==[canonical_sample_path(r['image_path']) for r in rows],'LP validation ordering mismatch')
        selection=[json.loads(s) for s in (root/'selection.jsonl').read_text().splitlines()]
        selection={canonical_sample_path(r['image_path']):r for r in selection if r['split']=='train'}
        require(len(train)==133815 and len(val)==14880, 'LP split size mismatch')
        mask=[]
        for p,y in zip(train['image_path'],train['label']):
            row=selection[canonical_sample_path(p)]; require(row['label']==int(y),'LP training label mismatch'); mask.append(row['selected'])
        require(list(val['label'])==[r['label'] for r in rows], 'LP validation label mismatch')
        x=store.features[[store.index_of(p) for p in train['image_path']]]
        y=torch.tensor(train['label'].to_numpy()).long()
        flip=torch.load(cache_path,map_location='cpu',weights_only=True)
        binding=flip['binding']
        require(binding['data_version']=='20260921' and binding['pretrained']=='openai' and
                binding['backbone']=='ViT-B/32' and binding['feature_manifest_sha256']==sha(c['features']['manifest_path'])
                and binding['val_csv_sha256']==sha(c['data']['val_csv']) and binding['tta']=='horizontal_flip', 'Unbound flip features')
        require([canonical_sample_path(p) for p in flip['paths']]==[canonical_sample_path(p) for p in val['image_path']], 'Flip cache paths mismatch')
        require(tuple(flip['features'].shape)==(14880,c['model']['feature_dim']) and bool(torch.isfinite(flip['features']).all()),'Invalid flip features')
        heads,audit=train_pair(x,y,torch.tensor(mask),c,root/'B',start+3500)
        predictions=[]; bindings={}
        vx=store.features[[store.index_of(p) for p in val['image_path']]]
        vy=torch.tensor(val['label'].to_numpy()).long()
        for arm,head in zip(('control','masked'),heads):
            with torch.no_grad():
                logits=head(F.normalize(vx,dim=1)); flipped=head(F.normalize(flip['features'].float(),dim=1))
                fused=fuse_paired_logits(logits,flipped,mode='mean_probabilities',temperature=1.4)
                bias,fit=fit_prior_bias(fused,max_iterations=50)
                predictions.append(apply_prior_bias(fused,bias,strength=.6).argmax(1).numpy())
            torch.save(dict(original_logits=logits,flip_logits=flipped,labels=vy,paths=list(val['image_path']),prior_bias=bias),root/f'B/{arm}_validation.pt')
            bindings[arm]=dict(validation_sha256=sha(root/f'B/{arm}_validation.pt'),checkpoint_sha256=sha(root/f'B/{arm}.pt'),flip_sha256=sha(cache_path),
                manifest_sha256=sha(root/'manifest.json'),fit=fit)
        require(time.monotonic()-start<3600,'LP evaluation exceeded budget')
        emit(root/'B',rows,predictions,bindings,'P75_MASK_LP',audit); state['status']='local_result'
    except BaseException as exc:
        state.update(status='incomplete',error=repr(exc))
        write(root/'B/conclusion.json',dict(status='incomplete',reason=repr(exc),FT_hypothesis_disproved=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old_handler)
        state['used_seconds']=time.monotonic()-start; write(root/'B/status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--authorization',type=Path,required=True)
    a=p.parse_args(); run(a.out.resolve(),a.authorization)
