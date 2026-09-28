"""Training-only R1 channels and conditional R2 group pairing; single student inference."""
from __future__ import annotations
import copy
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Sampler


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def channel_loss(logits, channels, targets):
    """Original=CE, conflict=soft CE, uncertain=no class target. Batch denominator unchanged."""
    if logits.shape != targets.shape or channels.shape != logits.shape[:1]:
        raise ValueError('Supervision batch does not align')
    ce=-(targets.float()*F.log_softmax(logits.float(),dim=1)).sum(1)
    return torch.where(channels != 2,ce,torch.zeros_like(ce))


def anchored_soft_targets(fixed, teacher, channels, coefficient=.25):
    """Teacher cannot create classes or change channel admission."""
    restricted=teacher.detach().float()*(fixed>0)
    mass=restricted.sum(1,keepdim=True)
    restricted=torch.where(mass>1e-8,restricted/mass.clamp_min(1e-8),fixed)
    mixed=(1-coefficient)*fixed+coefficient*restricted
    return torch.where((channels==1)[:,None],mixed,fixed).detach()


def supervised_contrastive(features, labels, groups, trusted, allowed_edges, temperature=.1):
    """Only different-group positives and preregistered reliable class relations."""
    z=F.normalize(features.float(),dim=1)
    logits=z@z.T/temperature
    size=len(z)
    different_group=groups[:,None]!=groups[None,:]
    reliable=trusted[:,None]&trusted[None,:]&different_group
    positive=reliable&(labels[:,None]==labels[None,:])
    relation=torch.zeros((size,size),device=z.device,dtype=torch.bool)
    for a,b in allowed_edges:
        relation|=((labels[:,None]==a)&(labels[None,:]==b))|((labels[:,None]==b)&(labels[None,:]==a))
    allowed=positive|(reliable&relation)
    usable=positive.any(1)&(allowed&~positive).any(1)
    if not usable.any():
        return features.sum()*0
    active=logits[usable]
    denominator=active.masked_fill(~allowed[usable],float('-inf')).logsumexp(1)
    positive_mean=(active*positive[usable]).sum(1)/positive[usable].sum(1)
    return (denominator-positive_mean).mean()


class ConfusionSampler(Sampler):
    """Half random batches, half 2 classes x 2 independent groups; fixed microbatch 4."""
    def __init__(self,rows,edges,seed=42):
        self.rows,self.seed,self.epoch=rows,seed,0
        by_class=defaultdict(dict)
        for i,row in enumerate(rows):
            if row['channel']=='original':
                by_class[int(row['original_label'])].setdefault(row['content_group'],i)
        self.members={c:list(groups.values()) for c,groups in by_class.items()}
        self.edges=[(a,b) for a,b in edges if len(self.members.get(a,[]))>=2 and len(self.members.get(b,[]))>=2]
        if not self.edges:
            raise ValueError('R2 has no trustworthy cross-group confusion pairs')
    def set_epoch(self,epoch):
        self.epoch=epoch
    def __len__(self):
        return len(self.rows)
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)
        random=rng.permutation(len(self.rows)).tolist()
        output=[]
        for offset in range(0,len(random),4):
            if (offset//4)%2==0 or len(random)-offset<4:
                output.extend(random[offset:offset+4])
            else:
                a,b=self.edges[int(rng.integers(len(self.edges)))]
                output.extend(rng.choice(self.members[a],2,replace=False).tolist())
                output.extend(rng.choice(self.members[b],2,replace=False).tolist())
        return iter(output)


class EvidenceState:
    def __init__(self,config,paths,labels):
        options=config['loss']['supervision_evidence']
        self.route=options['route']
        if self.route not in ('R1','R2'):
            raise ValueError('Unknown evidence route')
        evidence=Path(options['table'])
        if digest(evidence)!=options['sha256']:
            raise ValueError('Frozen intervention evidence changed')
        with evidence.open() as f:
            self.rows=[r for r in csv.DictReader(f) if r['split']=='train']
        if len(self.rows)!=len(paths):
            raise ValueError('Evidence train row count mismatch')
        for r,p,y in zip(self.rows,paths,labels):
            if r['image_path'].removeprefix('train/')!=str(p).removeprefix('train/') or int(r['original_label'])!=y:
                raise ValueError('Evidence train path/label order mismatch')
        if any(r['channel'] not in ('original','soft','uncertain') for r in self.rows):
            raise ValueError('Invalid channel')
        if config['loss'].get('mixup_probability',0)!=0 or config['trust']['enabled']:
            raise ValueError('Evidence recipe requires unmixed images and no legacy trust')
        self.channels=torch.tensor([{'original':0,'soft':1,'uncertain':2}[r['channel']] for r in self.rows])
        self.labels=torch.tensor(labels)
        group_ids={g:i for i,g in enumerate(sorted({r['content_group'] for r in self.rows}))}
        self.groups=torch.tensor([group_ids[r['content_group']] for r in self.rows])
        self.num_classes=int(config['model']['num_classes'])
        self.cached=bool(config['model'].get('use_cached_training',False))
        self.teacher=None
        self.batch_targets=None
        self.epoch=0
        slice_path=Path(options['slices'])
        if digest(slice_path)!=options['slices_sha256']:
            raise ValueError('Frozen mechanism slices changed')
        self.edges=json.loads(slice_path.read_text())['trusted_edges']
        if self.route=='R2' and config['train']['batch_size']!=4:
            raise ValueError('R2 requires microbatch 4 for the frozen 2x2 pairing')
    def begin_epoch(self,model,epoch):
        self.epoch=epoch
        if self.route=='R1' and not self.cached:
            if self.teacher is None:
                self.teacher=copy.deepcopy(model)
            else:
                self.teacher.load_state_dict(model.state_dict())
            self.teacher.eval().requires_grad_(False)
    def batch(self,indices,device):
        idx=indices.detach().cpu().long()
        channels=self.channels[idx].to(device)
        labels=self.labels[idx].to(device)
        targets=F.one_hot(labels,self.num_classes).float()
        for j,i in enumerate(idx.tolist()):
            if self.channels[i]==1:
                targets[j].zero_()
                votes={int(k):float(v) for k,v in json.loads(self.rows[i]['neighbor_votes']).items()}
                for c,count in votes.items():
                    if c<0 or c>=self.num_classes or count<0:
                        raise ValueError('Invalid external soft reference')
                    targets[j,c]=count
                if targets[j].sum()<=0:
                    raise ValueError('Empty external soft reference')
                targets[j]/=targets[j].sum()
        return channels,labels,targets,self.groups[idx].to(device)
    def global_loss(self,model,arguments,logits,encoded,indices,base,totals):
        channels,labels,targets,groups=self.batch(indices,logits.device)
        self.auxiliary_loss = logits.sum()*0
        if self.route=='R2':
            contrast=supervised_contrastive(encoded,labels,groups,channels==0,self.edges)
            self.auxiliary_loss = .1*contrast
            totals['evidence_contrastive_sum']=totals.get('evidence_contrastive_sum',0.)+float(contrast.detach())*len(logits)
            return base
        consistency=torch.zeros_like(base)
        if self.teacher is not None:
            if 'images' not in arguments:
                raise ValueError('R1 visual stage requires actual image views')
            with torch.no_grad():
                teacher_logits,teacher_features=self.teacher(images=arguments['images'].flip(3),return_features=True)
            targets=anchored_soft_targets(targets,teacher_logits.float().softmax(1),channels)
            consistency=(1-F.cosine_similarity(encoded.float(),teacher_features.detach().float(),dim=1))*(channels==2)
        self.batch_targets=targets
        self.batch_channels=channels
        for c,name in enumerate(('original','soft','uncertain')):
            key='evidence_'+name+'_examples'
            totals[key]=totals.get(key,0)+int((channels==c).sum())
        totals['evidence_uncertain_consistency_sum']=totals.get('evidence_uncertain_consistency_sum',0.)+float(consistency.detach().sum())
        return channel_loss(logits,channels,targets)+consistency
    def local_loss(self,logits,base,confidence,gate,inputs,global_loss):
        if self.route=='R2':
            admitted=confidence>=gate
            return torch.where(admitted,base,global_loss),admitted
        channels=self.batch_channels
        finite=torch.isfinite(inputs).flatten(1).all(1)
        varied=inputs.float().flatten(2).std(2).mean(1)>=.05
        admitted=finite&varied&((channels==0)|((channels==1)&(confidence>=gate)))
        local=channel_loss(logits,channels,self.batch_targets)
        # Unknown channel retains its global view-consistency term on fallback.
        # No class loss is injected by either global or local branch.
        return torch.where(admitted,local,global_loss),admitted
