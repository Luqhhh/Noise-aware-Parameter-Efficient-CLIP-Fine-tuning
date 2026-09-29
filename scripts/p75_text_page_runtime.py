"""Only global/local classification masking; immutable batch denominator and feature anchor."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch


def mask_classification(per_sample, selected, enabled):
    if per_sample.ndim != 1 or selected.shape != per_sample.shape or selected.dtype != torch.bool:
        raise ValueError('Classification mask must be an aligned boolean vector')
    return torch.where(selected, torch.zeros_like(per_sample), per_sample) if enabled else per_sample


class TextPageState:
    def __init__(self, config, paths, labels, device):
        spec = config['loss']['text_page_mask']
        self.enabled = spec['enabled']
        path = Path(spec['selection'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('Pixel evidence changed')
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rows = [r for r in rows if r['split']=='train']
        if len(rows)!=len(paths): raise ValueError('Training coverage mismatch')
        for r,p,y in zip(rows,paths,labels):
            if r['image_path'].removeprefix('train/') != str(p).removeprefix('train/') or r['label']!=y:
                raise ValueError('Training path/label order mismatch')
            if not isinstance(r['selected'],bool): raise ValueError('Non-boolean mask')
        if config['loss'].get('mixup_probability',0) or config['trust']['enabled']:
            raise ValueError('This fixed comparison requires original unmixed sampling')
        self.selected = torch.tensor([r['selected'] for r in rows],device=device)

    def global_loss(self, per_sample, logits, targets, indices, totals):
        self.batch_selected = self.selected[indices]
        # Aggregate each group's actual training views. These are new records,
        # never backfilled from final L05 inference. Grad-scale is relative to CE.
        with torch.no_grad():
            probs = logits.float().softmax(1)
            py = (probs*targets).sum(1)
            pm = probs.max(1).values
            for name, chosen in [('selected',self.batch_selected),('remaining',~self.batch_selected)]:
                for key, value in [('rows', chosen.sum()), ('py_sum', py[chosen].sum()),
                    ('sqrt_py_sum',py[chosen].sqrt().sum()),('pmax_sum',pm[chosen].sum()),
                    ('raw_classification_sum',per_sample[chosen].sum())]:
                    k='text_page_'+name+'_'+key
                    totals[k]=totals.get(k,0.)+float(value)
        return mask_classification(per_sample,self.batch_selected,self.enabled)

    def local_loss(self, per_sample):
        return mask_classification(per_sample,self.batch_selected,self.enabled)
