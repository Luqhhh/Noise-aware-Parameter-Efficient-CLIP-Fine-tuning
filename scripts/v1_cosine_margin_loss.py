"""Train-only cosine margin for each of the two original weighted Mixup sources."""
from __future__ import annotations

import contextlib
import torch


def component_loss(logits,probabilities,scale,margin):
    if not 0 <= margin <= .05:
        raise ValueError('This probe only supports margin in [0,0.05]')
    labels=probabilities.argmax(1)
    shift=torch.zeros_like(logits).scatter(1,labels[:,None],(-margin*scale.to(logits)).expand(len(logits),1))
    return -(probabilities*(logits+shift).log_softmax(1)).sum(1)


def margin_mixup_loss(logits,probabilities,weights,permutation,lam,scale,margin):
    a=component_loss(logits,probabilities,scale,margin)
    b=component_loss(logits,probabilities[permutation],scale,margin)
    return (lam*weights*a+(1-lam)*weights[permutation]*b).sum()/weights.sum().clamp_min(1e-8)


def mixed_backward(model,images,probabilities,weights,permutation,lam,micro_batch,scaler,margin,amp):
    # Same image mixture and total reliable mass as the delivered control.
    mixed=lam*images+(1-lam)*images[permutation]
    denominator=weights.sum().clamp_min(1e-8); total=0.
    for start in range(0,len(images),micro_batch):
        stop=start+micro_batch
        autocast=torch.autocast('cuda',dtype=torch.float16) if amp else contextlib.nullcontext()
        with autocast:
            logits=model(mixed[start:stop]).float()
            scale=model.head.logit_scale.exp().clamp(1,100)
            a=component_loss(logits,probabilities[start:stop],scale,margin)
            b=component_loss(logits,probabilities[permutation[start:stop]],scale,margin)
            loss=(lam*weights[start:stop]*a+(1-lam)*weights[permutation[start:stop]]*b).sum()/denominator
        if not torch.isfinite(loss):raise ValueError('Nonfinite cosine-margin loss')
        scaler.scale(loss).backward();total+=float(loss.detach())
    return total


def logical_update(model,opt,images,probabilities,weights,permutation,lam,micro_batch,scaler,margin,grad_clip,amp=True):
    """Keep original same-batch AMP retry semantics and exactly one successful update."""
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(images.device) if images.is_cuda else None
    initial_scale=scaler.get_scale();retries=[]
    for attempt in range(17):
        if attempt:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,images.device)
        opt.zero_grad(set_to_none=True)
        loss=mixed_backward(model,images,probabilities,weights,permutation,lam,micro_batch,scaler,margin,amp)
        scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
        if not torch.isfinite(norm):
            scale=scaler.get_scale()
            if not amp or scale<=1 or attempt==16:raise ValueError('Persistent nonfinite cosine-margin gradients')
            scaler.update()
            if scaler.get_scale()>=scale:raise ValueError('AMP scale failed to decrease')
            retries.append(dict(old_scale=scale,new_scale=scaler.get_scale()));continue
        old_scale=scaler.get_scale();scaler.step(opt);scaler.update()
        if scaler.get_scale()<old_scale:raise ValueError('Unexpected skipped optimizer update')
        return loss,dict(gradient_norm=float(norm),amp_initial_scale=initial_scale,
                        amp_final_scale=scaler.get_scale(),amp_retries=retries)
    raise AssertionError('Unreachable retry state')
