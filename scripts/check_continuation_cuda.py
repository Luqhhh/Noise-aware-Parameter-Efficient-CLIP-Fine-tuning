"""Bounded local CUDA numerics/memory check of the frozen continuation recipe."""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reproducibility/aegis_f1'))

import numpy as np
import torch
from torch.utils.data import DataLoader
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_pipeline import Images, image_transform, seed_training
from aegis_clip.v1_strategy import WeightAverage, target_probabilities
from v1_continuation.plan import verify
from v1_continuation.runtime import initial_model, mixed_backward, optimizer_for, scheduler_for, require_idle_cuda


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--execute', action='store_true')
    args=p.parse_args()
    if not args.execute:p.error('CUDA probe requires --execute')
    report_path=Path(args.report).resolve()
    if report_path.exists():raise FileExistsError(report_path)
    plan, context, targets=verify(args.plan)
    require_idle_cuda()
    model,zero=initial_model(plan,context)
    model=model.to('cuda')
    cfg,recipe=plan['config'],plan['config']['recipe']
    active=np.flatnonzero(targets['weights']>0)[:plan['logical_batch_size']*3]
    rows=[context.train[i] for i in active]
    data=Images(context.train_root,rows,image_transform(recipe['image_size'],training=True,
        crop_min=plan['inherited_train']['crop_min_scale']))
    loader=DataLoader(data,batch_size=plan['logical_batch_size'],num_workers=cfg['num_workers'])
    seed_training(cfg['seed'])
    tensors={k:torch.as_tensor(targets[k][active],device='cuda') for k in
             ('targets','labels','weights','original_alpha')}
    probabilities=target_probabilities(tensors['targets'],tensors['labels'],tensors['original_alpha'],
        len(plan['classes']),plan['inherited_train']['label_smoothing'])
    optimizer=optimizer_for(model,recipe)
    steps_per_epoch=math.ceil(plan['active_train_rows']/plan['logical_batch_size'])
    scheduler=scheduler_for(optimizer,recipe,steps_per_epoch)
    scaler=torch.amp.GradScaler('cuda',enabled=plan['inherited_train']['amp'])
    live=lambda:{n:p.detach() for n,p in model.named_parameters() if p.requires_grad}
    ema=WeightAverage(live(),recipe['ema_decay'])
    model.train(); torch.cuda.reset_peak_memory_stats()
    times,losses,norms=[],[],[]
    for images,indices in loader:
        started=time.monotonic()
        images,indices=images.to('cuda'),indices.to('cuda')
        permutation=torch.randperm(len(images),device='cuda')
        lam=float(np.random.beta(plan['inherited_train']['mixup_alpha'],plan['inherited_train']['mixup_alpha']))
        optimizer.zero_grad(set_to_none=True)
        loss=mixed_backward(model,images,probabilities[indices],tensors['weights'][indices],
            permutation,lam,cfg['micro_batch_size'],scaler,plan['inherited_train']['amp'])
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['grad_clip'])
        if not torch.isfinite(norm):raise ValueError('Nonfinite CUDA gradient')
        before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<before:raise ValueError('Probe optimizer update skipped')
        scheduler.step();ema.update(live());torch.cuda.synchronize()
        times.append(time.monotonic()-started);losses.append(loss);norms.append(float(norm))
        print(dict(probe_update=len(times),seconds=times[-1],loss=loss,grad_norm=norms[-1]),flush=True)
    peak=torch.cuda.max_memory_allocated()/2**20
    model.eval()
    eval_times=[]
    # Probe an inference batch32 separately: this does not change the training logical batch.
    eval_images=torch.stack([Images(context.train_root,rows[:32],image_transform(recipe['image_size']))[i][0]
                             for i in range(min(32,len(rows)))]).to('cuda')
    with torch.no_grad():
        for _ in range(3):
            start=time.monotonic();logits=model(eval_images);torch.cuda.synchronize()
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite FP32 evaluation logits')
            eval_times.append(time.monotonic()-start)
    eval_peak=torch.cuda.max_memory_allocated()/2**20
    report=dict(status='passed',experiment_id=plan['experiment_id'],plan_sha256=sha256_file(args.plan),
        device=torch.cuda.get_device_name(),zero_update=zero,successful_updates=len(times),
        logical_batch_size=plan['logical_batch_size'],micro_batch_size=cfg['micro_batch_size'],
        train_seconds=times,losses=losses,gradient_norms=norms,training_peak_mib=peak,
        fp32_inference_batch_size=len(eval_images),fp32_inference_seconds=eval_times,inference_peak_mib=eval_peak,
        fitted_weights_saved=False,train_estimated_hours=max(times[1:])*steps_per_epoch*4/3600,
        inference_estimate_seconds_per_image=max(eval_times[1:])/len(eval_images),
        estimated_cuda_work_hours=(max(times[1:])*steps_per_epoch*4+
            max(eval_times[1:])/len(eval_images)*(plan['val_rows']*10+context.manifest['test_samples'])*
            len(recipe['views'])*(2 if recipe['flip'] else 1))/3600,
        costs_are_estimates=True)
    atomic_json_dump(report,report_path)
    print(report,flush=True)
    del model,optimizer,ema,probabilities,tensors,eval_images,logits
    gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
