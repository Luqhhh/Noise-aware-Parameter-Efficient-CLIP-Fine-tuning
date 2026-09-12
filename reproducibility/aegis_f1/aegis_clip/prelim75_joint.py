"""Fixed 3-epoch global/local continuation of the complete FULLFT_DUAL model."""
from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from aegis_clip.checkpoint import _atomic_torch_save
from aegis_clip.local_inference import adapted_dual_local_view_logits
from aegis_clip.prelim75 import (
    OfficialImages, audit_parent, candidate_config, gpu_setup, load_composite,
    loader, read_cache, saved_candidate, source_manifest, SCALE_WEIGHTS,
)
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.trainer import _per_sample_loss, _warmup_cosine


def fixed_choices(paths, epoch):
    flips, scales = [], []
    for path in paths:
        digest = hashlib.sha256(f'42/{epoch}/{path}'.encode()).digest()
        flips.append(bool(digest[0] & 1))
        draw = int.from_bytes(digest[1:9], 'little') / 2**64
        scales.append(min(int(np.searchsorted(np.cumsum(SCALE_WEIGHTS), draw, side='right')), 3))
    return np.asarray(flips, dtype=bool), np.asarray(scales, dtype=np.int64)


def crop_with_bound_boxes(images, boxes):
    if len(images) != len(boxes):
        raise ValueError('Box/image row mismatch')
    crops = []
    for image, box in zip(images, boxes):
        x0, y0, x1, y1 = map(int, box)
        if not (0 <= x0 < x1 <= 224 and 0 <= y0 < y1 <= 224):
            raise ValueError('Invalid native-224 box')
        crop = image[:, y0:y1, x0:x1].unsqueeze(0)
        crops.append(F.interpolate(crop, size=(224, 224), mode='bilinear', align_corners=False))
    return torch.cat(crops, dim=0)


def official_anchor(device):
    import clip
    official, _ = clip.load('ViT-B/32', device='cpu', jit=False)
    visual = official.visual.float().to(device).eval().requires_grad_(False)
    del official
    return visual


def weighted_micro_loss(logits, targets, weights, denominator, loss_config, epoch):
    return (_per_sample_loss(logits, targets, loss_config, epoch) * weights).sum() / denominator


def train_joint(plan, name):
    if name not in ('V0', 'V1'):
        raise ValueError('Only the two registered visual candidates are authorized')
    output = Path(plan['output']) / name
    output.mkdir(parents=True, exist_ok=False)
    device = gpu_setup(); start = time.monotonic()
    data, arrays, cache_manifest = read_cache(plan)
    model, preprocess, checkpoint, o3, pta = load_composite(plan['parent'], device)
    # Reuse precisely the full-FT model's own visual permission mask.
    licensed_visual = {n for n,p in model.visual.named_parameters() if p.requires_grad}
    if 'conv1.weight' in licensed_visual or 'positional_embedding' in licensed_visual:
        raise ValueError('Native full-FT geometry freeze mask changed')
    model.classifier.requires_grad_(True)
    o3.requires_grad_(name == 'V1'); pta.requires_grad_(name == 'V1')
    model.train(); o3.train(name == 'V1'); pta.train(name == 'V1')
    teacher = official_anchor(device)
    config = candidate_config(checkpoint, plan, name)
    config['train'].update({'epochs': 3, 'schedule_epochs': 18, 'batch_size': 32,
                            'backbone_lr': 1e-6, 'head_lr': 1.5e-7, 'amp': False})
    audit_parent(config, plan, output)
    atomic_json_dump(config, output/'resolved_config.json')
    atomic_json_dump(source_manifest(), output/'source_manifest.json')
    frozen = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
    visual_parameters = [p for p in model.visual.parameters() if p.requires_grad]
    groups = [{'name':'backbone', 'params':visual_parameters, 'lr':1e-6, 'weight_decay':0.},
              {'name':'head', 'params':list(model.classifier.parameters()), 'lr':1.5e-7, 'weight_decay':1e-4}]
    adapter_parameters = list(o3.parameters()) + list(pta.parameters()) if name == 'V1' else []
    if adapter_parameters:
        groups.append({'name':'local_adapters', 'params':adapter_parameters, 'lr':3e-6, 'weight_decay':0.})
    optimizer = torch.optim.AdamW(groups)
    dataset = OfficialImages(data['paths'], plan['train_root'], preprocess)
    generator = torch.Generator().manual_seed(42)
    stream = loader(dataset, 32, plan, shuffle=True, generator=generator)
    total_steps = 18 * len(stream)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _warmup_cosine(step, 0, total_steps))
    all_parameters = visual_parameters + list(model.classifier.parameters()) + adapter_parameters
    recipe = {'epochs':3, 'schedule_epochs':18, 'batch_size':32, 'effective_batch_size':32,
              'microbatch_size':32, 'accumulation':1, 'global_weight':.6 if name == 'V1' else 1.,
              'local_weight':.4 if name == 'V1' else 0., 'anchor_weight':2.,
              'geometry':'native_clip_center_crop_224_plus_stateless_flip',
              'anchor':'frozen_official_same_global_pixel_tensor_online',
              'boxes':'frozen_P_native_224_orientation_and_scale_bound',
              'boxes_file_sha256':cache_manifest['files_sha256']['boxes.npy'],
              'licensed_visual_parameters': sorted(licensed_visual),
              'supervision':'fixed_parent_epoch3_w_q', 'selection':'last_epoch'}
    atomic_json_dump(recipe, output/'training_recipe.json')
    microbatch = 32; history = []; optimizer_step = 0
    for epoch in range(1, 4):
        flips, scales = fixed_choices(data['paths'], epoch)
        totals = {'global_loss':0., 'local_loss':0., 'anchor_loss':0., 'loss':0., 'samples':0}
        for step, batch in enumerate(stream):
            indices = batch['index'].numpy()
            full_weights = data['weights'][indices]
            denominator = full_weights.sum().to(device).clamp_min(1e-8)
            # A single effective group stays intact when OOM requires 16+16.
            while True:
                optimizer.zero_grad(set_to_none=True)
                parts = {'global_loss':0., 'local_loss':0., 'anchor_loss':0., 'loss':0.}
                try:
                    for offset in range(0, len(indices), microbatch):
                        sub_idx = indices[offset:offset+microbatch]
                        images = batch['images'][offset:offset+microbatch].to(device, non_blocking=True)
                        flip = flips[sub_idx]
                        if flip.any():
                            mask = torch.tensor(flip, device=device)
                            images[mask] = images[mask].flip(3)
                        q = data['targets'][sub_idx].to(device)
                        w = data['weights'][sub_idx].to(device)
                        with torch.no_grad():
                            reference = F.normalize(teacher(images).float(), dim=1)
                        z_global, h_global = model(images=images, return_features=True)
                        global_loss = weighted_micro_loss(z_global, q, w, denominator, checkpoint['config']['loss'], epoch+3)
                        anchor = (1.-F.cosine_similarity(h_global.float(), reference, dim=1)).sum()/len(indices)
                        loss = (.6 if name == 'V1' else 1.) * global_loss + 2.*anchor
                        parts['global_loss'] += float(global_loss.detach())
                        parts['anchor_loss'] += float(anchor.detach())
                        if name == 'V1':
                            boxes = arrays['boxes'][sub_idx, flip.astype(int), scales[sub_idx]]
                            local = crop_with_bound_boxes(images, boxes)
                            z_local = adapted_dual_local_view_logits(model, o3, pta, local)
                            local_loss = weighted_micro_loss(z_local, q, w, denominator, checkpoint['config']['loss'], epoch+3)
                            if optimizer_step == 0 and offset == 0:
                                probes = torch.autograd.grad(local_loss,
                                    [model.visual.proj, o3.up.weight, pta.up.weight], retain_graph=True)
                                local_audit = {key: float(g.detach().norm()) for key,g in
                                               zip(('visual_proj', 'o3_up', 'pta_up'), probes)}
                                if any(not math.isfinite(v) or v <= 0 for v in local_audit.values()):
                                    raise RuntimeError('Local-only loss does not reach visual/O3/PTA')
                                atomic_json_dump(local_audit, output/'local_only_gradient_audit.json')
                            loss = loss + .4*local_loss
                            parts['local_loss'] += float(local_loss.detach())
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite visual continuation loss')
                        parts['loss'] += float(loss.detach())
                        loss.backward()
                    break
                except torch.cuda.OutOfMemoryError:
                    if optimizer_step or microbatch != 32:
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    # The rejected first group made no optimizer mutation.
                    z_global = h_global = loss = global_loss = anchor = images = reference = None
                    z_local = local_loss = local = None
                    import gc
                    gc.collect(); torch.cuda.empty_cache()
                    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
                    microbatch = 16
                    recipe.update({'microbatch_size':16, 'accumulation':2,
                                   'oom_downgrade':'first_group_before_any_optimizer_step'})
                    atomic_json_dump(recipe, output/'training_recipe.json')
                    print('OOM engineering downgrade: batch16 accumulation2, effective32', flush=True)
            if optimizer_step == 0:
                def grad_norm(params):
                    return math.sqrt(sum(float(p.grad.detach().float().pow(2).sum()) for p in params if p.grad is not None))
                audit = {'visual_gradient_norm':grad_norm(visual_parameters),
                         'head_gradient_norm':grad_norm(model.classifier.parameters()),
                         'o3_gradient_norm':grad_norm(o3.parameters()),
                         'pta_gradient_norm':grad_norm(pta.parameters()),
                         'frozen_gradient_leaks':[n for n,p in model.named_parameters() if not p.requires_grad and p.grad is not None]}
                if audit['visual_gradient_norm'] <= 0 or audit['head_gradient_norm'] <= 0 or audit['frozen_gradient_leaks']:
                    raise RuntimeError('Invalid licensed visual/head gradient')
                if name == 'V1' and (audit['o3_gradient_norm'] <= 0 or audit['pta_gradient_norm'] <= 0):
                    raise RuntimeError('Joint local adapters did not receive gradients')
                atomic_json_dump(audit, output/'first_step_gradient_audit.json')
            torch.nn.utils.clip_grad_norm_(all_parameters, 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); optimizer_step += 1
            for key in parts:
                totals[key] += parts[key] * len(indices)
            totals['samples'] += len(indices)
            if step % 100 == 0:
                row = {'status':'training', 'candidate':name, 'epoch':epoch,
                       'completed_samples':totals['samples'], 'optimizer_steps':optimizer_step,
                       'elapsed_seconds':time.monotonic()-start}
                atomic_json_dump(row, output/'status.json'); print(json.dumps(row), flush=True)
        row = {'epoch':epoch, **{k:v/totals['samples'] for k,v in totals.items() if k!='samples'},
               'samples':totals['samples'], 'optimizer_steps':optimizer_step,
               'elapsed_seconds':time.monotonic()-start}
        history.append(row); atomic_json_dump(history, output/'training_history.json')
        print(json.dumps(row), flush=True)
    for n,p in model.named_parameters():
        if n in frozen and not torch.equal(p.detach().cpu(), frozen[n]):
            raise RuntimeError(f'Frozen geometry tensor changed: {n}')
    payload = saved_candidate(checkpoint, model, config, name, plan, optimizer, scheduler, 3,
                              o3 if name=='V1' else None, pta if name=='V1' else None)
    _atomic_torch_save(payload, output/'candidate.pt')
    result = {'status':'trained_pending_real_diagnostic', 'candidate':name, 'history':history,
              'checkpoint_sha256':sha256_file(output/'candidate.pt'),
              'elapsed_seconds':time.monotonic()-start,
              'max_cuda_memory_allocated':torch.cuda.max_memory_allocated(),
              'online_accuracy':None, 'validation_scope':'overlap_diagnostic'}
    atomic_json_dump(result, output/'training_result.json'); atomic_json_dump(result, output/'status.json')
    return output/'candidate.pt'
