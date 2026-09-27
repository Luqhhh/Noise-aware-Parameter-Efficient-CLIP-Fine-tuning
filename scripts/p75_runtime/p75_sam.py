"""Full-effective-batch SAM over the complete L05 global/local/anchor loss."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _second_loss(model, microbatch: dict, device: torch.device) -> torch.Tensor:
    from aegis_clip import trainer
    inputs = microbatch['forward_inputs'].to(device)
    reference = microbatch['mixed_reference'].to(device)
    targets = microbatch['mixed_targets'].to(device)
    weights = microbatch['mixed_weights'].to(device)
    arguments = {microbatch['forward_key']: inputs}
    if microbatch['mixed_gate'] is not None:
        arguments['gate'] = microbatch['mixed_gate'].to(device)
        arguments['reference_features'] = reference
    logits, encoded = model(**arguments, return_features=True)
    training_logits = trainer.class_prior_adjusted_logits(
        logits, microbatch['class_counts'], microbatch['prior_tau'])
    global_per = trainer._per_sample_loss(
        training_logits, targets, microbatch['loss_config'],
        microbatch['epoch'], (None if microbatch['batch_suspicious'] is None
                              else microbatch['batch_suspicious'].to(device)))
    global_loss = (global_per * weights).sum()/weights.sum().clamp_min(1e-8)
    attention = microbatch['local_attention']
    if attention is not None:
        if microbatch['forward_key'] != 'images':
            raise ValueError('SAM local supervision requires image input')
        local_inputs = trainer.attention_guided_crop(
            inputs, attention.to(device),
            crop_size=microbatch['local_crop_size'],
            top_patches=microbatch['local_top_patches'])
        local_logits = model(images=local_inputs)
        local_training_logits = trainer.class_prior_adjusted_logits(
            local_logits, microbatch['class_counts'], microbatch['prior_tau'])
        local_per = trainer._per_sample_loss(
            local_training_logits, targets, microbatch['loss_config'],
            microbatch['epoch'], (None if microbatch['batch_suspicious'] is None
                                  else microbatch['batch_suspicious'].to(device)))
        confidence = training_logits.detach().float().softmax(1).max(1).values
        local_per = torch.where(confidence >= microbatch['local_gate'],
                                local_per, global_per)
        local_loss = (local_per * weights).sum()/weights.sum().clamp_min(1e-8)
        local_weight = microbatch['local_weight']
        classification = (1. - local_weight)*global_loss + local_weight*local_loss
    else:
        classification = global_loss
    if microbatch['distill_weight']:
        drift = 1. - F.cosine_similarity(
            encoded.float(), F.normalize(reference.float(), dim=1), dim=1)
        classification = classification + microbatch['distill_weight']*drift.mean()
    return classification


def full_l05_sam_step(model, optimizer, *, microbatches: list[dict],
                      cycle_samples: int, rho: float, clip_norm: float,
                      gsam_alpha, rng_state: dict, device: torch.device,
                      **_kwargs) -> dict:
    """Perturb once after the entire effective-batch first gradient.

    The caller has accumulated first-pass gradients. Stored image tensors live
    on CPU; this routine replays the same images and detached crop geometry.
    Parameters are restored exactly before the sole AdamW update, including
    when the second pass raises.
    """
    if gsam_alpha not in (None, 0, 0.) or not 0. < rho < 1.:
        raise ValueError('P75 permits only fixed standard SAM with 0<rho<1')
    if not microbatches or cycle_samples != sum(x['sample_count'] for x in microbatches):
        raise ValueError('SAM microbatches do not form the complete effective batch')
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    active = [parameter for parameter in parameters if parameter.grad is not None]
    if not active or any(not torch.isfinite(parameter.grad).all() for parameter in active):
        raise FloatingPointError('SAM first gradient missing or nonfinite')
    first_norm = math.sqrt(sum(float(parameter.grad.detach().float().square().sum())
                               for parameter in active))
    if first_norm <= 0 or not math.isfinite(first_norm):
        raise FloatingPointError('SAM first gradient norm is invalid')
    first_loss = sum(x['first_loss']*x['sample_count'] for x in microbatches)/cycle_samples
    backups = [parameter.detach().clone() for parameter in parameters]
    after_first_rng = trainer_rng_state(device)
    actual_radius = None
    try:
        with torch.no_grad():
            for parameter in active:
                parameter.add_(parameter.grad, alpha=rho/(first_norm+1e-12))
            actual_radius = math.sqrt(sum(float((parameter-backup).float().square().sum())
                for parameter, backup in zip(parameters, backups)))
            if not math.isfinite(actual_radius) or abs(actual_radius-rho) > 1e-4:
                raise FloatingPointError('Actual SAM perturbation radius drifted')
        optimizer.zero_grad(set_to_none=not getattr(optimizer, 'is_fused_optimizer', False))
        from aegis_clip.trainer import _restore_rng_state
        _restore_rng_state(rng_state, device)
        second_loss_sum = 0.
        for microbatch in microbatches:
            loss = _second_loss(model, microbatch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError('SAM second loss is nonfinite')
            fraction = microbatch['sample_count']/cycle_samples
            (loss*fraction).backward()
            second_loss_sum += float(loss.detach())*fraction
    finally:
        with torch.no_grad():
            for parameter, backup in zip(parameters, backups):
                parameter.copy_(backup)
    if not equal_rng_state(trainer_rng_state(device), after_first_rng):
        raise RuntimeError('SAM second pass did not replay one random stream')
    second_active = [parameter for parameter in parameters if parameter.grad is not None]
    if {id(x) for x in second_active} != {id(x) for x in active}:
        raise RuntimeError('SAM first/second gradient parameter sets differ')
    if any(not torch.isfinite(parameter.grad).all() for parameter in second_active):
        raise FloatingPointError('SAM second gradient is nonfinite')
    second_norm = math.sqrt(sum(float(parameter.grad.detach().float().square().sum())
                                for parameter in second_active))
    if second_norm <= 0 or not math.isfinite(second_norm):
        raise FloatingPointError('SAM second gradient norm is invalid')
    preclip = float(torch.nn.utils.clip_grad_norm_(
        parameters, clip_norm, error_if_nonfinite=True))
    optimizer.step()
    return {'first_loss': first_loss, 'second_loss': second_loss_sum,
            'first_gradient_norm': first_norm, 'second_gradient_norm': second_norm,
            'preclip_gradient_norm': preclip, 'actual_radius': actual_radius,
            'gsam_alpha': None, 'gsam_projection': None}


def trainer_rng_state(device):
    state = {'cpu': torch.get_rng_state().clone()}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device).clone()
    return state


def equal_rng_state(left, right):
    return left.keys() == right.keys() and all(
        torch.equal(left[key], right[key]) for key in left)
