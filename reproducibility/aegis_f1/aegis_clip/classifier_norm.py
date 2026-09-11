"""A fixed, training-free shared linear-head norm alignment."""
from __future__ import annotations

import torch


def mean_norm_scale(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or not weight.is_floating_point() or min(weight.shape) == 0:
        raise ValueError("weight must be a nonempty floating [classes, features] tensor")
    if not torch.isfinite(weight).all():
        raise ValueError("weight must be finite")
    norms = weight.norm(dim=1)
    if not torch.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("classifier row norms must be finite and positive")
    scale = norms.mean() / norms
    if not torch.isfinite(scale).all():
        raise ValueError("norm alignment scale must be finite")
    return scale


def align_branch_logits(logits: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Re-express W' x + b from W x + b, before any probability fusion.

    Also applies to the existing additive dual-adapter feature residual because
    every term uses the same shared classifier weight and exactly one bias.
    """
    if logits.ndim != 2 or bias.ndim != 1 or scale.shape != bias.shape or logits.shape[1] != bias.numel():
        raise ValueError("logits, bias and scale class dimensions must agree")
    if not all(torch.isfinite(x).all() for x in (logits, bias, scale)) or (scale <= 0).any():
        raise ValueError("logits/bias must be finite and scale finite positive")
    return (logits - bias) * scale + bias


def aligned_inference_checkpoint(checkpoint: dict, *, parent_sha256: str, authorization: dict) -> dict:
    """Create an inference-only artifact, preserving all non-head model tensors."""
    import copy
    if authorization.get('action') != 'materialize_classifier_norm_candidate' or not authorization.get('decision_source'):
        raise ValueError('Explicit candidate materialization authorization required')
    state = checkpoint['model_state_dict']
    weight, bias = state['classifier.weight'], state['classifier.bias']
    if bias.shape != (weight.shape[0],) or not torch.isfinite(bias).all():
        raise ValueError('Classifier bias must be finite and match weight rows')
    scale = mean_norm_scale(weight)
    result = copy.deepcopy(checkpoint)
    result['model_state_dict']['classifier.weight'] = weight * scale[:, None]
    for key in ('optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict',
                'rng_state', 'data_generator_state', 'metrics', 'best_selector',
                'adaptive_cap_state', 'elr_state_dict', 'training_aux_state'):
        result.pop(key, None)
    result['classifier_norm_alignment'] = {
        'parent_sha256': parent_sha256, 'scale': scale,
        'inference_only': True, 'authorization': copy.deepcopy(authorization),
    }
    return result
