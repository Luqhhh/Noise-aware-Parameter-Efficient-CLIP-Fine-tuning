#!/usr/bin/env python3
"""Real CPU parent-load and zero-residual parity check; no GPU use."""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path

import torch

from aegis_clip.checkpoint import load_initial_weights
from aegis_clip.config import load_config
from aegis_clip.model import build_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'outputs/codex/p75_mechanisms_20260927/configs/P75_PATCH_READOUT.yaml'
PARENT = Path('/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus/local_parent/RM_LP/seed42/checkpoints/best.pt')


def main():
    torch.set_num_threads(4)
    config = load_config(CONFIG)
    assert config['model']['patch_readout'] == {'enabled': True, 'version': 1}
    image = torch.rand(1, 3, 224, 224)
    base = copy.deepcopy(config)
    base['model'].pop('patch_readout')
    ordinary, _ = build_model(base, torch.device('cpu'))
    load_initial_weights(ordinary, PARENT, torch.device('cpu'))
    ordinary.eval()
    with torch.no_grad():
        expected = ordinary(images=image)
    del ordinary
    gc.collect()
    candidate, _ = build_model(config, torch.device('cpu'))
    load_initial_weights(candidate, PARENT, torch.device('cpu'))
    candidate.eval()
    with torch.no_grad():
        observed = candidate(images=image)
    max_error = float((observed - expected).abs().max())
    same_top1 = bool(torch.equal(observed.argmax(1), expected.argmax(1)))
    groups = candidate.parameter_groups(
        head_lr=config['train']['head_lr'],
        head_weight_decay=config['train']['head_weight_decay'],
        backbone_lr=config['train']['backbone_lr'],
        backbone_weight_decay=config['train']['backbone_weight_decay'])
    names = [group['name'] for group in groups]
    readout_ids = {id(parameter) for parameter in candidate.patch_readout.parameters()}
    head_ids = {id(parameter) for group in groups if group['name'] == 'head'
                for parameter in group['params']}
    assert readout_ids <= head_ids
    assert max_error <= 1e-5 and same_top1, (max_error, same_top1)
    result = {'max_abs_initial_logit_error': max_error, 'same_top1': same_top1,
              'readout_in_head_optimizer_group': True, 'optimizer_groups': names,
              'device': 'cpu', 'gpu_used': False}
    out = CONFIG.parents[1]/'patch_cpu_smoke.json'
    out.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
