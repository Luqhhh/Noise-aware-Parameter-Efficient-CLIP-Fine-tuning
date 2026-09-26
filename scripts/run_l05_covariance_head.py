#!/usr/bin/env python3
"""Fixed CPU-only L05 Gaussian-head fit and cache/reload audit."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from l05_covariance_support import fit_discriminants
from run_l05_nonlinear_head import (
    preflight, metrics, fixed_decode, write_json, sha256_file,
    _atomic_torch_save, build_from_checkpoint,
)
from audit_l05_mixstyle_decode import numpy_decode


def load_shards(directory, split, rows, identity):
    parts, receipts = [], []
    for offset in range(0, len(rows), 4096):
        path = directory / f'{split}_{offset:06d}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        expected = {**identity, 'split': split, 'offset': offset,
                    'paths': [r['image_path'] for r in rows[offset:offset + 4096]]}
        if payload['identity'] != expected:
            raise ValueError(f'Feature identity mismatch: {path}')
        if payload['labels'].tolist() != [int(r['label']) for r in rows[offset:offset + 4096]]:
            raise ValueError(f'Feature label mismatch: {path}')
        for key in ('center', 'flip') if split == 'val' else ('center',):
            if not torch.isfinite(payload[key]).all():
                raise ValueError('Nonfinite features')
        parts.append(payload)
        receipts.append({'path': str(path), 'sha256': sha256_file(path)})
    keys = ('center', 'flip', 'labels') if split == 'val' else ('center', 'labels')
    return {key: torch.cat([p[key] for p in parts]) for key in keys}, receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = json.loads(Path(args.config).read_text())
    if (cfg['temperature'], cfg['prior_strength'], cfg['covariance'], cfg['class_priors']) != (
            1.4, .6, 'pooled_class_residuals_mle_oas', 'uniform'):
        raise ValueError('Unexpected fixed recipe')
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=False)
    source = json.loads(Path(cfg['source_config']).read_text())
    rows, reference, identity = preflight(source)
    directory = Path(cfg['source_features'])
    if json.loads((directory / 'complete.json').read_text()) != identity:
        raise ValueError('Incomplete or different feature cache')
    train, train_receipts = load_shards(directory, 'train', rows['train'], identity)
    val, val_receipts = load_shards(directory, 'val', rows['val'], identity)
    print('Preflight and all feature shards verified', flush=True)
    x = F.normalize(train['center'], dim=-1).numpy()
    labels = val['labels']
    classes = reference['original_logits'].shape[1]
    heads, means, fit = fit_discriminants(x, train['labels'].numpy(), classes)
    del train, x
    base_center = metrics(reference['original_logits'], labels)
    base_decode, _, base_pred, _ = fixed_decode(reference['original_logits'], reference['flip_logits'], labels, cfg)
    parent = torch.load(source['parent_checkpoint'], map_location='cpu', weights_only=False)
    report = {'experiment_id': cfg['experiment_id'], 'configuration': cfg,
              'config_sha256': sha256_file(args.config), 'source_identity': identity,
              'source_config_sha256': sha256_file(cfg['source_config']),
              'feature_shards': train_receipts + val_receipts, 'fit': fit,
              'base_center': base_center, 'base_decode': base_decode, 'arms': {},
              'test_data_used': False, 'platform_result': None,
              'limitation': 'Frozen feature cache evaluation, not independent full-image inference; val prior fitted in-sample'}
    for arm, head in heads.items():
        path = out / arm
        path.mkdir()
        weight, bias = (torch.from_numpy(head[key]).float() for key in ('weight', 'bias'))
        branches = [F.linear(F.normalize(val[key], dim=-1), weight, bias) for key in ('center', 'flip')]
        original, flip = branches
        center = metrics(original, labels)
        decode, prior, prediction, fit_report = fixed_decode(original, flip, labels, cfg)
        stopped = any(center[k] < base_center[k] - .02 for k in ('macro', 'micro'))
        # Independent float64 probability fusion and IPF, on the actual fp32 head outputs.
        np_prediction, np_bias, iterations = numpy_decode(original.numpy().astype(np.float64), flip.numpy().astype(np.float64))
        if not np.array_equal(np_prediction, prediction.numpy()):
            raise ValueError('Independent decode predictions differ')
        model_cfg = copy.deepcopy(parent['config'])
        model_cfg['model'].update(peft_mode='frozen', use_cached_training=True)
        model_cfg['project'].update(experiment_id=cfg['experiment_id']+'_'+arm,
                                    protocol='l05_covariance_head_v1', parent_kind='same_split_continue',
                                    parent_experiment_id='RM_V5_L05_CUDA_LOCAL')
        model_cfg['train'] = {'epochs': 0, 'backbone_lr': 0., 'head_lr': 0., 'amp': False,
                              'init_checkpoint': source['parent_checkpoint'],
                              'entrypoint': 'scripts/run_l05_covariance_head.py'}
        model_cfg['output']['root'] = str(out.resolve())
        state = dict(parent['model_state_dict'])
        state['classifier.weight'], state['classifier.bias'] = weight, bias
        checkpoint = {'format_version': 1, 'config': model_cfg, 'model_state_dict': state,
                      'epoch': 0, 'metrics': decode, 'refit_config': cfg, 'refit_identity': identity,
                      'inference_prior_bias': prior, 'inference_recipe': {
                          'tta': 'horizontal_flip', 'fusion': 'mean_probabilities',
                          'temperature': 1.4, 'prior_strength': .6}}
        _atomic_torch_save(checkpoint, path / 'best.pt')
        _atomic_torch_save({'original_logits': original, 'flip_logits': flip, 'labels': labels,
                           'prior_bias': prior, 'identity': identity}, path / 'val_logits.pt')
        _atomic_torch_save({'means': means, **head, 'fit': fit}, path / 'fit.pt')
        model, _, loaded = build_from_checkpoint(path / 'best.pt', torch.device('cpu'))
        model.eval().requires_grad_(False)
        assert loaded['refit_identity'] == identity
        assert all(torch.equal(value, parent['model_state_dict'][key])
                   for key, value in model.state_dict().items() if key.startswith('visual.'))
        max_error, agreements = 0., 0
        with torch.inference_mode():
            for key, expected in zip(('center', 'flip'), branches):
                actual = model(features=val[key])
                max_error = max(max_error, float((actual - expected).abs().max()))
                agreements += int((actual.argmax(1) == expected.argmax(1)).sum())
        if max_error > 1e-4 or agreements != 2 * len(labels):
            raise ValueError('Native checkpoint reload differs')
        del model, loaded
        result = {'center': center, 'decode': decode, 'stopped': stopped,
                  'delta_vs_l05_pp': {k: 100 * (decode[k] - base_decode[k]) for k in ('macro', 'micro')},
                  'corrections': int(((prediction == labels) & (base_pred != labels)).sum()),
                  'regressions': int(((prediction != labels) & (base_pred == labels)).sum()),
                  'checkpoint_sha256': sha256_file(path / 'best.pt'),
                  'val_logits_sha256': sha256_file(path / 'val_logits.pt'),
                  'prior_fit': fit_report, 'independent_decode_identical': len(labels),
                  'independent_prior_bias_max_difference': float(np.max(np.abs(np_bias - prior.numpy()))),
                  'independent_prior_iterations': iterations, 'reload_max_abs': max_error,
                  'reload_branch_prediction_agreements': agreements, 'visual_bitwise_unchanged': True}
        report['arms'][arm] = result
        write_json(path / 'result.json', result)
        print(arm, json.dumps(result), flush=True)
    def gate(candidate, baseline):
        return candidate['macro'] >= baseline['macro'] + .003 and candidate['micro'] >= baseline['micro']
    a, b = report['arms']['CH00'], report['arms']['CH01']
    report['CH00_pass'] = not a['stopped'] and gate(a['decode'], base_decode)
    report['CH01_pass'] = not b['stopped'] and gate(b['decode'], base_decode) and gate(b['decode'], a['decode'])
    report['status'] = 'candidate_requires_image_audit' if report['CH00_pass'] or report['CH01_pass'] else 'closed_below_gate'
    write_json(out / 'result.json', report)
    print('Result:', report['status'], flush=True)


if __name__ == '__main__':
    main()
