#!/usr/bin/env python3
"""CPU-only NumPy float64 replay of a completed MixStyle-arm validation decode."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from run_l05_mixstyle import ROOT, digest, dump, verify_framework


def softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    return probability / probability.sum(axis=1, keepdims=True)


def numpy_decode(original, flip):
    fused = np.log(np.maximum((softmax(original / 1.4) + softmax(flip / 1.4)) / 2, 1e-12))
    prior = np.full(fused.shape[1], 1 / fused.shape[1], dtype=np.float64)
    bias = np.zeros_like(prior)
    for iteration in range(1, 51):
        marginal = softmax(fused + bias).mean(axis=0)
        if np.max(np.abs(marginal - prior)) <= 1e-6:
            break
        bias += .5 * (np.log(prior) - np.log(np.maximum(marginal, 1e-12)))
        bias -= bias.mean()
    return (fused + .6 * bias).argmax(axis=1), bias, iteration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm', choices=('MS00', 'MS01'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = json.loads(Path(args.config).read_text())
    if (cfg['temperature'], cfg['prior_strength']) != (1.4, .6):
        raise ValueError('Fixed decode recipe changed')
    out = ROOT / cfg['output']
    framework = out / 'framework'
    receipt = verify_framework(cfg, framework)
    sys.path.insert(0, str(framework / 'reproducibility/aegis_f1'))
    from aegis_clip.config import load_config
    from aegis_clip.tta_prior_binding import checkpoint_identity, validation_cache_identity
    from aegis_clip.tta import fuse_paired_logits
    from aegis_clip.prior_alignment import fit_prior_bias, apply_prior_bias

    run = out / 'runs' / args.arm / f'seed{cfg["seed"]}'
    target = run / 'decode_audit.json'
    if target.exists():
        raise FileExistsError('Preserve the existing audit')
    complete = json.loads((run / 'training_complete.json').read_text())
    evaluation = json.loads((run / 'evaluation.json').read_text())
    checkpoint = run / 'checkpoints/best.pt'
    cache = run / 'val_branch_logits.pt'
    config = load_config(out / 'configs' / f'{args.arm}.yaml')
    identity = checkpoint_identity(checkpoint, config)
    binding = validation_cache_identity(cache, config, identity)
    if not (identity['checkpoint_sha256'] == complete['checkpoint_sha256'] == evaluation['checkpoint_sha256']):
        raise ValueError('Result/checkpoint identity mismatch')
    if binding != evaluation['binding'] or complete['smoke'] or evaluation['test_data_used']:
        raise ValueError('Unexpected evaluation provenance')
    payload = torch.load(cache, map_location='cpu', weights_only=False)
    original, flip = [payload[key].float() for key in ('original_logits', 'flip_logits')]
    labels = payload['labels'].long().numpy()
    fused = fuse_paired_logits(original, flip, mode='mean_probabilities', temperature=1.4)
    torch_bias, _ = fit_prior_bias(fused)
    torch_prediction = apply_prior_bias(fused, torch_bias, strength=.6).argmax(1).numpy()
    prediction, bias, iterations = numpy_decode(original.numpy().astype(np.float64), flip.numpy().astype(np.float64))
    if not np.array_equal(prediction, torch_prediction):
        raise ValueError('NumPy/Torch decoded predictions disagree')
    classes = original.shape[1]
    counts = np.bincount(labels, minlength=classes)
    if np.any(counts == 0):
        raise ValueError('Validation misses a class')
    correct = np.bincount(labels[prediction == labels], minlength=classes)
    metrics = {'macro': float((correct / counts).mean()), 'micro': float((prediction == labels).mean())}
    for key in metrics:
        if abs(metrics[key] - evaluation['decode'][key]) > 1e-6:
            raise ValueError(f'Reported {key} disagrees with independent replay')
    report = {'arm': args.arm, 'source_receipt': receipt, 'binding': binding,
              'script_sha256': digest(__file__), 'evaluation_sha256': digest(run / 'evaluation.json'),
              'validation_samples': len(labels), 'covered_classes': len(counts),
              'numpy_float64_metrics': metrics, 'identical_decoded_predictions': len(labels),
              'max_bias_difference': float(np.max(np.abs(bias - torch_bias.numpy()))),
              'prior_iterations': iterations, 'device': 'cpu', 'test_data_used': False,
              'limitation': 'Cache decode audit; does not independently rerun image inference or prove platform benefit'}
    dump(target, report)
    print(json.dumps({key: report[key] for key in ('arm', 'numpy_float64_metrics', 'identical_decoded_predictions', 'max_bias_difference')}, indent=2))


if __name__ == '__main__':
    main()
