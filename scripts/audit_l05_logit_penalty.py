#!/usr/bin/env python3
"""Independent CPU decode and provenance audit for completed SD01."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from run_l05_logit_penalty import ROOT, verify_control
from run_l05_mixstyle import verify_framework, digest, dump
from audit_l05_mixstyle_decode import numpy_decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    cfg = json.loads(Path(args.config).read_text())
    assert cfg['temperature'] == 1.4 and cfg['prior_strength'] == .6
    out = ROOT / cfg['output']
    framework = out / 'framework'
    source = verify_framework(cfg, framework)
    verify_control(cfg)
    sys.path.insert(0, str(framework / 'reproducibility/aegis_f1'))
    from aegis_clip.config import load_config
    from aegis_clip.tta_prior_binding import checkpoint_identity, validation_cache_identity
    from aegis_clip.tta import fuse_paired_logits
    from aegis_clip.prior_alignment import fit_prior_bias, apply_prior_bias
    run = out / 'runs/SD01/seed42'
    target = run / 'decode_audit.json'
    if target.exists():
        raise FileExistsError('Preserve existing audit')
    complete = json.loads((run / 'training_complete.json').read_text())
    evaluation = json.loads((run / 'evaluation.json').read_text())
    config = load_config(out / 'configs/SD01.yaml')
    checkpoint = run / 'checkpoints/best.pt'
    identity = checkpoint_identity(checkpoint, config)
    cache = run / 'val_branch_logits.pt'
    binding = validation_cache_identity(cache, config, identity)
    assert identity['checkpoint_sha256'] == complete['checkpoint_sha256'] == evaluation['checkpoint_sha256']
    assert binding == evaluation['binding'] and not complete['smoke'] and not evaluation['test_data_used']
    sidecars = {}
    for name in ('best.pt', 'epoch_1.pt', 'last.pt'):
        path = run / 'checkpoints' / name
        record = json.loads(Path(str(path) + '.logit_penalty.json').read_text())
        assert record['checkpoint_sha256'] == digest(path)
        assert record['options']['source_sha256'] == digest(ROOT / 'scripts/l05_logit_penalty_support.py')
        assert record['options']['coefficient_file_sha256'] == digest(out / 'coefficient.json')
        assert record['state']['coefficient'] == record['options']['coefficient']
        sidecars[name] = record
    payload = torch.load(cache, map_location='cpu', weights_only=False)
    original, flip = (payload[k].float() for k in ('original_logits', 'flip_logits'))
    labels = payload['labels'].long().numpy()
    fused = fuse_paired_logits(original, flip, mode='mean_probabilities', temperature=1.4)
    bias, _ = fit_prior_bias(fused)
    torch_pred = apply_prior_bias(fused, bias, strength=.6).argmax(1).numpy()
    pred, numpy_bias, iterations = numpy_decode(original.numpy().astype(np.float64), flip.numpy().astype(np.float64))
    assert np.array_equal(pred, torch_pred)
    counts = np.bincount(labels, minlength=original.shape[1])
    assert np.all(counts > 0)
    correct = np.bincount(labels[pred == labels], minlength=len(counts))
    measured = {'macro': float((correct/counts).mean()), 'micro': float((pred == labels).mean())}
    for k in measured:
        assert abs(measured[k] - evaluation['decode'][k]) < 1e-6
    report = {'source_receipt': source, 'binding': binding, 'sidecars': sidecars,
              'evaluation_sha256': digest(run / 'evaluation.json'), 'script_sha256': digest(__file__),
              'numpy_float64_metrics': measured, 'identical_decoded_predictions': len(labels),
              'max_bias_difference': float(np.max(np.abs(numpy_bias - bias.numpy()))),
              'prior_iterations': iterations, 'test_data_used': False,
              'limitation': 'Cache decode audit, not another independent image-inference pass or platform proof'}
    dump(target, report)
    print(json.dumps({k: report[k] for k in ('numpy_float64_metrics', 'identical_decoded_predictions', 'max_bias_difference')}, indent=2))


if __name__ == '__main__':
    main()
