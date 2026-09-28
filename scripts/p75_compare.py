#!/usr/bin/env python3
"""Paired L05 versus P75 validation changes under the frozen decode rule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aegis_clip.prior_alignment import fit_prior_bias, apply_prior_bias
from aegis_clip.tta import fuse_paired_logits

REFERENCE = Path('/home/lux1/noise/worktrees/rematch750_f05_focus/artifacts/f05_focus_l05/l05_val_branch_logits.pt')


def prediction(payload):
    logits = fuse_paired_logits(payload['original_logits'].float(),
                               payload['flip_logits'].float(),
                               mode='mean_probabilities', temperature=1.4)
    bias, _ = fit_prior_bias(logits)
    return apply_prior_bias(logits, bias, strength=.6).argmax(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-cache', required=True)
    parser.add_argument('--candidate-evaluation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    baseline = torch.load(REFERENCE, map_location='cpu', weights_only=False)
    candidate = torch.load(args.candidate_cache, map_location='cpu', weights_only=False)
    if baseline['paths'] != candidate['paths'] or not torch.equal(
        baseline['labels'], candidate['labels']
    ):
        raise ValueError('L05 and candidate validation rows do not align')
    labels = torch.as_tensor(baseline['labels']).long()
    if len(labels) != 14880 or int(labels.min()) != 0 or int(labels.max()) != 749:
        raise ValueError('Validation scope changed')
    control = prediction(baseline)
    challenger = prediction(candidate)
    correct_base = control == labels
    correct_new = challenger == labels
    corrected = (~correct_base) & correct_new
    regressed = correct_base & (~correct_new)
    counts = torch.bincount(labels, minlength=750)
    before = torch.bincount(labels[correct_base], minlength=750)
    after = torch.bincount(labels[correct_new], minlength=750)
    macro = float((after.double()/counts).mean())
    micro = float(correct_new.double().mean())
    measured = json.loads(Path(args.candidate_evaluation).read_text())
    if abs(macro-measured['decode']['macro']) > 1e-6 or abs(micro-measured['decode']['micro']) > 1e-6:
        raise ValueError('Independent frozen decode does not match official evaluation')
    by_class = []
    for index in range(750):
        by_class.append({'class_id': index, 'validation_samples': int(counts[index]),
                         'corrections': int((corrected & (labels == index)).sum()),
                         'regressions': int((regressed & (labels == index)).sum()),
                         'delta_accuracy': float((after[index]-before[index])/counts[index])})
    result = {'baseline': {'macro': float((before.double()/counts).mean()),
                            'micro': float(correct_base.double().mean())},
              'candidate': {'macro': macro, 'micro': micro},
              'corrections': int(corrected.sum()), 'regressions': int(regressed.sum()),
              'net_correct': int(corrected.sum()-regressed.sum()),
              'classes_with_corrections': sum(row['corrections'] > 0 for row in by_class),
              'classes_with_regressions': sum(row['regressions'] > 0 for row in by_class),
              'by_class': by_class, 'test_used': False}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('baseline','candidate','corrections',
                                                     'regressions','net_correct')}))


if __name__ == '__main__':
    main()
