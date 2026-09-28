#!/usr/bin/env python3
"""Replay incumbent decode and inspect the train-only feature bank without CUDA."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from p75_supported_ce import FIXED, OUT, preflight, read_json, read_rows, train_features, write_json
from p75_supported_ce_report import grouped_report, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Optional new JSON path; existing files are never overwritten')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise RuntimeError('This audit must run in a fresh CPU-only process')
    saved = preflight()
    fixed = read_json(FIXED)
    training = read_rows(Path(fixed['assets'])/'train_dev.csv')
    bank = train_features(training)
    norms = np.linalg.norm(bank, axis=1)
    if not np.isfinite(bank).all() or not np.allclose(norms, 1., atol=1e-4):
        raise ValueError('Frozen training feature bank is nonfinite or unnormalized')
    feature_check = dict(shape=list(bank.shape), min_norm=float(norms.min()),
        max_norm=float(norms.max()), validation_rows_in_bank=0, neighbor_queries_run=0)
    del bank
    payload = torch.load(fixed['reference_validation_cache'], map_location='cpu', weights_only=False)
    prediction = predictions(payload)
    comparison = grouped_report(payload['labels'].numpy(), prediction, prediction, [],
                                saved['identity']['num_classes'])
    if comparison['corrections'] or comparison['regressions'] or torch.cuda.is_initialized():
        raise ValueError('Incumbent CPU self-comparison failed')
    result = dict(check='CPU-only incumbent self-comparison and train_dev feature-bank verification',
        feature_check=feature_check, incumbent_metrics=comparison['baseline'], corrections=0,
        regressions=0, validation_rows=sum(group['validation_samples'] for group in comparison['groups']),
        cuda_initialized=False, candidate_metrics=None)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
