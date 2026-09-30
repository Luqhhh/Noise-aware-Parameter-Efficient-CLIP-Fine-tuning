"""Check actual ND-CW inputs while rejecting every attempt to open a test asset."""
from __future__ import annotations

import argparse
import builtins
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reproducibility/aegis_f1'))
from ndcw import audit, io, signals
from v2.plan import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/v2_ndcw/audit.yaml'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original_open, original_path_open = builtins.open, Path.open
    opened = []
    def guard(name):
        if isinstance(name, (str, Path)):
            path = Path(name)
            if 'test' in path.parts or path.name == 'test_manifest.csv':
                raise AssertionError('A0 attempted to open a test asset')
            opened.append(str(path))
    def protected_open(name, *args, **kwargs):
        guard(name)
        return original_open(name, *args, **kwargs)
    def protected_path_open(path, *args, **kwargs):
        guard(path)
        return original_path_open(path, *args, **kwargs)
    builtins.open, Path.open = protected_open, protected_path_open
    try:
        cfg = audit.config(args.config)
        rows, split, inputs = audit.source_binding(args.config, cfg)
        table, sources = signals.load(cfg, rows, inputs)
    finally:
        builtins.open, Path.open = original_open, original_path_open
    result = dict(status='engineering_inputs_verified_not_A0', total_images=len(rows),
                  train_dev=len(split['train']), val_dev=len(split['val']),
                  v1_trust_known=sum(s['v1_low_trust'] is not None for s in table.values()),
                  oof_probabilities_known=sum(s['oof_label_probability'] is not None for s in table.values()),
                  test_read=False, test_open_guard_passed=True, guarded_open_calls=len(opened),
                  training_started=False, thresholds_frozen=False, inputs={**inputs, **sources})
    output = io.fresh(args.output)
    dump(output / 'preflight.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'inputs'}, indent=2))


if __name__ == '__main__':
    main()
