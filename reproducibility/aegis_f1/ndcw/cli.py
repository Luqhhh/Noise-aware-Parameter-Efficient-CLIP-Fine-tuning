"""CPU-only ND-CW preparation. GPU execution uses the existing V2 runtime."""
from __future__ import annotations

import argparse
import json

from . import audit, io, paired, prepare, signals
from v2.plan import dump


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    pre = subs.add_parser('preflight')
    pre.add_argument('--config', required=True)
    pre.add_argument('--output', required=True)
    build = subs.add_parser('build-candidates')
    build.add_argument('--config', required=True)
    build.add_argument('--output', required=True)
    for name in ('perceptual-verify', 'audit-conflicts'):
        sub = subs.add_parser(name)
        sub.add_argument('--audit', required=True)
    freeze = subs.add_parser('build-manifest')
    freeze.add_argument('--audit', required=True)
    freeze.add_argument('--output', required=True)
    freeze.add_argument('--medium', required=True, type=float)
    freeze.add_argument('--strong', required=True, type=float)
    freeze.add_argument('--rationale', required=True)
    pair = subs.add_parser('prepare-pair')
    pair.add_argument('--manifest', required=True)
    pair.add_argument('--output', required=True)
    report = subs.add_parser('report')
    report.add_argument('--pair', required=True)
    report.add_argument('--output', required=True)
    report.add_argument('--probe', action='store_true')
    args = p.parse_args(argv)
    if args.command == 'preflight':
        cfg = audit.config(args.config)
        rows, split, inputs = audit.source_binding(args.config, cfg)
        table, sources = signals.load(cfg, rows, inputs)
        result = dict(status='engineering_inputs_verified_not_A0', total_images=len(rows),
                      train_dev=len(split['train']), val_dev=len(split['val']),
                      v1_trust_known=sum(s['v1_low_trust'] is not None for s in table.values()),
                      oof_probabilities_known=sum(s['oof_label_probability'] is not None for s in table.values()),
                      test_read=False, training_started=False, thresholds_frozen=False, inputs={**inputs, **sources})
        destination = io.fresh(args.output)
        dump(destination / 'preflight.json', result)
    elif args.command == 'build-candidates':
        result = dict(output=str(audit.build(args.config, args.output)))
    elif args.command == 'perceptual-verify':
        result = audit.verify(args.audit)
    elif args.command == 'audit-conflicts':
        result = audit.audit(args.audit)
    elif args.command == 'build-manifest':
        result = audit.freeze(args.audit, args.output, args.medium, args.strong, args.rationale)
    elif args.command == 'prepare-pair':
        result = prepare.prepare_pair(args.manifest, args.output)
    else:
        result = paired.report(args.pair, args.output, probe=args.probe)
    print(json.dumps({k: v for k, v in result.items() if k not in ('inputs', 'files', 'similarity_curve')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
