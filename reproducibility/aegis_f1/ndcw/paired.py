"""Frozen paired evaluation, all errors and class-sensitive promotion gates."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from v2.plan import dump, require, sha, verify_prepared
from .io import fresh, write_csv
from .prepare import verify_bundle

METRICS = ('macro', 'micro', 'tail_macro', 'remaining_macro', 'trusted_macro', 'trusted_micro',
           'affected_macro', 'unaffected_macro')


def metrics(labels, predictions, groups, trusted_mask):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    require(labels.shape == predictions.shape and labels.ndim == 1, 'Invalid prediction vectors')
    correct = labels == predictions
    def score(mask):
        mask = np.asarray(mask, dtype=bool)
        classes = np.unique(labels[mask])
        return (float(np.mean([correct[mask & (labels == c)].mean() for c in classes])) if len(classes) else None,
                float(correct[mask].mean()) if mask.any() else None)
    all_macro, all_micro = score(np.ones(len(labels), dtype=bool))
    tail = np.isin(labels, groups['tail_classes'])
    affected = np.isin(labels, groups['affected_classes'])
    trusted_macro, trusted_micro = score(trusted_mask)
    return dict(macro=all_macro, micro=all_micro, tail_macro=score(tail)[0], remaining_macro=score(~tail)[0],
                trusted_macro=trusted_macro, trusted_micro=trusted_micro,
                affected_macro=score(affected)[0], unaffected_macro=score(~affected)[0])


def compare(labels, control, treatment, groups, trusted_mask, *, probe=False):
    labels, control, treatment = map(np.asarray, (labels, control, treatment))
    c, t = metrics(labels, control, groups, trusted_mask), metrics(labels, treatment, groups, trusted_mask)
    delta = {k: (t[k] - c[k]) * 100 if c[k] is not None and t[k] is not None else None for k in METRICS}
    corrected = (control != labels) & (treatment == labels)
    regressed = (control == labels) & (treatment != labels)
    gate = dict(macro=delta['macro'] is not None and delta['macro'] >= .20 - 1e-10,
                trusted=delta['trusted_macro'] is not None and delta['trusted_macro'] > 0,
                tail=delta['tail_macro'] is not None and delta['tail_macro'] > -.30,
                affected=delta['affected_macro'] is not None and delta['affected_macro'] > 0)
    # Probe rejects catastrophic tail regression, never claims final gain.
    passed = delta['tail_macro'] is not None and delta['tail_macro'] > -2.0 if probe else all(gate.values())
    return dict(control=c, treatment=t, delta_pp=delta, corrections=int(corrected.sum()), regressions=int(regressed.sum()),
                net=int(corrected.sum() - regressed.sum()), promotion_gates=gate, passed=bool(passed),
                decision=('probe_pass' if passed else 'probe_fail_close') if probe else ('n1_pass_checkpoint' if passed else 'close_ndcw'),
                platform_score=None)


def report(pair_path, output, *, probe=False):
    from v2.runtime import read_checkpoint, records_and_split
    pair_path = Path(pair_path).resolve()
    pair = json.loads(pair_path.read_text())
    bundle, meta = verify_bundle(pair['manifest'])
    require(sha(bundle) == pair['manifest_sha256'], 'Changed paired manifest')
    groups = json.loads((bundle.parent / 'evaluation_groups.json').read_text())
    plans, checkpoints, vectors, hashes, orders = {}, {}, {}, {}, {}
    for arm, spec in pair['arms'].items():
        path = Path(spec['plan'])
        require(sha(path) == spec['plan_sha256'], 'Changed paired plan')
        plans[arm] = verify_prepared(path)
        records, split = records_and_split(path.parent)
        destination = path.parent / ('probes' if probe else 'runs') / 's1_384'
        if probe:
            cost = json.loads((destination / 'cost.json').read_text())
            numerics = json.loads((destination / 'ndcw_probe.json').read_text())
            require(cost['status'] == 'cost_probe_only' and numerics['status'] == 'numerics_pass_pending_paired_tail_check', 'Probe numeric checks failed')
            epoch, chosen = 0, 'ema' if plans[arm]['recipe']['ema_enabled'] else 'raw'
            orders[arm] = numerics['sample_order_sha256']
            checkpoints[arm] = dict(probe_updates=numerics['updates'], weights='fixed_probe_ema', numerics=numerics)
            hashes[str(destination / 'ndcw_probe.json')] = sha(destination / 'ndcw_probe.json')
            hashes[str(destination / 'cost.json')] = sha(destination / 'cost.json')
        else:
            cp = destination / 'best.pt'
            payload = read_checkpoint(cp, plans[arm], 's1_384')
            epoch, chosen = payload['epoch'], payload['metrics']['chosen']
            all_history = json.loads((destination / 'history.json').read_text())
            orders[arm] = [h['sample_order_sha256'] for h in all_history]
            hashes[str(destination / 'history.json')] = sha(destination / 'history.json')
            checkpoints[arm] = dict(path=str(cp), sha256=sha(cp), epoch=epoch, weights=chosen)
            hashes[str(cp)] = sha(cp)
        npz = destination / f'holdout_epoch{epoch}.npz'
        with np.load(npz) as f:
            require(np.array_equal(f['indices'], split['val']), 'Holdout order mismatch')
            names = ['train/' + records[i]['relative_path'] for i in f['indices']]
            require(set(names) == set(groups['val_paths']), 'Foreign validation rows')
            vectors[arm] = (f['labels'].copy(), f[chosen].copy(), names)
        hashes[str(npz)] = sha(npz)
    require(plans['control']['recipe']['ndcw']['arm'] == 'control' and plans['treatment']['recipe']['ndcw']['arm'] == 'treatment', 'Pair arms swapped')
    require(orders['control'] == orders['treatment'], 'Paired sample orders differ')
    if probe:
        require(checkpoints['control']['probe_updates'] == checkpoints['treatment']['probe_updates'], 'Paired probe lengths differ')
    for stage in ('s1_384',):
        configs = []
        for arm in ('control', 'treatment'):
            p = Path(pair['arms'][arm]['plan'])
            cfg = json.loads((p.parent / plans[arm]['stages'][stage]['config']).read_text())
            cfg.pop('ndcw'); cfg['paths'].pop('project_root')
            configs.append(cfg)
        require(configs[0] == configs[1], 'Control and treatment changed more than ND-CW')
    clabels, control, names = vectors['control']
    tlabels, treatment, tnames = vectors['treatment']
    require(np.array_equal(clabels, tlabels) and names == tnames, 'Paired labels/paths mismatch')
    trusted = np.isin(names, groups['trusted_val_paths'])
    result = compare(clabels, control, treatment, groups, trusted, probe=probe)
    if probe:
        used = checkpoints['treatment']['numerics']['weighted_samples_observed']
        result['weighted_samples_observed'] = used
        result['passed'] &= used > 0
        result['decision'] = 'probe_pass' if result['passed'] else 'probe_fail_close'
    result.update(pair_sha256=sha(pair_path), manifest_sha256=sha(bundle), checkpoints=checkpoints,
                  plans={a: spec['plan_sha256'] for a, spec in pair['arms'].items()},
                  inputs=hashes, trusted_val_rows=int(trusted.sum()), stage='probe' if probe else 'N1_s1_384',
                  no_platform_submission=True)
    output = fresh(output)
    write_csv(output / 'metrics.csv', [dict(metric=k, control=result['control'][k], treatment=result['treatment'][k], delta_pp=result['delta_pp'][k]) for k in METRICS],
              ['metric', 'control', 'treatment', 'delta_pp'])
    write_csv(output / 'paired_predictions.csv', [dict(image_path=name, label=int(y), control=int(c), treatment=int(t), corrected=bool(c != y and t == y), regressed=bool(c == y and t != y))
              for name, y, c, t in zip(names, clabels, control, treatment)], ['image_path', 'label', 'control', 'treatment', 'corrected', 'regressed'])
    write_csv(output / 'class_metrics.csv', [dict(label=int(c), samples=int((clabels == c).sum()),
              control=float((control[clabels == c] == c).mean()), treatment=float((treatment[clabels == c] == c).mean()),
              affected=int(c) in groups['affected_classes'], tail=int(c) in groups['tail_classes']) for c in np.unique(clabels)],
              ['label', 'samples', 'control', 'treatment', 'affected', 'tail'])
    dump(output / 'report.json', result)
    return result
