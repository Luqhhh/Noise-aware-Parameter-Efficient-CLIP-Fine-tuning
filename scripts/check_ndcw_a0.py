"""Independent checks on real A0 artifacts; no ND-CW implementation imports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as f:
        return list(csv.DictReader(f))


def truth(value):
    return str(value).lower() == 'true'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit, bundle = Path(args.audit).resolve(), Path(args.manifest).resolve()
    report = json.loads((audit / 'report.json').read_text())
    manifest = json.loads(bundle.read_text())
    config = yaml.safe_load(Path(json.loads((audit / 'candidates.json').read_text())['config']).read_text())
    base = Path(config['stage_artifacts'])
    rows = read(base / 'full_train.csv')
    lookup = {r['image_path']: i for i, r in enumerate(rows)}
    assert len(rows) == len(lookup) == 148695
    assert all(r['image_path'].startswith('train/') for r in rows)
    for path, h in report['inputs'].items():
        assert digest(path) == h, path
    for name, h in manifest['files'].items():
        assert digest(bundle.parent / name) == h, name
    assert digest(audit / 'report.json') == manifest['audit_report_sha256']
    assert digest(manifest['freeze_file']) == manifest['freeze_sha256']
    with np.load(audit / 'knn_top10.npz') as f:
        idx, scores = f['indices'], f['similarities']
    assert idx.shape == scores.shape == (len(rows), 10)
    assert np.isfinite(scores).all() and np.all(idx >= 0) and np.all(idx < len(rows))
    assert np.all(idx != np.arange(len(rows))[:, None])
    assert np.all(np.diff(scores, axis=1) <= 1e-7)
    assert all(len(set(neighbors)) == 10 for neighbors in idx)
    expected = set()
    for i, (neighbors, values) in enumerate(zip(idx, scores)):
        expected.update(tuple(sorted((i, int(j)))) for j, s in zip(neighbors, values) if float(s) >= .94)
    pairs = read(audit / 'verified_pairs.csv.gz')
    actual = {(int(p['i']), int(p['j'])) for p in pairs}
    assert expected == actual and len(actual) == len(pairs)
    for p in pairs:
        i, j = int(p['i']), int(p['j'])
        assert i < j and p['image_i'] == rows[i]['image_path'] and p['image_j'] == rows[j]['image_path']
        for key, r in (('i', rows[i]), ('j', rows[j])):
            assert p['label_' + key] == r['label'] and p['content_group_' + key] == r['content_group']
        assert truth(p['exact']) == (rows[i]['content_group'] == rows[j]['content_group'])
        assert truth(p['mutual_nn']) == (j in idx[i] and i in idx[j])
        retrieved = scores[i][idx[i] == j] if j in idx[i] else scores[j][idx[j] == i]
        assert abs(float(p['cosine_similarity']) - float(retrieved[0])) <= 2e-6
    medium, strong = manifest['thresholds']['medium'], manifest['thresholds']['strong']
    train_paths = {r['image_path'] for r in read(base / 'train_dev.csv')}
    totals, class_counts = {}, {}
    for scope, name in (('full', 'ndcw_weights.csv'), ('dev', 'ndcw_dev_weights.csv')):
        table = read(bundle.parent / name)
        by_name = {r['image_path']: r for r in table}
        assert len(table) == len(by_name) == len(lookup) and set(by_name) == set(lookup)
        raw_weights = {name: 1. for name in lookup}
        for p in pairs:
            a, b = p['image_i'], p['image_j']
            if scope == 'dev' and not {a, b} <= train_paths:
                continue
            if p['label_i'] == p['label_j'] or truth(p['exact']):
                continue
            s = float(p['cosine_similarity'])
            mutual, perceptual = truth(p['mutual_nn']), truth(p['perceptual_pass'])
            w = .2 if s >= strong and mutual and perceptual else .5 if s >= medium and (mutual or perceptual) else 1.
            raw_weights[a] = min(raw_weights[a], w)
            raw_weights[b] = min(raw_weights[b], w)
        counts, loss, affected = Counter(), Counter(), Counter()
        for name, r in by_name.items():
            original = rows[lookup[name]]
            assert r['label'] == original['label'] and r['content_group'] == original['content_group']
            raw, w = float(r['raw_nd_weight']), float(r['nd_weight'])
            assert raw == raw_weights[name] and w in (raw, 1.) and w in (.2, .5, 1.)
            assert r['raw_nd_level'] == {1.: 'weak', .5: 'medium', .2: 'strong'}[raw]
            assert r['nd_level'] == {1.: 'weak', .5: 'medium', .2: 'strong'}[w]
            if scope == 'dev' and name not in train_paths:
                assert w == raw == 1.
                continue
            counts[r['label']] += 1
            loss[r['label']] += 1 - w
            affected[r['label']] += w < 1
        assert all(loss[c] <= .1 * n + 1e-8 for c, n in counts.items())
        impact = read(bundle.parent / (scope + '_class_impact.csv'))
        for r in impact:
            c = r['label']
            assert int(r['samples']) == counts[c] and int(r['affected']) == affected[c]
            assert abs(float(r['equivalent_supervision_loss']) - loss[c]) < 1e-8
        totals[scope] = dict(raw_conflict_images=sum(w < 1 for w in raw_weights.values()),
                             weighted_images=sum(affected.values()), affected_classes=sum(n > 0 for n in affected.values()),
                             max_class_supervision_loss_rate=max(loss[c] / n for c, n in counts.items()))
        class_counts[scope] = counts
        assert totals[scope]['weighted_images'] == manifest['stats'][scope]['gates']['weighted_images']
    control = read(bundle.parent / 'control_weights.csv')
    assert len(control) == len(lookup) and {r['image_path'] for r in control} == set(lookup)
    assert all(float(r['nd_weight']) == 1 for r in control)
    groups = defaultdict(list)
    for r in rows:
        groups[r['content_group']].append(r)
    exact = [rs for rs in groups.values() if len(rs) > 1]
    conflicts = [rs for rs in exact if len({r['label'] for r in rs}) > 1]
    assert sum(map(len, exact)) == report['exact_duplicate_images']
    assert sum(map(len, conflicts)) == report['exact_cross_label_images']
    result = dict(status='passed', total_images=len(rows), candidate_pairs=len(pairs),
                  exact_duplicate_images=sum(map(len, exact)), exact_conflict_images=sum(map(len, conflicts)),
                  independent_manifest_checks=totals, source_sha256=digest(__file__),
                  audit_report_sha256=digest(audit / 'report.json'), manifest_sha256=digest(bundle), test_read=False)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
