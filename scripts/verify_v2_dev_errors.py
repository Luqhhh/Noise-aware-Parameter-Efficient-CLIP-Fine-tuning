"""Independently recount the V2 diagnostic using CSV rows and integer sets."""
import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(report_path):
    report = json.loads(Path(report_path).read_text())
    assert report['new_training'] is False and report['test_predictions_used'] is False
    assert digest(report['aligned_csv']) == report['aligned_csv_sha256']
    for path, expected in report['sources'].items():
        assert digest(path) == expected, path
    with Path(report['aligned_csv']).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report['validation_rows'] == len({r['image_path'] for r in rows})
    correct = {name: {i for i, r in enumerate(rows) if r[name] == r['label']}
               for name in report['metrics']}
    groups = {g: {i for i, r in enumerate(rows) if r[g] == '1'} for g in report['overlap']}
    assert len(groups['all']) == len(rows)
    counted_metrics = counted_pairs = 0
    for name, slices in report['metrics'].items():
        for group, expected in slices.items():
            population = groups[group]
            totals = Counter(rows[i]['label'] for i in population)
            wins = Counter(rows[i]['label'] for i in population & correct[name])
            assert expected['rows'] == len(population)
            assert expected['classes'] == len(totals)
            assert expected['correct'] == sum(wins.values())
            assert expected['errors'] == len(population) - sum(wins.values())
            assert math.isclose(expected['micro'], sum(wins.values()) / len(population), abs_tol=1e-12)
            assert math.isclose(expected['macro'], sum(wins[c] / n for c, n in totals.items()) / len(totals), abs_tol=1e-12)
            counted_metrics += 1
    for name, comparison in report['paired'].items():
        before, after = name.split('__to__')
        for group, expected in comparison['groups'].items():
            population = groups[group]
            corrections = len(population & (correct[after] - correct[before]))
            regressions = len(population & (correct[before] - correct[after]))
            assert expected == dict(rows=len(population), corrections=corrections, regressions=regressions,
                net=corrections-regressions, both_wrong=len(population - correct[before] - correct[after]),
                changed=sum(rows[i][before] != rows[i][after] for i in population))
            counted_pairs += 1
    all_old_wrong = groups['all'] - set.union(*(correct[n] for n in report['v1_candidates']))
    all_old_right = set.intersection(*(correct[n] for n in report['v1_candidates']))
    for group, expected in report['overlap'].items():
        p = groups[group]
        assert expected == dict(rows=len(p), v1_all_wrong=len(p & all_old_wrong),
            v2_recovers_v1_all_wrong=len(p & all_old_wrong & correct['s3_ema']),
            v1_all_correct=len(p & all_old_right),
            v2_loses_v1_all_correct=len((p & all_old_right) - correct['s3_ema']),
            all_five_wrong=len((p & all_old_wrong) - correct['s3_ema']))
    errors = groups['all'] - correct['s3_ema']
    budget = report['error_budget']
    proxies = {name: errors & groups[name] for name in budget['proxy_errors'] if name in groups}
    union = set.union(*proxies.values())
    proxies['s2_to_s3_regression'] = correct['s2_ema'] - correct['s3_ema']
    all_union = set.union(*proxies.values())
    assert budget['total_errors'] == len(errors)
    assert budget['proxy_errors'] == {k: len(v) for k, v in proxies.items()}
    assert budget['intersections'] == {a: {b: len(x & y) for b, y in proxies.items()} for a, x in proxies.items()}
    assert budget['proxy_union_errors'] == len(union)
    assert budget['errors_outside_proxy_union'] == len(errors - union)
    assert budget['union_including_stage_regressions'] == len(all_union)
    assert budget['errors_outside_all_recorded_proxies'] == len(errors - all_union)
    confusion = Counter(tuple(sorted([int(rows[i]['label']), int(rows[i]['s3_ema'])])) for i in errors)
    ranked = sorted(confusion.items(), key=lambda item: (-item[1], item[0]))
    assert report['confusion']['errors'] == len(errors)
    assert report['confusion']['distinct_unordered_pairs'] == len(confusion)
    for k, v in report['confusion']['concentration'].items():
        assert v['errors'] == sum(n for pair, n in ranked[:int(k)])
        assert math.isclose(v['fraction'], v['errors'] / len(errors), abs_tol=1e-12)
    assert [(tuple(r['class_indices']), r['errors']) for r in report['confusion']['top100']] == ranked[:100]

    # Independently join original NPZ indices to CSV paths; the analyzer is not imported.
    archive_path = next(p for p in report['sources'] if p.endswith('delivery_metadata.tar.gz'))
    blobs = {}
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if member.name in report['archive_members']:
                assert member.isfile() and member.name not in blobs
                blobs[member.name] = archive.extractfile(member).read()
                assert hashlib.sha256(blobs[member.name]).hexdigest() == report['archive_members'][member.name]
    assert set(blobs) == set(report['archive_members'])
    manifest = list(csv.DictReader(io.StringIO(blobs['train_manifest.csv'].decode())))
    holds = {'s2_ema': ('s2_448', 5, 'ema'), 's3_ema': ('s3_576', 3, 'ema'), 's3_raw': ('s3_576', 3, 'raw')}
    holds.update({f's3_e{e}_ema': ('s3_576', e, 'ema') for e in range(3)})
    for name, (stage, epoch, weights) in holds.items():
        with np.load(io.BytesIO(blobs[f'runs/{stage}/holdout_epoch{epoch}.npz']), allow_pickle=False) as data:
            source = {'train/' + manifest[int(index)]['relative_path']: (int(y), int(p))
                      for index, y, p in zip(data['indices'], data['labels'], data[weights])}
        assert len(source) == len(rows)
        for row in rows:
            assert source[row['image_path']] == (int(row['label']), int(row[name]))
    return dict(status='passed', rows=len(rows), metrics_recounted=counted_metrics,
        paired_groups_recounted=counted_pairs, overlap_groups_recounted=len(groups),
        original_v2_predictions_rejoined=len(holds)*len(rows), proxy_intersections_recounted=len(proxies)**2,
        confusion_pairs_recounted=len(confusion), sources_verified=len(report['sources']),
        archive_members_verified=len(blobs), report_sha256=digest(report_path),
        aligned_csv_sha256=report['aligned_csv_sha256'], verifier_sha256=digest(__file__),
        bootstrap_note='Existing fixed group-bootstrap implementation; not independently resampled here',
        new_training=False, new_platform_score=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = verify(args.report)
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
