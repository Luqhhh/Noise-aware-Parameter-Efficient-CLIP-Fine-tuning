"""Independent stdlib/set recount of every row, metric, and paired cohort."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def verify(report_path, config_path, output):
    report_path, config_path = Path(report_path), Path(config_path)
    report, config = [json.loads(p.read_text()) for p in (report_path, config_path)]
    root = Path(__file__).resolve().parents[1]
    assert sha(config_path) == report['config_sha256']
    assert sha(root / 'scripts/diagnose_v1_candidate_errors.py') == report['script_sha256']
    for path, expected in report['source_files'].items():
        assert sha(path) == expected, path
    csv_path = Path(report['aligned_csv'])
    assert sha(csv_path) == report['aligned_csv_sha256']
    with csv_path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    with (Path(config['stage_root']) / 'val_dev.csv').open(newline='') as handle:
        canonical = list(csv.DictReader(handle))
    assert len(rows) == len(canonical) == report['validation_rows']
    assert len({r['image_path'] for r in rows}) == len(rows)
    for a, b in zip(rows, canonical):
        assert all(a[k] == b[k] for k in ('image_path', 'label', 'content_group'))
    names = list(report['metrics'])
    all_ids = set(range(len(rows)))
    correct = {name: {i for i, row in enumerate(rows) if row[name] == row['label']} for name in names}
    masks = {group: {i for i, row in enumerate(rows) if row[group] == '1'}
             for group in report['candidate_error_overlap']}
    assert masks['all'] == all_ids
    for name in names:
        for group, selected in masks.items():
            total = Counter(rows[i]['label'] for i in selected)
            hits = Counter(rows[i]['label'] for i in selected & correct[name])
            expected = report['metrics'][name][group]
            assert expected['rows'] == len(selected) and expected['correct'] == sum(hits.values())
            assert expected['errors'] == len(selected - correct[name])
            micro = sum(hits.values()) / len(selected) if selected else None
            macro = sum(hits[label] / count for label, count in total.items()) / len(total) if total else None
            for key, value in [('micro', micro), ('macro', macro)]:
                assert (expected[key] is None and value is None) or abs(expected[key] - value) < 1e-12
    for key, comparison in report['paired'].items():
        a, b = key.split('__to__')
        assert comparison['same_decoder'] == (report['decoders'][a] == report['decoders'][b])
        for group, selected in masks.items():
            corrections = len(selected & (correct[b] - correct[a]))
            regressions = len(selected & (correct[a] - correct[b]))
            expected = comparison['groups'][group]
            assert expected['corrections'] == corrections and expected['regressions'] == regressions
            assert expected['net'] == corrections - regressions
            assert expected['changed'] == sum(rows[i][a] != rows[i][b] for i in selected)
            assert expected['both_wrong'] == len(selected - (correct[a] | correct[b]))
    candidates = report['candidate_names']
    for group, selected in masks.items():
        remaining = selected.copy()
        for name in candidates:
            remaining -= correct[name]
        shared_correct = set.intersection(*(correct[name] for name in candidates))
        any_correct = set.union(*(correct[name] for name in candidates))
        agreement = sum(len({rows[i][name] for name in candidates}) == 1 for i in remaining)
        expected = report['candidate_error_overlap'][group]
        assert expected['all_candidates_wrong'] == len(remaining)
        assert expected['all_wrong_same_prediction'] == agreement
        assert expected['all_wrong_disagree'] == len(remaining) - agreement
        assert expected['correctness_disagreement'] == len(selected & (any_correct - shared_correct))
        for name in candidates:
            others = set.union(*(correct[n] for n in candidates if n != name))
            assert expected['uniquely_correct'][name] == len(selected & (correct[name] - others))
    for name in candidates:
        counts = Counter(tuple(sorted((int(row['label']), int(row[name]))))
                         for row in rows if row['label'] != row[name])
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        for k, result in report['confusion'][name]['concentration'].items():
            assert result['errors'] == sum(n for _, n in ordered[:int(k)])
    groups = defaultdict(Counter)
    for row in canonical:
        groups[row['content_group']][row['label']] += 1
    lower_bound = sum(sum(c.values()) - max(c.values()) for c in groups.values())
    assert report['identical_content_label_conflicts']['minimum_original_label_errors'] == lower_bound
    proxy_names = ['tail75', 'a_target_classes', 'cross_label_duplicate',
                   'short_edge_below224', 'aspect_ratio_at_least2']
    union = set.union(*(masks[name] for name in proxy_names))
    lr_errors = all_ids - correct['lr512_swa']
    error_proxies = {name: masks[name] & lr_errors for name in proxy_names}
    error_proxies['native_parent_regression'] = correct['lr512_parent'] & lr_errors
    error_union = set.union(*error_proxies.values())
    incumbent = Path('/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip')
    assert sha(incumbent) == '1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e'
    assert 'All checks passed' in (root / config['incumbent_check']).read_text()
    verification = dict(status='passed', rows=len(rows), metrics_recounted=len(names) * len(masks),
        paired_groups_recounted=len(report['paired']) * len(masks), overlap_groups_recounted=len(masks),
        source_checksums_verified=len(report['source_files']), group_conflicts_lower_bound=lower_bound,
        incumbent_zip=str(incumbent), incumbent_sha256=sha(incumbent),
        incumbent_existing_submission_check=config['incumbent_check'],
        proxy_error_union=dict(groups=proxy_names, rows=len(union), lr512_errors=len(union & lr_errors),
                              lr512_errors_outside_union=len(lr_errors - union),
                              interpretation='Overlapping retrospective proxies, not exhaustive mechanisms or clean truth'),
        error_proxy_counts={name: len(ids) for name, ids in error_proxies.items()},
        error_proxy_intersections={a: {b: len(error_proxies[a] & error_proxies[b]) for b in error_proxies}
                                   for a in error_proxies},
        error_union_including_native_regressions=len(error_union),
        errors_outside_all_recorded_proxies=len(lr_errors - error_union),
        report_sha256=sha(report_path), aligned_csv_sha256=sha(csv_path),
        verifier_sha256=sha(__file__), new_training=False, new_platform_score=None)
    Path(output).write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify(args.report, args.config, args.output)
