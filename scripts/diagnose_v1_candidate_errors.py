"""Compare audited DEV recipes on aligned original labels; never form predictions.

Different native resolutions/view counts are retained and reported. Cross-recipe
comparisons describe error overlap, not the effect of a single training factor.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def load_aligned_npz(path, image_paths, labels):
    with np.load(path, allow_pickle=False) as z:
        require(np.array_equal(z['image_paths'], image_paths), 'Prediction path/order mismatch')
        require(np.array_equal(z['labels'], labels), 'Prediction label mismatch')
        return {k: z[k].copy() for k in z.files if k != 'logits'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(labels, prediction, classes, mask):
    counts = np.bincount(labels[mask], minlength=classes)
    correct = np.bincount(labels[mask & (labels == prediction)], minlength=classes)
    present = counts > 0
    return dict(rows=int(mask.sum()), correct=int(correct.sum()),
                errors=int(mask.sum() - correct.sum()), classes=int(present.sum()),
                micro=float(correct.sum() / mask.sum()) if mask.any() else None,
                macro=float((correct[present] / counts[present]).mean()) if present.any() else None)


def paired(labels, before, after, mask):
    a, b = before == labels, after == labels
    corrections = int((mask & ~a & b).sum())
    regressions = int((mask & a & ~b).sum())
    return dict(rows=int(mask.sum()), corrections=corrections, regressions=regressions,
                net=corrections - regressions, changed=int((mask & (before != after)).sum()),
                both_wrong=int((mask & ~a & ~b).sum()))


def confusion(labels, prediction, names):
    pairs = Counter(tuple(sorted((int(y), int(p))))
                    for y, p in zip(labels, prediction) if y != p)
    ordered = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))
    errors = int((labels != prediction).sum())
    return dict(errors=errors, distinct_unordered_pairs=len(ordered),
                concentration={str(k): dict(errors=sum(n for _, n in ordered[:k]),
                    fraction=sum(n for _, n in ordered[:k]) / errors if errors else 0.)
                    for k in (10, 50, 100)},
                top100=[dict(class_indices=list(pair), class_names=[names[i] for i in pair],
                             errors=count) for pair, count in ordered[:100]])


def bootstrap_intervals(labels, predictions, comparisons, group_indices, protocol):
    """Resample whole decoded-content groups, retaining every row of each group."""
    n = int(group_indices.max()) + 1
    sizes = np.bincount(group_indices, minlength=n)
    signed = np.stack([np.bincount(group_indices,
        weights=(predictions[b] == labels).astype(int) - (predictions[a] == labels).astype(int),
        minlength=n) for a, b in comparisons], axis=1)
    rng = np.random.default_rng(protocol['seed'])
    samples = []
    for start in range(0, protocol['resamples'], protocol['chunk']):
        size = min(protocol['chunk'], protocol['resamples'] - start)
        indices = rng.integers(0, n, size=(size, n))
        samples.append(100 * signed[indices].sum(axis=1) / sizes[indices].sum(axis=1)[:, None])
    bounds = np.quantile(np.concatenate(samples), [.025, .975], axis=0, method='linear')
    return {f'{a}__to__{b}': dict(delta_micro_pp_95_percentile_interval=bounds[:, i].tolist(),
                independent_content_groups=n, resamples=protocol['resamples'], seed=protocol['seed'],
                interpretation='Sampling uncertainty conditional on original noisy labels; not platform uncertainty')
            for i, (a, b) in enumerate(comparisons)}


def analyze(config_path, output):
    started = time.monotonic()
    cfg = read_json(config_path)
    require(cfg['new_training'] is False and cfg['test_predictions_used'] is False,
            'This diagnostic accepts DEV predictions only')
    provenance = {}

    def checked(path, expected=None):
        path = Path(path)
        digest = sha(path)
        require(expected is None or digest == expected, f'Checksum mismatch: {path}')
        provenance[str(path)] = digest
        return path

    stage = Path(cfg['stage_root'])
    manifest = read_json(checked(stage / 'dataset_manifest.json', cfg['dataset_manifest_sha256']))
    require(manifest['data_version'] == cfg['data_version'] and manifest['stage'] == 'repechage',
            'Wrong data stage')
    for name in ('class_to_idx.json', 'train_dev.csv', 'val_dev.csv', 'full_train.csv'):
        checked(stage / name, manifest['files'][name])
    mapping = read_json(stage / 'class_to_idx.json')
    names = [name for name, i in sorted(mapping.items(), key=lambda item: item[1])]
    require(sorted(mapping.values()) == list(range(len(names))), 'Noncontiguous class mapping')
    require(len(names) == manifest['num_classes'], 'Wrong number of classes')
    train, val, full = [read_rows(stage / name) for name in ('train_dev.csv', 'val_dev.csv', 'full_train.csv')]
    paths = np.array([r['image_path'] for r in val])
    labels = np.array([int(r['label']) for r in val])
    require(len(paths) == len(set(paths)) == manifest['val_dev_samples'], 'Wrong DEV population')
    require(len(train) == manifest['train_dev_samples'], 'Wrong training population')
    require(not set(paths) & {r['image_path'] for r in train}, 'Train/val image overlap')
    require(not {r['content_group'] for r in val} & {r['content_group'] for r in train},
            'Train/val decoded-content overlap')
    require({r['image_path'] for r in train + val} == {r['image_path'] for r in full}, 'Incomplete split')
    for row in train + val:
        require(0 <= int(row['label']) < len(names) and
                Path(row['image_path']).parts[:2] == ('train', names[int(row['label'])]),
                'Class label and path disagree')

    def load_npz(path, expected):
        return load_aligned_npz(checked(path, expected), paths, labels)

    closure = read_json(checked(ROOT / cfg['head_closure_report']))
    heads_report = read_json(checked(ROOT / cfg['head_delivery_report']))
    require(heads_report['validation_independent'] is True, 'Heads used overlapping validation')
    require(heads_report['binding']['dataset_manifest_sha256'] == cfg['dataset_manifest_sha256'],
            'Head dataset binding mismatch')
    head = load_npz(closure['validation_npz'], closure['validation_npz_sha256'])
    predictions = dict(parent448_center=head['parent512'], head768=head['unbalanced768'],
                       balanced768=head['balanced768'])
    decoders = {name: dict(input_size=448, resize_short_edges=[448], flip=False, bias=None)
                for name in predictions}
    native_pairs = [('parent448_center', 'head768'), ('parent448_center', 'balanced768')]
    reports = {}
    for prefix, relative in cfg['continuation_reports'].items():
        report = read_json(checked(ROOT / relative))
        plan = read_json(checked(report['plan'], report['plan_sha256']))
        require(report['validation_is_independent'] is True and report['partition'] == 'train_dev',
                'Continuation used overlapping validation')
        require(plan['parent_binding']['dataset_manifest_sha256'] == cfg['dataset_manifest_sha256'],
                'Continuation dataset binding mismatch')
        recipe = report['recipe']
        for policy, suffix in [('parent', 'baseline.npz'), ('ema', 'val_epoch04_ema.npz'),
                               ('swa', 'val_epoch04_ema_swa_2_4.npz')]:
            matched = [p for p in report['artifacts'] if Path(p).name == suffix]
            require(len(matched) == 1, f'Missing/ambiguous artifact: {suffix}')
            path = matched[0]
            name = f'{prefix}_{policy}'
            predictions[name] = load_npz(path, report['artifacts'][path])['predictions']
            decoders[name] = dict(input_size=recipe['image_size'], resize_short_edges=recipe['views'],
                                 flip=recipe['flip'], bias=None)
        native_pairs.extend([(f'{prefix}_parent', f'{prefix}_ema'), (f'{prefix}_parent', f'{prefix}_swa')])
        reports[prefix] = report
    for name, prediction in predictions.items():
        require(prediction.shape == labels.shape and prediction.dtype.kind in 'iu' and
                np.all((prediction >= 0) & (prediction < len(names))), f'Invalid predictions: {name}')
    require(np.array_equal(head['tail'], reports['lr512']['frozen_groups']['tail_classes']),
            'Tail definitions differ')
    a_group = read_json(checked(ROOT / cfg['a_target_groups']))
    require(a_group['source_sha256'] == closure['validation_npz_sha256'], 'A group parent mismatch')
    group_labels = defaultdict(set)
    for row in full:
        group_labels[row['content_group']].add(int(row['label']))
    masks = dict(all=np.ones(len(labels), dtype=bool),
                 tail75=np.isin(labels, head['tail']),
                 head75=np.isin(labels, reports['lr512']['frozen_groups']['head_classes']),
                 a_target_classes=np.isin(labels, a_group['target_classes']),
                 cross_label_duplicate=np.array([len(group_labels[r['content_group']]) > 1 for r in val]),
                 short_edge_below224=np.array([min(int(r['width']), int(r['height'])) < 224 for r in val]),
                 aspect_ratio_at_least2=np.array([max(int(r['width']), int(r['height'])) >=
                                                 2 * min(int(r['width']), int(r['height'])) for r in val]))
    masks['non_tail'] = ~masks['tail75']
    masks['outside_a_target'] = ~masks['a_target_classes']
    results = {name: {group: metrics(labels, pred, len(names), mask) for group, mask in masks.items()}
               for name, pred in predictions.items()}
    # Reproduce every published native comparison before making new comparisons.
    for prefix, report in reports.items():
        for policy, key in [('ema', 'last_ema'), ('swa', 'ema_swa_2_4')]:
            measured = paired(labels, predictions[f'{prefix}_parent'], predictions[f'{prefix}_{policy}'], masks['all'])
            reference = report['paired'][key]['slices']['all']
            require(all(measured[k] == reference[k] for k in ('corrections', 'regressions', 'net')),
                    'Native continuation result no longer reproduces')
            for metric in ('micro', 'macro'):
                require(abs(results[f'{prefix}_{policy}']['all'][metric] - reference['candidate'][metric]) < 1e-12,
                        'Published continuation score mismatch')
    for name, key in [('parent448_center', 'parent'), ('head768', 'unbalanced'), ('balanced768', 'balanced')]:
        for metric in ('micro', 'macro'):
            require(abs(results[name]['all'][metric] - closure['metrics'][key][metric]) < 1e-12,
                    'Published head score mismatch')
    candidates = cfg['candidate_names']
    comparisons = list(dict.fromkeys(native_pairs + list(itertools.combinations(candidates, 2)) +
                  [('parent448_center', 'lr512_parent'), ('parent448_center', 'wft448_parent'),
                   ('lr512_ema', 'lr512_swa')]))
    pair_reports = {f'{a}__to__{b}': dict(same_decoder=decoders[a] == decoders[b],
                    native_parent_comparison=(a, b) in native_pairs,
                    groups={group: paired(labels, predictions[a], predictions[b], mask)
                            for group, mask in masks.items()}) for a, b in comparisons}
    correct = np.stack([predictions[name] == labels for name in candidates])
    overlap = {}
    for group, mask in masks.items():
        all_wrong = mask & ~correct.any(axis=0)
        candidate_matrix = np.stack([predictions[name] for name in candidates])
        unanimous = (candidate_matrix == candidate_matrix[:1]).all(axis=0)
        overlap[group] = dict(rows=int(mask.sum()), all_candidates_wrong=int(all_wrong.sum()),
            all_wrong_same_prediction=int((all_wrong & unanimous).sum()),
            all_wrong_disagree=int((all_wrong & ~unanimous).sum()),
            correctness_disagreement=int((mask & correct.any(axis=0) & ~correct.all(axis=0)).sum()),
            uniquely_correct={name: int((mask & correct[i] & (correct.sum(axis=0) == 1)).sum())
                              for i, name in enumerate(candidates)},
            errors={name: int((mask & ~correct[i]).sum()) for i, name in enumerate(candidates)})
    _, group_indices = np.unique([r['content_group'] for r in val], return_inverse=True)
    val_group_counts = defaultdict(Counter)
    for row in val:
        val_group_counts[row['content_group']][int(row['label'])] += 1
    contradiction_lower_bound = sum(sum(counts.values()) - max(counts.values())
                                    for counts in val_group_counts.values())
    ci = bootstrap_intervals(labels, predictions, comparisons, group_indices, cfg['bootstrap'])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'aligned_predictions.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image_path', 'content_group', 'label'] + list(predictions) + list(masks))
        for i, row in enumerate(val):
            writer.writerow([row['image_path'], row['content_group'], int(labels[i])] +
                            [int(p[i]) for p in predictions.values()] + [int(m[i]) for m in masks.values()])
    report = dict(experiment_id=cfg['experiment_id'], status='completed_verified_diagnostic',
                  data_version=cfg['data_version'], classes=len(names), validation_rows=len(labels),
                  training_rows=len(train), content_groups=int(group_indices.max()) + 1,
                  analysis_type=cfg['analysis_type'], decoder_comparison_is_causal=False,
                  original_noisy_labels=True, clean_labels_known=False, platform_gain_known=False,
                  new_candidate=False, training_started=False, test_predictions_used=False,
                  native_results_reproduced=True, decoders=decoders, metrics=results,
                  paired=pair_reports, sampling_intervals=ci, candidate_names=candidates,
                  candidate_error_overlap=overlap,
                  confusion={name: confusion(labels, predictions[name], names) for name in candidates},
                  cohort_definitions=dict(tail75='Original frozen train_dev tail75 classes',
                    head75='Original frozen train_dev head75 classes',
                    a_target_classes='Previously frozen A target class union; retrospective for these old candidates',
                    cross_label_duplicate='Exact decoded content group has multiple original labels in full_train',
                    short_edge_below224='Official dimensions: shorter side <224; descriptive, not true source',
                    aspect_ratio_at_least2='Official dimensions: longer/shorter >=2; descriptive, not true source'),
                  cohorts_overlap=True, source_files=provenance, config_sha256=sha(config_path),
                  identical_content_label_conflicts=dict(
                    validation_conflicting_groups=sum(len(c) > 1 for c in val_group_counts.values()),
                    minimum_original_label_errors=contradiction_lower_bound,
                    interpretation='Lower bound for a deterministic image-content classifier on these contradictory original labels; does not identify clean labels'),
                  elapsed_seconds=time.monotonic() - started,
                  script_sha256=sha(__file__), aligned_csv=str(output / 'aligned_predictions.csv'),
                  aligned_csv_sha256=sha(output / 'aligned_predictions.csv'),
                  decision='Use existing independent A/B pairs to test visual learning/generalization; no new full training, fusion, or parameter scan justified by error overlap alone',
                  incumbent_reference=cfg['incumbent_reference'], incumbent_check=cfg['incumbent_check'])
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], rows=len(labels),
                         candidates={n: results[n]['all'] for n in candidates}, overlap=overlap['all']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.config, args.output)
