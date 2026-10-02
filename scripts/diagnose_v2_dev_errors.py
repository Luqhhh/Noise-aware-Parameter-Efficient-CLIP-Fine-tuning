"""Audit existing V2 DEV predictions against fixed V1 recipes, without inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np

from diagnose_v1_candidate_errors import (
    bootstrap_intervals, confusion, metrics, paired, read_json, read_rows, require, sha,
)

ROOT = Path(__file__).resolve().parents[1]


def align_predictions(payload, manifest, val, split_indices, classes):
    """Join by manifest index and image identity, never by incidental row order."""
    indices, labels = payload['indices'], payload['labels']
    require(indices.ndim == 1 and indices.dtype.kind in 'iu', 'Invalid index array')
    require(len(indices) == len(set(indices.tolist())) == len(val), 'Duplicate/missing indices')
    require(set(indices.tolist()) == set(split_indices), 'Wrong holdout population')
    require(labels.shape == indices.shape and labels.dtype.kind in 'iu', 'Invalid labels')
    positions = {}
    for position, (index, label) in enumerate(zip(indices.tolist(), labels.tolist())):
        row = manifest[index]
        require(int(row['label']) == label, 'Holdout label mismatch')
        path = 'train/' + row['relative_path']
        require(path not in positions, 'Duplicate manifest path')
        positions[path] = position
    require(set(positions) == {r['image_path'] for r in val}, 'Holdout path mismatch')
    order = [positions[r['image_path']] for r in val]
    require(np.array_equal(labels[order], [int(r['label']) for r in val]), 'Aligned label mismatch')
    predictions = {}
    for key in ('raw', 'ema'):
        p = payload[key]
        require(p.shape == indices.shape and p.dtype.kind in 'iu' and
                np.all((p >= 0) & (p < classes)), 'Invalid class predictions')
        predictions[key] = p[order]
    return predictions


def archive_members(path, names):
    result = {}
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if member.name in names:
                require(member.isfile() and member.name not in result, 'Ambiguous archive member')
                result[member.name] = archive.extractfile(member).read()
    require(set(result) == set(names), 'Missing archive member')
    return result


def analyze(config_path, output):
    started = time.monotonic()
    cfg = read_json(config_path)
    require(cfg['new_training'] is False and cfg['test_predictions_used'] is False, 'DEV only')
    require(set(cfg['selected']) == {'s2_448', 's3_576'}, 'Only independent DEV stages allowed')
    sources = {}

    def checked(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, f'Checksum mismatch: {path}')
        sources[str(path)] = digest
        return path

    checked(config_path)
    for script in (__file__, ROOT / 'scripts/diagnose_v1_candidate_errors.py'):
        checked(script)
    stage = Path(cfg['stage_root'])
    dm = read_json(checked(stage / 'dataset_manifest.json', cfg['dataset_manifest_sha256']))
    require(dm['data_version'] == cfg['data_version'] and dm['stage'] == 'repechage', 'Wrong stage')
    for key in ('class_to_idx.json', 'train_dev.csv', 'val_dev.csv'):
        checked(stage / key, dm['files'][key])
    mapping = read_json(stage / 'class_to_idx.json')
    classes = len(mapping)
    require(sorted(mapping.values()) == list(range(classes)), 'Invalid class mapping')
    class_names = [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]
    train, val = read_rows(stage / 'train_dev.csv'), read_rows(stage / 'val_dev.csv')
    require(len(train) == dm['train_dev_samples'] and len(val) == dm['val_dev_samples'], 'Row mismatch')
    official = {r['image_path']: r for r in train + val}
    require(len(official) == len(train) + len(val), 'Duplicate official paths')
    require(not {r['content_group'] for r in train} & {r['content_group'] for r in val}, 'Content leakage')
    labels = np.array([int(r['label']) for r in val])

    v1 = read_json(checked(ROOT / cfg['v1_report'], cfg['v1_report_sha256']))
    old = read_rows(checked(v1['aligned_csv'], v1['aligned_csv_sha256']))
    require(len(old) == len(val), 'V1 population mismatch')
    for a, b in zip(old, val):
        require(all(a[k] == b[k] for k in ('image_path', 'label', 'content_group')), 'V1 alignment mismatch')
    predictions = {name: np.array([int(r[name]) for r in old]) for name in cfg['v1_candidates']}
    masks = {name: np.array([int(r[name]) for r in old], dtype=bool) for name in cfg['groups']}
    require(masks['all'].all(), 'Incomplete all population')
    decoders = {name: v1['decoders'][name] for name in predictions}

    delivery = Path(cfg['delivery_root'])
    receipt = read_json(checked(delivery / 'delivery_receipt.json'))
    require(receipt['status'] == 'shutdown_endpoint_offline' and receipt['shutdown_issued'] is True,
            'Delivery incomplete; never reconnect or start work on the server')
    names = ['plan.json', 'split.json', 'train_manifest.csv']
    for name in cfg['selected']:
        names += [f'configs/{name}.json', f'runs/{name}/history.json', f'runs/{name}/best.binding.json']
    holds = {'s2_ema': ('s2_448', 5, 'ema'), 's3_ema': ('s3_576', 3, 'ema'),
             's3_raw': ('s3_576', 3, 'raw')}
    holds.update({f's3_e{epoch}_ema': ('s3_576', epoch, 'ema') for epoch in range(3)})
    names += sorted({f'runs/{s}/holdout_epoch{e}.npz' for s, e, w in holds.values()})
    blobs = archive_members(checked(delivery / 'delivery_metadata.tar.gz', cfg['archive_sha256']), names)
    blob_sha = {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()}
    require(blob_sha['plan.json'] == cfg['plan_sha256'] == receipt['plan_sha256'], 'Plan mismatch')
    plan, split = [json.loads(blobs[k]) for k in ('plan.json', 'split.json')]
    manifest = list(csv.DictReader(io.StringIO(blobs['train_manifest.csv'].decode())))
    require(len(manifest) == len(official), 'Manifest population mismatch')
    require([int(r['index']) for r in manifest] == list(range(len(manifest))), 'Manifest indices mismatch')
    for row in manifest:
        match = official.get('train/' + row['relative_path'])
        require(match is not None and all(row[a] == match[b] for a, b in
                [('label', 'label'), ('source_sha256', 'file_sha256'), ('content_group', 'content_group')]),
                'V2 image/label/content identity mismatch')
    for partition, rows in [('train', train), ('val', val)]:
        ids = split[partition]
        require(len(ids) == len(set(ids)) == len(rows) and
                all(isinstance(i, int) and 0 <= i < len(manifest) for i in ids), 'Invalid split indices')
        require({'train/' + manifest[i]['relative_path'] for i in ids} == {r['image_path'] for r in rows},
                'V2 partition differs from official DEV')

    # These are owned training checkpoints, authenticated before full RNG-state loading.
    import torch
    torch.set_num_threads(2)
    histories, selections = {}, {}
    for name, selected in cfg['selected'].items():
        stage_plan = plan['stages'][name]
        require(stage_plan['holdout_independent'] is True and plan['data_version'] == cfg['data_version'],
                'Overlapping or foreign stage')
        config = json.loads(blobs[f'configs/{name}.json'])
        require(blob_sha[f'configs/{name}.json'] == stage_plan['config_sha256'], 'Stage config mismatch')
        history = json.loads(blobs[f'runs/{name}/history.json'])
        require(all(h['validation_is_independent'] is True for h in history), 'Overlapping validation')
        best = max(history, key=lambda h: h['selection_score'])
        require(best['epoch'] == selected['epoch'] and best['chosen'] == selected['weights'], 'Selection mismatch')
        binding = json.loads(blobs[f'runs/{name}/best.binding.json'])
        require(binding['sha256'] == selected['checkpoint_sha256'] and
                binding['binding']['plan_sha256'] == cfg['plan_sha256'] and
                binding['binding']['manifest_sha256'] == blob_sha['train_manifest.csv'], 'Binding mismatch')
        cp = torch.load(checked(delivery / 'runs' / name / 'best.pt', selected['checkpoint_sha256']),
                        map_location='cpu', weights_only=False, mmap=True)
        require(cp['binding'] == binding['binding'] and cp['binding']['complete'] is True and
                cp['epoch'] == selected['epoch'] and cp['metrics'] == best and cp['history'] == history,
                'Checkpoint selection/history mismatch')
        selections[name] = dict(epoch_zero_based=cp['epoch'], weights=cp['metrics']['chosen'],
            checkpoint_sha256=selected['checkpoint_sha256'], independent_of_training=True,
            validation_used_for_original_selection=True)
        del cp
        histories[name] = history
        for prediction_name, (s, epoch, weights) in holds.items():
            if s != name:
                continue
            with np.load(io.BytesIO(blobs[f'runs/{s}/holdout_epoch{epoch}.npz']), allow_pickle=False) as data:
                predictions[prediction_name] = align_predictions(data, manifest, val, split['val'], classes)[weights]
            data_cfg = config['data']
            decoders[prediction_name] = dict(input_size=data_cfg['eval_size'],
                resize_short_edges=[round(data_cfg['eval_size'] * data_cfg['eval_resize_ratio'])],
                flip=False, bias=None)
            observed = metrics(labels, predictions[prediction_name], classes, masks['all'])
            reference = history[epoch]['val_ema' if weights == 'ema' else 'val']
            require(abs(observed['micro'] - reference['accuracy']) < 1e-12 and
                    abs(observed['macro'] - reference['macro_accuracy']) < 1e-12, 'History metrics mismatch')

    scores = {name: {group: metrics(labels, p, classes, mask) for group, mask in masks.items()}
              for name, p in predictions.items()}
    for name in cfg['v1_candidates']:
        require(scores[name] == v1['metrics'][name], 'V1 metrics changed')
    comparisons = [(name, 's3_ema') for name in cfg['v1_candidates']]
    comparisons += [('s2_ema', 's3_ema'), ('s3_raw', 's3_ema'), ('s3_e0_ema', 's3_ema'),
                    ('s3_e0_ema', 's3_e1_ema'), ('s3_e1_ema', 's3_e2_ema'), ('s3_e2_ema', 's3_ema')]
    pairs = {f'{a}__to__{b}': dict(same_decoder=decoders[a] == decoders[b],
        groups={group: paired(labels, predictions[a], predictions[b], mask) for group, mask in masks.items()})
        for a, b in comparisons}
    correct_v1 = np.stack([predictions[k] == labels for k in cfg['v1_candidates']])
    v2_correct = predictions['s3_ema'] == labels
    overlap = {group: dict(rows=int(mask.sum()),
        v1_all_wrong=int((mask & ~correct_v1.any(0)).sum()),
        v2_recovers_v1_all_wrong=int((mask & ~correct_v1.any(0) & v2_correct).sum()),
        v1_all_correct=int((mask & correct_v1.all(0)).sum()),
        v2_loses_v1_all_correct=int((mask & correct_v1.all(0) & ~v2_correct).sum()),
        all_five_wrong=int((mask & ~correct_v1.any(0) & ~v2_correct).sum()))
        for group, mask in masks.items()}
    proxies = {k: mask & ~v2_correct for k, mask in masks.items() if k in cfg['proxy_groups']}
    proxies['s2_to_s3_regression'] = (predictions['s2_ema'] == labels) & ~v2_correct
    proxy_union = np.logical_or.reduce([proxies[k] for k in cfg['proxy_groups']])
    total_union = np.logical_or.reduce(list(proxies.values()))
    budget = dict(total_errors=int((~v2_correct).sum()), proxy_errors={k: int(m.sum()) for k, m in proxies.items()},
        intersections={a: {b: int((x & y).sum()) for b, y in proxies.items()} for a, x in proxies.items()},
        proxy_union_errors=int(proxy_union.sum()), errors_outside_proxy_union=int((~v2_correct & ~proxy_union).sum()),
        union_including_stage_regressions=int(total_union.sum()),
        errors_outside_all_recorded_proxies=int((~v2_correct & ~total_union).sum()),
        interpretation='Overlapping descriptive proxies; attribution and recoverability are unknown')
    _, group_indices = np.unique([r['content_group'] for r in val], return_inverse=True)
    intervals = bootstrap_intervals(labels, predictions, comparisons, group_indices, cfg['bootstrap'])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    aligned = output / 'aligned_predictions.csv'
    with aligned.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image_path', 'content_group', 'label'] + list(predictions) + list(masks))
        for i, row in enumerate(val):
            writer.writerow([row['image_path'], row['content_group'], int(labels[i])] +
                            [int(p[i]) for p in predictions.values()] + [int(m[i]) for m in masks.values()])
    elapsed = time.monotonic() - started
    require(elapsed <= cfg['cpu_run_limit_seconds'], 'CPU diagnostic budget exceeded')
    report = dict(experiment_id=cfg['experiment_id'], status='completed_diagnostic', data_version=cfg['data_version'],
        training_rows=len(train), validation_rows=len(val), classes=classes, content_groups=len(np.unique(group_indices)),
        new_training=False, test_predictions_used=False, new_candidate=False, platform_gain_known=False,
        original_noisy_labels=True, validation_used_for_original_selection=True, causal_recipe_comparison=False,
        selections=selections, decoders=decoders, metrics=scores, paired=pairs, sampling_intervals=intervals,
        v1_candidates=cfg['v1_candidates'], overlap=overlap, error_budget=budget,
        confusion=confusion(labels, predictions['s3_ema'], class_names),
        aligned_csv=str(aligned), aligned_csv_sha256=sha(aligned), sources=sources, archive_members=blob_sha,
        elapsed_seconds=elapsed, decision=cfg['decision'])
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.config, args.output)
    print(json.dumps(dict(status=result['status'], rows=result['validation_rows'],
                         s3=result['metrics']['s3_ema']['all'], seconds=result['elapsed_seconds'])))
