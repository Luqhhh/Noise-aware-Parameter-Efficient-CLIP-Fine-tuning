"""Independently recount the fixed image-detail probe from saved CUDA logits."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def compare(before, after, labels, selected):
    a = {i for i in selected if before[i] == labels[i]}
    b = {i for i in selected if after[i] == labels[i]}
    return dict(rows=len(selected), corrections=len(b - a), regressions=len(a - b),
                net=len(b) - len(a), changed=sum(before[i] != after[i] for i in selected),
                both_wrong=len(selected - (a | b)))


def measure(prediction, labels, selected):
    total = Counter(labels[i] for i in selected)
    hits = Counter(labels[i] for i in selected if prediction[i] == labels[i])
    return dict(rows=len(selected), correct=sum(hits.values()), errors=len(selected) - sum(hits.values()),
                classes=len(total), micro=sum(hits.values()) / len(selected) if selected else None,
                macro=sum(hits[label] / n for label, n in total.items()) / len(total) if total else None)


def verify(config_path, run_root, output):
    root = Path(__file__).resolve().parents[1]
    config = json.loads(Path(config_path).read_text())
    run_root = Path(run_root).resolve()
    report = json.loads((run_root / 'report.json').read_text())
    preflight = json.loads((run_root / 'preflight.json').read_text())
    assert report['config_sha256'] == preflight['config_sha256'] == sha(config_path)
    assert report['implementation_sha256'] == sha(root / 'scripts/probe_v1_resolution_degradation.py')
    assert report['preflight_sha256'] == sha(run_root / 'preflight.json')
    for path, expected in report['source_files'].items():
        assert sha(path) == expected, path
    source = json.loads((root / config['source_report']).read_text())
    plan = json.loads(Path(source['plan']).read_text())
    for path, expected in plan['inputs'].items():
        assert sha(path) == expected, path
    stage = Path('/home/lux1/noise/artifacts/stages/repechage/20260921')
    manifest = json.loads((stage / 'dataset_manifest.json').read_text())
    assert sha(stage / 'dataset_manifest.json') == plan['source_binding']['dataset_manifest_sha256']
    assert sha(stage / 'val_dev.csv') == manifest['files']['val_dev.csv']
    with (stage / 'val_dev.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    labels = [int(r['label']) for r in rows]
    masks = dict(all=set(range(len(rows))),
        reduced={i for i, r in enumerate(rows) if min(int(r['width']), int(r['height'])) > 224},
        unchanged={i for i, r in enumerate(rows) if min(int(r['width']), int(r['height'])) <= 224},
        originally_small={i for i, r in enumerate(rows) if min(int(r['width']), int(r['height'])) < 224},
        tail75={i for i, r in enumerate(rows) if int(r['label']) in plan['groups']['tail_classes']})
    assert sha(run_root / 'frozen_groups.csv') == preflight['groups_sha256']
    with (run_root / 'frozen_groups.csv').open(newline='') as handle:
        frozen = list(csv.DictReader(handle))
    assert len(frozen) == len(rows) == report['validation_rows']
    for i, row in enumerate(frozen):
        assert row['image_path'] == rows[i]['image_path'] and int(row['label']) == labels[i]
        assert all(int(row[group]) == int(i in selected) for group, selected in masks.items())
    arrays, losses, gains = {}, {}, {}
    for name in config['models']:
        path = run_root / (name + '.npz')
        assert report['artifacts'][str(path)] == sha(path)
        with np.load(path, allow_pickle=False) as artifact:
            assert artifact['labels'].tolist() == labels
            assert artifact['image_paths'].tolist() == [r['image_path'] for r in rows]
            assert artifact['reduced_indices'].tolist() == sorted(masks['reduced'])
            logits = artifact['perturbed_logits']
            assert logits.shape == (len(masks['reduced']), len(plan['classes'])) and np.isfinite(logits).all()
            original, perturbed = artifact['original'].tolist(), artifact['perturbed'].tolist()
            assert [perturbed[i] for i in sorted(masks['reduced'])] == logits.argmax(1).tolist()
        native_file = 'baseline.npz' if name == 'native_parent' else 'val_epoch04_ema_swa_2_4.npz'
        native_path = next(p for p in source['artifacts'] if Path(p).name == native_file)
        with np.load(native_path, allow_pickle=False) as artifact:
            assert artifact['predictions'].tolist() == original
        assert all(original[i] == perturbed[i] for i in masks['unchanged'])
        for group, selected in masks.items():
            expected = report['results'][name]
            assert expected['paired'][group] == compare(original, perturbed, labels, selected)
            for policy, predictions in [('original', original), ('perturbed', perturbed)]:
                measured = measure(predictions, labels, selected)
                for key, value in measured.items():
                    actual = expected[policy][group][key]
                    assert (actual == value if key not in ('micro', 'macro') or value is None
                            else abs(actual - value) < 1e-12), (name, group, policy, key)
        assert report['native_parity'][name]['matching_predictions'] == config['parity_rows']
        arrays[name] = (original, perturbed)
        losses[name] = {i for i in masks['reduced'] if original[i] == labels[i] and perturbed[i] != labels[i]}
        gains[name] = {i for i in masks['reduced'] if original[i] != labels[i] and perturbed[i] == labels[i]}
    candidate_vs_parent = {policy: {group: compare(arrays['native_parent'][i], arrays['lr512_swa'][i], labels, selected)
        for group, selected in masks.items()} for i, policy in enumerate(('original', 'perturbed'))}
    comparison = candidate_vs_parent['perturbed']['all']['net'] - candidate_vs_parent['original']['all']['net']
    parent = report['results']['native_parent']
    delta = 100 * (parent['perturbed']['reduced']['micro'] - parent['original']['reduced']['micro'])
    gate = config['mechanism_gate']
    supported = (parent['paired']['reduced']['regressions'] >= gate['minimum_parent_regressions'] and
                 delta <= gate['parent_delta_micro_pp_at_most'])
    assert report['detail_loss_mechanism_gate'] == supported
    assert report['training_updates'] == 0 and report['new_candidate'] is False and report['test_data_used'] is False
    checked = dict(status='passed', experiment_id=config['experiment_id'], rows=len(rows),
        group_sizes={group: len(selected) for group, selected in masks.items()},
        metrics_recounted=20, paired_groups_recounted=10, cold_native_matches=128,
        all_reduced_predictions_replayed_from_logits=True, unchanged_predictions_preserved=True,
        frozen_group_rows_independently_verified=True, original_code_inputs_sha_verified=len(plan['inputs']),
        mechanism_gate_independently_verified=supported,
        report_sha256=sha(run_root / 'report.json'), verifier_sha256=sha(__file__),
        candidate_vs_parent=candidate_vs_parent, change_in_lr512_advantage_after_detail_loss=comparison,
        shared_regressions=len(losses['native_parent'] & losses['lr512_swa']),
        parent_only_regressions=len(losses['native_parent'] - losses['lr512_swa']),
        lr512_only_regressions=len(losses['lr512_swa'] - losses['native_parent']),
        shared_corrections=len(gains['native_parent'] & gains['lr512_swa']),
        new_training=False, new_candidate=False, platform_gain=None)
    Path(output).write_text(json.dumps(checked, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(checked, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify(args.config, args.run_root, args.output)
