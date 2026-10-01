"""Independently recompute the frozen pair and replay CSV bytes from saved logits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

from aegis_clip.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def count_metrics(labels, prediction, mask, classes):
    y, p = labels[mask], prediction[mask]
    unique, counts = np.unique(y, return_counts=True)
    accuracy = [np.mean(p[y == c] == c) for c in unique]
    correct = int(np.count_nonzero(y == p))
    return dict(rows=len(y), correct=correct, errors=len(y)-correct, classes=len(unique),
                micro=correct/len(y), macro=float(np.mean(accuracy)))


def count_pair(labels, before, after, mask):
    y, a, b = labels[mask], before[mask], after[mask]
    corr = sum(bool(x != t and z == t) for t, x, z in zip(y, a, b))
    reg = sum(bool(x == t and z != t) for t, x, z in zip(y, a, b))
    return dict(rows=len(y), corrections=corr, regressions=reg, net=corr-reg,
                changed=int(np.count_nonzero(a != b)), both_wrong=int(np.count_nonzero((a != y) & (b != y))))


def same(expected, actual, label):
    require(set(expected) == set(actual), f'{label}: fields differ')
    for key in expected:
        if isinstance(expected[key], float):
            require(np.isclose(expected[key], actual[key], rtol=0, atol=1e-12), f'{label}/{key}: value differs')
        else:
            require(expected[key] == actual[key], f'{label}/{key}: value differs')


def val_prediction(path, labels, paths, classes):
    with np.load(path, allow_pickle=False) as data:
        require(np.array_equal(data['labels'], labels) and np.array_equal(data['image_paths'], paths),
                'Validation row/label alignment differs')
        logits = data['logits']
        require(logits.shape == (len(labels), classes) and np.isfinite(logits).all(), 'Invalid validation logits')
        prediction = logits.argmax(1)
        require(np.array_equal(prediction, data['predictions']), 'Saved prediction differs from logits argmax')
    return prediction


def verify(config, output):
    output = Path(output).resolve()
    cfg, pre, report = read_json(config), read_json(output/'preflight.json'), read_json(output/'report.json')
    bind = pre['binding']
    require(bind == report['binding'] and bind['config_sha256'] == sha(config) and
            bind['implementation_sha256'] == sha(ROOT/'scripts/train_v1_detail_aug_pair.py'), 'Frozen implementation changed')
    require(report['status'] == 'completed_packaged_pair' and report['experiment_id'] == cfg['experiment_id'] and
            report['validation_rows'] == 14880 and report['primary_checkpoint'] == 'last_raw_step1024' and
            report['automatic_full'] is False and report['platform_gain_known'] is False, 'Not a complete fixed pair')
    require(read_json(output/'status.json')['status'] == 'completed_packaged_pair', 'Formal status differs')
    artifacts = {}
    def record(path, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None: require(digest == expected, f'Checksum differs: {path}')
        artifacts[str(path.resolve())] = digest
    for path, digest in bind['parent_sources'].items(): record(path, digest)
    plan = read_json(pre['source_plan'])
    for path, digest in plan['inputs'].items(): record(path, digest)
    source_cfg = yaml.safe_load(Path(plan['config']['parent_config']).read_text())
    reference = load_config(Path(source_cfg['source']['root'])/source_cfg['source']['reference_config'])
    base = Path(reference['data']['dataset_manifest']).parent
    record(base/'dataset_manifest.json', plan['source_binding']['dataset_manifest_sha256'])
    record(base/'class_to_idx.json', plan['source_binding']['class_mapping_sha256'])
    train, val, test = rows(base/'train_dev.csv'), rows(base/'val_dev.csv'), rows(base/'test_manifest.csv')
    mapping = read_json(base/'class_to_idx.json')
    classes = [name for name, index in sorted(mapping.items(), key=lambda item: item[1])]
    require(classes == plan['classes'] and len(classes) == 750, 'Class order differs')
    labels, paths = np.array([int(r['label']) for r in val]), np.array([r['image_path'] for r in val])
    masks = dict(all=np.ones(len(val), bool), small=np.array([min(int(r['width']), int(r['height'])) < 224 for r in val]),
                 other=np.array([min(int(r['width']), int(r['height'])) >= 224 for r in val]),
                 tail75=np.isin(labels, plan['groups']['tail_classes']))
    record(output/'frozen_val_groups.csv', pre['val_groups_sha256'])
    frozen_groups = rows(output/'frozen_val_groups.csv')
    require(np.array_equal([r['image_path'] for r in frozen_groups], paths) and
            np.array_equal([int(r['label']) for r in frozen_groups], labels), 'Frozen group rows differ')
    for group, mask in masks.items():
        require(np.array_equal([int(r[group]) for r in frozen_groups], mask), f'{group}: frozen group definition differs')
    record(output/'schedule.npz', bind['schedule_sha256'])
    targets = torch.load(plan['config']['targets'], map_location='cpu', weights_only=False)
    active = np.flatnonzero(np.asarray(targets['weights']) > 0)
    with np.load(output/'schedule.npz', allow_pickle=False) as schedule:
        idx = schedule['train_indices']
        require(np.array_equal(idx, np.random.default_rng(42).permutation(active)[:32768]) and
                len(np.unique(idx)) == 32768, 'Training population/order differs')
        flags = schedule['detail_flags'].reshape(1024, 32)
        require(np.all(flags.sum(1) == 16), 'Not exactly half of each shared batch')
        rng = np.random.default_rng(43)
        expected_flags = np.zeros((1024, 32), dtype=bool)
        for flag in expected_flags: flag[rng.permutation(32)[:16]] = True
        require(np.array_equal(flags, expected_flags), 'Detail flags changed')
        require(np.array_equal(schedule['mixup_lambdas'], np.random.default_rng(44).beta(.2, .2, 1024)), 'Mixup changed')
        rng = np.random.default_rng(45)
        require(np.array_equal(schedule['mixup_permutations'], np.stack([rng.permutation(32) for _ in range(1024)])), 'Mixup order changed')
    require(not {train[i]['content_group'] for i in idx} & {r['content_group'] for r in val}, 'Train/val content overlap')
    baseline_path = next(p for p in bind['parent_sources'] if Path(p).name == 'val_epoch04_ema_swa_2_4.npz')
    with np.load(baseline_path, allow_pickle=False) as data:
        require(np.array_equal(data['labels'], labels) and np.array_equal(data['image_paths'], paths), 'Parent rows differ')
        baseline = data['predictions']
    for group, mask in masks.items(): same(pre['baseline'][group], count_metrics(labels, baseline, mask, 750), f'baseline/{group}')
    cost = read_json(output/'cost.json')
    require(cost['status'] == 'passed_discarded_probe_weights' and cost['wall_clock_time_limit'] is None and
            all(cost['arms'][arm]['real_updates'] == 8 for arm in cfg['arms']), 'Cost check differs')
    final, packages = {}, {}
    for arm in cfg['arms']:
        directory = output/arm
        arm_report = read_json(directory/'report.json')
        require(arm_report == report['arms'][arm] and arm_report['status'] == 'completed_packaged' and
                arm_report['optimizer_updates'] == 1024, 'Incomplete arm/report mismatch')
        trajectory = read_json(directory/'trajectory.json')
        require(trajectory['schedule_sha256'] == bind['schedule_sha256'] and trajectory['logical_updates'] == 1024 and
                trajectory['initial_checkpoint_sha256'] == next(v for p,v in bind['parent_sources'].items() if Path(p).name == 'ema_swa.pt'), 'Arm initialization differs')
        history = read_json(directory/'history.json')
        require(history == arm_report['history'] and [h['step'] for h in history] == [512, 1024], 'Evaluation points differ')
        for entry in history:
            step = entry['step']; cached = directory/f'val_step{step:04d}.npz'
            pred = val_prediction(cached, labels, paths, 750); record(cached)
            for group, mask in masks.items():
                same(entry['metrics'][group], count_metrics(labels, pred, mask, 750), f'{arm}/{step}/{group}')
                same(entry['paired_to_parent'][group], count_pair(labels, baseline, pred, mask), f'{arm}/{step}/{group}/pair')
            require(100*(np.mean(pred == labels)-np.mean(baseline == labels)) >= cfg['stop_micro_pp_vs_parent_below'], 'Stop-loss breached')
            checkpoint = directory/('selected.pt' if step == 1024 else 'step_0512.pt')
            sidecar = read_json(checkpoint.with_suffix('.sha256.json'))
            record(checkpoint, sidecar['sha256']); record(checkpoint.with_suffix('.sha256.json'))
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            require(sidecar['binding'] == bind == state['binding'] and state['classes'] == classes and
                    state['complete'] is (step == 1024) and state['optimizer_updates'] == step and
                    state['primary'] == 'last_raw' and state['bias'] is None and state['image_size'] == 512 and
                    state['arm'] == arm, 'Wrong checkpoint export')
            if step == 1024: final[arm] = pred
        package = arm_report['package']; submission = directory/'submission'
        require(package['checkpoint'] == str(directory/'selected.pt') and package['rows'] == 37444 and
                package['checks_passed'] == 9 and package['cold_validation_matches'] == 64 and
                package['zip_csv_identical'] is True, 'Incomplete package')
        record(directory/'selected.pt', package['checkpoint_sha256'])
        for filename, key in [('pred_results.csv', 'csv'), ('submission.zip', 'zip')]:
            require(package[key] == str(submission/filename), 'Package path differs')
            record(submission/filename, package[key+'_sha256'])
        with np.load(directory/'test_predictions.npz', allow_pickle=False) as data:
            names, logits = data['names'], data['logits']
            require(np.array_equal(names, [Path(r['image_path']).name for r in test]) and
                    logits.shape == (37444, 750) and np.isfinite(logits).all(), 'Wrong test population/logits')
            buffer = io.StringIO(newline=''); writer = csv.writer(buffer)
            for name, label in zip(names, logits.argmax(1)): writer.writerow([name, ' '+classes[label]])
        record(directory/'test_predictions.npz')
        expected_csv = buffer.getvalue().encode()
        require(expected_csv == (submission/'pred_results.csv').read_bytes(), 'Independent test logits-to-CSV replay differs')
        with zipfile.ZipFile(submission/'submission.zip') as archive:
            require(archive.namelist() == ['pred_results.csv'] and archive.read('pred_results.csv') == expected_csv, 'ZIP replay differs')
        checked = subprocess.run([sys.executable, str(ROOT/'scripts/check_submission.py'), '--test_dir', reference['data']['test_root'],
            '--class-mapping', str(base/'class_to_idx.json'), '--csv', str(submission/'pred_results.csv'), '--zip', str(submission/'submission.zip')],
            capture_output=True, text=True)
        require(checked.returncode == 0 and (checked.stdout+checked.stderr).count('✅') == 9, 'Independent nine checks failed')
        require((submission/'submission_check.log').read_text().count('✅') == 9, 'Original check log incomplete')
        manifest = read_json(submission/'manifest.json')
        require(manifest['binding'] == bind and manifest['decoder'] == cfg['test_decoder'] and manifest['training_updates'] == 1024 and
                manifest['checkpoint_sha256'] == package['checkpoint_sha256'] and manifest['prediction_csv_sha256'] == package['csv_sha256'] and
                manifest['submission_zip_sha256'] == package['zip_sha256'], 'Submission lineage differs')
        packages[arm] = dict(rows=37444, independent_checks=9, logits_to_csv_identical=True, zip_csv_identical=True)
    comparisons = {group: count_pair(labels, final['control'], final['detail_aug'], mask) for group,mask in masks.items()}
    for group in masks: same(report['candidate_vs_control'][group], comparisons[group], f'candidate/control/{group}')
    gate = cfg['review_gate']; full, small = comparisons['all'], comparisons['small']
    supported = bool(full['net'] >= gate['net_vs_control_min'] and
        count_pair(labels, baseline, final['detail_aug'], masks['all'])['net'] >= gate['net_vs_parent_min'] and
        small['net'] >= gate['small_group_net_vs_control_min'] and
        full['corrections'] >= gate['corrections_to_regressions_min']*full['regressions'])
    require(report['supports_review'] is supported and report['decision'] ==
            ('supports_review' if supported else 'close_fixed_detail_augmentation_recipe'), 'Frozen review decision differs')
    for path in [output/'preflight.json', output/'cost.json', output/'report.json', output/'status.json']: record(path)
    return dict(status='passed_independent_full_replay', experiment_id=cfg['experiment_id'], output=str(output),
                validation_predictions_replayed=4*14880, test_predictions_replayed=2*37444,
                metric_groups_checked=20, paired_groups_checked=20, source_and_artifact_sha256=artifacts,
                packages=packages, candidate_vs_control=comparisons, decision=report['decision'],
                platform_gain_known=False, automatic_full=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    require(not Path(args.result).exists(), 'Independent result must use a fresh path')
    result = verify(args.config, args.output)
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key != 'source_and_artifact_sha256'}))
