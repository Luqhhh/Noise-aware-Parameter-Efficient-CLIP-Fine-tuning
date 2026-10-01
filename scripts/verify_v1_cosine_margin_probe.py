"""Independent frozen-group, paired-count and submission replay; never trains."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

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


def metrics(labels, prediction, mask):
    totals, correct = Counter(), Counter()
    for label, pred, include in zip(labels, prediction, mask):
        if include:
            totals[int(label)] += 1
            correct[int(label)] += int(label == pred)
    n, c = sum(totals.values()), sum(correct.values())
    return dict(rows=n, correct=c, errors=n-c, classes=len(totals), micro=c/n,
                macro=sum(correct[k]/totals[k] for k in sorted(totals))/len(totals))


def pair(labels, before, after, mask):
    corr = reg = changed = wrong = n = 0
    for y, a, b, include in zip(labels, before, after, mask):
        if include:
            n += 1
            corr += int(a != y and b == y)
            reg += int(a == y and b != y)
            changed += int(a != b)
            wrong += int(a != y and b != y)
    return dict(rows=n, corrections=corr, regressions=reg, net=corr-reg,
                changed=changed, both_wrong=wrong)


def same(expected, actual, label):
    require(set(expected) == set(actual), f'{label}: keys differ')
    for key, value in actual.items():
        require(abs(expected[key]-value) <= 1e-12 if isinstance(value, float)
                else expected[key] == value, f'{label}/{key}: value differs')


def prediction(path, labels, paths, classes):
    with np.load(path, allow_pickle=False) as z:
        require(np.array_equal(z['labels'], labels) and np.array_equal(z['image_paths'], paths),
                f'Validation alignment differs: {path}')
        require(z['logits'].shape == (len(labels), classes) and np.isfinite(z['logits']).all(), 'Invalid logits')
        pred = z['logits'].argmax(1)
        require(np.array_equal(pred, z['predictions']), 'Argmax differs')
    return pred


def verify(config, output):
    cfg = read_json(config)
    out = Path(output).resolve()
    pre, report, cost = [read_json(out/f'{name}.json') for name in ['preflight', 'report', 'cost']]
    bind = pre['binding']
    require(bind == report['binding'] and bind['config_sha256'] == sha(config) and
            bind['implementation_sha256'] == sha(ROOT/'scripts/train_v1_cosine_margin_probe.py') and
            bind['loss_sha256'] == sha(ROOT/'scripts/v1_cosine_margin_loss.py'), 'Frozen implementation differs')
    require(report['status'] == 'completed_packaged_single_candidate' and report['experiment_id'] == cfg['experiment_id'] and
            read_json(out/'status.json')['status'] == report['status'] and report['optimizer_updates'] == cfg['steps'] == 1024 and
            report['primary_checkpoint'] == cfg['primary_checkpoint'] == 'last_raw_step1024' and
            report['read_only_control'] is True and report['control_training_repeated'] is False and
            report['cosine_margin'] == cfg['cosine_margin'] == .05 and report['accuracy_early_stop'] is False and
            report['margin_applied_at_inference'] is False and all(report[k] is False for k in
                ['automatic_full', 'parameter_search', 'platform_upload', 'platform_gain_known']), 'Incomplete fixed candidate')
    artifacts = {}
    def record(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, f'Checksum differs: {path}')
        artifacts[str(path)] = digest
    for path, digest in bind['parent_sources'].items():
        record(path, digest)
    record(config, bind['config_sha256'])
    record(ROOT/'scripts/train_v1_cosine_margin_probe.py', bind['implementation_sha256'])
    record(ROOT/'scripts/v1_cosine_margin_loss.py', bind['loss_sha256'])
    plan = read_json(pre['source_plan'])
    for path, digest in plan['inputs'].items():
        record(path, digest)
    parent_cfg = yaml.safe_load(Path(plan['config']['parent_config']).read_text())
    reference = load_config(Path(parent_cfg['source']['root'])/parent_cfg['source']['reference_config'])
    base = Path(reference['data']['dataset_manifest']).parent
    record(base/'dataset_manifest.json', plan['source_binding']['dataset_manifest_sha256'])
    record(base/'class_to_idx.json', plan['source_binding']['class_mapping_sha256'])
    mapping = read_json(base/'class_to_idx.json')
    classes = [name for name, index in sorted(mapping.items(), key=lambda item: item[1])]
    require(classes == plan['classes'], 'Class order differs')
    train, val, test = [rows(base/name) for name in ['train_dev.csv', 'val_dev.csv', 'test_manifest.csv']]
    labels, paths = np.array([int(r['label']) for r in val]), np.array([r['image_path'] for r in val])
    require(report['validation_rows'] == len(labels), 'Validation population differs')
    source_native = next(p for p in bind['parent_sources'] if Path(p).name == 'val_epoch04_ema_swa_2_4.npz')
    with np.load(source_native, allow_pickle=False) as z:
        require(np.array_equal(z['labels'], labels) and np.array_equal(z['image_paths'], paths), 'Parent alignment differs')
        native = z['predictions'].copy()
    views = next(p for p in bind['parent_sources'] if Path(p).name == 'lr512_swa.npz')
    with np.load(views, allow_pickle=False) as z:
        require(np.array_equal(z['labels'], labels) and np.array_equal(z['image_paths'], paths), 'Gap source alignment differs')
        logits = (z['original_logits']+z['flipped_logits'])/2
        require(np.isfinite(logits).all() and np.array_equal(logits.argmax(1), native), 'Gap source native replay differs')
        top = np.sort(logits, axis=1)[:, -2:]
        gap = top[:, 1]-top[:, 0]
        common = z['common_candidate_wrong'].copy()
    rank = sorted(range(len(val)), key=lambda i: (float(gap[i]), paths[i]))
    low = np.zeros(len(val), bool)
    low[rank[:len(val)//4]] = True
    groups = dict(all=np.ones(len(val), bool), low_gap=low, other_gap=~low,
        tail75=np.isin(labels, plan['groups']['tail_classes']),
        small=np.array([min(int(r['width']), int(r['height'])) < 224 for r in val]), common_candidate_wrong=common)
    require(cfg['target_rule'] == pre['target_group_rule'] ==
            'lowest_quartile_native_top1_minus_top2_logit_gap;ties_image_path_ascending' and
            int((low & (native != labels)).sum()) >= cfg['target_parent_errors_min'], 'Frozen target differs')
    record(out/'frozen_groups.npz', pre['frozen_groups_sha256'])
    with np.load(out/'frozen_groups.npz', allow_pickle=False) as z:
        for key, value in dict(labels=labels, image_paths=paths, native_gap=gap, **groups).items():
            require(np.array_equal(z[key], value), f'Frozen {key} differs')
    source = Path(cfg['source_control_root'])
    old_cfg, old_report = read_json(cfg['source_control_config']), read_json(source/'report.json')
    require(old_report == read_json(ROOT/cfg['source_control_report']) and old_report['status'] == 'completed_packaged_pair' and
            report['reused_control_package'] == old_report['arms']['control']['package'], 'Read-only control differs')
    control_path = source/'control/val_step1024.npz'
    control = prediction(control_path, labels, paths, len(classes))
    require(str(control_path) in bind['parent_sources'], 'Control prediction digest was not frozen')
    record(out/'schedule.npz', cfg['source_schedule_sha256'])
    require(bind['schedule_sha256'] == cfg['source_schedule_sha256'] and
            (out/'schedule.npz').read_bytes() == (source/'schedule.npz').read_bytes(), 'Original schedule bytes differ')
    targets = torch.load(plan['config']['targets'], map_location='cpu', weights_only=False)
    with np.load(out/'schedule.npz', allow_pickle=False) as z:
        frozen = {k: z[k].copy() for k in z.files}
    active = np.flatnonzero(np.asarray(targets['weights']) > 0)
    idx = frozen['train_indices']
    require(np.array_equal(idx, np.random.default_rng(42).permutation(active)[:32768]) and
            len(np.unique(idx)) == 32768 and len(idx) == pre['training_rows'] and
            bool((targets['original_alpha'][idx] == 0).all()), 'Training source/targets/order differ')
    require(not {train[i]['content_group'] for i in idx} & {r['content_group'] for r in val}, 'Training/validation overlap')
    require(np.array_equal(frozen['mixup_lambdas'], np.random.default_rng(44).beta(.2, .2, 1024)), 'Mixup lambdas differ')
    rng = np.random.default_rng(45)
    require(np.array_equal(frozen['mixup_permutations'], np.stack([rng.permutation(32) for _ in range(1024)])), 'Mixup pairing differs')
    absent = np.flatnonzero(np.bincount(np.asarray(targets['targets'])[idx], minlength=len(classes)) == 0).tolist()
    require(absent == pre['training_target_classes_absent'], 'Target class coverage differs')
    # Reproduce the original deterministic CPU image pipeline without loading a model.
    import train_v1_detail_aug_pair as old
    context = SimpleNamespace(train=train, train_root=Path(reference['data']['train_root']))
    images, _ = next(iter(old.loader(context, old_cfg, frozen, 'control', cfg['cost_steps'])))
    pixel_sha = hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()
    require(pixel_sha == pre['source_first_batch_pixel_sha256'] == report['source_first_batch_pixel_sha256'] and
            report['common_augmentation_pixel_hash_matches'] is True, 'Original augmentation replay differs')
    require(cost['status'] == 'passed_discarded_probe_weights' and cost['real_updates'] == cfg['cost_steps'] == 8 and
            cost['wall_clock_time_limit'] is None and cost['common_augmentation_pixel_hash_matches'] is True and
            len(cost['seconds']) == 8 and all(t > 0 for t in cost['seconds']), 'Real cost check differs')
    metric_count = pair_count = 0
    for group, mask in groups.items():
        same(pre['parent'][group], metrics(labels, native, mask), f'parent/{group}')
        same(pre['reused_control'][group], metrics(labels, control, mask), f'control/{group}')
        same(report['reused_control'][group], metrics(labels, control, mask), f'report/control/{group}')
        metric_count += 3
    directory = out/'candidate'
    original_checkpoint = next(p for p in bind['parent_sources'] if Path(p).name == 'ema_swa.pt')
    original_state = torch.load(original_checkpoint, map_location='cpu', weights_only=False)['selected_state']
    history = read_json(directory/'history.json')
    require(history == report['history'] and [h['step'] for h in history] == cfg['evaluation_steps'] == [512, 1024], 'Evaluation points differ')
    for item in history:
        step = item['step']
        cached = directory/f'val_step{step:04d}.npz'
        pred = prediction(cached, labels, paths, len(classes))
        record(cached)
        for group, mask in groups.items():
            same(item['metrics'][group], metrics(labels, pred, mask), f'{step}/{group}')
            same(item['paired_to_parent'][group], pair(labels, native, pred, mask), f'{step}/parent/{group}')
            same(item['paired_to_control'][group], pair(labels, control, pred, mask), f'{step}/control/{group}')
            metric_count += 1
            pair_count += 2
        checkpoint = directory/('selected.pt' if step == 1024 else 'step_0512.pt')
        sidecar = read_json(checkpoint.with_suffix('.sha256.json'))
        record(checkpoint, sidecar['sha256'])
        record(checkpoint.with_suffix('.sha256.json'))
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        selected_state = state['selected_state']
        require(selected_state.keys() == original_state.keys() and all(
                selected_state[k].shape == original_state[k].shape and torch.isfinite(selected_state[k]).all()
                for k in original_state), 'Export changed parameter layout or contains nonfinite tensors')
        require(state['binding'] == sidecar['binding'] == bind and state['classes'] == classes and state['image_size'] == 512 and
                state['optimizer_updates'] == step and state['complete'] is (step == 1024) and state['primary'] == 'last_raw' and
                state['arm'] == 'cosine_margin' and state['cosine_margin'] == .05 and state['bias'] is None and
                state['margin_applied_at_inference'] is False and state['parent_checkpoint_sha256'] ==
                next(d for p, d in bind['parent_sources'].items() if Path(p).name == 'ema_swa.pt'), 'Checkpoint lineage differs')
    retries = read_json(directory/'amp_retries.json')
    require(len(retries) == report['amp_retry_batches'] and all(1 <= r['step'] <= 1024 and
            np.isfinite(r['gradient_norm']) and all(a['new_scale'] < a['old_scale'] for a in r['amp_retries']) for r in retries),
            'Numerical retry evidence differs')
    pkg = report['package']
    submit = directory/'submission'
    require(pkg['checkpoint'] == str(directory/'selected.pt') and pkg['rows'] == len(test) and pkg['checks_passed'] == 9 and
            pkg['cold_validation_matches'] == 64 and pkg['zip_csv_identical'] is True and pkg['margin_applied_at_inference'] is False,
            'Incomplete candidate package')
    record(directory/'selected.pt', pkg['checkpoint_sha256'])
    for name, key in [('pred_results.csv', 'csv'), ('submission.zip', 'zip')]:
        require(pkg[key] == str(submit/name), 'Package path differs')
        record(submit/name, pkg[key+'_sha256'])
    with np.load(directory/'test_predictions.npz', allow_pickle=False) as z:
        require(np.array_equal(z['names'], [Path(r['image_path']).name for r in test]) and
                z['logits'].shape == (len(test), len(classes)) and np.isfinite(z['logits']).all(), 'Test population/logits differ')
        buffer = io.StringIO(newline='')
        writer = csv.writer(buffer)
        for name, label in zip(z['names'], z['logits'].argmax(1)):
            writer.writerow([name, ' '+classes[label]])
    expected_csv = buffer.getvalue().encode()
    require(expected_csv == (submit/'pred_results.csv').read_bytes(), 'Independent logits-to-CSV bytes differ')
    with zipfile.ZipFile(submit/'submission.zip') as z:
        require(z.namelist() == ['pred_results.csv'] and z.read('pred_results.csv') == expected_csv, 'ZIP replay differs')
    checked = subprocess.run([sys.executable, str(ROOT/'scripts/check_submission.py'), '--test_dir', reference['data']['test_root'],
        '--class-mapping', str(base/'class_to_idx.json'), '--csv', str(submit/'pred_results.csv'), '--zip', str(submit/'submission.zip')],
        text=True, capture_output=True)
    checked_text = checked.stdout+checked.stderr
    require(checked.returncode == 0 and checked_text.count('✅') == 8 and 'All checks passed!' in checked_text,
            'Independent nine-constraint submission check failed')
    original = (submit/'submission_check.log').read_text()
    require(original.count('✅') == 8 and 'All checks passed!' in original, 'Original submission check incomplete')
    manifest = read_json(submit/'manifest.json')
    require(manifest['binding'] == bind and manifest['decoder'] == cfg['test_decoder'] and manifest['training_updates'] == 1024 and
            manifest['checkpoint_sha256'] == pkg['checkpoint_sha256'] and manifest['prediction_csv_sha256'] == pkg['csv_sha256'] and
            manifest['submission_zip_sha256'] == pkg['zip_sha256'] and manifest['cosine_margin_training_only'] == .05 and
            manifest['margin_applied_at_inference'] is False, 'Submission provenance differs')
    item, gate = history[-1], cfg['review_gate']
    full = pair(labels, control, pred, groups['all'])
    criteria = dict(net_vs_control=full['net'] >= gate['net_vs_control_min'],
        net_vs_parent=pair(labels, native, pred, groups['all'])['net'] >= gate['net_vs_parent_min'],
        target_net_vs_control=pair(labels, control, pred, low)['net'] >= gate['target_net_vs_control_min'],
        macro_vs_control=100*(metrics(labels, pred, groups['all'])['macro']-metrics(labels, control, groups['all'])['macro'])
            >= gate['macro_delta_pp_vs_control_min'],
        correction_regression_ratio=full['corrections'] >= gate['corrections_to_regressions_min']*full['regressions'])
    passed = all(criteria.values())
    require(report['supports_review'] is passed and report['decision'] ==
            ('supports_bounded_margin_review' if passed else 'close_fixed_cosine_margin_recipe'), 'Frozen decision differs')
    for path in [out/'preflight.json', out/'cost.json', out/'report.json', out/'status.json', directory/'history.json',
                 directory/'amp_retries.json', directory/'test_predictions.npz', submit/'manifest.json', submit/'submission_check.log']:
        record(path)
    return dict(status='passed_independent_full_replay', experiment_id=cfg['experiment_id'], output=str(out),
        verifier_sha256=sha(__file__), validation_predictions_replayed=2*len(val), test_predictions_replayed=len(test),
        metric_groups_checked=metric_count, paired_groups_checked=pair_count, native_gap_rows_replayed=len(val),
        original_augmentation_first_batch_replayed=32, original_control_sources_unchanged=True,
        read_only_control=True, control_training_repeated=False, successful_optimizer_updates=1024,
        package=dict(rows=len(test), independent_checks=9, logits_to_csv_identical=True, zip_csv_identical=True,
                     cold_validation_matches=64, margin_applied_at_inference=False),
        gate_criteria=criteria, decision=report['decision'], source_and_artifact_sha256=artifacts,
        automatic_full=False, platform_gain_known=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    require(not Path(args.result).exists(), 'Independent result must use a fresh path')
    torch.set_num_threads(2)
    result = verify(args.config, args.output)
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_and_artifact_sha256'}))
