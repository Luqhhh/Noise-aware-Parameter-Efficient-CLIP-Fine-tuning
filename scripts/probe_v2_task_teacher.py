"""One fixed V2 DEV forward to audit a confidence-gated transfer budget."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diagnose_v1_candidate_errors import metrics, paired, read_json, read_rows, require, sha
from diagnose_v1_frozen_teacher_recovery import supported, gate_pass
from run_v2_fixed_prior import ensure_idle_cuda, load_model
from v2.runtime import Budget, InferenceDataset, amp
from v2.training_utils import build_eval_transform

ROOT = Path(__file__).resolve().parents[1]


def score_logits(logits):
    require(logits.ndim == 2 and np.isfinite(logits).all(), 'Invalid teacher logits')
    z = logits.astype(np.float64)
    z -= z.max(1, keepdims=True)
    probabilities = np.exp(z)
    probabilities /= probabilities.sum(1, keepdims=True)
    top = np.partition(probabilities, -2, axis=1)[:, -2:]
    return dict(prediction=probabilities.argmax(1), confidence=top.max(1),
                margin=top.max(1)-top.min(1))


def replay_check(original, scores, support, protocol):
    changed = original != scores['prediction']
    count = int(changed.sum())
    margin = float(scores['margin'][changed].max()) if changed.any() else 0.
    supported_count = int((changed & support).sum())
    require(count <= protocol['max_top1_changes'], 'Too many native top1 replay changes')
    require(margin <= protocol['changed_probability_margin_max'], 'Confident native replay change')
    require(supported_count <= protocol['supported_changes_allowed'], 'Supported native replay change')
    return dict(rows=len(original), top1_changes=count, changed_probability_margin_max=margin,
                supported_top1_changes=supported_count, exact_top1_replay=count == 0)


def check_teacher(payload, cfg, classes):
    require(payload['binding']['stage'] == 's3_576' == cfg['teacher_stage'] and
            payload['binding']['data_version'] == '20260921' and payload['binding']['complete'] is True and
            payload['binding']['plan_sha256'] == cfg['plan_sha256'], 'Wrong/incomplete teacher stage')
    require(payload['epoch'] == cfg['teacher_epoch_zero_based'] == 3 and
            payload['metrics']['chosen'] == cfg['teacher_weights'] == 'ema' and
            payload['metrics']['validation_is_independent'] is True and
            all(h['validation_is_independent'] is True for h in payload['history']), 'Not the selected DEV EMA')
    data = payload['config']['data']
    require(payload['config']['local_replay']['final_stage'] is False and payload['num_classes'] == classes and
            data['eval_size'] == cfg['image_size'] == 576 and
            data['eval_resize_ratio'] == cfg['resize_ratio'] == 1.14 and
            payload['config']['local_replay']['micro_batch_size'] == cfg['batch_size'] == 16,
            'Native evaluation recipe changed')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(config_path, output):
    cfg = read_json(config_path)
    require(all(cfg[k] is False for k in ('new_training', 'test_data_used', 'fusion_predictions_created',
            'automatic_training', 'parameter_search')), 'Diagnostic scope changed')
    inherited = read_json(ROOT / 'configs/v1_frozen_teacher_recovery_20261001.json')
    require(cfg['support_rule'] == inherited['support_rule'] and cfg['feasibility_gate'] == inherited['feasibility_gate'],
            'Fixed support rule or investment gate changed')
    require(cfg['operation_budget_seconds'] + cfg['consumed_seconds_before_native_correction'] <=
            cfg['total_diagnostic_budget_seconds'] == 900, 'Cumulative diagnostic budget exceeded')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(cfg['operation_budget_seconds'])
    sources = {}

    def check(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or expected == digest, f'Source SHA mismatch: {path}')
        sources[str(path)] = digest
        budget.check()
        return path

    try:
        dump(output / 'progress.json', dict(status='validating_inputs', rows=0))
        check(config_path)
        for name in (__file__, ROOT / 'scripts/diagnose_v1_candidate_errors.py',
                ROOT / 'scripts/diagnose_v1_frozen_teacher_recovery.py', ROOT / 'scripts/run_v2_fixed_prior.py',
                ROOT / 'reproducibility/aegis_f1/v2/model.py', ROOT / 'reproducibility/aegis_f1/v2/runtime.py',
                ROOT / 'reproducibility/aegis_f1/v2/training_utils.py', ROOT / 'reproducibility/aegis_f1/aegis_clip/model.py',
                ROOT / 'configs/v1_frozen_teacher_recovery_20261001.json'):
            check(name)
        stage = Path(cfg['stage_root'])
        manifest = read_json(check(stage / 'dataset_manifest.json', cfg['dataset_manifest_sha256']))
        require(manifest['data_version'] == '20260921' and manifest['stage'] == 'repechage', 'Foreign data stage')
        for name in ('class_to_idx.json', 'train_dev.csv', 'val_dev.csv'):
            check(stage / name, manifest['files'][name])
        val, train = read_rows(stage / 'val_dev.csv'), read_rows(stage / 'train_dev.csv')
        classes, n = manifest['num_classes'], len(val)
        require(n == manifest['val_dev_samples'] and len(train) == manifest['train_dev_samples'], 'Wrong population')
        require(not {r['content_group'] for r in val} & {r['content_group'] for r in train}, 'Content leakage')
        mapping = read_json(stage / 'class_to_idx.json')
        require(sorted(mapping.values()) == list(range(classes)), 'Invalid class mapping')
        previous = read_json(check(ROOT / cfg['v2_error_report'], cfg['v2_error_report_sha256']))
        require(previous['new_training'] is False and previous['test_predictions_used'] is False and
                previous['selections']['s3_576']['checkpoint_sha256'] == cfg['checkpoint_sha256'], 'Wrong source diagnostic')
        old = read_rows(check(previous['aligned_csv'], previous['aligned_csv_sha256']))
        require(len(old) == n, 'Wrong cached population')
        for a, b in zip(val, old):
            require(all(a[k] == b[k] for k in ('image_path', 'label', 'content_group')), 'Cached image alignment mismatch')
        labels = np.array([int(r['label']) for r in val])
        paths = np.array([r['image_path'] for r in val])
        require(len(set(paths)) == n, 'Duplicate validation path')
        original = np.array([int(r['s3_ema']) for r in old])
        students = {k: np.array([int(r[k]) for r in old]) for k in cfg['v1_candidates']}
        groups = {k: np.array([int(r[k]) for r in old], dtype=bool) for k in cfg['groups']}
        common = np.logical_and.reduce([p != labels for p in students.values()])
        require(int(common.sum()) == previous['overlap']['all']['v1_all_wrong'], 'Common error mismatch')
        groups['common_v1_errors'] = common
        groups['other_primary_errors'] = (students[cfg['primary_student']] != labels) & ~common
        for row in val:
            path = Path(manifest['train_root']) / Path(row['image_path']).relative_to('train')
            require(sha(path) == row['file_sha256'], 'Validation image bytes changed: ' + row['image_path'])
            budget.check()
        check(cfg['official_checkpoint'], cfg['official_sha256'])
        checkpoint = check(cfg['checkpoint'], cfg['checkpoint_sha256'])
        # Owned checkpoint with authenticated bytes, including its original RNG metadata.
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        check_teacher(payload, cfg, classes)
        ensure_idle_cuda()
        model = load_model(cfg, payload['ema'], classes)
        del payload
        data = DataLoader(InferenceDataset(Path(manifest['train_root']),
            [str(Path(p).relative_to('train')) for p in paths], build_eval_transform(cfg['image_size'], cfg['resize_ratio'])),
            batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True,
            persistent_workers=False, prefetch_factor=2)
        logits = np.empty((n, classes), np.float32)
        visited = np.zeros(n, bool)
        batch_seconds, probe_seconds = [], None
        torch.cuda.reset_peak_memory_stats()
        gpu_started = time.monotonic()
        last_batch_end = gpu_started
        with torch.inference_mode():
            for batch, (images, indices) in enumerate(data):
                budget.check()
                with amp():
                    values = model(images.cuda(non_blocking=True)).float().cpu().numpy()
                ix = indices.numpy()
                require(np.isfinite(values).all() and not visited[ix].any(), 'Invalid/repeated model output')
                logits[ix] = values
                visited[ix] = True
                now = time.monotonic()
                batch_seconds.append(now - last_batch_end)
                last_batch_end = now
                if batch + 1 == cfg['cost_probe_batches']:
                    probe_seconds = max(batch_seconds[1:]) * (len(data) - batch - 1)
                    require(probe_seconds + now - budget.start < budget.limit, 'Projected native forward exceeds budget')
                    dump(output / 'cost.json', dict(first_batch_seconds=batch_seconds[0],
                        warmed_batch_seconds=batch_seconds[1:], projected_remaining_seconds=probe_seconds,
                        elapsed_seconds=now-budget.start, reused_in_full_forward=True))
                if batch % 16 == 0:
                    progress = dict(status='dev_forward', rows=int(visited.sum()), total=n, seconds=now-budget.start)
                    dump(output / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
        require(visited.all(), 'Incomplete DEV forward')
        gpu_seconds = time.monotonic() - gpu_started
        environment = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            batch_size=cfg['batch_size'], amp_dtype='bfloat16', peak_allocated_bytes=torch.cuda.max_memory_allocated())
        del model
        torch.cuda.empty_cache()
        scores = score_logits(logits)
        support = supported(scores, cfg['support_rule'])
        np.savez_compressed(output / 'scores.npz', logits=logits, image_paths=paths, labels=labels,
            original_v2_prediction=original, teacher_prediction=scores['prediction'], confidence=scores['confidence'],
            margin=scores['margin'], supported=support, **students, **{f'group_{k}': v for k, v in groups.items()})
        replay = replay_check(original, scores, support, cfg['native_replay'])
        comparisons = {}
        for name in (cfg['primary_student'], cfg['secondary_student']):
            student = students[name]
            comparisons[name] = {k: dict(teacher=metrics(labels, scores['prediction'], classes, mask),
                student=metrics(labels, student, classes, mask), paired=paired(labels, student, scores['prediction'], mask),
                supported_rows=int((mask & support).sum()),
                supported_teacher=metrics(labels, scores['prediction'], classes, mask & support),
                supported_student=metrics(labels, student, classes, mask & support),
                supported_paired=paired(labels, student, scores['prediction'], mask & support),
                unsupported_paired=paired(labels, student, scores['prediction'], mask & ~support))
                for k, mask in groups.items()}
        primary = comparisons[cfg['primary_student']]
        common_recovered = primary['common_v1_errors']['supported_paired']['corrections']
        passed = gate_pass(primary['all']['supported_paired'], common_recovered, cfg['feasibility_gate'])
        budget.check()
        report = dict(experiment_id=cfg['experiment_id'], status='completed_diagnostic',
            validation_rows=n, classes=classes, validation_independent_of_training=True,
            validation_used_for_original_selection=True, sources=sources, config=str(Path(config_path).resolve()),
            score_file=str(output / 'scores.npz'), score_sha256=sha(output / 'scores.npz'),
            native_replay=replay, support_rows=int(support.sum()), comparisons=comparisons,
            primary_student=cfg['primary_student'], gate_passed=passed,
            decision='supports_transfer_review' if passed else 'close_fixed_v2_confident_teacher_transfer',
            cost_probe_projected_remaining_seconds=probe_seconds, gpu_forward_seconds=gpu_seconds,
            elapsed_seconds=time.monotonic()-budget.start, environment=environment,
            new_training=False, test_data_used=False, new_candidate=False, automatic_training=False,
            fusion_predictions_created=False, platform_gain_known=False, causal_comparison=False,
            limitation='Conditional recovery counts under original noisy labels are not a deployable model, attainable distillation gain or evidence against/above the incumbent 768 full.')
        dump(output / 'report.json', report)
        dump(output / 'progress.json', dict(status='completed_diagnostic', rows=n, seconds=report['elapsed_seconds']))
        print(json.dumps(dict(status=report['status'], support_rows=report['support_rows'], native_replay=replay,
            supported=primary['all']['supported_paired'], common_supported_recoveries=common_recovered,
            gate_passed=passed, decision=report['decision'], seconds=report['elapsed_seconds'])), flush=True)
        return report
    except Exception as exc:
        dump(output / 'failure.json', dict(status='failed_closed', error_type=type(exc).__name__, error=str(exc),
            elapsed_seconds=time.monotonic()-budget.start, new_training=False, new_candidate=False,
            automatic_restart=False, sources=sources))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.config, args.output)
