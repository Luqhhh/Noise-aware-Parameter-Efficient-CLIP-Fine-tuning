"""Explicitly authorized epoch-boundary recovery of the frozen machine-B pair.

The original implementation remains byte-for-byte unchanged. Checked, narrow
source substitutions reuse its actual loops and finalization; no training,
augmentation, sampling, averaging, or evaluation expression is replaced.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    from aegis_clip.runtime import atomic_json_dump
    atomic_json_dump(value, path)


def frozen_runtime(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root / 'reproducibility/aegis_f1'))
    from strong_aug_pair import runtime
    require(Path(runtime.__file__).resolve() == root / 'reproducibility/aegis_f1/strong_aug_pair/runtime.py',
            'Must import the original, byte-frozen runtime')
    return runtime


def substitute_once(source, old, new):
    require(source.count(old) == 1, 'Frozen source anchor missing or ambiguous: ' + old[:90])
    return source.replace(old, new, 1)


def resumed_train_source(runtime):
    source = inspect.getsource(runtime.train_arm)
    source = substitute_once(source, 'directory.mkdir(parents=True, exist_ok=False)',
                             'require(directory.is_dir() and not probe, "Expected migrated weak arm")')
    source = substitute_once(source, 'with log_path.open("x", encoding="utf-8") as draws:',
                             'with log_path.open("a", encoding="utf-8") as draws:')
    source = substitute_once(source, 'for epoch in range(1, 2 if probe else 5):',
                             'for epoch in range(4, 5):')
    anchor = '    dump(dict(status="probing" if probe else "training", epochs_per_arm=4), directory / "status.json")'
    return substitute_once(source, anchor,
        '    history, updates, average = restore_epoch3(runtime, directory, binding, plan, source, '
        'model, optimizer, scheduler, scaler, ema)\n' + anchor)


def restored_average(runtime, payload, device):
    average = runtime.WeightAverage.restore(payload)
    average.state = {name: value.to(device) for name, value in average.state.items()}
    return average


def restore_state(runtime, payload, model, optimizer, scheduler, scaler, ema):
    """Restore every state that determines the next update, without RNG guessing."""
    runtime.load_trainable_state(model, payload['raw_state'])
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    scaler.load_state_dict(payload['scaler'])
    device = next(model.parameters()).device
    saved_ema = restored_average(runtime, payload['ema'], device)
    ema.state, ema.decay, ema.count = saved_ema.state, saved_ema.decay, saved_ema.count
    return restored_average(runtime, payload['average'], device)


def validate_epoch3(payload, batches, classes):
    require(payload['epoch'] == 3 and payload['updates'] == 3 * batches and not payload['complete'],
            'Recovery requires the complete epoch-3 boundary, not partial epoch-4 updates')
    require(payload['classes'] == classes and payload['bias'] is None and payload['selected_policy'] == 'ema',
            'Checkpoint architecture/decoder changed')
    require(payload['ema']['count'] == 3 * batches and payload['ema']['decay'] == .999,
            'EMA trajectory/count changed')
    require(payload['average']['count'] == 2 and payload['average']['decay'] is None,
            'Expected arithmetic average of epoch-2 and epoch-3 EMA')
    require(payload['scheduler']['last_epoch'] == 3 * batches
            and payload['scheduler']['total_steps'] == 4 * batches, 'Four-epoch scheduler boundary changed')


def restore_epoch3(runtime, directory, binding, plan, source, model, optimizer, scheduler, scaler, ema):
    payload = runtime.load_artifact(directory / 'epoch_03.pt', binding)
    validate_epoch3(payload, plan['logical_batches_per_epoch'], source.classes)
    require(not any(isinstance(layer, torch.nn.modules.dropout._DropoutNd) and layer.p > 0
                    for layer in model.modules()), 'Unsaved stochastic model RNG prohibits exact boundary recovery')
    history = json.loads((directory / 'history.json').read_text())
    require([row['epoch'] for row in history] == [1, 2, 3]
            and [row['optimizer_updates'] for row in history] ==
            [e * plan['logical_batches_per_epoch'] for e in (1, 2, 3)], 'Migrated history boundary differs')
    average = restore_state(runtime, payload, model, optimizer, scheduler, scaler, ema)
    runtime.progress(directory.parents[1], 'restored_epoch3', arm='original_augmentation',
                     updates=payload['updates'], scheduler_step=scheduler.last_epoch,
                     ema_count=ema.count, average_count=average.count, next_epoch=4,
                     incomplete_epoch4_updates_discarded=True)
    return history, payload['updates'], average


def accepted_prefix(source, destination, batches):
    """Keep only checkpoint-covered draws; archive the old partial fourth epoch."""
    accepted, discarded = 0, 0
    with Path(source).open(encoding='utf-8') as old, Path(destination).open('x', encoding='utf-8') as new:
        for line in old:
            row = json.loads(line)
            expected = accepted + discarded + 1
            require(row['update'] == expected and row['epoch'] == (expected - 1) // batches + 1
                    and row['batch'] == (expected - 1) % batches + 1, 'Old draw order discontinuity')
            if row['epoch'] <= 3:
                accepted += 1
                new.write(line)
            else:
                require(row['epoch'] == 4, 'Old run exceeded four epochs')
                discarded += 1
    require(accepted == 3 * batches, 'Missing checkpoint-covered draws')
    return dict(accepted_prefix_updates=accepted, discarded_partial_epoch4_updates=discarded)


def reused_preflight(runtime, plan, source, output):
    baseline_data = np.load(output / 'baseline_center.npz', allow_pickle=False)
    labels = np.array([int(row['label']) for row in source.val])
    require(np.array_equal(baseline_data['labels'], labels)
            and np.array_equal(baseline_data['image_paths'], [row['image_path'] for row in source.val])
            and np.array_equal(baseline_data['predictions'], baseline_data['logits'].argmax(1))
            and np.array_equal(baseline_data['predictions'], source.parent_predictions), 'Frozen parent replay changed')
    frozen = json.loads((output / 'frozen_groups.json').read_text())
    require(digest(output / 'frozen_groups.csv') == frozen['group_file_sha256'], 'Frozen grouping changed')
    with (output / 'frozen_groups.csv').open() as handle:
        groups = list(csv.DictReader(handle))
    training = [row for row in groups if row['partition'] == 'train_dev']
    rows = [row for row in groups if row['partition'] == 'val_dev']
    require(len(training) == plan['train_rows'] and [row['image_path'] for row in rows]
            == [row['image_path'] for row in source.val], 'Frozen group population/alignment differs')
    require(float(np.quantile([float(row['max_cosine']) for row in training], .1, method='linear'))
            == frozen['threshold'], 'Training-only frozen quantile changed')
    require(all(int(row['target']) == int(float(row['max_cosine']) < frozen['threshold']) for row in groups),
            'Frozen target flags changed')
    masks = dict(all=np.ones(len(labels), dtype=bool), target=np.array([r['target'] == '1' for r in rows]),
                 complement=np.array([r['target'] == '0' for r in rows]), tail75=np.array([r['tail75'] == '1' for r in rows]))
    require({name: int(mask.sum()) for name, mask in masks.items()} == frozen['masks']
            and int(((source.parent_predictions != labels) & masks['target']).sum())
            == frozen['parent_target_errors'] >= 75, 'Frozen mechanism gate changed')
    return baseline_data['predictions'], labels, masks, frozen


def continued_run_source(runtime):
    source = inspect.getsource(runtime.run)
    start = source.index('        seed_training(42)\n')
    end = source.index('        bindings = {}\n', start)
    return source[:start] + (
        '        baseline, labels, masks, frozen = reused_preflight(runtime, plan, source, output)\n'
        '        dump(dict(status="formal_training", epochs_per_arm=4, pid=os.getpid(), '
        'training_started=True, explicitly_authorized_resume=True), output / "status.json")\n') + source[end:]


def resource_source(runtime, context):
    # Only this independently inspected Chromium GPU-display PID is added.
    # The original C/G-type restriction and all unknown-workload stops remain.
    source = inspect.getsource(runtime.resource_check)
    anchor = 'kind in ("G", "C+G") and Path(name).name.lower() in known_display'
    replacement = ('kind in ("G", "C+G") and (Path(name).name.lower() in known_display '
                   f'or (pid == {int(context["pid"])} and Path(name).name.lower() == {context["name"]!r}))')
    return substitute_once(source, anchor, replacement)


def verify_display_context(runtime, context):
    result = runtime.subprocess.run(['powershell.exe', '-NoProfile', '-Command',
        f'Get-CimInstance Win32_Process -Filter "ProcessId = {int(context["pid"])}" | '
        'Select-Object Name,ExecutablePath,CommandLine | ConvertTo-Json -Compress'],
        check=True, capture_output=True, text=True)
    if not result.stdout.strip():
        return  # Exited display context will also be ignored by the old check.
    process = json.loads(result.stdout)
    require(process['Name'].lower() == context['name']
            and Path(process['ExecutablePath']).resolve() == Path(context['executable']).resolve()
            and context['command_flag'] in process['CommandLine'], 'Inspected display PID changed identity')


def compiled(function_source, name, namespace):
    exec(compile(function_source, '<frozen-machine-b-explicit-resume>', 'exec'), namespace)
    return namespace[name]


def prepare(request):
    runtime = frozen_runtime(request['frozen_code_root'])
    old = Path(request['old_run']).resolve()
    output = Path(request['output']).resolve()
    require(request['authorization'] == 'user_continue_after_reported_disk_full', 'Explicit continuation required')
    require(json.loads((old / 'status.json').read_text())['status'] == 'aborted', 'Old run is not stopped')
    require(not output.exists() and output.drive.lower() == 'd:', 'Recovery needs a fresh D-drive directory')
    require(shutil.disk_usage(output.parent).free >= 10 * 1024**3, 'Insufficient D-drive storage')
    plan, source = runtime.verify(old / 'plan.json')
    binding = dict(plan_sha256=digest(old / 'plan.json'), source_binding=source.binding,
                   frozen_groups_sha256=digest(old / 'frozen_groups.csv'), arm='original_augmentation')
    checkpoint = old / 'arms/original_augmentation/epoch_03.pt'
    validate_epoch3(runtime.load_artifact(checkpoint, binding), plan['logical_batches_per_epoch'], source.classes)
    require(not (old / 'arms/strong_augmentation').exists(), 'Strong arm already started; do not duplicate it')
    output.mkdir()
    archive = output / 'original_abort_archive'
    shutil.copytree(old, archive)
    archive_hashes = {str(path.relative_to(old)): digest(path) for path in old.rglob('*') if path.is_file()}
    require(all(digest(archive / name) == value for name, value in archive_hashes.items()), 'Archive copy differs')
    for name in ('plan.json', 'location_map.json', 'parent_replay.json', 'baseline_center.npz',
                 'frozen_groups.csv', 'frozen_groups.json', 'cost.json'):
        shutil.copy2(archive / name, output / name)
    shutil.copytree(archive / 'probe', output / 'probe')
    arm = output / 'arms/original_augmentation'
    arm.mkdir(parents=True)
    for path in (archive / 'arms/original_augmentation').iterdir():
        if path.is_file() and path.name not in ('draws.jsonl', 'status.json'):
            shutil.copy2(path, arm / path.name)
    prefix = accepted_prefix(archive / 'arms/original_augmentation/draws.jsonl', arm / 'draws.jsonl',
                             plan['logical_batches_per_epoch'])
    snapshot = output / 'resume_engine.py'
    shutil.copy2(__file__, snapshot)
    if request.get('previous_preflight_attempt'):
        previous = Path(request['previous_preflight_attempt'])
        require(not (previous / 'status.json').exists(), 'Prior attempt may have entered training')
        evidence = output / 'previous_preflight_attempt'
        evidence.mkdir()
        for name in ('run.err', 'run.log', 'launch.json', 'resume_manifest.json'):
            shutil.copy2(previous / name, evidence / name)
    immutable = {str(path.relative_to(output)): digest(path) for path in output.rglob('*') if path.is_file()
                 and not path.is_relative_to(archive) and path.name not in ('history.json', 'draws.jsonl')}
    manifest = dict(schema=1, **request, **prefix, original_checkpoint_sha256=digest(checkpoint),
        original_plan_sha256=binding['plan_sha256'], source_binding=source.binding,
        archive_sha256=archive_hashes, immutable_sha256=immutable,
        accepted_draw_prefix_sha256=digest(arm / 'draws.jsonl'),
        history_prefix_sha256=digest(arm / 'history.json'), engine_sha256=digest(__file__),
        training_source_sha256=hashlib.sha256(resumed_train_source(runtime).encode()).hexdigest(),
        finalization_source_sha256=hashlib.sha256(continued_run_source(runtime).encode()).hexdigest(),
        resource_source_sha256=hashlib.sha256(resource_source(runtime, request['display_context']).encode()).hexdigest(),
        epochs_per_arm=4, average_epochs=[2, 3, 4], weak_next_epoch=4, strong_starts_from_original_parent=True,
        original_artifacts_modified=False, automatic_restart=False, wall_clock_cutoff=None)
    write_json(output / 'resume_manifest.json', manifest)
    print(json.dumps(dict(status='recovery_prepared', output=str(output), **prefix)), flush=True)


def run(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    runtime = frozen_runtime(manifest['frozen_code_root'])
    output = manifest_path.parent
    require(str(output) == str(Path(manifest['output']).resolve()), 'Manifest/output location differs')
    require(digest(__file__) == manifest['engine_sha256'], 'Resume engine changed')
    for name, expected in manifest['archive_sha256'].items():
        require(digest(output / 'original_abort_archive' / name) == expected, 'Original evidence archive changed: ' + name)
    for name, expected in manifest['immutable_sha256'].items():
        require(digest(output / name) == expected, 'Migrated immutable artifact changed: ' + name)
    arm = output / 'arms/original_augmentation'
    require(digest(arm / 'draws.jsonl') == manifest['accepted_draw_prefix_sha256']
            and digest(arm / 'history.json') == manifest['history_prefix_sha256'], 'Recovery already attempted or prefix changed')
    train_source, run_source = resumed_train_source(runtime), continued_run_source(runtime)
    require(hashlib.sha256(train_source.encode()).hexdigest() == manifest['training_source_sha256']
            and hashlib.sha256(run_source.encode()).hexdigest() == manifest['finalization_source_sha256'], 'Frozen loop adaptations changed')
    namespace = dict(vars(runtime), runtime=runtime, restore_epoch3=restore_epoch3, reused_preflight=reused_preflight)
    resume_weak = compiled(train_source, 'train_arm', dict(namespace))
    def dispatch(plan, source, output, arm, masks, baseline, *, probe=False):
        require(not probe, 'Cost probes must remain frozen and discarded')
        implementation = resume_weak if arm == 'original_augmentation' else runtime.train_arm
        return implementation(plan, source, output, arm, masks, baseline)
    namespace['train_arm'] = dispatch
    checked_resources = compiled(resource_source(runtime, manifest['display_context']),
                                 'resource_check', dict(vars(runtime)))
    require(hashlib.sha256(resource_source(runtime, manifest['display_context']).encode()).hexdigest()
            == manifest['resource_source_sha256'], 'Resource-check adaptation changed')
    def resource_check(output):
        verify_display_context(runtime, manifest['display_context'])
        report = checked_resources(output)
        report['additional_inspected_display_context'] = manifest['display_context']
        write_json(output / 'resources.json', report)
        return report
    namespace['resource_check'] = resource_check
    continued_run = compiled(run_source, 'run', namespace)
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        continued_run(output / 'plan.json')
        if (output / 'report.json').exists():
            spec = importlib.util.spec_from_file_location('frozen_strong_aug_verifier',
                Path(manifest['frozen_code_root']) / 'scripts/verify_strong_aug_pair.py')
            verifier = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(verifier)
            verifier.verify_report(output / 'plan.json')
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--request')
    parser.add_argument('--manifest')
    args = parser.parse_args()
    if args.action == 'prepare':
        require(args.request, 'Preparation requires a request JSON')
        prepare(json.loads(Path(args.request).read_text()))
    else:
        require(args.manifest, 'Continuation requires the verified migration manifest')
        run(args.manifest)
