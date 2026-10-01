"""One bounded low-detail training intervention with identical paired schedules."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from aegis_clip.runtime import atomic_json_dump, sha256_file, seed_worker
from aegis_clip.submission import create_submission
from aegis_clip.v1_pipeline import (Images, MEAN, STD, load_artifact,
                                  read_rows, save_artifact, seed_training)
from aegis_clip.v1_strategy import load_trainable_state, target_probabilities, trainable_state
from v1_continuation.runtime import initial_model, logical_update, optimizer_for, evaluate, require_idle_cuda
from v1_continuation.plan import verify as verify_source_plan
from probe_v1_resolution_degradation import inspect as inspect_parent
from diagnose_v1_candidate_errors import require, metrics, paired

ROOT = Path(__file__).resolve().parents[1]


def inspect(config):
    cfg = json.loads(Path(config).read_text())
    require(cfg['steps_per_arm'] == 1024 and cfg['evaluation_steps'] == [512, 1024] and
        cfg['cost_steps_per_arm'] == 8 and cfg['batch_size'] == 32 and cfg['micro_batch_size'] == 8 and
        cfg['workers'] == 2 and cfg['seed'] == 42 and cfg['arms'] == ['control', 'detail_aug'] and
        cfg['primary_checkpoint'] == 'last_raw_step1024' and cfg['ema'] is False and
        cfg['automatic_full_training'] is False and cfg['parameter_search'] is False and
        cfg['platform_upload'] is False, 'Only the fixed limited pair is supported')
    require(cfg['test_decoder'] == dict(input_size=512, scales=[512], flip=True, bias=None), 'Decoder changed')
    require(cfg['lora_lr'] == 5e-5 and cfg['head_lr'] == 2.5e-4 and
            cfg['head_weight_decay'] == .01 and cfg['lora_weight_decay'] == 0 and cfg['grad_clip'] == 1 and
            cfg['scheduler'] == 'OneCycleLR_pct0.1_cos', 'Optimizer protocol changed')
    _, plan, context, payload, originals, labels, paths, groups, provenance = inspect_parent(ROOT / cfg['source_probe_config'])
    _, _, targets = verify_source_plan(next(p for p in provenance if Path(p).name == 'plan.json'))
    require(cfg['mixup_alpha'] == plan['inherited_train']['mixup_alpha'] and
            cfg['label_smoothing'] == plan['inherited_train']['label_smoothing'], 'Supervision changed')
    baseline = originals['lr512_swa']
    groups = dict(all=groups['all'], small=groups['originally_small'],
                  other=~groups['originally_small'], tail75=groups['tail75'])
    require(int(((baseline != labels) & groups['small']).sum()) >= 75, 'Insufficient target error budget')
    return cfg, plan, context, payload, targets, baseline, labels, paths, groups, provenance


def schedule(active, cfg):
    count = cfg['steps_per_arm'] * cfg['batch_size']
    require(len(active) >= count, 'Insufficient independent active training rows')
    indices = np.random.default_rng(cfg['seed']).permutation(active)[:count]
    rng = np.random.default_rng(cfg['seed'] + 1)
    flags = np.zeros(count, dtype=bool)
    for start in range(0, count, cfg['batch_size']):
        flags[start + rng.permutation(cfg['batch_size'])[:cfg['batch_size']//2]] = True
    rng = np.random.default_rng(cfg['seed'] + 2)
    lambdas = rng.beta(cfg['mixup_alpha'], cfg['mixup_alpha'], cfg['steps_per_arm'])
    rng = np.random.default_rng(cfg['seed'] + 3)
    permutations = np.stack([rng.permutation(cfg['batch_size']) for _ in lambdas])
    return dict(train_indices=indices, detail_flags=flags, mixup_lambdas=lambdas,
                mixup_permutations=permutations)


def pil_augmentation():
    return T.Compose([T.RandomResizedCrop(512, scale=(.8, 1.), interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(), T.RandAugment(num_ops=2, magnitude=7)])


def crop_tensor(image, reduce):
    if reduce:
        image = T.Resize(224, interpolation=T.InterpolationMode.BICUBIC)(image)
        image = T.Resize(512, interpolation=T.InterpolationMode.BICUBIC)(image)
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])(image)


class TrainingImages(Dataset):
    def __init__(self, root, rows, flags, candidate):
        self.images = Images(root, rows, pil_augmentation())
        self.flags, self.candidate = flags, candidate
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image, _ = self.images[index]
        return crop_tensor(image, self.candidate and bool(self.flags[index])), index


def model_from_parent(plan, context, payload):
    model, check = initial_model(plan, context)
    load_trainable_state(model, payload['selected_state'])
    state = trainable_state(model)
    require(all(torch.equal(value, state[name])
                for name, value in payload['selected_state'].items()), 'Initial weights differ')
    return model, check


def binding(config, frozen, provenance):
    return dict(config_sha256=sha256_file(config), schedule_sha256=sha256_file(frozen),
                implementation_sha256=sha256_file(__file__), parent_sources=provenance,
                stage='repechage', data_version='20260921', partition='train_dev')


def prepare(config, output):
    cfg, plan, context, payload, targets, baseline, labels, paths, groups, provenance = inspect(config)
    seed_training(cfg['seed'])
    model, check = model_from_parent(plan, context, payload)
    frozen = schedule(np.flatnonzero(targets['weights'] > 0), cfg)
    indices = frozen['train_indices']
    original_counts = np.bincount(targets['labels'][indices], minlength=len(context.classes))
    target_counts = np.bincount(targets['targets'][indices], minlength=len(context.classes))
    require(not {context.train[i]['content_group'] for i in indices} &
            {row['content_group'] for row in context.val}, 'Training/validation content overlap')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'schedule.npz', **frozen)
    with (output / 'frozen_val_groups.csv').open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['image_path', 'label'] + list(groups))
        for i, path in enumerate(paths):
            writer.writerow([path, int(labels[i])] + [int(mask[i]) for mask in groups.values()])
    checkpoint_binding = binding(config, output / 'schedule.npz', provenance)
    preflight = dict(status='prepared_verified_cpu', binding=checkpoint_binding,
        zero_update=check, source_plan=next(p for p in provenance if Path(p).name == 'plan.json'),
        training_rows=len(indices), active_source_rows=int((targets['weights'] > 0).sum()),
        train_original_classes_present=int((original_counts > 0).sum()),
        train_target_classes_present=int((target_counts > 0).sum()),
        train_original_classes_absent=np.flatnonzero(original_counts == 0).tolist(),
        train_target_classes_absent=np.flatnonzero(target_counts == 0).tolist(),
        training_targets_unchanged=True, validation_is_independent=True,
        baseline={g: metrics(labels, baseline, len(context.classes), mask) for g, mask in groups.items()},
        val_groups_sha256=sha256_file(output / 'frozen_val_groups.csv'),
        primary_checkpoint=cfg['primary_checkpoint'], wall_clock_time_limit=None)
    atomic_json_dump(preflight, output / 'preflight.json')
    print(json.dumps(preflight), flush=True)


def frozen_context(config, output):
    cfg, plan, ctx, parent, targets, baseline, labels, paths, groups, provenance = inspect(config)
    output = Path(output).resolve()
    preflight = json.loads((output / 'preflight.json').read_text())
    expected = binding(config, output / 'schedule.npz', provenance)
    require(preflight['binding'] == expected and preflight['val_groups_sha256'] ==
            sha256_file(output / 'frozen_val_groups.csv'), 'Frozen inputs changed')
    with np.load(output / 'schedule.npz', allow_pickle=False) as artifact:
        frozen = {key: artifact[key] for key in artifact.files}
    recomputed = schedule(np.flatnonzero(targets['weights'] > 0), cfg)
    require(all(np.array_equal(frozen[k], recomputed[k]) for k in recomputed), 'Frozen schedule differs')
    return cfg, plan, ctx, parent, targets, baseline, labels, paths, groups, frozen, expected


def learning_state(model, cfg, targets, frozen):
    recipe = dict(optimizer='v1_adamw', head_lr=cfg['head_lr'], backbone_lr=cfg['lora_lr'])
    opt = optimizer_for(model, recipe)
    require(sorted({g['weight_decay'] for g in opt.param_groups}) == [0., .01], 'Weight decay differs')
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, [g['lr'] for g in opt.param_groups],
        total_steps=cfg['steps_per_arm'], pct_start=.1, anneal_strategy='cos')
    idx = frozen['train_indices']
    selected = {key: torch.as_tensor(targets[key][idx], device='cuda')
                for key in ('labels', 'targets', 'weights', 'original_alpha')}
    probabilities = target_probabilities(selected['targets'], selected['labels'], selected['original_alpha'],
                                        model.head.weight.shape[0], cfg['label_smoothing'])
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    return opt, scheduler, probabilities, selected['weights'], scaler


def loader(ctx, cfg, frozen, arm, steps):
    count = steps * cfg['batch_size']
    dataset = TrainingImages(ctx.train_root, [ctx.train[i] for i in frozen['train_indices'][:count]],
                             frozen['detail_flags'][:count], arm == 'detail_aug')
    generator = torch.Generator().manual_seed(cfg['seed'] + 4)
    return DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'],
                      shuffle=False, worker_init_fn=seed_worker, generator=generator)


def update(model, opt, scheduler, probabilities, weights, scaler, images, indices, step, frozen, cfg):
    permutation = torch.tensor(frozen['mixup_permutations'][step - 1], device='cuda')
    loss, numeric = logical_update(model, opt, images.to('cuda'), probabilities[indices.to('cuda')],
        weights[indices.to('cuda')], permutation, float(frozen['mixup_lambdas'][step - 1]),
        cfg['micro_batch_size'], scaler, True, cfg['grad_clip'])
    scheduler.step()
    return loss, numeric


def cost(config, output):
    cfg, plan, ctx, parent, targets, baseline, labels, paths, groups, frozen, _ = frozen_context(config, output)
    output = Path(output).resolve()
    require(not (output / 'cost.json').exists(), 'Cost probe must not be repeated')
    require_idle_cuda()
    results = {}
    for arm in cfg['arms']:
        seed_training(cfg['seed']); torch.cuda.reset_peak_memory_stats()
        model, _ = model_from_parent(plan, ctx, parent); model.to('cuda').train()
        opt, scheduler, probabilities, weights, scaler = learning_state(model, cfg, targets, frozen)
        durations, retries = [], 0
        for step, (images, indices) in enumerate(loader(ctx, cfg, frozen, arm, cfg['cost_steps_per_arm']), 1):
            torch.cuda.synchronize(); start = time.monotonic()
            loss, numeric = update(model, opt, scheduler, probabilities, weights, scaler,
                                   images, indices, step, frozen, cfg)
            torch.cuda.synchronize(); durations.append(time.monotonic() - start)
            retries += len(numeric['amp_retries'])
            print(json.dumps(dict(phase='cost', arm=arm, update=step, loss=loss, seconds=durations[-1])), flush=True)
        results[arm] = dict(real_updates=len(durations), seconds=durations, amp_retries=retries,
                           steady_seconds_per_update=float(np.mean(durations[4:])),
                           peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated())
        del model, opt, scheduler, probabilities, weights, scaler
        torch.cuda.empty_cache()
    record = dict(status='passed_discarded_probe_weights', arms=results,
        estimated_training_seconds=sum(r['steady_seconds_per_update'] * cfg['steps_per_arm'] for r in results.values()),
        estimated_eval_and_package_seconds=2 * (3 * 240 + 600),
        eval_estimate_source='prior complete native512 two-view validation/inference throughput',
        formal_initialization='original LR512 EMA2-4, probe states discarded', wall_clock_time_limit=None)
    atomic_json_dump(record, output / 'cost.json'); print(json.dumps(record), flush=True)


def summary(labels, baseline, prediction, groups, classes):
    return dict(metrics={g: metrics(labels, prediction, classes, m) for g, m in groups.items()},
                paired_to_parent={g: paired(labels, baseline, prediction, m) for g, m in groups.items()})


def package(plan, ctx, model, checkpoint, arm_root, expected_val, bind):
    selected = load_artifact(checkpoint, bind)
    require(selected['complete'] is True and selected['optimizer_updates'] == 1024 and
            selected['primary'] == 'last_raw' and selected['classes'] == ctx.classes and
            selected['bias'] is None, 'Only the complete fixed raw checkpoint may be packaged')
    cold, _ = initial_model(plan, ctx); load_trainable_state(cold, selected['selected_state']); cold.to('cuda').eval()
    probe = np.sort(np.random.default_rng(42).choice(len(ctx.val), 64, replace=False))
    parity = evaluate(plan, ctx, cold, [ctx.val[i] for i in probe], 'cuda').argmax(1).numpy()
    require(np.array_equal(parity, expected_val[probe]), 'Cold single-checkpoint validation replay differs')
    del cold; torch.cuda.empty_cache()
    model.eval()
    rows = read_rows(Path(ctx.reference['data']['dataset_manifest']).parent / 'test_manifest.csv')
    names = [Path(r['image_path']).name for r in rows]
    require(len(names) == ctx.manifest['test_samples'] and set(names) ==
            {p.name for p in ctx.test_root.iterdir() if p.is_file()}, 'Wrong official test coverage')
    logits = evaluate(plan, ctx, model, rows, 'cuda', ctx.test_root,
                      progress_path=arm_root / 'inference_progress.json', phase='fixed_native512_flip')
    np.savez_compressed(arm_root / 'test_predictions.npz', names=names, logits=logits.numpy())
    directory = arm_root / 'submission'
    create_submission([(name, ctx.classes[i]) for name, i in zip(names, logits.argmax(1).tolist())],
        names, directory, checkpoint, inference_mode='fixed_detail_aug_pair_native512_flip',
        tta_risk_acknowledged=True, valid_labels=set(ctx.classes), space_after_comma=True,
        extra_manifest=dict(binding=bind, decoder=dict(input_size=512, scales=[512], flip=True, bias=None),
                            training_updates=selected['optimizer_updates'], primary='last_raw'))
    checked = subprocess.run([sys.executable, str(ROOT / 'scripts/check_submission.py'),
        '--test_dir', str(ctx.test_root), '--class-mapping', ctx.reference['data']['class_mapping'],
        '--csv', str(directory / 'pred_results.csv'), '--zip', str(directory / 'submission.zip')],
        text=True, capture_output=True)
    (directory / 'submission_check.log').write_text(checked.stdout + checked.stderr)
    checked.check_returncode()
    csv_bytes = (directory / 'pred_results.csv').read_bytes()
    with zipfile.ZipFile(directory / 'submission.zip') as archive:
        require(archive.namelist() == ['pred_results.csv'] and archive.read('pred_results.csv') == csv_bytes,
                'ZIP and external CSV differ')
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
        csv=str(directory / 'pred_results.csv'), zip=str(directory / 'submission.zip'),
        csv_sha256=sha256_file(directory / 'pred_results.csv'), zip_sha256=sha256_file(directory / 'submission.zip'),
        rows=len(names), checks_passed=9, zip_csv_identical=True, cold_validation_matches=64)


def run(config, output):
    cfg, plan, ctx, parent, targets, baseline, labels, paths, groups, frozen, bind = frozen_context(config, output)
    output = Path(output).resolve()
    require(json.loads((output / 'cost.json').read_text())['status'] == 'passed_discarded_probe_weights', 'Cost probe missing')
    require(not (output / 'status.json').exists(), 'No implicit formal restart')
    require_idle_cuda(); started = time.monotonic()
    atomic_json_dump(dict(status='running', steps_per_arm=cfg['steps_per_arm'], current_arm='control'), output / 'status.json')
    reports, final_predictions = {}, {}
    try:
        for arm in cfg['arms']:
            arm_root = output / arm; arm_root.mkdir(exist_ok=False)
            seed_training(cfg['seed'])
            model, _ = model_from_parent(plan, ctx, parent); model.to('cuda').train()
            opt, scheduler, probabilities, weights, scaler = learning_state(model, cfg, targets, frozen)
            atomic_json_dump(dict(arm=arm, initial_checkpoint_sha256='c103dc3e6039c9fe156a56dfcd196500975b5bd21354be6506447bb1ee636e2e',
                schedule_sha256=bind['schedule_sha256'], logical_updates=cfg['steps_per_arm'],
                sample_order='shared frozen schedule', mixup='shared frozen lambdas and permutations'), arm_root / 'trajectory.json')
            history, retry_events, last_report, loss_values = [], [], 0., []
            for step, (images, indices) in enumerate(loader(ctx, cfg, frozen, arm, cfg['steps_per_arm']), 1):
                model.train()
                loss, numeric = update(model, opt, scheduler, probabilities, weights, scaler,
                                       images, indices, step, frozen, cfg)
                loss_values.append(loss)
                if numeric['amp_retries']: retry_events.append(dict(step=step, **numeric))
                now = time.monotonic()
                if step == 1 or now - last_report >= 30 or step in cfg['evaluation_steps']:
                    progress = dict(status='training', arm=arm, update=step, updates=cfg['steps_per_arm'],
                                    loss=float(np.mean(loss_values[-32:])), elapsed_seconds=now - started)
                    atomic_json_dump(progress, output / 'progress.json'); print(json.dumps(progress), flush=True)
                    last_report = now
                if step in cfg['evaluation_steps']:
                    checkpoint = arm_root / ('selected.pt' if step == cfg['steps_per_arm'] else f'step_{step:04d}.pt')
                    save_artifact(checkpoint, dict(selected_state=trainable_state(model), classes=ctx.classes,
                        primary='last_raw', complete=step == cfg['steps_per_arm'], optimizer_updates=step,
                        parent_checkpoint_sha256='c103dc3e6039c9fe156a56dfcd196500975b5bd21354be6506447bb1ee636e2e',
                        image_size=512, bias=None, arm=arm), bind)
                    logits = evaluate(plan, ctx, model, ctx.val, 'cuda', progress_path=output / 'evaluation_progress.json',
                                      phase=f'{arm}_step{step}_raw')
                    prediction = logits.argmax(1).numpy()
                    np.savez_compressed(arm_root / f'val_step{step:04d}.npz', labels=labels, image_paths=paths,
                                        predictions=prediction, logits=logits.numpy())
                    item = dict(step=step, **summary(labels, baseline, prediction, groups, len(ctx.classes)))
                    history.append(item); atomic_json_dump(history, arm_root / 'history.json')
                    require(100 * (item['metrics']['all']['micro'] - np.mean(baseline == labels)) >=
                            cfg['stop_micro_pp_vs_parent_below'], 'Frozen full-validation stop-loss reached')
                    model.train()
            require(step == cfg['steps_per_arm'], 'Incomplete fixed update schedule')
            atomic_json_dump(retry_events, arm_root / 'amp_retries.json')
            del opt, scheduler, probabilities, weights, scaler
            final_predictions[arm] = prediction
            delivered = package(plan, ctx, model, checkpoint, arm_root, prediction, bind)
            reports[arm] = dict(status='completed_packaged', optimizer_updates=step, history=history, package=delivered,
                               amp_retry_batches=len(retry_events))
            atomic_json_dump(reports[arm], arm_root / 'report.json')
            del model; torch.cuda.empty_cache()
        comparison = {g: paired(labels, final_predictions['control'], final_predictions['detail_aug'], m)
                      for g, m in groups.items()}
        overall, small = comparison['all'], comparison['small']
        parent_net = reports['detail_aug']['history'][-1]['paired_to_parent']['all']['net']
        gate = cfg['review_gate']
        supported = (overall['net'] >= gate['net_vs_control_min'] and parent_net >= gate['net_vs_parent_min'] and
                     small['net'] >= gate['small_group_net_vs_control_min'] and
                     overall['corrections'] >= gate['corrections_to_regressions_min'] * overall['regressions'])
        report = dict(status='completed_packaged_pair', experiment_id=cfg['experiment_id'], binding=bind,
            arms=reports, candidate_vs_control=comparison, supports_review=supported,
            decision='supports_review' if supported else 'close_fixed_detail_augmentation_recipe',
            training_population='32768 frozen active train_dev rows, shared without replacement',
            primary_checkpoint=cfg['primary_checkpoint'], validation_rows=len(labels),
            original_labels_noisy=True, platform_gain_known=False, automatic_full=False,
            elapsed_seconds=time.monotonic() - started)
        atomic_json_dump(report, output / 'report.json')
        atomic_json_dump(dict(status='completed_packaged_pair', report=str(output / 'report.json')), output / 'status.json')
        print(json.dumps(dict(status=report['status'], decision=report['decision'], comparison=comparison)), flush=True)
    except Exception as error:
        atomic_json_dump(dict(status='failed', error=str(error), completed_arms=list(reports),
                              automatically_restarted=False), output / 'status.json')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'cost', 'run'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.action != 'prepare' and not args.execute: parser.error('CUDA requires --execute')
    dict(prepare=prepare, cost=cost, run=run)[args.action](args.config, args.output)
