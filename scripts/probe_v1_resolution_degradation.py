"""One fixed synthetic detail-loss intervention on independent DEV images.

Use the same native512+flip decoder for the parent and existing LR512 average.
Only images larger than short-edge224 are reduced; smaller images stay intact.
No parameters, biases, supervision, or test predictions are fitted.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_pipeline import Images, image_transform, load_artifact, seed_training
from aegis_clip.v1_strategy import load_trainable_state
from v1_continuation.plan import verify
from v1_continuation.runtime import initial_model, require_idle_cuda
from diagnose_v1_candidate_errors import metrics, paired, require, load_aligned_npz

ROOT = Path(__file__).resolve().parents[1]


class DetailLossTransform:
    def __init__(self, short_edge=224, native_size=512):
        self.short_edge = short_edge
        self.reduce = T.Resize(short_edge, interpolation=T.InterpolationMode.BICUBIC)
        self.native = image_transform(native_size)

    def __call__(self, image):
        if min(image.size) > self.short_edge:
            image = self.reduce(image)
        return self.native(image)


def inspect(config_path):
    cfg = json.loads(Path(config_path).read_text())
    require(cfg['training_updates'] == 0 and cfg['test_data_used'] is False and
            cfg['automatic_training'] is False and cfg['parameter_search'] is False,
            'Only a nontraining DEV diagnostic is supported')
    require(cfg['short_edge'] == 224 and cfg['decoder'] ==
            dict(input_size=512, scales=[512], flip=True, bias=None), 'Protocol changed')
    require(cfg['models'] == ['native_parent', 'lr512_swa'] and cfg['batch_size'] == 32 and
            cfg['workers'] == 2 and cfg['parity_rows'] == 64 and cfg['seed'] == 42 and
            cfg['interpolation'] == 'PIL_bicubic_torchvision_Resize', 'Fixed probe changed')
    report_path = ROOT / cfg['source_report']
    source = json.loads(report_path.read_text())
    require(sha256_file(source['plan']) == cfg['plan_sha256'], 'Plan checksum changed')
    plan, context, _ = verify(source['plan'])
    require(plan['config']['partition'] == 'train_dev' and plan['config']['route'] == 'LR512' and
            plan['source_binding']['data_version'] == cfg['data_version'] and len(context.val),
            'A current-stage independent DEV source is required')
    require(source['validation_is_independent'] is True, 'Overlapping validation')
    labels = np.array([int(r['label']) for r in context.val])
    paths = np.array([r['image_path'] for r in context.val])
    sources = {}
    for name, filename in [('native_parent', 'baseline.npz'), ('lr512_swa', 'val_epoch04_ema_swa_2_4.npz')]:
        matched = [path for path in source['artifacts'] if Path(path).name == filename]
        require(len(matched) == 1, 'Missing native predictions')
        path = matched[0]
        require(sha256_file(path) == source['artifacts'][path], 'Native prediction checksum changed')
        sources[name] = load_aligned_npz(path, paths, labels)['predictions']
    checkpoints = [path for path in source['artifacts'] if Path(path).name == 'ema_swa.pt']
    require(len(checkpoints) == 1, 'Missing fixed average checkpoint')
    checkpoint = checkpoints[0]
    require(sha256_file(checkpoint) == cfg['checkpoint_sha256'] == source['artifacts'][checkpoint],
            'Checkpoint checksum changed')
    binding = dict(plan_sha256=sha256_file(source['plan']), source_binding=context.binding,
                   parent_sha256=sha256_file(plan['config']['parent_checkpoint']),
                   targets_sha256=sha256_file(plan['config']['targets']))
    payload = load_artifact(checkpoint, binding)
    require(payload['complete'] is True and payload['selected_policy'] == 'ema_swa_2_4' and
            payload['average_epochs'] == [2, 3, 4] and payload['classes'] == context.classes,
            'Incorrect fixed export')
    eligible = np.array([min(int(row['width']), int(row['height'])) > cfg['short_edge']
                         for row in context.val])
    groups = dict(all=np.ones(len(labels), dtype=bool), reduced=eligible, unchanged=~eligible,
                  originally_small=np.array([min(int(row['width']), int(row['height'])) < 224
                                             for row in context.val]),
                  tail75=np.isin(labels, plan['groups']['tail_classes']))
    provenance = {str(report_path): sha256_file(report_path), source['plan']: cfg['plan_sha256'],
                  checkpoint: cfg['checkpoint_sha256'], str(Path(checkpoint).with_suffix('.sha256.json')):
                  sha256_file(Path(checkpoint).with_suffix('.sha256.json'))}
    for path in source['artifacts']:
        if Path(path).name in ('baseline.npz', 'val_epoch04_ema_swa_2_4.npz'):
            provenance[path] = source['artifacts'][path]
    return cfg, plan, context, payload, sources, labels, paths, groups, provenance


def prepare(config, output):
    cfg, plan, context, payload, sources, labels, paths, groups, provenance = inspect(config)
    seed_training(cfg['seed'])
    model, zero = initial_model(plan, context)
    load_trainable_state(model, payload['selected_state'])
    # Check the real official model/checkpoint and two real held-out images on CPU.
    model.eval()
    indices = [int(np.flatnonzero(groups['reduced'])[0]), int(np.flatnonzero(groups['originally_small'])[0])]
    native = Images(context.train_root, [context.val[i] for i in indices], image_transform(512))
    degraded = Images(context.train_root, [context.val[i] for i in indices], DetailLossTransform())
    with torch.no_grad():
        logits = model(torch.stack([native[i][0] for i in range(2)]))
        reduced_logits = model(torch.stack([degraded[i][0] for i in range(2)]))
    require(torch.isfinite(logits).all() and torch.isfinite(reduced_logits).all(), 'Nonfinite CPU forward')
    require(torch.equal(native[1][0], degraded[1][0]), 'Small images must remain byte-identical tensors')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'frozen_groups.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image_path', 'label'] + list(groups))
        for i, path in enumerate(paths):
            writer.writerow([path, int(labels[i])] + [int(mask[i]) for mask in groups.values()])
    preflight = dict(status='prepared_verified_cpu', config_sha256=sha256_file(config),
        implementation_sha256=sha256_file(__file__), source_files=provenance,
        groups_sha256=sha256_file(output / 'frozen_groups.csv'),
        validation_rows=len(labels), groups={name: int(mask.sum()) for name, mask in groups.items()},
        original_metrics={name: metrics(labels, prediction, len(context.classes), groups['all'])
                          for name, prediction in sources.items()},
        zero_update=zero, cpu_forward_finite=True, cpu_images=2,
        small_image_transform_identity=True, torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        frozen_before_gpu_output=True, training_started=False, test_data_used=False)
    atomic_json_dump(preflight, output / 'preflight.json')
    print(json.dumps(preflight), flush=True)


@torch.no_grad()
def predict(model, context, rows, transform, cfg, output, phase):
    dataset = Images(context.train_root, rows, transform)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'], shuffle=False)
    logits = torch.empty(len(rows), len(context.classes))
    started, last = time.monotonic(), 0.
    model.eval()
    for batch, (images, indices) in enumerate(loader, 1):
        images = images.to('cuda')
        result = (model(images) + model(images.flip(3))) / 2
        require(torch.isfinite(result).all(), 'Nonfinite CUDA logits')
        logits[indices] = result.cpu()
        now = time.monotonic()
        if batch == 1 or batch == len(loader) or now - last >= 30:
            progress = dict(status='evaluating', phase=phase, batch=batch, batches=len(loader),
                            rows=len(rows), elapsed_seconds=now - started)
            atomic_json_dump(progress, output / 'progress.json')
            print(json.dumps(progress), flush=True)
            last = now
    return logits.numpy()


def run(config, output):
    cfg, plan, context, payload, originals, labels, paths, groups, provenance = inspect(config)
    output = Path(output).resolve()
    preflight = json.loads((output / 'preflight.json').read_text())
    require(preflight['config_sha256'] == sha256_file(config) and
            preflight['implementation_sha256'] == sha256_file(__file__) and
            preflight['groups_sha256'] == sha256_file(output / 'frozen_groups.csv') and
            preflight['source_files'] == provenance, 'Frozen preflight changed')
    require(not (output / 'status.json').exists(), 'No implicit restart or output overwrite')
    require_idle_cuda()
    seed_training(cfg['seed'])
    torch.set_num_threads(2)
    started = time.monotonic()
    results, parity = {}, {}
    selected = np.flatnonzero(groups['reduced'])
    probe = np.sort(np.random.default_rng(cfg['seed']).choice(len(labels), cfg['parity_rows'], replace=False))
    atomic_json_dump(dict(status='running', models=cfg['models'], training_updates=0), output / 'status.json')
    try:
        for name in cfg['models']:
            model, _ = initial_model(plan, context)
            if name == 'lr512_swa':
                load_trainable_state(model, payload['selected_state'])
            model.to('cuda').eval()
            native = predict(model, context, [context.val[i] for i in probe], image_transform(512),
                             cfg, output, name + '_native_parity')
            matches = int((native.argmax(1) == originals[name][probe]).sum())
            require(matches == len(probe), 'Native cold checkpoint replay differs')
            parity[name] = dict(rows=len(probe), matching_predictions=matches)
            logits = predict(model, context, [context.val[i] for i in selected], DetailLossTransform(),
                             cfg, output, name + '_detail_loss224')
            prediction = originals[name].copy()
            prediction[selected] = logits.argmax(1)
            np.savez_compressed(output / (name + '.npz'), image_paths=paths, labels=labels,
                                original=originals[name], perturbed=prediction,
                                reduced_indices=selected, perturbed_logits=logits)
            results[name] = dict(original={g: metrics(labels, originals[name], len(context.classes), m)
                                          for g, m in groups.items()},
                perturbed={g: metrics(labels, prediction, len(context.classes), m) for g, m in groups.items()},
                paired={g: paired(labels, originals[name], prediction, m) for g, m in groups.items()})
            require(all(p.grad is None for p in model.parameters()), 'Inference unexpectedly computed gradients')
            del model
            torch.cuda.empty_cache()
        parent = results['native_parent']
        delta = 100 * (parent['perturbed']['reduced']['micro'] - parent['original']['reduced']['micro'])
        gate = cfg['mechanism_gate']
        supported = parent['paired']['reduced']['regressions'] >= gate['minimum_parent_regressions'] and delta <= gate['parent_delta_micro_pp_at_most']
        report = dict(experiment_id=cfg['experiment_id'], status='completed_diagnostic',
            validation_rows=len(labels), classes=len(context.classes), results=results,
            native_parity=parity, config_sha256=sha256_file(config), preflight_sha256=sha256_file(output / 'preflight.json'),
            implementation_sha256=sha256_file(__file__), source_files=provenance,
            artifacts={str(output / (name + '.npz')): sha256_file(output / (name + '.npz')) for name in cfg['models']},
            detail_loss_mechanism_gate=supported,
            decision='supports_resolution_robustness_review' if supported else 'close_fixed_detail_loss_hypothesis',
            automatic_training=False, training_updates=0, new_candidate=False, test_data_used=False,
            original_labels_noisy=True, actual_small_image_causal_share_known=False,
            platform_gain_known=False, elapsed_seconds=time.monotonic() - started,
            peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated())
        atomic_json_dump(report, output / 'report.json')
        atomic_json_dump(dict(status='completed_diagnostic', report=str(output / 'report.json')), output / 'status.json')
        print(json.dumps(dict(status=report['status'], elapsed_seconds=report['elapsed_seconds'],
                              decision=report['decision'], results=results)), flush=True)
    except Exception as error:
        atomic_json_dump(dict(status='failed', error=str(error), automatically_restarted=False), output / 'status.json')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.action == 'run' and not args.execute:
        parser.error('CUDA diagnostic requires --execute')
    (prepare if args.action == 'prepare' else run)(args.config, args.output)
