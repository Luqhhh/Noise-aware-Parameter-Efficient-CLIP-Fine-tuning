"""Bounded current-stage frozen-feature and conditional calibration diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_pipeline import (
    Images, StageContext, image_transform, load_artifact, load_recipe, seed_training,
)
from aegis_clip.v1_strategy import CosineHead, fit_training_bias, knn_signals

SOURCE = Path('/home/lux1/noise')
SWA = SOURCE / 'worktrees/v1_swa_trial_20260930/outputs/codex/v1_swa_trial_20260930/candidate'
TARGET = SOURCE / 'worktrees/v1_train_20260929/outputs/codex/v1_train_20260929/targets.pt'
BASELINE = SOURCE / 'worktrees/wft448_dev_20261001/outputs/codex/wft448_dev_20261001_r2/run/baseline.npz'


def context():
    return StageContext(load_recipe(SWA / 'config.yaml'))


def historical_artifact(path, ctx):
    """Check historical code binding plus unchanged current data/recipe identity."""
    metadata = json.loads(path.with_suffix('.sha256.json').read_text())
    binding = metadata['binding']
    if set(binding) != set(ctx.binding) or any(
        v != ctx.binding[k] for k,v in binding.items() if k != 'implementation_sha256'
    ):
        raise ValueError('Historical artifact data, recipe, or official weight identity changed')
    return load_artifact(path, binding)


def metric(prediction, labels, classes, tail):
    prediction, labels = np.asarray(prediction), np.asarray(labels)
    counts = np.bincount(labels, minlength=classes)
    correct = np.bincount(labels[prediction == labels], minlength=classes)
    supported = counts > 0
    per_class = correct / np.maximum(counts, 1)
    selected = np.isin(labels, tail)
    return dict(samples=len(labels), correct=int(correct.sum()),
        micro=float(np.mean(prediction == labels)), macro=float(per_class[supported].mean()),
        covered_classes=int(supported.sum()), tail75_samples=int(selected.sum()),
        tail75_macro=float(per_class[np.intersect1d(np.where(supported)[0], tail)].mean()),
        per_class_correct=correct.tolist(), per_class_samples=counts.tolist())


def paired(before, after, labels):
    before, after, labels = np.asarray(before), np.asarray(after), np.asarray(labels)
    return dict(changed=int(np.sum(before != after)),
        corrections=int(np.sum((before != labels) & (after == labels))),
        regressions=int(np.sum((before == labels) & (after != labels))),
        net_correct=int(np.sum(after == labels) - np.sum(before == labels)))


def grouped_two_folds(rows):
    """Balance groups within classes; unsplittable classes are fit-only anchors."""
    by_class = defaultdict(set)
    for row in rows:
        by_class[int(row['label'])].add(row['content_group'])
    assignment = {}
    key = lambda g: hashlib.sha256(('42:' + g).encode()).hexdigest()
    for label in sorted(by_class, key=lambda c: (len(by_class[c]), c)):
        groups = sorted(by_class[label], key=key)
        counts = [sum(assignment.get(g) == f for g in groups) for f in (0, 1)]
        for g in groups:
            if g not in assignment:
                f = 0 if counts[0] <= counts[1] else 1
                assignment[g] = f
                counts[f] += 1
    anchors = {c for c, groups in by_class.items() if len({assignment[g] for g in groups}) < 2}
    # Whole mixed-label content groups containing an anchor are fit-only as well.
    anchor_groups = {g for c in anchors for g in by_class[c]}
    for g in anchor_groups:
        assignment[g] = -1
    # Removing a mixed-label anchor group can render another class unsplittable.
    while True:
        extra = {c for c, groups in by_class.items()
                 if c not in anchors and len({assignment[g] for g in groups if assignment[g] >= 0}) < 2}
        if not extra:
            break
        anchors |= extra
        for c in extra:
            for g in by_class[c]:
                assignment[g] = -1
    folds = np.array([assignment[r['content_group']] for r in rows], dtype=np.int64)
    for f in (0, 1):
        fit_groups = {r['content_group'] for r, flag in zip(rows, folds != f) if flag}
        hold_groups = {r['content_group'] for r, flag in zip(rows, folds == f) if flag}
        if fit_groups & hold_groups:
            raise ValueError('Calibration content-group leakage')
        if len({int(r['label']) for r, flag in zip(rows, folds != f) if flag}) != len(by_class):
            raise ValueError('Calibration fit pool lost class coverage')
    return folds, sorted(anchors)


def begin(out, mode):
    if out.exists():
        raise FileExistsError(f'Preserve existing diagnostic: {out}')
    out.mkdir(parents=True)
    ctx = context()
    counts = np.bincount([int(r['label']) for r in ctx.train], minlength=len(ctx.classes))
    tail = np.lexsort((np.arange(len(counts)), counts))[:75]
    atomic_json_dump(dict(experiment_id='V1_FEATURE_CALIBRATION_20261001', mode=mode,
        binding=ctx.binding, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=sha256_file(__file__), tail75_classes=tail.tolist(),
        diagnostic_only=True, test_data_used=False, candidate_training=False), out/'protocol.json')
    return ctx, tail


def calibration(out):
    ctx, tail = begin(out, 'calibration')
    started = time.monotonic()
    # Validate existing artifacts against the effective original v1 recipe.
    parent = historical_artifact(SWA/'selected.pt', ctx)
    calibration_artifact = historical_artifact(SWA/'calibration.pt', ctx)
    if parent['classes'] != ctx.classes or calibration_artifact['checkpoint_sha256'] != sha256_file(SWA/'selected.pt'):
        raise ValueError('Parent or calibration binding mismatch')
    delivered = json.loads((SOURCE/'results/wft448_dev_20261001/report.json').read_text())
    if delivered['artifacts'][str(BASELINE)] != sha256_file(BASELINE):
        raise ValueError('Baseline differs from completed WFT delivery audit')
    cache = np.load(BASELINE)
    by_path = {p: i for i, p in enumerate(cache['image_paths'])}
    if set(by_path) != {r['image_path'] for r in ctx.val}:
        raise ValueError('Baseline paths mismatch')
    ix = [by_path[r['image_path']] for r in ctx.val]
    logits = torch.from_numpy(cache['logits'][ix].copy())
    labels = np.array([int(r['label']) for r in ctx.val])
    if not np.array_equal(cache['labels'][ix], labels):
        raise ValueError('Baseline labels mismatch')
    frozen = (logits + calibration_artifact['bias']).argmax(1).numpy()
    raw = logits.argmax(1).numpy()
    with (SWA/'validation_predictions.csv').open() as handle:
        recorded = {r['image_path']: r for r in csv.DictReader(handle)}
    if any(int(recorded[r['image_path']]['raw_prediction']) != raw[i]
           or int(recorded[r['image_path']]['calibrated_prediction']) != frozen[i]
           for i, r in enumerate(ctx.val)):
        raise ValueError('Baseline does not replay all v1 SWA predictions')
    folds, anchors = grouped_two_folds(ctx.val)
    prediction = np.full(len(labels), -1, dtype=np.int64)
    biases, reports = [], []
    for f in (0, 1):
        fit, hold = folds != f, folds == f
        bias = fit_training_bias(logits[fit], torch.from_numpy(labels[fit]), iterations=200)
        fit_prediction = (logits[fit] + bias).argmax(1).numpy()
        prediction[hold] = (logits[hold] + bias).argmax(1).numpy()
        biases.append(bias.numpy())
        reports.append(dict(fold=f, fit_samples=int(fit.sum()), heldout_samples=int(hold.sum()),
            fit_raw=metric(raw[fit], labels[fit], len(ctx.classes), tail),
            fit_calibrated=metric(fit_prediction, labels[fit], len(ctx.classes), tail),
            heldout_raw=metric(raw[hold], labels[hold], len(ctx.classes), tail),
            heldout_calibrated=metric(prediction[hold], labels[hold], len(ctx.classes), tail)))
    hold = folds >= 0
    np.savez_compressed(out/'predictions.npz', logits=logits.numpy(), labels=labels, folds=folds,
        raw=raw, frozen=frozen, crossfit=prediction, biases=np.stack(biases), tail=tail)
    report = dict(mode='calibration', samples=len(labels), scored_samples=int(hold.sum()),
        fit_only_anchor_classes=anchors, fit_only_anchor_samples=int((~hold).sum()),
        baseline_sha256=sha256_file(BASELINE), parent_checkpoint_sha256=sha256_file(SWA/'selected.pt'),
        historical_parent_binding=parent['binding'],
        original_predictions_replayed=len(labels), raw_all=metric(raw, labels, len(ctx.classes), tail),
        frozen_all=metric(frozen, labels, len(ctx.classes), tail),
        raw_heldout_population=metric(raw[hold], labels[hold], len(ctx.classes), tail),
        frozen_reference_heldout_population=metric(frozen[hold], labels[hold], len(ctx.classes), tail),
        crossfit=metric(prediction[hold], labels[hold], len(ctx.classes), tail),
        paired_vs_raw=paired(raw[hold], prediction[hold], labels[hold]),
        paired_vs_full_fit=paired(frozen[hold], prediction[hold], labels[hold]), folds=reports,
        seconds=time.monotonic()-started, test_data_used=False, platform_calibration_gain=None,
        limitation='Conditional bias cross-fit on noisy validation labels; not independent-source or platform evaluation')
    atomic_json_dump(report, out/'report.json')
    print(json.dumps({k: v for k, v in report.items() if k in ('scored_samples', 'fit_only_anchor_classes', 'paired_vs_raw', 'paired_vs_full_fit', 'seconds')}), flush=True)


def fit_head(features, labels, initial, out, dimension, device):
    seed_training(42)
    model = CosineHead(dimension, int(labels.max())+1, dropout=.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20, eta_min=.0005)
    selected = torch.where(initial)[0]
    order_rng = torch.Generator(device=device).manual_seed(42)
    for epoch in range(20):
        model.train()
        order = selected[torch.randperm(len(selected), device=device, generator=order_rng)]
        for indices in order.split(8192):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features[indices]), labels[indices], label_smoothing=.1)
            loss.backward()
            opt.step()
        scheduler.step()
    model.eval()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out/f'head{dimension}.pt')
    return model


def features(out, device):
    ctx, tail = begin(out, 'features')
    started = time.monotonic()
    if device == 'cuda':
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if busy:
            raise RuntimeError(f'Existing CUDA task; do not compete: {busy}')
    import clip
    model, _ = clip.load(str(ctx.official), device='cpu', jit=False)
    visual = model.visual.float().to(device).eval()
    del model
    # This hook records the same native forward, immediately before visual.proj.
    captured = []
    hook = visual.ln_post.register_forward_hook(lambda m, a, z: captured.append(z.detach()))
    loader = DataLoader(Images(ctx.train_root, ctx.full, image_transform(224)), batch_size=128,
        shuffle=False, num_workers=2, pin_memory=device=='cuda')
    pre = torch.empty((len(ctx.full), 768))
    post = torch.empty((len(ctx.full), 512))
    algebra_error, maximum_batch_seconds = 0., 0.
    projection = visual.proj.detach().cpu().float()
    with torch.inference_mode():
        for batch, (images, indices) in enumerate(loader):
            if batch == 0:
                # Warm the kernels before the fixed first-batch cost estimate.
                for _ in range(2):
                    visual(images.to(device))
                if device == 'cuda':
                    torch.cuda.synchronize()
            batch_started = time.monotonic()
            captured.clear()
            z512 = visual(images.to(device, non_blocking=True))
            z768 = captured[0]
            algebra_error = max(algebra_error, float((z768@visual.proj-z512).abs().max()))
            pre[indices], post[indices] = z768.cpu(), F.normalize(z512.float(), dim=1).cpu()
            elapsed = time.monotonic()-started
            maximum_batch_seconds = max(maximum_batch_seconds, time.monotonic()-batch_started)
            if batch == 0 and maximum_batch_seconds*len(loader) > 900:
                atomic_json_dump(dict(status='cost_gate_closed', estimated_encoding_seconds=maximum_batch_seconds*len(loader)), out/'report.json')
                return
            if elapsed > 1800:
                raise TimeoutError('Fixed 30-minute diagnostic cap')
            if batch % 100 == 0:
                print(json.dumps(dict(stage='encoding', batch=batch, batches=len(loader), seconds=elapsed)), flush=True)
                atomic_json_dump(dict(batch=batch, batches=len(loader), seconds=elapsed), out/'progress.json')
    hook.remove()
    del visual
    cached = torch.load(ctx.reference['features']['tensor_path'], map_location='cpu', weights_only=True).float()
    cache_max_error = float((cached-post).abs().max())
    cache_mean_cosine = float((F.normalize(cached,dim=1)*post).sum(1).mean())
    if algebra_error > 1e-5 or cache_max_error > 2e-4 or cache_mean_cosine < .99999:
        raise ValueError('Extracted projection does not match native forward/current-stage cache')
    full_index = {r['image_path']: i for i,r in enumerate(ctx.full)}
    tr = [full_index[r['image_path']] for r in ctx.train]
    va = [full_index[r['image_path']] for r in ctx.val]
    labels = torch.tensor([int(r['label']) for r in ctx.train], device=device)
    val_labels = np.array([int(r['label']) for r in ctx.val])
    targets = historical_artifact(TARGET, ctx)
    from aegis_clip.v1_strategy import class_top_keep
    initial = class_top_keep(targets['agreement'], labels.cpu(), len(ctx.classes), .9) | (targets['agreement']>=.7)
    initial = initial.to(device)
    outputs, knn_outputs, metrics, signals = {}, {}, {}, {}
    group_ids = {g:i for i,g in enumerate(sorted({r['content_group'] for r in ctx.full}))}
    tg = torch.tensor([group_ids[r['content_group']] for r in ctx.train], device=device)
    vg = torch.tensor([group_ids[r['content_group']] for r in ctx.val], device=device)
    for dimension, tensor in [(512, cached), (768, pre)]:
        train_x = F.normalize(tensor[tr], dim=1).to(device)
        val_x = F.normalize(tensor[va], dim=1).to(device)
        head = fit_head(train_x, labels, initial, out, dimension, device)
        with torch.inference_mode():
            outputs[dimension] = torch.cat([head(x).argmax(1).cpu() for x in val_x.split(8192)]).numpy()
        agreement, votes = knn_signals(val_x, torch.from_numpy(val_labels).to(device), train_x, labels,
            len(ctx.classes), k=16, query_groups=vg, gallery_groups=tg)
        knn_outputs[dimension] = votes.cpu().numpy()
        metrics[str(dimension)] = dict(head=metric(outputs[dimension],val_labels,len(ctx.classes),tail),
            knn=metric(knn_outputs[dimension],val_labels,len(ctx.classes),tail),
            mean_label_agreement=float(agreement.mean()))
        signals[dimension] = agreement.cpu().numpy()
        del head, train_x, val_x
    np.savez_compressed(out/'predictions.npz', labels=val_labels, head512=outputs[512], head768=outputs[768],
        knn512=knn_outputs[512],knn768=knn_outputs[768],agreement512=signals[512],agreement768=signals[768],tail=tail)
    torch.save(dict(preprojection=pre, projection=projection, image_paths=[r['image_path'] for r in ctx.full]),out/'features768.pt')
    report = dict(mode='features', source='official frozen CLIP at 224 center; original v1 fixed kNN teacher population',
        train_samples=len(tr), fit_samples=int(initial.sum()), val_samples=len(va), seed=42, epochs=20,
        native_projection_max_error=algebra_error, cached512_max_error=cache_max_error,
        cached512_mean_cosine=cache_mean_cosine, metrics=metrics,
        head_paired=paired(outputs[512],outputs[768],val_labels),
        knn_paired=paired(knn_outputs[512],knn_outputs[768],val_labels),seconds=time.monotonic()-started,
        official_weight_sha256=sha256_file(ctx.official), targets_sha256=sha256_file(TARGET),
        historical_target_binding=targets['binding'],
        test_data_used=False, candidate_training=False, platform_score=None,
        limitation='Single fixed frozen-feature diagnostic; does not establish 768 LoRA/student or platform gain')
    atomic_json_dump(report,out/'report.json')
    print(json.dumps({k:v for k,v in report.items() if k in ('fit_samples','head_paired','knn_paired','seconds')}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['calibration','features'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.mode=='calibration': calibration(args.output)
    else: features(args.output,args.device)


if __name__=='__main__':main()
