"""Fixed local V2 512/768 paired continuation, including checked submissions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from aegis_clip.runtime import atomic_json_dump
from aegis_clip.v1_test_bias import fit_test_uniform_bias
from diagnose_v1_candidate_errors import metrics, paired, read_json, read_rows, require, sha
from run_v2_fixed_prior import ensure_idle_cuda, inference_view, probability_scores
from v2 import training_utils as tu
from v2.core import logical_backward, sample_weights
from v2.preprojection import V2FeatureClassifier, fold_projection
from v2.runtime import Budget, InferenceDataset, amp, write_submission

ROOT = Path(__file__).resolve().parents[1]


def tensor_sha(value):
    return hashlib.sha256(memoryview(value.detach().cpu().contiguous().numpy())).hexdigest()


def progress(out, status, **extra):
    value = dict(status=status, **extra)
    atomic_json_dump(value, out / 'progress.json')
    print(json.dumps(value), flush=True)


def schedule(labels, classes, cfg):
    count = cfg['steps'] * cfg['logical_batch_size']
    generator = torch.Generator().manual_seed(cfg['seed'])
    indices = torch.multinomial(sample_weights(labels, classes), count, replacement=True,
                                generator=generator).numpy()
    rng = np.random.default_rng(cfg['seed'])
    return dict(indices=indices, image_seeds=rng.integers(0, 2**31, count, dtype=np.int64),
                mix_seeds=rng.integers(0, 2**31, cfg['steps'], dtype=np.int64))


class ScheduledImages(Dataset):
    def __init__(self, root, rows, schedule, transform, limit):
        self.root, self.rows, self.schedule = Path(root), rows, schedule
        self.transform, self.limit = transform, limit

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        seed = int(self.schedule['image_seeds'][index])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        row = self.rows[int(self.schedule['indices'][index])]
        image = tu.load_image(self.root / Path(row['image_path']).relative_to('train'))
        return self.transform(image), int(row['label']), index


def loader(dataset, batch, cfg, shuffle=False):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=cfg['num_workers'],
                      pin_memory=True, prefetch_factor=2, persistent_workers=False,
                      generator=torch.Generator().manual_seed(cfg['seed']))


def model_from_state(cfg, state, dimension, classes):
    model = V2FeatureClassifier(dict(official_checkpoint=cfg['official_checkpoint'],
        official_sha256=cfg['official_sha256'], num_classes=classes, gradient_checkpointing=True), dimension)
    model.load_state_dict(state if dimension == 512 else fold_projection(state), strict=True)
    return model.cuda()


def run_updates(model, root, rows, frozen, cfg, parent_cfg, steps, out, budget, phase):
    tu.set_seed(cfg['seed'])
    settings = dict(parent_cfg['train'], lr_backbone=cfg['lr_backbone'], lr_head=cfg['lr_head'],
                    weight_decay=cfg['weight_decay'], llrd_gamma=1.0)
    optimizer = tu.build_optimizer(model, settings)
    scheduler = tu.build_scheduler(optimizer, 1, cfg['warmup_fraction'], cfg['steps'], cfg['min_lr_ratio'])
    dataset = ScheduledImages(root, rows, frozen, tu.build_train_transform(cfg['image_size'], parent_cfg['augment']),
                              steps * cfg['logical_batch_size'])
    batches = loader(dataset, cfg['logical_batch_size'], cfg)
    trace, seconds, losses = [], [], []
    model.train()
    aug = parent_cfg['augment']
    started = time.monotonic()
    last = started
    for step, (images, labels, positions) in enumerate(batches):
        budget.check()
        tu.set_seed(int(frozen['mix_seeds'][step]))
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        targets = tu.one_hot(labels, model.head.out_features, aug['label_smoothing'])
        images, targets, _ = tu.apply_mixup_cutmix(images, targets, aug['mixup'], aug['cutmix'],
                                                 aug['mix_prob'], model.head.out_features)
        entry = dict(step=step + 1, positions_sha256=tensor_sha(positions),
                     images_sha256=tensor_sha(images), targets_sha256=tensor_sha(targets))
        trace.append(entry)
        optimizer.zero_grad(set_to_none=True)
        loss = logical_backward(model, images, targets, cfg['micro_batch_size'], amp)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        require(math.isfinite(loss) and bool(torch.isfinite(norm)), 'Nonfinite loss or gradient')
        optimizer.step(); scheduler.step()
        torch.cuda.synchronize()
        now = time.monotonic()
        seconds.append(now - last); last = now
        losses.append(dict(step=step + 1, loss=loss, gradient_norm=float(norm)))
        if step % 16 == 0 or step + 1 == steps:
            progress(out, phase, step=step + 1, total_steps=steps, loss=loss,
                     elapsed_seconds=now - budget.start)
    require(len(trace) == steps, 'Incomplete training schedule')
    step_counts = [int(v['step']) for v in optimizer.state.values()]
    require(step_counts and all(x == steps for x in step_counts), 'Incomplete optimizer updates')
    require(len(step_counts) == sum(p.requires_grad for p in model.parameters()), 'Missing trained parameters')
    result = dict(updates=steps, elapsed_seconds=time.monotonic()-started, step_seconds=seconds,
                  optimizer_parameter_steps=step_counts, scheduler_last_epoch=scheduler.last_epoch,
                  losses=losses, input_trace=trace)
    del optimizer, scheduler, batches, images, targets
    return result


@torch.inference_mode()
def evaluate(model, root, val, cfg, out, budget, name, initial=False):
    data = loader(InferenceDataset(Path(root), [str(Path(r['image_path']).relative_to('train')) for r in val],
        tu.build_eval_transform(cfg['image_size'], cfg['resize_ratio'])), cfg['micro_batch_size'], cfg)
    logits = np.empty((len(val), model.head.out_features), np.float32)
    twin = np.empty_like(logits) if initial else None
    native = np.empty_like(logits) if initial else None
    folded = fold_projection(model.state_dict()) if initial else None
    model.eval()
    started = time.monotonic()
    for batch, (images, indices) in enumerate(data):
        budget.check()
        with amp():
            features = model.preprojection(images.cuda(non_blocking=True))
        values = model.readout(features)
        ix = indices.numpy()
        logits[ix] = values.cpu().numpy()
        if initial:
            twin[ix] = torch.nn.functional.linear(features.float(), folded['head.weight'], folded['head.bias']).cpu().numpy()
            with amp():
                old = torch.nn.functional.linear(features @ model.visual.proj, model.head.weight, model.head.bias)
            native[ix] = old.float().cpu().numpy()
        if batch % 64 == 0:
            progress(out, name, rows=min((batch+1)*cfg['micro_batch_size'], len(val)),
                     total_rows=len(val), elapsed_seconds=time.monotonic()-budget.start)
    require(np.isfinite(logits).all(), 'Nonfinite validation output')
    payload = dict(logits=logits, predictions=logits.argmax(1),
                   labels=np.array([int(r['label']) for r in val]),
                   image_paths=np.array([r['image_path'] for r in val]))
    if initial:
        require(np.isfinite(twin).all() and np.isfinite(native).all(), 'Nonfinite initial readout')
        payload.update(folded_logits=twin, native_bf16_logits=native)
    np.savez_compressed(out / f'{name}.npz', **payload)
    return payload, time.monotonic()-started


@torch.inference_mode()
def probabilities(model, root, files, cfg, out, budget, phase):
    sums = np.zeros((len(files), model.head.out_features), np.float32)
    model.eval()
    timings = []
    for view in cfg['test_views']:
        data = loader(InferenceDataset(Path(root), files, inference_view(dict(decoder_profile='multiscale_flip_six'), view)),
                      cfg['micro_batch_size'], cfg)
        visited = np.zeros(len(files), bool)
        last = time.monotonic(); seconds = []
        for batch, (images, indices) in enumerate(data):
            budget.check()
            with amp():
                values = model(images.cuda(non_blocking=True))
            p = values.float().softmax(1).cpu().numpy()
            ix = indices.numpy()
            require(not visited[ix].any() and np.isfinite(p).all(), 'Invalid inference batch')
            sums[ix] += p; visited[ix] = True
            now = time.monotonic(); seconds.append(now-last); last=now
            if batch % 128 == 0:
                progress(out, phase, view=view, rows=int(visited.sum()), total_rows=len(files),
                         elapsed_seconds=now-budget.start)
        require(visited.all(), 'Incomplete view')
        timings.append(dict(view=view, seconds=seconds))
    return sums / len(cfg['test_views']), timings


def save_checkpoint(model, path, cfg, sources, arm):
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(dict(model=state, feature_dim=model.feature_dim, config=cfg,
                    experiment_id=cfg['experiment_id'], arm=arm, updates=cfg['steps'],
                    selected='fixed_last_raw', sources=sources, partition='train_dev',
                    num_classes=model.head.out_features), path)


def package(model, root, files, cfg, arm_out, budget, classes, stage):
    p, timings = probabilities(model, root, files, cfg, arm_out, budget, 'test_sixview')
    scores = probability_scores(p)
    bias = fit_test_uniform_bias(torch.from_numpy(scores).cuda(), iterations=cfg['bias_iterations']).numpy()
    raw, corrected = p.argmax(1), (scores + bias).argmax(1)
    np.savez_compressed(arm_out/'test_predictions.npz', probabilities=p, bias=bias,
                        names=np.asarray(files), raw=raw, calibrated=corrected)
    packages = {}
    for name, prediction in (('submission_raw', raw), ('submission_bias', corrected)):
        destination = arm_out / name
        write_submission(destination, files, prediction, classes)
        command = [sys.executable, str(ROOT/'scripts/check_submission.py'), '--test_dir', str(root),
            '--class-mapping', str(stage/'class_to_idx.json'), '--csv', str(destination/'pred_results.csv'),
            '--zip', str(destination/'submission.zip')]
        checked = subprocess.run(command, capture_output=True, text=True)
        (destination/'check.log').write_text(checked.stdout+checked.stderr)
        checked.check_returncode()
        packages[name] = dict(csv=str(destination/'pred_results.csv'), zip=str(destination/'submission.zip'),
            csv_sha256=sha(destination/'pred_results.csv'), zip_sha256=sha(destination/'submission.zip'), rows=len(files),
            checks_passed=9, checkpoint_sha256=sha(arm_out/'selected.pt'))
    return dict(packages=packages, bias_changed_rows=int((raw!=corrected).sum()), view_timings=timings,
                predictions_sha256=sha(arm_out/'test_predictions.npz'))


def review(metrics_by_arm, comparisons, gate):
    c = comparisons['candidate_vs_control']['all']
    parent = comparisons['candidate_vs_parent']['all']
    common = comparisons['candidate_vs_control']['common_five_errors']
    return bool(c['net'] >= gate['net_vs_control_min'] and
        parent['net'] >= gate['net_vs_parent_min'] and
        common['net'] >= gate['common_error_net_vs_control_min'] and
        metrics_by_arm['candidate']['all']['macro'] - metrics_by_arm['control']['all']['macro'] >= gate['macro_delta_vs_control_min'] and
        (c['regressions'] == 0 or c['corrections']/c['regressions'] >= gate['corrections_to_regressions_min']))


def run(config_path, output):
    cfg = read_json(config_path)
    require(cfg['steps']==512 and cfg['cost_steps']==8 and cfg['logical_batch_size']==80 and
            cfg['micro_batch_size']==16 and cfg['primary_checkpoint']=='last_raw_step512', 'Fixed recipe changed')
    require(not any(cfg[k] for k in ('automatic_full_training','parameter_search','platform_upload')), 'Scope changed')
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    budget = Budget(cfg['budget_seconds'])
    sources = {}
    def check(path, expected=None):
        path=Path(path).resolve(); digest=sha(path)
        require(expected is None or digest==expected, f'Source changed: {path}')
        sources[str(path)]=digest; budget.check(); return path
    try:
        progress(out, 'validating_inputs')
        check(config_path)
        for p in [__file__, ROOT/'reproducibility/aegis_f1/v2/preprojection.py',
            ROOT/'reproducibility/aegis_f1/v2/model.py', ROOT/'reproducibility/aegis_f1/v2/training_utils.py',
            ROOT/'reproducibility/aegis_f1/v2/core.py', ROOT/'scripts/run_v2_fixed_prior.py']:
            check(p)
        stage=Path(cfg['stage_root'])
        manifest=read_json(check(stage/'dataset_manifest.json',cfg['dataset_manifest_sha256']))
        require(manifest['data_version']=='20260921' and manifest['stage']=='repechage', 'Foreign stage')
        for name in ('train_dev.csv','val_dev.csv','test_manifest.csv','class_to_idx.json'):
            check(stage/name,manifest['files'][name])
        train, val = read_rows(stage/'train_dev.csv'), read_rows(stage/'val_dev.csv')
        classes=manifest['num_classes']
        require(len(train)==133815 and len(val)==14880 and classes==750, 'Wrong split')
        require(not {r['content_group'] for r in train}&{r['content_group'] for r in val}, 'Leaking DEV split')
        check(cfg['official_checkpoint'],cfg['official_sha256'])
        check(cfg['checkpoint'],cfg['checkpoint_sha256'])
        payload=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        require(payload['binding']['stage']=='s3_576' and payload['binding']['complete'] is True and
            payload['binding']['plan_sha256']==cfg['plan_sha256'] and payload['epoch']==3 and
            payload['metrics']['chosen']=='ema' and payload['metrics']['validation_is_independent'] is True and
            payload['config']['local_replay']['final_stage'] is False and payload['num_classes']==classes,
            'Wrong DEV parent')
        parent_cfg=payload['config']; state={k:v.detach().clone() for k,v in payload['ema'].items()}; del payload
        old=read_json(check(ROOT/cfg['error_report'],cfg['error_report_sha256']))
        aligned=read_rows(check(old['aligned_csv'],old['aligned_csv_sha256']))
        require(len(aligned)==len(val) and all(all(a[k]==b[k] for k in ('image_path','label','content_group'))
            for a,b in zip(aligned,val)), 'Error-budget population mismatch')
        labels=np.array([int(r['label']) for r in val]); paths=np.array([r['image_path'] for r in val])
        original=np.array([int(r['s3_ema']) for r in aligned])
        groups={k:np.array([int(r[k]) for r in aligned],bool) for k in
            ('all','tail75','head75','a_target_classes','cross_label_duplicate','short_edge_below224','aspect_ratio_at_least2')}
        groups['common_five_errors']=np.logical_and.reduce([
            np.array([int(r[k]) for r in aligned])!=labels for k in
            ('s3_ema','lr512_swa','head768','balanced768','wft448_ema')])
        require(int(groups['common_five_errors'].sum())==2411, 'Frozen error target changed')
        np.savez_compressed(out/'groups.npz', labels=labels,image_paths=paths,original_s3=original,**groups)
        frozen=schedule([int(r['label']) for r in train],classes,cfg)
        np.savez_compressed(out/'schedule.npz',**frozen)
        image_rows=[train[i] for i in np.unique(frozen['indices'])]+val
        for row in image_rows:
            require(sha(Path(manifest['train_root'])/Path(row['image_path']).relative_to('train'))==row['file_sha256'],
                    'Official training pixels changed')
            budget.check()
        atomic_json_dump(dict(sources=sources,groups_sha256=sha(out/'groups.npz'),schedule_sha256=sha(out/'schedule.npz'),
            parent_config=parent_cfg,unique_training_rows=len(np.unique(frozen['indices'])),
            sampled_training_rows=len(frozen['indices']),verified_training_val_images=len(image_rows),
            actual_frozen_target_errors=int(groups['common_five_errors'].sum())),out/'binding.json')
        ensure_idle_cuda(); torch.cuda.reset_peak_memory_stats()
        model=model_from_state(cfg,state,512,classes)
        parent, val_seconds=evaluate(model,manifest['train_root'],val,cfg,out,budget,'parent',initial=True)
        initial=dict(max_abs_logit_difference=float(np.max(np.abs(parent['logits']-parent['folded_logits']))),
            top1_changes=int((parent['predictions']!=parent['folded_logits'].argmax(1)).sum()),
            fp32_readout_vs_original_archive_changes=int((parent['predictions']!=original).sum()),
            native_local_bf16_vs_archive_changes=int((parent['native_bf16_logits'].argmax(1)!=original).sum()))
        atomic_json_dump(initial,out/'initial_equivalence.json')
        require(initial['max_abs_logit_difference']<=cfg['initial_max_abs_logit_difference'] and
            initial['top1_changes']<=cfg['initial_top1_changes_allowed'], 'Initial function-equivalence check failed')
        del model; torch.cuda.empty_cache()
        costs={}
        preview_files=[str(Path(r['image_path']).relative_to('train')) for r in val[:64]]
        for arm,dimension in (('control',512),('candidate',768)):
            model=model_from_state(cfg,state,dimension,classes)
            timing=run_updates(model,manifest['train_root'],train,frozen,cfg,parent_cfg,cfg['cost_steps'],out,budget,'probe_'+arm)
            _,views=probabilities(model,manifest['train_root'],preview_files,cfg,out,budget,'cost_sixview_'+arm)
            inference=sum(v['seconds'][0]+max(v['seconds'][1:])*(math.ceil(manifest['test_samples']/cfg['micro_batch_size'])-1) for v in views)
            costs[arm]=dict(update_max_seconds=max(timing['step_seconds'][1:]),
                projected_training_seconds=max(timing['step_seconds'][1:])*cfg['steps'],
                projected_test_seconds=inference,probe_training=timing,probe_views=views)
            del model; torch.cuda.empty_cache()
        require(costs['control']['probe_training']['input_trace']==costs['candidate']['probe_training']['input_trace'],
                'Cost-probe paired inputs differ')
        estimate=sum(v['projected_training_seconds']+v['projected_test_seconds'] for v in costs.values())+2*val_seconds+180
        cost=dict(arms=costs,remaining_estimate_seconds=estimate,reserved_seconds=estimate*cfg['cost_reserve_factor'],
                  elapsed_seconds=time.monotonic()-budget.start,budget_seconds=budget.limit)
        cost['passed']=cost['reserved_seconds']+cost['elapsed_seconds']<budget.limit
        atomic_json_dump(cost,out/'cost.json')
        require(cost['passed'], 'Bounded pair and delivery cost gate failed')
        histories={}; predictions={'parent':parent['predictions']}
        for arm,dimension in (('control',512),('candidate',768)):
            arm_out=out/arm; arm_out.mkdir()
            model=model_from_state(cfg,state,dimension,classes)
            history=run_updates(model,manifest['train_root'],train,frozen,cfg,parent_cfg,cfg['steps'],out,budget,'train_'+arm)
            histories[arm]=history
            atomic_json_dump(history,arm_out/'training.json')
            save_checkpoint(model,arm_out/'selected.pt',cfg,sources,arm)
            evaluation,_=evaluate(model,manifest['train_root'],val,cfg,arm_out,budget,'validation')
            predictions[arm]=evaluation['predictions']
            cold=V2FeatureClassifier(dict(official_checkpoint=cfg['official_checkpoint'],official_sha256=cfg['official_sha256'],
                num_classes=classes,gradient_checkpointing=True),dimension)
            loaded=torch.load(arm_out/'selected.pt',map_location='cpu',weights_only=False,mmap=True)
            cold.load_state_dict(loaded['model'],strict=True)
            del model,loaded; torch.cuda.empty_cache(); cold=cold.cuda()
            cold_eval,_=evaluate(cold,manifest['train_root'],val[:64],cfg,arm_out,budget,'cold_validation64')
            require(np.array_equal(cold_eval['logits'],evaluation['logits'][:64]),'Cold checkpoint mismatch')
            del cold; torch.cuda.empty_cache()
        require(histories['control']['input_trace']==histories['candidate']['input_trace'], 'Paired training inputs differ')
        metric={a:{g:metrics(labels,p,classes,m) for g,m in groups.items()} for a,p in predictions.items()}
        comparisons={name:{g:paired(labels,predictions[b],predictions[a],m) for g,m in groups.items()}
            for name,a,b in [('candidate_vs_control','candidate','control'),('candidate_vs_parent','candidate','parent'),
                             ('control_vs_parent','control','parent')]}
        passed=review(metric,comparisons,cfg['review_gate'])
        diagnostic=dict(metrics=metric,comparisons=comparisons,review_gate_passed=passed,initial_equivalence=initial)
        atomic_json_dump(diagnostic,out/'diagnostic.json')
        progress(out,'paired_training_complete',review_gate_passed=passed,paired=comparisons['candidate_vs_control']['all'])
        tests=read_rows(stage/'test_manifest.csv'); files=sorted(Path(r['image_path']).name for r in tests)
        require(len(files)==len(set(files))==manifest['test_samples'], 'Invalid test population')
        for row in tests:
            require(sha(Path(manifest['test_root'])/Path(row['image_path']).name)==row['file_sha256'],'Test pixels changed')
            budget.check()
        packages={}
        for arm,dimension in (('control',512),('candidate',768)):
            arm_out=out/arm
            model=V2FeatureClassifier(dict(official_checkpoint=cfg['official_checkpoint'],official_sha256=cfg['official_sha256'],
                num_classes=classes,gradient_checkpointing=True),dimension)
            payload=torch.load(arm_out/'selected.pt',map_location='cpu',weights_only=False,mmap=True)
            model.load_state_dict(payload['model'],strict=True); del payload
            model=model.cuda()
            packages[arm]=package(model,manifest['test_root'],files,cfg,arm_out,budget,classes,stage)
            del model; torch.cuda.empty_cache()
        artifacts={str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('progress.json',)}
        report=dict(status='completed_packaged_pair_pending_independent_verification',experiment_id=cfg['experiment_id'],
            output=str(out),config=str(Path(config_path).resolve()),sources=sources,artifacts=artifacts,
            validation_rows=len(val),test_rows=len(files),classes=classes,steps_per_arm=cfg['steps'],
            training_inputs_identical=True,precision=cfg['precision'],primary_checkpoint=cfg['primary_checkpoint'],
            **diagnostic,packages=packages,cost=cost,elapsed_seconds=time.monotonic()-budget.start,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),torch_version=torch.__version__,
            gpu=torch.cuda.get_device_name(),platform_score=None,
            decision='supports_full_training_review' if passed else 'close_fixed_short_continuation',
            automatic_full_training=False,parameter_search=False,platform_uploaded=False)
        atomic_json_dump(report,out/'report.json'); progress(out,'completed_packaged_pair',decision=report['decision'])
    except Exception as error:
        atomic_json_dump(dict(status='failed_closed',error_type=type(error).__name__,error=str(error),
            elapsed_seconds=time.monotonic()-budget.start,sources=sources,automatic_restart=False),out/'failure.json')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); torch.set_num_threads(2)
    run(args.config,args.output)
