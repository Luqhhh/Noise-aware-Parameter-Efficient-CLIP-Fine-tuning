"""Fixed preliminary-75 experiment: real composite features and shared-head refit.

Training priors are computed from the parent's final soft supervision, never
from diagnostic/test predictions. Existing Aegis loaders and lineage guards
remain in use. The historical validation remains an overlap diagnostic.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from aegis_clip.balanced_inference import prediction_metrics
from aegis_clip.checkpoint import _atomic_torch_save, build_from_checkpoint
from aegis_clip.config import validate_config
from aegis_clip.data import TrustBundle, load_class_mapping, resolve_image_path
from aegis_clip.features import canonical_sample_path
from aegis_clip.lineage import run_lineage_audit
from aegis_clip.local_feature_adapter import load_local_feature_adapter
from aegis_clip.local_inference import (
    adapted_dual_local_view_logits, native_visual_forward_with_patch_features,
)
from aegis_clip.localization import (
    extract_attention_crops, forward_features_with_last_block_attention,
    fuse_global_multilocal_flip_probabilities,
)
from aegis_clip.losses import class_prior_adjusted_logits, corrected_targets
from aegis_clip.part_token_adapter import load_part_token_adapter, pool_cls_aligned_patch_features
from aegis_clip.runtime import atomic_json_dump, environment_manifest, seed_worker, set_seed, sha256_file

SCALES = (112, 128, 144, 160)
SCALE_WEIGHTS = (.2, .3, .4, .1)
VIEW_WEIGHTS = (.3, .3, .04, .06, .08, .02, .04, .06, .08, .02)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_plan(path):
    source = Path(path).resolve()
    plan = yaml.safe_load(source.read_text())
    allowed = {'plan_id', 'seed', 'parent', 'trust', 'train_csv', 'val_csv',
               'class_mapping', 'groups', 'train_root', 'test_root', 'output',
               'cache_batch_size', 'num_workers', 'parent_sha256', 'trust_sha256',
               'train_csv_sha256', 'val_csv_sha256', 'class_mapping_sha256'}
    if set(plan) != allowed or plan['plan_id'] != 'PRELIM75_V2_20260912' or plan['seed'] != 42:
        raise ValueError('Unknown or incomplete fixed experiment plan')
    for key in ('parent', 'trust', 'train_csv', 'val_csv', 'class_mapping',
                'groups', 'train_root', 'test_root', 'output'):
        plan[key] = str((source.parent / plan[key]).resolve())
    for key in ('parent', 'trust', 'train_csv', 'val_csv', 'class_mapping'):
        if sha256_file(plan[key]) != plan[key + '_sha256']:
            raise ValueError(f'Frozen asset hash mismatch: {key}')
    if Path(plan['train_root']) == Path(plan['test_root']):
        raise ValueError('Train/test roots must differ')
    plan['_config_path'] = str(source)
    plan['_config_sha256'] = sha256_file(source)
    return plan


def repository_root():
    for directory in Path(__file__).resolve().parents:
        if (directory / '.git').exists():
            return directory
    raise RuntimeError('Experiment must run inside its repository')


def source_manifest():
    root = repository_root()
    package = Path(__file__).parent
    files = {str(p.relative_to(root)): sha256_file(p) for p in sorted(package.rglob('*.py'))}
    return {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'dirty_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=root)).hexdigest(),
        'source_files_sha256': files,
        'runtime': environment_manifest(),
    }


def gpu_setup():
    if not torch.cuda.is_available():
        raise RuntimeError('The authorized experiment requires CUDA')
    set_seed(42, deterministic=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    return torch.device('cuda')


def supervision(plan):
    mapping, _ = load_class_mapping(plan['class_mapping'])
    if len(mapping) != 500:
        raise ValueError('Frozen preliminary mapping must have 500 classes')
    frame = pd.read_csv(plan['train_csv'])
    paths = [canonical_sample_path(p) for p in frame.image_path.astype(str)]
    if len(paths) != 103218 or len(set(paths)) != len(paths):
        raise ValueError('Official final fitting list identity mismatch')
    labels = torch.tensor(frame.label.to_numpy(), dtype=torch.long)
    if (labels < 0).any() or (labels >= 500).any():
        raise ValueError('Invalid original labels')
    train_root = Path(plan['train_root']).resolve()
    for path, label in zip(paths, labels.tolist()):
        resolved = resolve_image_path(train_root, path).resolve()
        if not resolved.is_relative_to(train_root) or mapping.get(Path(path).parts[0]) != label:
            raise ValueError('Training row outside official root or inconsistent mapping')
    trust = TrustBundle(plan['trust'])
    trust.verify_coverage(paths)
    idx = torch.tensor([trust.path_to_index[p] for p in paths])
    clean = trust.clean_probability[idx]
    w = torch.where(clean >= .60, .6 + .4 * clean, torch.zeros_like(clean))
    q = corrected_targets(labels, trust.pseudo_label[idx], trust.correction_alpha[idx], 500)
    if not torch.isfinite(q).all() or not torch.isfinite(w).all() or not torch.allclose(q.sum(1), torch.ones(len(q))):
        raise ValueError('Nonfinite/unnormalized parent supervision')
    groups = json.loads(Path(plan['groups']).read_text())
    group_ids = [groups[p] for p in paths]
    n_eff = effective_counts(w, q)
    return {'paths': paths, 'labels': labels, 'weights': w, 'targets': q,
            'clean_probability': clean, 'pseudo_label': trust.pseudo_label[idx],
            'correction_alpha': trust.correction_alpha[idx], 'group_ids': group_ids,
            'n_eff': n_eff, 'trust': trust}


def effective_counts(weights, targets):
    if targets.ndim != 2 or weights.shape != (len(targets),):
        raise ValueError('Supervision row shapes differ')
    if not torch.isfinite(targets).all() or not torch.isfinite(weights).all() or (weights < 0).any() or (targets < 0).any():
        raise ValueError('Invalid supervision')
    if not torch.allclose(targets.sum(1), torch.ones(len(targets), device=targets.device)):
        raise ValueError('Soft targets must sum to one')
    return (weights.double().unsqueeze(1) * targets.double()).sum(0)


class OfficialImages(Dataset):
    def __init__(self, paths, root, preprocess):
        self.paths = paths
        self.root = Path(root).resolve()
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = resolve_image_path(self.root, self.paths[index]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Image escaped official training root')
        try:
            with Image.open(path) as image:
                tensor = self.preprocess(image.convert('RGB'))
        except Exception as exc:
            raise RuntimeError(f'Official training image decode failed: {path}') from exc
        return {'images': tensor, 'index': index, 'path': self.paths[index]}


def loader(dataset, batch_size, plan, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=plan['num_workers'], pin_memory=True,
                      timeout=120 if plan['num_workers'] else 0, worker_init_fn=seed_worker,
                      **({'prefetch_factor': 1, 'persistent_workers': True} if plan['num_workers'] else {}))


def load_composite(path, device):
    model, preprocess, checkpoint = build_from_checkpoint(path, device)
    validate_config(checkpoint['config'])
    permitted_mode = model.peft_mode == 'full_finetune' or (
        model.peft_mode == 'frozen' and checkpoint.get('prelim75_training', {}).get('name') in ('H0', 'H1', 'C'))
    if not permitted_mode or not isinstance(model.classifier, torch.nn.Linear):
        raise ValueError('Complete full-finetune shared linear model required')
    o3 = load_local_feature_adapter(checkpoint, device)
    pta = load_part_token_adapter(checkpoint, device)
    model.eval(); o3.eval(); pta.eval()
    return model, preprocess, checkpoint, o3, pta


def candidate_config(checkpoint, plan, name):
    config = copy.deepcopy(checkpoint['config'])
    config['project']['experiment_id'] = f'PRELIM75_V2_{name}'
    for key in ('train_csv', 'val_csv', 'class_mapping', 'train_root', 'test_root'):
        config['data'][key] = plan[key]
    config['data']['train_augmentation'] = 'clip_center_crop'
    config['train']['init_checkpoint'] = plan['parent']
    config['train']['require_lineage_for_init_checkpoint'] = True
    config['lineage'] = {'enabled': True, 'parent_experiment_id': plan.get('_feature_parent_experiment_id', 'F1_FLAT_FULL_FT_R3MS'),
                         'parent_train_csv': plan['train_csv'], 'parent_val_csv': plan['val_csv'],
                         'require_same_train': True, 'require_same_val': True,
                         'allow_parent_val_in_child_train': True,
                         'allow_parent_train_in_child_val': True}
    config['evaluation']['selection_policy'] = 'last_epoch'
    config['output']['root'] = plan['output']
    config['features'] = {
        'tensor_path': str(Path(plan['output']).parent / 'stage_readiness/20260912_full_rebuild_r1/features/features.pt'),
        'paths_path': str(Path(plan['output']).parent / 'stage_readiness/20260912_full_rebuild_r1/features/image_paths.json'),
        'manifest_path': str(Path(plan['output']).parent / 'stage_readiness/20260912_full_rebuild_r1/features/manifest.json'),
    }
    validate_config(config)
    return config


def audit_parent(config, plan, output):
    return run_lineage_audit(config, child_train_csv=plan['train_csv'],
                             child_val_csv=plan['val_csv'], checkpoint_path=plan['parent'],
                             output_path=output / 'split_lineage_audit.json')


def cache_binding(plan, paths):
    return {'parent_sha256': plan['parent_sha256'], 'train_csv_sha256': plan['train_csv_sha256'],
            'trust_sha256': plan['trust_sha256'], 'class_mapping_sha256': plan['class_mapping_sha256'],
            'groups_sha256': sha256_file(plan['groups']),
            'paths_sha256': hashlib.sha256('\n'.join(paths).encode()).hexdigest(),
            'scales': list(SCALES), 'view_weights': list(VIEW_WEIGHTS),
            'top_k': 5, 'part_top_patches': 8, 'part_temperature': .07,
            'temperature': 1.5, 'prior_in_inference': False, 'dtype': 'float32',
            'geometry': 'native_clip_224_center_crop_and_tensor_flip'}


@torch.no_grad()
def cache_features(plan):
    output = Path(plan['output']) / plan.get('_cache_directory', 'cache')
    output.mkdir(parents=True, exist_ok=False)
    device = gpu_setup(); start = time.monotonic()
    data = supervision(plan)
    model, preprocess, checkpoint, o3, pta = load_composite(plan['parent'], device)
    model.requires_grad_(False); o3.requires_grad_(False); pta.requires_grad_(False)
    audit_parent(candidate_config(checkpoint, plan, 'CACHE'), plan, output)
    atomic_json_dump(source_manifest(), output / 'source_manifest.json')
    atomic_json_dump(plan, output / 'resolved_plan.json')
    atomic_json_dump(data['paths'], output / 'paths.json')
    _atomic_torch_save({k: v for k, v in data.items() if k != 'trust'}, output / 'supervision.pt')
    arrays = {
        'global': np.lib.format.open_memmap(output / 'global.npy', mode='w+', dtype=np.float32, shape=(len(data['paths']), 2, 512)),
        'local_base': np.lib.format.open_memmap(output / 'local_base.npy', mode='w+', dtype=np.float32, shape=(len(data['paths']), 8, 512)),
        'local_residual': np.lib.format.open_memmap(output / 'local_residual.npy', mode='w+', dtype=np.float32, shape=(len(data['paths']), 8, 512)),
        'boxes': np.lib.format.open_memmap(output / 'boxes.npy', mode='w+', dtype=np.int16, shape=(len(data['paths']), 2, 4, 4)),
    }
    dataset = OfficialImages(data['paths'], plan['train_root'], preprocess)
    stream = loader(dataset, plan['cache_batch_size'], plan)
    numeric = {'calls': 0, 'max_abs': 0., 'argmax_disagreements': 0}
    completed = 0
    for batch_idx, batch in enumerate(stream):
        images = batch['images'].to(device, non_blocking=True)
        indices = batch['index'].numpy()
        for orientation, view in enumerate((images, images.flip(3))):
            global_logits, global_features, attention = forward_features_with_last_block_attention(model, view)
            if not torch.isfinite(global_features).all():
                raise RuntimeError('Nonfinite global cache')
            arrays['global'][indices, orientation] = global_features.cpu().numpy()
            for scale_idx, scale in enumerate(SCALES):
                local, boxes = extract_attention_crops(view, attention, crop_size=scale, top_k=5)
                base_logits, base, patches = native_visual_forward_with_patch_features(model, local)
                part = pool_cls_aligned_patch_features(base, patches, top_patches=8, temperature=.07)
                adapted = o3(base) + pta(base, part) - base
                residual = adapted - base
                if not torch.isfinite(base).all() or not torch.isfinite(residual).all():
                    raise RuntimeError('Nonfinite local cache')
                slot = orientation * 4 + scale_idx
                arrays['local_base'][indices, slot] = base.cpu().numpy()
                arrays['local_residual'][indices, slot] = residual.cpu().numpy()
                arrays['boxes'][indices, orientation, scale_idx] = np.asarray(boxes, dtype=np.int16)
                if batch_idx == 0:
                    native = adapted_dual_local_view_logits(model, o3, pta, local)
                    reconstructed = model.classifier(base) + F.linear(residual, model.classifier.weight)
                    torch.testing.assert_close(native, reconstructed, atol=1e-4, rtol=1e-5)
                    numeric['calls'] += 1
                    numeric['max_abs'] = max(numeric['max_abs'], float((native-reconstructed).abs().max()))
                    numeric['argmax_disagreements'] += int((native.argmax(1) != reconstructed.argmax(1)).sum())
            if batch_idx == 0:
                torch.testing.assert_close(model.classifier(global_features), global_logits, atol=1e-4, rtol=1e-5)
        completed += len(indices)
        if batch_idx % 20 == 0 or completed == len(dataset):
            progress = {'status': 'running', 'completed_samples': completed, 'total_samples': len(dataset),
                        'elapsed_seconds': time.monotonic()-start}
            atomic_json_dump(progress, output / 'status.json')
            print(json.dumps(progress), flush=True)
    if numeric['argmax_disagreements']:
        raise RuntimeError('Cache/native initial argmax mismatch')
    for value in arrays.values():
        value.flush()
    binding = cache_binding(plan, data['paths'])
    binding.update({'status': 'complete', 'rows': len(dataset), 'numerical_check': numeric,
                    'elapsed_seconds': time.monotonic()-start,
                    'max_cuda_memory_allocated': torch.cuda.max_memory_allocated(),
                    'files_sha256': {p.name: sha256_file(p) for p in output.iterdir()
                                     if p.suffix in ('.npy', '.pt') or p.name == 'paths.json'}})
    atomic_json_dump(binding, output / 'manifest.json')
    atomic_json_dump(binding, output / 'status.json')
    return output


def read_cache(plan, verify=True):
    output = Path(plan['output']) / plan.get('_cache_directory', 'cache')
    manifest = json.loads((output / 'manifest.json').read_text())
    paths = json.loads((output / 'paths.json').read_text())
    if manifest['status'] != 'complete' or manifest['rows'] != 103218:
        raise ValueError('Incomplete feature cache')
    for key, value in cache_binding(plan, paths).items():
        if manifest.get(key) != value:
            raise ValueError(f'Cache identity mismatch: {key}')
    if verify:
        for name, value in manifest['files_sha256'].items():
            if sha256_file(output / name) != value:
                raise ValueError(f'Cache hash mismatch: {name}')
    data = torch.load(output / 'supervision.pt', map_location='cpu', weights_only=False)
    if data['paths'] != paths or not torch.equal(effective_counts(data['weights'], data['targets']), data['n_eff']):
        raise ValueError('Cache supervision mismatch')
    arrays = {k: np.load(output / f'{k}.npy', mmap_mode='r') for k in ('global', 'local_base', 'local_residual', 'boxes')}
    return data, arrays, manifest


def cached_logits(head, arrays, indices, device):
    global_features = torch.from_numpy(np.array(arrays['global'][indices], copy=True)).to(device)
    base = torch.from_numpy(np.array(arrays['local_base'][indices], copy=True)).to(device)
    residual = torch.from_numpy(np.array(arrays['local_residual'][indices], copy=True)).to(device)
    globals_ = head(global_features)
    locals_ = head(base) + F.linear(residual, head.weight)
    return torch.cat((globals_, locals_), dim=1)


def head_loss(logits, targets, weights, counts=None, view_weights=VIEW_WEIGHTS):
    scaled = logits.float() / 1.5
    if counts is not None:
        scaled = class_prior_adjusted_logits(scaled.reshape(-1, scaled.shape[-1]), counts, 1.0).reshape_as(scaled)
    log_probs = F.log_softmax(scaled, dim=-1)
    per_view = -(targets[:, None, :] * log_probs).sum(-1)
    view_weight = torch.tensor(view_weights, device=logits.device)
    if per_view.shape[1] != len(view_weight) or not math.isclose(float(view_weight.sum()), 1., abs_tol=1e-6):
        raise ValueError('Invalid view weights')
    return ((per_view * view_weight).sum(1) * weights).mean()


def fused_logits(views):
    return fuse_global_multilocal_flip_probabilities(
        views[:, 0], [views[:, i] for i in range(2, 6)],
        views[:, 1], [views[:, i] for i in range(6, 10)],
        local_weight=.4, flip_weight=.5, temperature=1.5,
        global_temperature=1.5, local_temperature=1.5, local_scale_weights=SCALE_WEIGHTS)


def saved_candidate(checkpoint, model, config, name, plan, optimizer, scheduler, epoch, o3=None, pta=None):
    stale = {'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict', 'rng_state',
             'data_generator_state', 'metrics', 'best_selector', 'adaptive_cap_state',
             'elr_state_dict', 'training_aux_state'}
    result = {k: v for k, v in checkpoint.items() if k not in stale}
    result['model_state_dict'] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    result['config'] = config
    result['effective_model_spec'] = model.effective_spec()
    result['epoch'] = epoch
    result['optimizer_state_dict'] = optimizer.state_dict()
    result['scheduler_state_dict'] = scheduler.state_dict()
    result['prelim75_training'] = {'name': name, 'plan_id': plan['plan_id'],
                                  'parent_sha256': plan.get('_root_parent_sha256', plan['parent_sha256']),
                                  'feature_parent_sha256': plan['parent_sha256'], 'selection': 'last_epoch',
                                  'supervision': 'P_epoch3_fixed_w_q', 'prior_in_inference': False,
                                  'upstream_provenance_complete': False}
    if o3 is not None:
        for key, adapter in (('local_feature_adapter', o3), ('part_token_adapter', pta)):
            result[key] = dict(result[key])
            result[key]['state_dict'] = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
            result[key]['gate'] = {'scope': 'fixed_joint_continuation_no_separate_selection',
                                   'historical_gate_reference': checkpoint[key].get('gate'),
                                   'selection_policy': 'last_epoch'}
    return result


def train_head(plan, name):
    if name not in ('H0', 'H1') and not (name == 'C' and plan.get('_combination_approved') is True):
        raise ValueError('Only the two registered head candidates are authorized')
    output = Path(plan['output']) / name
    output.mkdir(parents=True, exist_ok=False)
    device = gpu_setup(); start = time.monotonic()
    data, arrays, manifest = read_cache(plan)
    model, _, checkpoint, o3, pta = load_composite(plan['parent'], device)
    model.requires_grad_(False); o3.requires_grad_(False); pta.requires_grad_(False)
    model.classifier.requires_grad_(True)
    head = model.classifier
    head_variant = plan.get('_head_variant', name)
    counts = data['n_eff'].float().to(device) if head_variant == 'H1' else None
    if counts is not None and (counts <= 0).any():
        raise ValueError('Zero effective class support; H1 closed')
    config = candidate_config(checkpoint, plan, name)
    # Frozen/full-FT own exactly the same plain visual weights and linear head.
    # Record the actual head-only scope without weakening the PEFT LR guard.
    config['model']['peft_mode'] = 'frozen'
    config['model']['use_cached_training'] = True
    model.peft_mode = 'frozen'
    config['train'].update({'epochs': 10, 'schedule_epochs': 10, 'batch_size': 1024,
                            'head_lr': 1e-3, 'head_weight_decay': 1e-4, 'backbone_lr': 0., 'amp': False})
    config['loss'].update({'name': 'cross_entropy', 'feature_distillation_weight': 0., 'class_prior_adjustment_tau': 0.})
    validate_config(config)
    audit_parent(config, plan, output)
    atomic_json_dump(config, output / 'resolved_config.json')
    atomic_json_dump(source_manifest(), output / 'source_manifest.json')
    atomic_json_dump({'cache_manifest_sha256': sha256_file(Path(plan['output'])/plan.get('_cache_directory','cache')/'manifest.json'),
                      'n_eff': data['n_eff'].tolist(), 'training_prior_tau': int(head_variant == 'H1'),
                      'supervision_state': 'parent_epoch3_all_ten_epochs',
                      'trainable_parameters': [n for n,p in model.named_parameters() if p.requires_grad],
                      'frozen_modules_eval': True}, output / 'training_recipe.json')
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(data['paths']) / 1024)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10*steps_per_epoch)
    generator = torch.Generator().manual_seed(42)
    history = []
    for epoch in range(1, 11):
        order = torch.randperm(len(data['paths']), generator=generator).numpy()
        total_loss = 0.
        for offset in range(0, len(order), 1024):
            idx = order[offset:offset+1024]
            targets = data['targets'][idx].to(device)
            weights = data['weights'][idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = cached_logits(head, arrays, idx, device)
            loss = head_loss(logits, targets, weights, counts)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite shared-head loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            if epoch == 1 and offset == 0:
                if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('classifier.')):
                    raise RuntimeError('Head-only gradient leaked')
            optimizer.step(); scheduler.step()
            total_loss += float(loss.detach()) * len(idx)
        row = {'epoch': epoch, 'loss': total_loss/len(order), 'head_lr': scheduler.get_last_lr()[0],
               'elapsed_seconds': time.monotonic()-start}
        history.append(row)
        atomic_json_dump({'status': 'training', 'history': history}, output / 'status.json')
        print(json.dumps(row), flush=True)
    payload = saved_candidate(checkpoint, model, config, name, plan, optimizer, scheduler, 10)
    for k, value in checkpoint['model_state_dict'].items():
        if not k.startswith('classifier.') and not torch.equal(payload['model_state_dict'][k], value.cpu()):
            raise RuntimeError(f'Frozen parent tensor changed: {k}')
    _atomic_torch_save(payload, output / 'candidate.pt')
    final = {'status': 'trained_pending_real_diagnostic', 'history': history,
             'checkpoint_sha256': sha256_file(output/'candidate.pt'),
             'elapsed_seconds': time.monotonic()-start,
             'max_cuda_memory_allocated': torch.cuda.max_memory_allocated(),
             'online_accuracy': None, 'validation_scope': 'overlap_diagnostic'}
    atomic_json_dump(final, output / 'training_result.json')
    atomic_json_dump(final, output / 'status.json')
    return output / 'candidate.pt'
