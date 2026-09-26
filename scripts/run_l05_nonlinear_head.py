#!/usr/bin/env python3
"""Fixed paired head refit on this stage's frozen L05 image features."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import torch
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reproducibility/aegis_f1'))
from aegis_clip.checkpoint import build_from_checkpoint, _atomic_torch_save
from aegis_clip.model import ResidualFeatureAdapter
from aegis_clip.prior_alignment import fit_prior_bias, apply_prior_bias
from aegis_clip.runtime import sha256_file
from aegis_clip.tta import fuse_paired_logits


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def preflight(cfg):
    assets = Path(cfg['assets'])
    manifest = json.loads((assets / 'dataset_manifest.json').read_text())
    assert manifest['stage'] == 'repechage' and manifest['data_version'] == '20260921'
    assert not manifest['validation_overlap_with_training']
    for name, digest in manifest['files'].items():
        assert sha256_file(assets / name) == digest, name
    assert sha256_file(cfg['parent_checkpoint']) == cfg['parent_sha256']
    rows = {}
    for split in ('train', 'val'):
        with (assets / f'{split}_dev.csv').open(newline='') as f:
            rows[split] = list(csv.DictReader(f))
        assert len(rows[split]) == manifest[f'{split}_dev_samples']
        assert all(r['image_path'].startswith('train/') for r in rows[split])
    for key in ('image_path', 'content_group'):
        assert not ({r[key] for r in rows['train']} & {r[key] for r in rows['val']}), key
    reference = torch.load(cfg['reference_cache'], map_location='cpu', weights_only=False)
    assert reference['checkpoint_sha256'] == cfg['parent_sha256']
    assert reference['validation_csv_sha256'] == manifest['files']['val_dev.csv']
    by_path = {r['image_path'].removeprefix('train/'): r for r in rows['val']}
    assert set(by_path) == set(reference['paths']) and len(by_path) == len(reference['paths'])
    rows['val'] = [by_path[p] for p in reference['paths']]
    assert [int(r['label']) for r in rows['val']] == reference['labels'].tolist()
    identity = dict(parent=cfg['parent_sha256'], manifest=sha256_file(assets/'dataset_manifest.json'),
                    config=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                    reference_cache=sha256_file(cfg['reference_cache']), format=1)
    return rows, reference, identity


class Images(Dataset):
    def __init__(self, rows, image_root, transform):
        self.rows, self.image_root, self.transform = rows, Path(image_root), transform
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.image_root / row['image_path'].removeprefix('train/')).resolve()
        assert path.is_relative_to(self.image_root.resolve())
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['file_sha256'], str(path)
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(io.BytesIO(raw)) as image:
            return self.transform(image.convert('RGB'))


def metrics(logits, labels):
    pred = logits.argmax(1)
    counts = torch.bincount(labels, minlength=logits.shape[1])
    assert bool((counts > 0).all())
    correct = torch.bincount(labels[pred == labels], minlength=logits.shape[1])
    return {'macro': float((correct.double()/counts).mean()),
            'micro': float((pred == labels).double().mean())}


def cache(cfg, rows, reference, identity):
    device = torch.device('cuda:0')
    model, transform, _ = build_from_checkpoint(cfg['parent_checkpoint'], device)
    model.eval().requires_grad_(False)
    out = Path(cfg['output']) / 'features'
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    # Validation first: fail before the expensive train feature pass if replay differs.
    for split in ('val', 'train'):
        for offset in range(0, len(rows[split]), 4096):
            selected = rows[split][offset:offset+4096]
            path = out / f'{split}_{offset:06d}.pt'
            metadata = {**identity, 'split': split, 'offset': offset,
                        'paths': [r['image_path'] for r in selected]}
            if path.exists():
                old = torch.load(path, map_location='cpu', weights_only=False)
                assert old['identity'] == metadata and torch.isfinite(old['center']).all()
                print(f'reusing {path}', flush=True)
                continue
            loader = DataLoader(Images(selected, cfg['image_root'], transform),
                                batch_size=cfg['inference_batch_size'], num_workers=cfg['workers'],
                                shuffle=False, pin_memory=True)
            features, flipped = [], []
            with torch.inference_mode():
                for batch_i, images in enumerate(loader):
                    images = images.to(device, non_blocking=True)
                    with torch.autocast('cuda'):
                        feat = model.encode_image(images)
                        flip = model.encode_image(images.flip(3)) if split == 'val' else None
                        if split == 'val' and offset == 0 and batch_i == 0:
                            checks = []
                            for x, key in ((feat, 'original_logits'), (flip, 'flip_logits')):
                                logits = model.classifier(x).float().cpu()
                                ref = reference[key][:len(logits)]
                                error = float((logits-ref).abs().max())
                                agreement = bool(torch.equal(logits.argmax(1), ref.argmax(1)))
                                assert error <= .02 and agreement, (key, error, agreement)
                                checks.append(dict(branch=key, max_abs=error, top1_identical=agreement))
                            write_json(Path(cfg['output'])/'native_replay.json', checks)
                            print(json.dumps(checks), flush=True)
                    features.append(feat.float().cpu())
                    if flip is not None:
                        flipped.append(flip.float().cpu())
            payload = {'identity': metadata, 'center': torch.cat(features),
                       'labels': torch.tensor([int(r['label']) for r in selected])}
            if flipped:
                payload['flip'] = torch.cat(flipped)
            assert torch.isfinite(payload['center']).all()
            _atomic_torch_save(payload, path)
            print(f'cached {split} {offset+len(selected)}/{len(rows[split])}; elapsed={time.monotonic()-start:.1f}s', flush=True)
    write_json(out/'complete.json', identity)


def load_features(cfg, rows, identity, split):
    parts = []
    for offset in range(0, len(rows[split]), 4096):
        p = torch.load(Path(cfg['output'])/'features'/f'{split}_{offset:06d}.pt',
                       map_location='cpu', weights_only=False)
        assert p['identity'] == {**identity, 'split': split, 'offset': offset,
                                'paths': [r['image_path'] for r in rows[split][offset:offset+4096]]}
        assert p['labels'].tolist() == [int(r['label']) for r in rows[split][offset:offset+4096]]
        parts.append(p)
    keys = ['center', 'labels'] + (['flip'] if split == 'val' else [])
    return {key: torch.cat([p[key] for p in parts]).to('cuda:0') for key in keys}


def fixed_decode(original, flip, labels, cfg):
    fused = fuse_paired_logits(original, flip, mode='mean_probabilities', temperature=cfg['temperature'])
    bias, report = fit_prior_bias(fused)
    adjusted = apply_prior_bias(fused, bias, strength=cfg['prior_strength'])
    return metrics(adjusted, labels), bias, adjusted.argmax(1), report


def train(cfg, rows, reference, identity):
    out = Path(cfg['output'])
    if (out/'result.json').exists():
        raise FileExistsError('Completed result exists; refusing to overwrite')
    train_data = load_features(cfg, rows, identity, 'train')
    val = load_features(cfg, rows, identity, 'val')
    parent = torch.load(cfg['parent_checkpoint'], map_location='cpu', weights_only=False)
    w = parent['model_state_dict']['classifier.weight']
    b = parent['model_state_dict']['classifier.bias']
    labels = val['labels']
    base_decode, _, base_pred, _ = fixed_decode(reference['original_logits'].cuda(),
                                                reference['flip_logits'].cuda(), labels, cfg)
    base_center = metrics(reference['original_logits'].cuda(), labels)
    results = {'identity': identity, 'base_center': base_center, 'base_decode': base_decode,
               'test_used': False, 'prior_scope': 'same validation fit; not independent', 'arms': {}}
    for arm in ('NH00', 'NH01'):
        arm_dir = out/arm
        if arm_dir.exists():
            raise FileExistsError(f'{arm_dir} exists; preserve evidence and explicitly resume/relocate')
        arm_dir.mkdir()
        torch.manual_seed(cfg['seed'])
        adapter = (ResidualFeatureAdapter(w.shape[1], cfg['adapter_dim'], cfg['adapter_scale'])
                   if arm == 'NH01' else nn.Identity())
        classifier = nn.Linear(w.shape[1], w.shape[0])
        classifier.load_state_dict({'weight': w, 'bias': b})
        head = nn.Sequential(adapter, classifier).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
        gen = torch.Generator(device='cuda').manual_seed(cfg['seed'])
        with torch.no_grad():
            initial = head(val['center'])
            # Cached f32 head and original AMP head can differ slightly; report separately.
            initial_metrics = metrics(initial, labels)
            audit = {'metrics': initial_metrics,
                     'top1_agreement': float((initial.argmax(1)==reference['original_logits'].cuda().argmax(1)).float().mean())}
        best = (initial_metrics['macro'], initial_metrics['micro'])
        best_state = copy.deepcopy(head.state_dict())
        best_epoch, history, stopped = 0, [], False
        for epoch in range(1, cfg['epochs']+1):
            head.train()
            order = torch.randperm(len(train_data['labels']), generator=gen, device='cuda')
            total = 0.
            for idx in order.split(cfg['batch_size']):
                logits = head(train_data['center'][idx])
                prob = logits.softmax(1).gather(1, train_data['labels'][idx, None]).squeeze(1)
                loss = ((1-prob.clamp_min(1e-12).pow(cfg['gce_q']))/cfg['gce_q']).mean()
                assert bool(torch.isfinite(loss))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                total += float(loss.detach())*len(idx)
            scheduler.step()
            head.eval()
            with torch.no_grad():
                measured = metrics(head(val['center']), labels)
            row = {'epoch': epoch, 'train_loss': total/len(order), **measured}
            history.append(row)
            score = (measured['macro'], measured['micro'])
            if score > best:
                best, best_epoch = score, epoch
                best_state = copy.deepcopy(head.state_dict())
            _atomic_torch_save({'epoch': epoch, 'head': head.state_dict(), 'optimizer': opt.state_dict(),
                                'scheduler': scheduler.state_dict(), 'generator': gen.get_state(),
                                'best_state': best_state, 'best_epoch': best_epoch, 'identity': identity}, arm_dir/'last_head.pt')
            write_json(arm_dir/'history.json', history)
            print(arm, json.dumps(row), flush=True)
            if any(measured[k] < base_center[k]-.02 for k in ('macro', 'micro')):
                stopped = True
                break
        head.load_state_dict(best_state)
        with torch.no_grad():
            original, flip = head(val['center']), head(val['flip'])
            decoded, bias, prediction, fit_report = fixed_decode(original, flip, labels, cfg)
        config = copy.deepcopy(parent['config'])
        config['model'].update(peft_mode='feature_adapter' if arm=='NH01' else 'frozen',
                               adapter_dim=cfg['adapter_dim'], adapter_scale=cfg['adapter_scale'])
        config['project'].update(experiment_id=cfg['experiment_id']+'_'+arm,
                                 parent_kind='same_split_continue', parent_experiment_id='RM_V5_L05_CUDA_LOCAL')
        state = {k: v for k,v in parent['model_state_dict'].items() if not k.startswith('classifier.')}
        state.update({'classifier.'+k: v.cpu() for k,v in head[1].state_dict().items()})
        if arm == 'NH01':
            state.update({'feature_adapter.'+k: v.cpu() for k,v in head[0].state_dict().items()})
        payload = {'format_version': 1, 'config': config, 'model_state_dict': state,
                   'epoch': best_epoch, 'metrics': decoded, 'refit_config': cfg,
                   'refit_identity': identity, 'inference_prior_bias': bias.cpu(),
                   'inference_recipe': {'tta': 'horizontal_flip', 'fusion': 'mean_probabilities',
                                         'temperature': cfg['temperature'], 'prior_strength': cfg['prior_strength']}}
        _atomic_torch_save(payload, arm_dir/'best.pt')
        _atomic_torch_save({'original_logits': original.cpu(), 'flip_logits': flip.cpu(),
                           'labels': labels.cpu(), 'prior_bias': bias.cpu(), 'identity': identity}, arm_dir/'val_logits.pt')
        result = {'selected_epoch': best_epoch, 'stopped': stopped, 'initial_cache_audit': audit,
                  'center': metrics(original, labels), 'decode': decoded,
                  'delta_macro_pp': (decoded['macro']-base_decode['macro'])*100,
                  'delta_micro_pp': (decoded['micro']-base_decode['micro'])*100,
                  'corrections': int(((base_pred!=labels)&(prediction==labels)).sum()),
                  'regressions': int(((base_pred==labels)&(prediction!=labels)).sum()),
                  'checkpoint_sha256': sha256_file(arm_dir/'best.pt'), 'prior_fit': fit_report}
        results['arms'][arm] = result
        write_json(arm_dir/'result.json', result)
    a, b = results['arms']['NH00'], results['arms']['NH01']
    def gate(x, ref):
        return x['decode']['macro'] >= ref['macro']+cfg['gate_macro_pp']/100 and x['decode']['micro'] >= ref['micro']
    results['nh00_pass'] = not a['stopped'] and gate(a, base_decode)
    results['nh01_pass'] = not b['stopped'] and gate(b, base_decode) and gate(b, a['decode'])
    write_json(out/'result.json', results)
    print(json.dumps(results, indent=2), flush=True)


def audit(cfg, rows, reference, identity):
    """Reconstruct each complete checkpoint and audit image/cache equivalence."""
    out = Path(cfg['output'])
    parent = torch.load(cfg['parent_checkpoint'], map_location='cpu', weights_only=False)
    val = load_features(cfg, rows, identity, 'val')
    reports = {}
    for arm in ('NH00', 'NH01'):
        model, transform, checkpoint = build_from_checkpoint(out/arm/'best.pt', torch.device('cuda:0'))
        model.eval().requires_grad_(False)
        assert checkpoint['refit_identity'] == identity
        unchanged = all(torch.equal(v.cpu(), parent['model_state_dict'][k])
                        for k,v in model.state_dict().items() if k.startswith('visual.'))
        assert unchanged, 'Frozen visual weights changed'
        saved = torch.load(out/arm/'val_logits.pt', map_location='cpu', weights_only=False)
        max_error, agreements = 0., []
        with torch.inference_mode():
            for key, refkey in (('center','original_logits'), ('flip','flip_logits')):
                logits = model(features=val[key]).cpu()
                error = float((logits-saved[refkey]).abs().max())
                agreement = float((logits.argmax(1)==saved[refkey].argmax(1)).float().mean())
                assert error < 0.0001 and agreement == 1., (error, agreement)
                max_error = max(max_error, error)
                agreements.append(agreement)
            # Full validation direct image inference; fixed fp32 head after AMP encoder.
            loader = DataLoader(Images(rows['val'], cfg['image_root'], transform),
                                batch_size=cfg['inference_batch_size'], num_workers=cfg['workers'])
            image_max_error, seen, image_agree = 0., 0, 0
            direct_first, direct_flip = [], []
            for images in loader:
                images = images.cuda()
                for flipped, refkey, dest in ((False,'original_logits',direct_first), (True,'flip_logits',direct_flip)):
                    with torch.autocast('cuda'):
                        # Encode through the frozen backbone only, then execute the trained head in fp32.
                        features = torch.nn.functional.normalize(model._encode_visual(images.flip(3) if flipped else images).float(), dim=-1)
                    logits = model(features=features).cpu()
                    expected = saved[refkey][seen:seen+len(images)]
                    image_max_error = max(image_max_error, float((logits-expected).abs().max()))
                    image_agree += int((logits.argmax(1)==expected.argmax(1)).sum())
                    dest.append(logits)
                seen += len(images)
            assert image_max_error < 0.02 and image_agree == 2*len(rows['val']), (image_max_error,image_agree)
            direct_fused = fuse_paired_logits(torch.cat(direct_first), torch.cat(direct_flip),
                                             mode='mean_probabilities', temperature=cfg['temperature'])
            direct_logits = apply_prior_bias(direct_fused, checkpoint['inference_prior_bias'], strength=cfg['prior_strength'])
            direct_metrics = metrics(direct_logits, saved['labels'])
        reports[arm] = {'visual_bitwise_unchanged': unchanged, 'cache_reload_max_abs': max_error,
                        'all_validation_image_branch_top1_agreements': image_agree,
                        'all_validation_image_branch_comparisons': 2*seen,
                        'image_reload_max_abs': image_max_error, 'decode': direct_metrics,
                        'checkpoint_sha256': sha256_file(out/arm/'best.pt')}
        write_json(out/'reload_audit.json', reports)
        print(arm, json.dumps(reports[arm]), flush=True)
        del model
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--phase', choices=('cache','train','audit'), required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    rows, reference, identity = preflight(cfg)
    {'cache': cache, 'train': train, 'audit': audit}[args.phase](cfg, rows, reference, identity)

if __name__ == '__main__':
    main()
