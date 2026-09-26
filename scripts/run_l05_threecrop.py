#!/usr/bin/env python3
"""Fixed L05 same-checkpoint three-position crop + flip validation protocol."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from run_l05_nonlinear_head import preflight, metrics, write_json
from aegis_clip.checkpoint import build_from_checkpoint, _atomic_torch_save
from aegis_clip.prior_alignment import fit_prior_bias, apply_prior_bias
from aegis_clip.runtime import sha256_file
from aegis_clip.tta import fuse_paired_logits


def endpoint_boxes(width: int, height: int, size: int):
    if min(width, height) != size:
        raise ValueError('Resize must make the shorter edge exactly crop_size')
    return (0, 0, size, size), (width-size, height-size, width, height)


def fuse_views(branches, temperature):
    if len(branches) != 6 or temperature <= 0:
        raise ValueError('Protocol requires exactly six views and positive temperature')
    shape = branches[0].shape
    if len(shape) != 2 or any(b.shape != shape or not torch.isfinite(b).all() for b in branches):
        raise ValueError('Views must be finite matching [N,C] tensors')
    return (sum((b.float()/temperature).softmax(1) for b in branches)/6).clamp_min(1e-12).log()


class ThreeCropImages(Dataset):
    def __init__(self, rows, root, native, size, offset):
        self.rows, self.root, self.native = rows, Path(root).resolve(), native
        self.resize = native.transforms[0]
        self.center = native.transforms[1]
        self.finish = Compose(native.transforms[2:])
        self.size, self.offset = size, offset
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        path = (self.root/row['image_path'].removeprefix('train/')).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Image escapes declared root')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['file_sha256']:
            raise ValueError('Image hash disagrees with official manifest')
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert('RGB')
            resized = self.resize(image)
            boxes = endpoint_boxes(*resized.size, self.size)
            edges = torch.stack([self.finish(resized.crop(box)) for box in boxes])
            center = self.native(image) if self.offset+idx < 32 else torch.empty(0)
        return edges, center


def evaluate(cfg, rows, reference, identity, branch_files):
    device = torch.device('cuda:0')
    out = Path(cfg['output'])
    labels = reference['labels'].long().to(device)
    original = reference['original_logits'].float().to(device)
    flip = reference['flip_logits'].float().to(device)
    extra = [[], [], [], []]
    for file in branch_files:
        part = torch.load(file, map_location='cpu', weights_only=False)
        for i in range(4):
            extra[i].append(part['branches'][i])
    extra = [torch.cat(parts).to(device) for parts in extra]
    baseline = fuse_paired_logits(original, flip, mode='mean_probabilities', temperature=cfg['temperature'])
    candidate = fuse_views([original, flip, *extra], cfg['temperature'])
    values = {}
    decoded = {}
    for name, logits in [('baseline', baseline), ('threecrop', candidate)]:
        bias, report = fit_prior_bias(logits, max_iterations=50)
        adjusted = apply_prior_bias(logits, bias, strength=cfg['prior_strength'])
        values[name] = {'no_prior': metrics(logits, labels), 'decode': metrics(adjusted, labels), 'prior_fit': report}
        decoded[name] = adjusted.argmax(1)
        _atomic_torch_save({'identity': identity, 'recipe': cfg, 'fused_logits': logits.cpu(),
                           'bias': bias.cpu(), 'labels': labels.cpu()}, out/f'{name}_validation.pt')
    base, new = values['baseline']['decode'], values['threecrop']['decode']
    delta = {k: 100*(new[k]-base[k]) for k in ('macro','micro')}
    corrections = int(((decoded['baseline'] != labels)&(decoded['threecrop']==labels)).sum())
    regressions = int(((decoded['baseline'] == labels)&(decoded['threecrop']!=labels)).sum())
    gate = delta['macro'] >= cfg['gate_macro_pp'] and delta['micro'] >= 0
    result = {'identity': identity, 'protocol': cfg['protocol'], 'metrics': values, 'delta_pp': delta,
              'paired': {'corrections': corrections, 'regressions': regressions,
                         'changed': int((decoded['baseline']!=decoded['threecrop']).sum())},
              'preliminary_gate_pass': gate, 'test_data_used': False,
              'independent_end_to_end_validation': False,
              'validation_samples': len(labels), 'new_branch_count': 4,
              'calibration': 'uniform prior fitted on full validation; same-set diagnostic'}
    if gate:
        fold_ids = torch.tensor([int.from_bytes(hashlib.sha256(
            (str(cfg['fold_seed'])+':'+r['content_group']).encode()).digest()[:8], 'big')%cfg['folds']
            for r in rows], device=device)
        cf = {}
        for name, logits in [('baseline', baseline), ('threecrop', candidate)]:
            logits_cf = torch.empty_like(logits)
            for fold in range(cfg['folds']):
                mask = fold_ids == fold
                bias, _ = fit_prior_bias(logits[~mask], max_iterations=50)
                logits_cf[mask] = apply_prior_bias(logits[mask], bias, strength=cfg['prior_strength'])
            cf[name] = metrics(logits_cf, labels)
        cf['delta_pp'] = {k:100*(cf['threecrop'][k]-cf['baseline'][k]) for k in ('macro','micro')}
        cf['fold_sizes'] = torch.bincount(fold_ids).tolist()
        cf['fold_sha256'] = hashlib.sha256(fold_ids.cpu().numpy().tobytes()).hexdigest()
        cf['limitation'] = 'Conditional prior cross-fit only; checkpoint and T/strength previously selected on whole validation'
        result['conditional_crossfit'] = cf
        result['promotion_pass'] = cf['delta_pp']['macro'] >= cfg['crossfit_gate_macro_pp'] and cf['delta_pp']['micro'] >= 0
    else:
        result['promotion_pass'] = False
    result['branch_cache_sha256'] = {p.name:sha256_file(p) for p in branch_files}
    result['native_replay'] = json.loads((out/'native_replay.json').read_text())
    write_json(out/'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg['protocol'] != 'l05_threecrop_v1' or cfg['crop_size'] != 384 or cfg['inference_batch_size'] != 32:
        raise ValueError('Unregistered protocol or replay geometry')
    if (cfg['temperature'], cfg['prior_strength'], cfg['fold_seed'], cfg['folds']) != (1.4, 0.6, 42, 3):
        raise ValueError('Fixed calibration recipe must not drift')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    rows, reference, identity = preflight(cfg)
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists():
        raise FileExistsError('Completed result exists; refusing to overwrite')
    write_json(out/'implementation.json', {'script_sha256':sha256_file(__file__), 'config_sha256':sha256_file(args.config),
               'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               'torch':torch.__version__, 'command':os.sys.argv, 'pid':os.getpid()})
    model, native, _ = build_from_checkpoint(cfg['parent_checkpoint'], torch.device('cuda:0'))
    model.eval().requires_grad_(False)
    if model.input_resolution != cfg['crop_size']:
        raise ValueError('Parent input resolution changed')
    files = []
    started = time.monotonic()
    for offset in range(0, len(rows['val']), 1024):
        selected = rows['val'][offset:offset+1024]
        file = out/f'val_edges_{offset:06d}.pt'; files.append(file)
        metadata = {**identity, 'offset':offset, 'paths':[r['image_path'] for r in selected]}
        if file.exists():
            payload = torch.load(file,map_location='cpu',weights_only=False)
            if payload['identity'] != metadata or len(payload['branches']) != 4:
                raise ValueError('Existing branch cache identity mismatch')
            if any(b.shape != (len(selected), model.num_classes) or not torch.isfinite(b).all() for b in payload['branches']):
                raise ValueError('Invalid branch cache')
            continue
        loader = DataLoader(ThreeCropImages(selected,cfg['image_root'],native,cfg['crop_size'],offset),
                            batch_size=32, num_workers=cfg['workers'], shuffle=False, pin_memory=True)
        branches = [[], [], [], []]
        with torch.inference_mode():
            for batch, (edges, center) in enumerate(loader):
                edges = edges.cuda(non_blocking=True)
                with torch.autocast('cuda'):
                    if offset==0 and batch==0:
                        center = center.cuda()
                        replay=[]
                        for tensor,key in [(center,'original_logits'),(center.flip(3),'flip_logits')]:
                            logits=model(images=tensor).float().cpu(); ref=reference[key][:len(logits)]
                            error=float((logits-ref).abs().max()); agree=bool(torch.equal(logits.argmax(1),ref.argmax(1)))
                            if error>.02 or not agree:
                                raise ValueError('Native center replay failed')
                            replay.append({'branch':key,'max_abs':error,'top1_identical':agree})
                        write_json(out/'native_replay.json',replay); print(json.dumps(replay),flush=True)
                    for i,view in enumerate([edges[:,0], edges[:,0].flip(3), edges[:,1], edges[:,1].flip(3)]):
                        branches[i].append(model(images=view).float().cpu())
        _atomic_torch_save({'identity':metadata, 'branches':[torch.cat(p) for p in branches]},file)
        print(f'cached {offset+len(selected)}/{len(rows["val"])}; elapsed={time.monotonic()-started:.1f}s',flush=True)
    del model
    torch.cuda.empty_cache()
    evaluate(cfg,rows['val'],reference,identity,files)
    write_json(out/'timing.json',{'total_seconds':time.monotonic()-started})


if __name__=='__main__':
    main()
