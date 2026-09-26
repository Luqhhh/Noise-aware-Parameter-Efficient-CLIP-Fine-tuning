#!/usr/bin/env python3
"""Fixed full-frame rectangular inference through the existing Aegis model."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from l05_rectangular_support import rectangle_size, rectangular_logits
from run_l05_nonlinear_head import preflight, metrics, write_json, Images, fixed_decode
from aegis_clip.checkpoint import build_from_checkpoint, _atomic_torch_save
from aegis_clip.runtime import sha256_file
from aegis_clip.tta import fuse_paired_logits
from aegis_clip.prior_alignment import fit_prior_bias, apply_prior_bias
from audit_l05_mixstyle_decode import numpy_decode


class FullFrameImages(Dataset):
    def __init__(self, rows, root, finish):
        self.rows, self.root, self.finish = rows, Path(root).resolve(), finish

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row['image_path'].removeprefix('train/')).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Image escapes declared root')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['file_sha256']:
            raise ValueError('Image hash mismatch')
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (int(row['width']), int(row['height'])):
                raise ValueError('Declared image geometry mismatch')
            image = image.convert('RGB')
            image = image.resize(rectangle_size(*image.size), Image.Resampling.BICUBIC)
            tensor = self.finish(image)
        return index, tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    assert (cfg['protocol'], cfg['short_edge'], cfg['long_cap'], cfg['patch_size'], cfg['batch_size']) == (
        'l05_rectangular_fullframe_v1', 384, 576, 32, 32)
    assert (cfg['temperature'], cfg['prior_strength'], cfg['fold_seed'], cfg['folds']) == (1.4, .6, 42, 3)
    assert cfg['autocast'] is True and cfg['autocast_dtype'] == 'float16'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    rows, reference, identity = preflight(cfg)
    rows = rows['val']
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    if (out/'implementation.json').exists():
        raise FileExistsError('Preserve the existing run; do not restart a live process')
    write_json(out/'implementation.json', {'config_sha256': sha256_file(args.config),
        'script_sha256': sha256_file(__file__), 'support_sha256': sha256_file(Path(__file__).with_name('l05_rectangular_support.py')),
        'model_source_sha256': sha256_file(Path(__file__).resolve().parents[1]/'reproducibility/aegis_f1/aegis_clip/model.py'),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'torch': torch.__version__, 'autocast': True, 'autocast_dtype': 'float16',
        'command': os.sys.argv, 'pid': os.getpid()})
    model, native, checkpoint = build_from_checkpoint(cfg['parent_checkpoint'], torch.device('cuda:0'))
    model.eval().requires_grad_(False)
    assert isinstance(model.feature_adapter, torch.nn.Identity)
    assert isinstance(model.classifier, torch.nn.Linear) and model.input_resolution == 384
    original_state = checkpoint['model_state_dict']
    loader = DataLoader(Images(rows[:32], cfg['image_root'], native), batch_size=32, num_workers=2)
    square = next(iter(loader)).cuda()
    replay = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for flip, key in ((False, 'original_logits'), (True, 'flip_logits')):
            images = square.flip(3) if flip else square
            logits = model(images=images)
            independent = rectangular_logits(model, images)
            assert torch.equal(logits, independent), 'Square native/independent forward mismatch'
            expected = reference[key][:32].cuda()
            error = float((logits-expected).abs().max())
            assert error <= 1e-4 and torch.equal(logits.argmax(1), expected.argmax(1))
            replay.append({'branch': key, 'native_reference_max_abs': error, 'independent_forward_max_abs': 0.})
    write_json(out/'native_replay.json', replay)
    del square, loader
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        buckets[rectangle_size(int(row['width']), int(row['height']))].append(i)
    batches = [indices[offset:offset+32] for size, indices in sorted(buckets.items())
               for offset in range(0, len(indices), 32)]
    dataset = FullFrameImages(rows, cfg['image_root'], Compose(native.transforms[2:]))
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=cfg['workers'], pin_memory=True)
    original = torch.empty(len(rows), model.num_classes)
    flipped = torch.empty_like(original)
    seen = torch.zeros(len(rows), dtype=torch.bool)
    geometry_audit = {}
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for indices, images in loader:
            assert not seen[indices].any()
            images = images.cuda(non_blocking=True)
            logits = model(images=images)
            flip_logits = model(images=images.flip(3))
            assert torch.isfinite(logits).all() and torch.isfinite(flip_logits).all()
            shape = 'x'.join(map(str, images.shape[-2:]))
            if shape not in geometry_audit:
                independent = rectangular_logits(model, images)
                difference = float((logits-independent).abs().max())
                assert difference <= 1e-5 and torch.equal(logits.argmax(1), independent.argmax(1))
                geometry_audit[shape] = {'max_abs': difference, 'samples': len(images)}
            original[indices], flipped[indices] = logits.cpu(), flip_logits.cpu()
            seen[indices] = True
    assert seen.all()
    inference_seconds = time.monotonic() - started
    assert all(torch.equal(value.cpu(), original_state[key]) for key, value in model.state_dict().items())
    assert sha256_file(cfg['parent_checkpoint']) == cfg['parent_sha256']
    del model, checkpoint
    labels = reference['labels'].long()
    base, _, base_pred, _ = fixed_decode(reference['original_logits'], reference['flip_logits'], labels, cfg)
    candidate, bias, prediction, prior_fit = fixed_decode(original, flipped, labels, cfg)
    independent, numpy_bias, iterations = numpy_decode(original.numpy().astype(np.float64), flipped.numpy().astype(np.float64))
    assert np.array_equal(independent, prediction.numpy())
    delta = {k: 100*(candidate[k]-base[k]) for k in ('macro', 'micro')}
    passed = delta['macro'] >= .30 and delta['micro'] >= 0
    result = {'experiment_id': cfg['experiment_id'], 'identity': identity, 'config': cfg,
        'baseline_decode': base, 'candidate_decode': candidate, 'delta_pp': delta,
        'center_without_prior': metrics(original, labels),
        'prior_fit': prior_fit, 'native_replay': replay, 'geometry_audit': geometry_audit,
        'geometry_counts': {str(size): len(indices) for size, indices in sorted(buckets.items())},
        'validation_samples': len(rows), 'model_state_bitwise_unchanged': True,
        'parent_sha256_unchanged': True, 'test_used': False,
        'preliminary_gate_pass': passed, 'promotion_pass': False,
        'independent_decoded_predictions_identical': len(labels),
        'independent_bias_max_difference': float(np.max(np.abs(numpy_bias-bias.numpy()))),
        'independent_prior_iterations': iterations,
        'corrections': int(((prediction==labels)&(base_pred!=labels)).sum()),
        'regressions': int(((prediction!=labels)&(base_pred==labels)).sum()),
        'inference_seconds': inference_seconds,
        'limitation': 'Same-validation prior fit; independent NumPy decode is not a second full-image inference pass'}
    if passed:
        fold_ids = torch.tensor([int.from_bytes(hashlib.sha256(
            ('42:'+r['content_group']).encode()).digest()[:8], 'big') % 3 for r in rows])
        crossfit = {}
        for name, first, second in [('baseline',reference['original_logits'],reference['flip_logits']),
                                     ('candidate',original,flipped)]:
            fused = fuse_paired_logits(first, second, mode='mean_probabilities', temperature=1.4)
            adjusted = torch.empty_like(fused)
            for fold in range(3):
                mask = fold_ids==fold
                fold_bias, _ = fit_prior_bias(fused[~mask])
                adjusted[mask] = apply_prior_bias(fused[mask], fold_bias, strength=.6)
            crossfit[name] = metrics(adjusted, labels)
        crossfit['delta_pp'] = {k:100*(crossfit['candidate'][k]-crossfit['baseline'][k]) for k in ('macro','micro')}
        crossfit['fold_sizes'] = torch.bincount(fold_ids).tolist()
        crossfit['limitation'] = 'Conditional calibration cross-fit; model and recipe previously selected on whole validation'
        result['conditional_crossfit'] = crossfit
        result['promotion_pass'] = crossfit['delta_pp']['macro'] >= .20 and crossfit['delta_pp']['micro'] >= 0
    _atomic_torch_save({'original_logits': original, 'flip_logits': flipped, 'labels': labels,
        'prior_bias': bias, 'identity': identity, 'config': cfg}, out/'val_logits.pt')
    result['cache_sha256'] = sha256_file(out/'val_logits.pt')
    write_json(out/'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
