#!/usr/bin/env python3
"""One immutable final-L05 snapshot: train_dev center/flip raw probabilities."""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset

from p75_supported_ce import (FIXED, OUT, canonical, preflight, read_json, read_rows,
                              sha, write_json, write_rows, ROW_FIELDS)


class SnapshotDataset(Dataset):
    """Only explicit training rows; verify bytes bound to audited RGB groups."""
    def __init__(self, training, root, preprocess):
        self.rows, self.root, self.preprocess = training, Path(root), preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root/canonical(row['image_path'])
        if self.root.resolve() not in path.resolve().parents:
            raise ValueError('Training image escaped the fixed root')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['file_sha256']:
            raise ValueError(f'Training image bytes changed: {path}')
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert('RGB')
            # Group hash encoding is already audited by the stage split. Byte
            # verification binds that audited group; do not invent a new hash.
            tensor = self.preprocess(image)
        if tensor.shape != (3, 384, 384) or not torch.isfinite(tensor).all():
            raise ValueError('Expected finite deterministic 384px center preprocessing')
        return tensor, index


@torch.no_grad()
def export_snapshot(batch_size=32, num_workers=2):
    saved = preflight()
    fixed = read_json(FIXED)
    original = read_json(OUT/'configs/l05_original.json')
    training = read_rows(Path(fixed['assets'])/'train_dev.csv')
    destination = OUT/'snapshot'
    # Refuse both completed snapshots and partial exports. No teacher training,
    # source overwrites, resume with another model, or repeated snapshot grid.
    destination.mkdir(exist_ok=False)
    from aegis_clip.checkpoint import build_from_checkpoint
    from aegis_clip.rematch_protocol import validate_checkpoint
    device = torch.device(original['train']['device'])
    validate_checkpoint(fixed['control_checkpoint'], original, parent=False)
    model, preprocess, checkpoint = build_from_checkpoint(
        fixed['control_checkpoint'], device, config_override=original)
    epoch = int(checkpoint['epoch'])
    del checkpoint
    model.eval()
    model.requires_grad_(False)
    dataset = SnapshotDataset(training, original['data']['train_root'], preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        timeout=int(original['train'].get('loader_timeout', 120)) if num_workers else 0,
        pin_memory=device.type == 'cuda', persistent_workers=num_workers > 0)
    shape = (len(training), saved['identity']['num_classes'])
    first = np.lib.format.open_memmap(destination/'center_probabilities.npy',
                                    mode='w+', dtype=np.float32, shape=shape)
    second = np.lib.format.open_memmap(destination/'flip_probabilities.npy',
                                     mode='w+', dtype=np.float32, shape=shape)
    offset = 0
    for images, indices in loader:
        expected = torch.arange(offset, offset+len(images))
        if not torch.equal(indices, expected):
            raise ValueError('Snapshot loader training row order changed')
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type,
                            enabled=bool(original['train']['amp']) and device.type == 'cuda'):
            logits_center = model(images=images)
            logits_flip = model(images=torch.flip(images, dims=(3,)))
        # No fusion, prior, temperature scaling, or local branch inference.
        center = logits_center.float().softmax(1).cpu().numpy()
        flip = logits_flip.float().softmax(1).cpu().numpy()
        if not np.isfinite(center).all() or not np.isfinite(flip).all():
            raise ValueError('Nonfinite L05 training snapshot')
        end = offset+len(images)
        first[offset:end], second[offset:end] = center, flip
        offset = end
        if offset % (batch_size*100) == 0:
            print(f'L05 snapshot {offset}/{len(training)}', flush=True)
    if offset != len(training):
        raise ValueError('Incomplete training snapshot')
    first.flush()
    second.flush()
    del first, second
    write_rows(destination/'rows.csv', ROW_FIELDS,
               ({key: row[key] for key in ROW_FIELDS} for row in training))
    if sha(fixed['control_checkpoint']) != saved['identity']['control_sha256']:
        raise ValueError('L05 checkpoint changed during snapshot export')
    manifest = dict(format_version=1, identity=saved['identity'],
        checkpoint=str(fixed['control_checkpoint']), checkpoint_epoch=epoch,
        temperature=1., prior_applied=False, tta_fusion_applied=False,
        views=['center_384', 'horizontal_flip_384'],
        preprocessing=re.sub(r'0x[0-9a-fA-F]+', '<address>', repr(preprocess)),
        amp=bool(original['train']['amp']), device=str(device),
        source='final L05 state; not epoch history or evidence of persistent suppression',
        validation_images_read=False, test_images_read=False,
        files={name: sha(destination/name) for name in
               ('center_probabilities.npy', 'flip_probabilities.npy', 'rows.csv')})
    write_json(destination/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-gpu', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    if not args.execute_gpu:
        raise ValueError('Snapshot export requires explicit --execute-gpu')
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError('Invalid snapshot loader dimensions')
    from run_p75_supported_ce import ensure_idle_device
    ensure_idle_device()
    print(export_snapshot(args.batch_size, args.num_workers)['source'])


if __name__ == '__main__':
    raise SystemExit(main())
