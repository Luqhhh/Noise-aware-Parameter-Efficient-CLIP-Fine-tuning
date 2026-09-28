#!/usr/bin/env python3
"""Bounded final-L05 inference snapshot for P0, with original config binding."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from p75_supervision_evidence import context, OUT, ROOT
from p75_supported_ce import read_json, sha, write_json, write_rows, ROW_FIELDS
from p75_supported_ce_snapshot import SnapshotDataset

CONTROL_CONFIG = Path('/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/_runtime_configs/L05_RM_V5_L05_CUDA_LOCAL_cuda_0_mb4.yaml')


def original_config(source):
    from aegis_clip.config import load_config
    meta = read_json(Path(source['control_checkpoint']).with_suffix('.binding.json'))
    if sha(CONTROL_CONFIG) != meta['training_config_sha256']:
        raise ValueError('Original L05 YAML does not match checkpoint binding')
    config = load_config(CONTROL_CONFIG)
    # load_config restores _config_path omitted by checkpoint public_config().
    if config['_config_path'] != str(CONTROL_CONFIG):
        raise ValueError('Resolved original config path mismatch')
    return config


@torch.no_grad()
def snapshot(batch_size=32, workers=2):
    start = time.monotonic()
    rule, source, identity, train, _, _ = context()
    config = original_config(source)
    from aegis_clip.checkpoint import build_from_checkpoint
    from aegis_clip.rematch_protocol import validate_checkpoint
    validate_checkpoint(source['control_checkpoint'], config, parent=False)
    from run_p75_supported_ce import ensure_idle_device
    ensure_idle_device()
    torch.set_num_threads(4)
    device=torch.device(config['train']['device'])
    model,preprocess,checkpoint=build_from_checkpoint(source['control_checkpoint'],device,config_override=config)
    epoch=int(checkpoint['epoch'])
    del checkpoint
    model.eval().requires_grad_(False)
    output=OUT/'snapshot'
    output.mkdir(parents=True,exist_ok=False)
    dataset=SnapshotDataset(train,config['data']['train_root'],preprocess)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,
        pin_memory=True,timeout=120 if workers else 0)
    shape=(len(train),identity['num_classes'])
    arrays=[np.lib.format.open_memmap(output/name,mode='w+',dtype=np.float32,shape=shape)
        for name in ('center_probabilities.npy','flip_probabilities.npy')]
    offset=0
    for images,indices in loader:
        if time.monotonic()-start > rule['snapshot_wall_seconds']:
            raise TimeoutError('P0 snapshot budget exhausted; partial arrays are not evidence')
        if not torch.equal(indices,torch.arange(offset,offset+len(images))):
            raise ValueError('Snapshot row order changed')
        images=images.to(device,non_blocking=True)
        with torch.autocast(device_type=device.type,enabled=config['train']['amp']):
            logits=(model(images=images),model(images=images.flip(3)))
        for array,value in zip(arrays,logits):
            probs=value.float().softmax(1).cpu().numpy()
            if not np.isfinite(probs).all():
                raise ValueError('Nonfinite snapshot')
            array[offset:offset+len(images)]=probs
        offset+=len(images)
        if offset % (batch_size*50)==0:
            print(f'L05 snapshot {offset}/{len(train)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    if offset!=len(train):
        raise ValueError('Incomplete snapshot')
    for array in arrays:
        array.flush()
    write_rows(output/'rows.csv',ROW_FIELDS,({k:r[k] for k in ROW_FIELDS} for r in train))
    if sha(source['control_checkpoint'])!=identity['control_sha256']:
        raise ValueError('Checkpoint changed during export')
    write_json(output/'manifest.json',dict(identity=identity,checkpoint_epoch=epoch,
        temperature=1.,prior_applied=False,tta_fusion_applied=False,
        views=['center_384','horizontal_flip_384'],amp=config['train']['amp'],
        preprocessing=re.sub(r'0x[0-9a-fA-F]+','<address>',repr(preprocess)),
        config=str(CONTROL_CONFIG),config_sha256=sha(CONTROL_CONFIG),
        seconds=time.monotonic()-start,source='final checkpoint inference; not training history',
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        code_sha256=sha(Path(__file__)),test_images_read=False,validation_images_read=False))
    print(f'Completed {output}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-gpu',action='store_true')
    a=p.parse_args()
    if not a.execute_gpu:
        p.error('--execute-gpu required')
    snapshot()
