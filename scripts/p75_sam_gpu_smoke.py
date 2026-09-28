#!/usr/bin/env python3
"""Bounded real-data SAM engineering check, never a candidate training run."""
import json
from pathlib import Path
import sys
import time

import torch
import yaml

from run_p75_full_sam import CONFIG, FRAMEWORK, OUT, ROOT, preflight


def main():
    preflight()
    sys.path[:0] = [str(ROOT/'scripts/p75_runtime'),
                   str(FRAMEWORK/'reproducibility/aegis_f1')]
    from aegis_clip.config import load_config
    from aegis_clip.trainer import train
    import p75_sam

    config = yaml.safe_load(CONFIG.read_text())
    config['project']['experiment_id'] = 'P75_FULL_SAM_ENGINEERING_SMOKE'
    # Exercise local loss immediately and on all samples, including the
    # optimizer-state allocation on the second effective batch.
    config['loss']['attention_local_training']['start_epoch'] = 1
    config['loss']['attention_local_training']['confidence_gate'] = 0.0
    smoke_config = OUT/'configs/P75_FULL_SAM_ENGINEERING_SMOKE.yaml'
    smoke_config.write_text(yaml.safe_dump(config, sort_keys=False))
    report = OUT/'sam_gpu_smoke.json'
    if report.exists():
        raise FileExistsError(report)
    original = p75_sam.full_l05_sam_step
    updates = []
    started = time.monotonic()

    class Finished(Exception):
        pass

    def bounded(*args, **kwargs):
        assert kwargs['cycle_samples'] == 1024
        assert all(m['local_attention'] is not None for m in kwargs['microbatches'])
        result = original(*args, **kwargs)
        updates.append(result)
        if len(updates) == 2:
            torch.cuda.synchronize()
            report.write_text(json.dumps({
                'status': 'passed', 'effective_batch': 1024,
                'local_path_active': True, 'updates': updates,
                'amp': False, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'seconds': time.monotonic()-started,
                'candidate_checkpoint_created': False,
            }, indent=2)+'\n')
            raise Finished()
        return result

    p75_sam.full_l05_sam_step = bounded
    torch.set_num_threads(4)
    try:
        train(load_config(smoke_config))
    except Finished:
        print(report.read_text())
    else:
        raise RuntimeError('Smoke did not reach two effective-batch updates')


if __name__ == '__main__':
    main()
