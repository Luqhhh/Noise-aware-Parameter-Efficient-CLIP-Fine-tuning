"""Audit a completed fixed eight-epoch REMATCH750_V2 run; never train or upload."""
import argparse
import csv
import json
from pathlib import Path

import torch

from aegis_clip.config import load_config
from aegis_clip.rematch_assets import validate_checkpoint
from aegis_clip.runtime import sha256_file


def audit(config_path):
    config = load_config(config_path)
    run = Path(config['output']['root']) / config['project']['experiment_id'] / f"seed{config['project']['seed']}"
    with open(config['data']['train_csv']) as f:
        count = sum(1 for _ in csv.DictReader(f))
    batch = config['train']['batch_size']
    steps = (count + batch - 1) // batch
    assert config['train']['epochs'] == config['train']['schedule_epochs'] == 8
    curves = []
    for epoch in (2, 4, 6, 8):
        m = json.loads((run / f'logs/evaluation_epoch_{epoch}.json').read_text())
        assert m['successful_optimizer_updates'] == m['attempted_optimizer_updates'] == epoch * steps
        curves.append(dict(epoch=epoch, raw_macro=m['raw_macro'], raw_micro=m['raw_micro'],
                           successful_optimizer_updates=m['successful_optimizer_updates'],
                           cumulative_training_seconds=m['cumulative_training_seconds']))
    selected = max(curves, key=lambda m: (m['raw_macro'], m['raw_micro']))
    records = {}
    for name in ('best', 'last'):
        path = run / f'checkpoints/{name}.pt'
        validate_checkpoint(path, config)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        epoch = ckpt['epoch']
        expected = epoch * steps
        states = sorted({int(v['step']) for v in ckpt['optimizer_state_dict']['state'].values() if 'step' in v})
        assert states == [expected]
        assert ckpt['global_step'] == ckpt['scheduler_state_dict']['last_epoch'] == expected
        assert all(bool(torch.isfinite(v).all()) for v in ckpt['model_state_dict'].values())
        assert all(bool(torch.isfinite(v).all())
                   for state in ckpt['optimizer_state_dict']['state'].values()
                   for v in state.values() if isinstance(v, torch.Tensor))
        if name == 'best':
            selected_per_class = ckpt['metrics']['per_class']
        assert epoch == (selected['epoch'] if name == 'best' else 8)
        records[name] = dict(epoch=epoch, global_step=ckpt['global_step'], optimizer_steps=states,
                             sha256=sha256_file(path), raw_macro=ckpt['metrics']['raw_macro'],
                             raw_micro=ckpt['metrics']['raw_micro'])
        del ckpt
    reloaded = json.loads((run / 'checkpoints/best_evaluation.json').read_text())
    assert all(reloaded[k] == records['best'][k] for k in ('raw_macro', 'raw_micro'))
    assert reloaded['per_class'] == selected_per_class
    per_class = reloaded['per_class']
    tail = sorted(per_class, key=lambda r: (r['train_samples'], r['label']))[:75]
    tail_ids = {r['label'] for r in tail}
    def macro(rows):
        return sum(r['recall'] for r in rows) / len(rows)
    progress = [json.loads(line) for line in (run / 'logs/progress.jsonl').read_text().splitlines()]
    assert all(p['successful_optimizer_updates'] == p['global_step'] for p in progress)
    report = dict(candidate=config['project']['experiment_id'], status='passed',
                  config_sha256=sha256_file(config_path), init_checkpoint_sha256=sha256_file(config['train']['init_checkpoint']),
                  train_csv_sha256=sha256_file(config['data']['train_csv']), val_csv_sha256=sha256_file(config['data']['val_csv']),
                  curves=curves, checkpoints=records, reload_metrics_exact=True, reload_per_class_exact=True, optimizer_tensors_finite=True,
                  selected_epoch=selected['epoch'], raw_macro=reloaded['raw_macro'], raw_micro=reloaded['raw_micro'],
                  fixed_tail75_macro=macro(tail), rest675_macro=macro([r for r in per_class if r['label'] not in tail_ids]),
                  fixed_tail75_class_ids=sorted(tail_ids), successful_optimizer_updates=8*steps,
                  training_seconds=curves[-1]['cumulative_training_seconds'],
                  # Same-host file times bracket CLI source snapshot through final
                  # re-evaluation/per-class export, including validation and saves.
                  train_cli_seconds=(run / 'checkpoints/selected_report.json').stat().st_mtime
                      - (run.parent / 'training_code_manifest.json').stat().st_mtime,
                  platform_score=None)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = audit(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
