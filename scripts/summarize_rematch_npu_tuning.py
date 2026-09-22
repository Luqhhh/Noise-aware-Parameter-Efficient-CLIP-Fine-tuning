"""Summarize saved NPU probes and two-epoch acceptance, without running training."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    selection = read('results/npu_tuning/selection.json')
    acceptance = read('results/rematch750_npu_tuned_acceptance.json')
    assert selection['status'] == acceptance['status'] == 'passed'
    assert selection['user_selected_gce_status'] == 'passed'
    assert selection['user_selected']['batch'] == 1024
    rows = []
    for path in sorted((ROOT / 'results/npu_tuning').glob('*.json')):
        result = json.loads(path.read_text())
        if 'arguments' not in result:
            continue
        args = result['arguments']
        row = {k: args.get(k) for k in ('name', 'batch', 'workers', 'prefetch', 'threads',
                                       'optimizer', 'norm', 'production', 'pin_memory',
                                       'loss_phase', 'warmup', 'steps', 'profile')}
        row.update({k: result.get(k) for k in ('status', 'images_per_second', 'mean_step_seconds',
                                              'mean_loader_wait_seconds', 'peak_allocated_gib', 'error')})
        rows.append(row)
    with (ROOT / 'results/rematch750_npu_tuning_cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    run = ROOT / 'outputs/npu_tuning/RM_FT_NPU_TUNED_E2/seed42'
    progress = []
    first = selected = None
    for line in (run / 'logs/train.log').read_text().splitlines():
        timestamp = datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S,%f')
        if 'First-step audit passed' in line:
            first = timestamp
        if 'Selected checkpoint at epoch 2' in line:
            selected = timestamp
        match = re.search(r'Progress epoch=(\d+)/\d+ step=(\d+)/', line)
        if match:
            progress.append((int(match[1]), int(match[2]), timestamp))
    intervals = [(b[2]-a[2]).total_seconds()/(b[1]-a[1])
                 for a,b in zip(progress, progress[1:]) if a[0] == b[0]]
    assert len(intervals) == 39 and first and selected
    timing = dict(batch_size=32, interval_count=len(intervals),
                  median_seconds_per_step=statistics.median(intervals),
                  mean_seconds_per_step=statistics.mean(intervals),
                  first_step_to_epoch2_saved_seconds=(selected-first).total_seconds())
    baseline = read('results/rematch750_npu_throughput.json')
    binding = json.loads((run / 'submission/registry_binding.json').read_text())
    for filename, key in [('pred_results.csv', 'csv_sha256'), ('submission.zip', 'zip_sha256')]:
        assert hashlib.sha256((run/'submission'/filename).read_bytes()).hexdigest() == binding[key]
    checkpoint = run / 'checkpoints/best.pt'
    digest = hashlib.sha256()
    with checkpoint.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == binding['checkpoint_sha256']
    report = dict(status='passed', source_commit=read('results/rematch750_npu_tuning_source_manifest.json')['source_commit'],
                  remote_snapshot_commit=binding['code_commit'],
                  probe_count=len(rows), passed_probes=sum(r['status']=='passed' for r in rows),
                  failed_probes=[r['name'] for r in rows if r['status']!='passed'],
                  recipe_preserving_config='configs/rematch750_ft_npu_tuned.yaml',
                  throughput_only_config='configs/rematch750_ft_npu_throughput.yaml',
                  selection=selection, formal_timing=timing,
                  original_npu_to_tuned_speedup=baseline['npu']['median_seconds_per_step']/timing['median_seconds_per_step'],
                  original_gpu_to_tuned_speedup=baseline['gpu']['median_seconds_per_step']/timing['median_seconds_per_step'],
                  acceptance=acceptance, submission=binding,
                  local_submission_validation='outputs/npu_tuning/local_submission_check.log',
                  hardware_validation={'optimizer_clip_amp_restore':'6 passed in 50.99s',
                                       'loader_pinned_and_unpinned':'2 passed in 9.21s; overlaps one earlier case'},
                  local_tests={'passed':662, 'skipped_hardware':7, 'known_scope_asset_failures':2},
                  limitations=['Single machine/device, bounded empirical search; not a global optimum.',
                               'Large batch changes updates per epoch; no convergence or accuracy claim.',
                               'Formal acceptance covers two CE epochs; GCE has bounded throughput/update checks only.',
                               'No platform upload or score; full eight-epoch accuracy not reproduced.',
                               'Hardware/CPU placement differs from local GPU; this is a measured system comparison.'])
    (ROOT / 'results/rematch750_npu_tuning_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('acceptance','submission','selection')},indent=2))


if __name__ == '__main__':
    main()
