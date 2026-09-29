#!/usr/bin/env python3
"""User-authorized epoch-4 evaluation/delivery after supervisor interruption.

Never trains, changes frozen inputs, resets the 21600-second budget, or retries.
Interrupted task exit code remains unknown; charge through reconciliation time.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time

from run_p75_semantic_pair import (
    ROOT, OUT, CONFIGS, BUDGET, run_dir, write, verify, execute, remaining, sha,
)
from p75_semantic_pair_report import report, training_audit


def close():
    verify()
    state = json.loads((OUT / 'status.json').read_text())
    entry = state.get('current')
    if state['status'] != 'running' or not entry or entry['name'] != 'masked_train_to_4':
        raise RuntimeError('Only the audited interrupted masked-to-4 state is accepted')
    if Path(f"/proc/{entry['pid']}").exists():
        raise RuntimeError('Recorded training child is still alive; do not interfere')
    training_audit(4)
    for arm in ('control', 'masked'):
        checkpoint = run_dir(arm) / 'checkpoints/epoch_4.pt'
        binding = json.loads(checkpoint.with_suffix('.binding.json').read_text())
        if binding['epoch'] != 4 or sha(checkpoint) != binding['checkpoint_sha256']:
            raise ValueError(f'{arm} epoch-4 checkpoint binding failed')
    # Stop only this campaign's stale CPU finalizer before completing its status.
    finalizer_pid = int((OUT / 'finalizer.pid').read_text())
    cmdline = Path(f'/proc/{finalizer_pid}/cmdline')
    if cmdline.exists():
        command = cmdline.read_bytes().split(b'\0')
        if b'scripts/finalize_p75_semantic_pair.py' not in command:
            raise RuntimeError('Finalizer PID was reused; no signal sent')
        os.kill(finalizer_pid, signal.SIGTERM)
    with (OUT / 'interrupted_status.json').open('x') as stream:
        json.dump(state, stream, indent=2)
        stream.write('\n')
    reconciled_at = time.time()
    charged = reconciled_at - entry['start_unix']
    if charged < 0:
        raise ValueError('Invalid wall clock')
    entry.update(seconds=charged, returncode=None, recovered_after_supervisor_exit=True,
                 completion_evidence='Both epoch-4 bindings, complete order/update logs, '
                     'masked stdout checkpoint path and artifact manifest; exit code unknown',
                 time_accounting='Conservative charge from child start through reconciliation, '
                     'including idle interruption gap; budget never reset')
    state['used_seconds'] += charged
    state['history'].append(entry)
    state.update(current=None, matched_epoch=4,
                 closure_reason='user_requested_epoch4_evaluation_and_delivery_after_interrupt',
                 primary_complete=False, reconciliation_unix=reconciled_at,
                 closure_source_sha256=sha(Path(__file__)))
    if remaining(state) <= 0:
        state['status'] = 'budget_incomplete_validation'
        write(OUT / 'status.json', state)
        raise RuntimeError('No remaining original budget')
    write(OUT / 'status.json', state)
    print(json.dumps(dict(reconciled=True, used_seconds=state['used_seconds'],
                          remaining_seconds=remaining(state), budget_seconds=BUDGET)), flush=True)
    try:
        for arm in ('control', 'masked'):
            checkpoint = run_dir(arm) / 'checkpoints/epoch_4.pt'
            cache = run_dir(arm) / 'val_epoch_4.pt'
            if cache.exists():
                raise FileExistsError('Do not overwrite or silently reuse a cache')
            args = ['scripts/cache_validation_tta_logits.py', '--checkpoint', str(checkpoint),
                    '--config', str(CONFIGS / f'{arm}.json'), '--output', str(cache),
                    '--tta-temperature', '1.4', '--device', 'cuda:0',
                    '--batch-size', '32', '--num-workers', '2']
            if not execute(args, f'{arm}_cache_4', state, remaining(state)):
                state['status'] = 'budget_incomplete_validation'
                return
        report(OUT / 'e4_diagnostic_report')
        args = ['scripts/build_l05_tta_prior_submission_final.py', '--checkpoint',
                str(run_dir('masked') / 'checkpoints/epoch_4.pt'), '--config',
                str(CONFIGS / 'masked.json'), '--val-branch-cache',
                str(run_dir('masked') / 'val_epoch_4.pt'), '--temperature', '1.4',
                '--prior-strength', '.6', '--fusion', 'mean_probabilities',
                '--tta', 'horizontal_flip', '--tag', 'P75_SEMANTIC_MASKED_E4',
                '--output-root', str(OUT / 'deliveries'), '--skip-desktop-copy',
                '--device', 'cuda:0', '--batch-size', '32', '--num-workers', '2']
        delivered = execute(args, 'masked_deliver_4', state, remaining(state))
        state['status'] = ('epoch4_only_primary_incomplete' if delivered
                           else 'budget_incomplete_delivery')
    except Exception as exc:
        state.update(status='failed_epoch4_closure', error=repr(exc))
        raise
    finally:
        state['ended_unix'] = time.time()
        write(OUT / 'status.json', state)


if __name__ == '__main__':
    close()
