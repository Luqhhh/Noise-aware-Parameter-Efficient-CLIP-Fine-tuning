"""Summarize completed recovery logs; unavailable peak measurements stay null."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(run_dir):
    root = Path(run_dir)
    plan_path = root/'recovery_plan.json'
    plan = json.loads(plan_path.read_text())
    rows = []
    for node in plan['nodes']:
        name = node['node_id']
        directory = root/'recovery_models'/name/'seed42'
        log = root/('e2_recovery.log' if node['parent'] is None else name+'.log')
        text = log.read_text(errors='replace')
        stamps = re.findall(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})', text, re.M)
        if not stamps:
            raise ValueError('Missing timestamped training log')
        epochs = [int(e) for e in re.findall(r'\| INFO \| Epoch (\d+) \|', text)]
        ledger_files = sorted((directory/'logs').glob('longtail_epoch_*.json'))
        ledger_epochs = {int(p.stem.rsplit('_',1)[1]) for p in ledger_files}
        if ledger_epochs != set(epochs) or len(epochs) != len(set(epochs)):
            raise ValueError('Training epochs and supervision ledgers disagree')
        ledgers = [json.loads(p.read_text()) for p in ledger_files]
        files = [p for p in directory.rglob('*') if p.is_file() and not p.is_symlink()]
        start, end = [dt.datetime.strptime(s,'%Y-%m-%d %H:%M:%S,%f') for s in (stamps[0],stamps[-1])]
        rows.append(dict(node_id=name,registered_max_epochs=node['epochs'],completed_epochs=len(epochs),
            first_log_time=stamps[0],last_log_time=stamps[-1],logged_span_seconds=(end-start).total_seconds(),
            observed_training_batches=sum(x['optimizer_steps'] for x in ledgers),
            actual_sample_draws=sum(x['actual_draws'] for x in ledgers),
            current_output_bytes=sum(p.stat().st_size for p in files),
            log_sha256=digest(log),ledger_sha256={p.name:digest(p) for p in ledger_files},
            peak_gpu_bytes=None,peak_ram_bytes=None,peak_disk_bytes=None))
    return dict(schema_version=1,plan_sha256=digest(plan_path),nodes=rows,
        total_logged_span_seconds=sum(x['logged_span_seconds'] for x in rows),
        total_observed_training_batches=sum(x['observed_training_batches'] for x in rows),
        total_sample_draws=sum(x['actual_sample_draws'] for x in rows),
        limits='Log spans exclude setup before the first record and teardown after the last record. Batch counts do not prove GradScaler updates were never skipped. Current disk size is not historical peak. CPU/GPU peaks were not instrumented; no synthetic throughput extrapolation.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=summarize(args.run_dir)
    with Path(args.output).open('x') as stream:
        json.dump(report,stream,indent=2)


if __name__=='__main__':main()
