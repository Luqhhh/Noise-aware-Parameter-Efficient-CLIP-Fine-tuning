"""Audited, explicit-path cleanup for the user-requested 2026-10-02 segment.

Plan is read-only with respect to existing artifacts. Apply requires the recorded
plan hash, rejects changed/active/tracked files, and writes a deletion ledger.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

REPO = Path('/home/lux1/noise')
RECORD = Path(__file__).resolve().parent
RUNTIME = REPO / 'outputs/cleanup/20261002'
BASE = REPO / 'worktrees'
EXTERNAL = Path('/home/lux1/noise-worktrees')
HASH_CACHE = {}


def sha(path):
    s = Path(path).stat()
    key = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if key in HASH_CACHE:
        return HASH_CACHE[key]
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def snapshot(path, digest=False):
    path = Path(path)
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode), path
    x = dict(path=str(path), size=s.st_size, allocated=s.st_blocks * 512,
             dev=s.st_dev, inode=s.st_ino, nlink=s.st_nlink, mtime_ns=s.st_mtime_ns)
    if digest:
        x['sha256'] = sha(path)
    return x


def unchanged(row, digest=False, removed_links=None):
    x = snapshot(row['path'], digest)
    for k in ('size', 'dev', 'inode', 'nlink', 'mtime_ns'):
        expected = row[k]
        if k == 'nlink' and removed_links:
            expected -= removed_links.get((row['dev'], row['inode']), 0)
        assert x[k] == expected, (row['path'], k, expected, x[k])
    if digest:
        assert x['sha256'] == row['sha256'], row['path']


def worktrees():
    out = subprocess.check_output(['git', '-C', str(REPO), 'worktree', 'list', '--porcelain'], text=True)
    return [Path(line[9:]) for line in out.splitlines() if line.startswith('worktree ')]


def artifacts(root):
    for directory, ds, fs in os.walk(root, followlinks=False):
        ds[:] = [d for d in ds if d not in {'.git', '.secrets', '.venv', '__pycache__'}
                 and not Path(directory, d).is_symlink()]
        for name in fs:
            p = Path(directory, name)
            if p.is_file() and not p.is_symlink():
                yield p


def proposed():
    rows = []

    def add(p, category, evidence, retained, equivalent=None):
        assert p.exists(), p
        assert p.suffix == '.pt' and p.resolve() == p, p
        assert all(q.exists() for q in retained), retained
        x = snapshot(p, True)
        x.update(category=category, evidence=evidence, retained=[str(q) for q in retained])
        if equivalent:
            assert sha(equivalent) == x['sha256'], (p, equivalent)
            x['retained_identical'] = str(equivalent)
        rows.append(x)

    for experiment, arms, evidence in [
        ('l05_mixstyle', ['MS00', 'MS01'], 'results/l05_mixstyle_20260926.json'),
        ('l05_logit_penalty', ['SD01'], 'results/l05_logit_penalty_20260927.json'),
        ('l05_rsc', ['RC01'], 'results/l05_rsc_20260927.json'),
        ('l05_fourier_mix', ['FM01'], 'results/l05_fourier_mix_20260927.json'),
    ]:
        for arm in arms:
            run = EXTERNAL / experiment / 'outputs/codex' / experiment / 'runs' / arm / 'seed42'
            for epoch in (1, 2):
                add(run / 'checkpoints' / f'epoch_{epoch}.pt', 'closed_l05_epoch_snapshot', evidence,
                    [run / 'checkpoints/best.pt', run / 'checkpoints/last.pt', run / 'val_branch_logits.pt'])

    e4 = EXTERNAL / 'p75_semantic_mask_pair_20260929/outputs/codex/p75_semantic_mask_pair_20260929/runs'
    for arm in ('P75_SEMANTIC_CONTROL', 'P75_SEMANTIC_MASKED'):
        run = e4 / arm / 'seed42'
        add(run / 'checkpoints/epoch_2.pt', 'superseded_p75_e2_snapshot',
            'results/p75_mask_e6_execution_20260929/completion.json',
            [run / 'checkpoints/epoch_4.pt', run / 'checkpoints/best.pt', run / 'checkpoints/last.pt'])

    e6 = BASE / 'p75_mask_e6_execution_20260929/outputs/codex/p75_mask_e6_execution_20260929/A'
    for p in sorted((e6 / 'cost_probe').rglob('*.pt')):
        add(p, 'completed_cost_probe_checkpoint', 'results/p75_mask_e6_execution_20260929/cost_probe.json',
            [e6 / 'runs/P75_MASK_E6_CONTROL/seed42/checkpoints/epoch_6.pt',
             e6 / 'runs/P75_MASK_E6_MASKED/seed42/checkpoints/epoch_6.pt'])
    for arm in ('P75_MASK_E6_CONTROL', 'P75_MASK_E6_MASKED'):
        run = e6 / 'runs' / arm / 'seed42'
        add(run / 'checkpoints/epoch_5.pt', 'superseded_p75_e5_snapshot',
            'results/p75_mask_e6_execution_20260929/completion.json',
            [run / 'checkpoints/epoch_6.pt', run / 'checkpoints/best.pt', run / 'checkpoints/last.pt',
             run / 'checkpoints/continuation_E6.pt', run / 'val_epoch_6.pt'])
    pipeline = BASE / 'p75_supervision_pipeline_20260929/outputs/codex'
    for stage in ('ready', 'final'):
        for arm in ('control', 'masked'):
            p = pipeline / f'p75_supervision_pipeline_20260929_{stage}/A/{arm}_resume_E4.pt'
            retained = e6 / f'{arm}_resume_E4.pt'
            add(p, 'duplicate_obsolete_e4_handoff', 'results/p75_mask_e6_execution_20260929/completion.json',
                [retained], equivalent=retained)

    for experiment, suffix in [('wft448_dev_20261001', 'wft448_dev_20261001_r2'),
                               ('wft448_full_20261001', 'wft448_full_20261001')]:
        training = BASE / experiment / 'outputs/codex' / suffix / 'run/training'
        retained = [training / f for f in ('selected.pt', 'ema_swa.pt', 'epoch_02_ema.pt',
                                          'epoch_03_ema.pt', 'epoch_04_ema.pt', 'epoch_04_raw.pt')]
        for name in ('epoch_01_raw.pt', 'epoch_02_raw.pt', 'epoch_03_raw.pt', 'epoch_01_ema.pt'):
            add(training / name, 'unused_completed_wft_intermediate', f'results/{experiment}/final_validation.json', retained)

    v3 = BASE / 'v3_after_v1_20260930/outputs/codex/v3_after_v1_20260930/prepared_exif_recovery/runs/train'
    for arm in ('original', 'v1_supervision'):
        add(v3 / arm / 'epoch01.pt', 'superseded_v3_e1_resume', 'results/v3_final_delivery_20261001/final_validation.json',
            [v3 / arm / 'epoch02.pt', v3 / arm / 'selected.pt'])
    return rows


def tracked_guard(rows):
    roots = sorted(worktrees(), key=lambda p: len(str(p)), reverse=True)
    checked = {}
    for row in rows:
        p = Path(row['path'])
        root = next(r for r in roots if p.is_relative_to(r))
        if root not in checked:
            checked[root] = set(subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0'))
        assert str(p.relative_to(root)) not in checked[root], ('tracked', p)


def active_guard(rows):
    wanted = {(r['dev'], r['inode']) for r in rows}
    found = []
    inspected = 0
    roots = sorted(worktrees(), key=lambda p: len(str(p)), reverse=True)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            comm = (proc / 'comm').read_text().strip()
            if not any(s in comm.lower() for s in ('python', 'ssh', 'rsync', 'scp')):
                continue
            inspected += 1
            argv = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = (proc / 'cwd').resolve()
            for row in rows:
                p = Path(row['path'])
                wt = next(r for r in roots if p.is_relative_to(r) and r != REPO)
                if str(p) in argv or cwd.is_relative_to(wt):
                    found.append(dict(pid=int(proc.name), comm=comm, artifact=str(p)))
            for fd in (proc / 'fd').iterdir():
                try:
                    s = fd.stat()
                    if (s.st_dev, s.st_ino) in wanted:
                        found.append(dict(pid=int(proc.name), comm=comm, open_file=str(fd.resolve())))
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            # Other users' daemons cannot own these user-writable experiment jobs.
            continue
    assert not found, ('active candidate', found)
    return dict(relevant_processes_inspected=inspected, conflicts=found)


def plan():
    assert not (RUNTIME / 'plan.json').exists(), 'plan already exists'
    rows = proposed()
    print(f'Candidate checksums complete: {len(rows)} files', flush=True)
    tracked_guard(rows)
    protected = set()
    for row in rows:
        protected.update(row['retained'])
    large = []
    for root in worktrees():
        for sub in ('outputs', 'artifacts'):
            for p in artifacts(root / sub):
                if str(p) in {r['path'] for r in rows}:
                    continue
                if p.stat().st_size >= 32 * 1024**2:
                    large.append(snapshot(p))
                if p.name == 'pred_results.csv' or (p.suffix == '.zip' and p.stat().st_size <= 16 * 1024**2):
                    protected.add(str(p))
    best = EXTERNAL / 'v1_768_full_test_bias_20261001/outputs/codex/v1_768_full_test_bias_20261001'
    protected.update(str(best / f) for f in ('training/selected.pt', 'test_logits.pt',
                                           'test_calibration.pt', 'test_bias_report.json'))
    for exp in ('v1_train_20260929', 'v1_full_swa_20260930'):
        protected.add(str(BASE / exp / 'outputs/codex' / exp / 'training/selected.pt'))
    protected.add(str(BASE / 'lr512_dev_20261001/outputs/codex/lr512_dev_20261001/run/training/ema_swa.pt'))
    protected.add(str(BASE / 'p75_full_sam_20260928/outputs/codex/p75_mechanisms_20260927/runs/P75_FULL_SAM/seed42/checkpoints/last.pt'))
    assert not ({r['path'] for r in rows} & protected)
    print(f'Hashing {len(protected)} protected artifacts', flush=True)
    checks = [snapshot(p, True) for p in sorted(protected)]
    write(RUNTIME / 'protected.json', checks)
    write(RUNTIME / 'retained_large.json', large)
    value = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 experiment_id='STORAGE_CLEANUP_20261002', base_commit=subprocess.check_output(
                     ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                 files=len(rows), logical_bytes=sum(r['size'] for r in rows),
                 allocated_bytes=sum(r['allocated'] for r in rows if r['nlink'] == 1),
                 categories=dict(collections.Counter(r['category'] for r in rows)), candidates=rows,
                 protected_count=len(checks), retained_large_count=len(large),
                 protected_sha256=sha(RUNTIME / 'protected.json'),
                 retained_large_sha256=sha(RUNTIME / 'retained_large.json'))
    write(RUNTIME / 'plan.json', value)
    print(json.dumps({k:v for k,v in value.items() if k != 'candidates'}, indent=2), flush=True)
    print('plan_sha256', sha(RUNTIME / 'plan.json'), flush=True)


def apply(expected):
    assert sha(RUNTIME / 'plan.json') == expected, 'plan hash mismatch'
    assert not (RUNTIME / 'deleted.jsonl').exists(), 'already applied; inspect ledger'
    plan = json.loads((RUNTIME / 'plan.json').read_text())
    assert sha(RUNTIME / 'protected.json') == plan['protected_sha256']
    assert sha(RUNTIME / 'retained_large.json') == plan['retained_large_sha256']
    rows = plan['candidates']
    tracked_guard(rows)
    active = active_guard(rows)
    for row in rows:
        unchanged(row, True)
    checks = json.loads((RUNTIME / 'protected.json').read_text())
    for row in checks:
        unchanged(row, True)
    free_before = os.statvfs(REPO).f_bavail * os.statvfs(REPO).f_frsize
    removed_links = collections.Counter()
    with (RUNTIME / 'deleted.jsonl').open('x') as ledger:
        for row in rows:
            unchanged(row, removed_links=removed_links)
            if row.get('retained_identical'):
                assert sha(row['retained_identical']) == row['sha256']
            Path(row['path']).unlink()
            removed_links[(row['dev'], row['inode'])] += 1
            ledger.write(json.dumps(row) + '\n')
            ledger.flush()
    HASH_CACHE.clear()
    for row in checks:
        unchanged(row, True, removed_links)
    retained = json.loads((RUNTIME / 'retained_large.json').read_text())
    for row in retained:
        unchanged(row, removed_links=removed_links)
    assert all(not Path(row['path']).exists() for row in rows)
    result = dict(status='completed_verified', completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  files_deleted=len(rows), logical_bytes_deleted=plan['logical_bytes'],
                  allocated_bytes_released=sum(r['allocated'] for r in rows if r['nlink'] == 1),
                  free_bytes_before=free_before,
                  free_bytes_after=os.statvfs(REPO).f_bavail * os.statvfs(REPO).f_frsize,
                  protected_sha256_unchanged=len(checks), retained_large_metadata_unchanged=len(retained),
                  active_preflight=active, plan_sha256=expected,
                  best=dict(platform_score_percent=74.41512658903963, views=6, scales=[448,512,576], flip=True,
                            zip_sha256='0f83a2458199d524d3c70f19744fc7043497a531ba968d740ab6f15ebfaf4fca'),
                  categories=plan['categories'], ledger=str(RUNTIME / 'deleted.jsonl'),
                  hardlink_removals_release_no_data_blocks=sum(r['nlink'] > 1 for r in rows),
                  submission_validation='results/v1_768_bias_platform_20261002/validation.json',
                  v2_download_directory_preserved=True, raw_images_preserved=True,
                  historical_replay_limit='Deleted intermediate checkpoint replay is unavailable; retained endpoints, predictions, logs and fixed averaging inputs remain.')
    write(RUNTIME / 'result.json', result)
    for name in ('plan.json', 'result.json', 'deleted.jsonl'):
        (RECORD / name).write_bytes((RUNTIME / name).read_bytes())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def verify():
    result = json.loads((RUNTIME / 'result.json').read_text())
    assert result['status'] == 'completed_verified'
    assert sha(RUNTIME / 'plan.json') == result['plan_sha256']
    rows = [json.loads(line) for line in (RUNTIME / 'deleted.jsonl').read_text().splitlines()]
    assert len(rows) == result['files_deleted']
    assert all(not Path(row['path']).exists() for row in rows)
    removed = collections.Counter((row['dev'], row['inode']) for row in rows)
    retained = json.loads((RUNTIME / 'retained_large.json').read_text())
    for row in retained:
        unchanged(row, removed_links=removed)
    best = EXTERNAL / 'v1_768_full_test_bias_20261001/outputs/codex/v1_768_full_test_bias_20261001'
    source = json.loads((REPO / 'results/v1_768_bias_platform_20261002/record.json').read_text())
    assert sha(best / 'training/selected.pt') == source['checkpoint_sha256']
    assert sha(best / 'submission/submission.zip') == result['best']['zip_sha256']
    assert json.loads((best / 'test_bias_report.json').read_text())['views'] == 6
    command = ['python3', str(REPO / 'scripts/check_submission.py'), '--test_dir', str(REPO / 'test'),
               '--class-mapping', str(REPO / 'artifacts/stages/repechage/20260921/class_to_idx.json'),
               '--csv', str(best / 'submission/pred_results.csv'), '--zip', str(best / 'submission/submission.zip')]
    check = subprocess.run(command, text=True, capture_output=True)
    (RECORD / 'submission_check.log').write_text(check.stdout + check.stderr)
    assert check.returncode == 0, check.stdout + check.stderr
    validation = dict(status='verified', deleted_absent=len(rows), retained_large=len(retained),
                      best_checkpoint_sha256=source['checkpoint_sha256'], views=6,
                      best_submission_sha256=result['best']['zip_sha256'], submission_check_command=command,
                      submission_checker_passed=True,
                      complete_before_after_hash_audit=RUNTIME.as_posix() + '/protected.json')
    write(RECORD / 'validation.json', validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('plan', 'apply', 'verify'))
    p.add_argument('--plan-sha256')
    args = p.parse_args()
    if args.action == 'plan':
        plan()
    elif args.action == 'verify':
        verify()
    else:
        assert args.plan_sha256
        apply(args.plan_sha256)
