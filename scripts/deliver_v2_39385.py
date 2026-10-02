"""Resume and verify the completed V2 download before an explicitly authorized shutdown.

The private config binds authorization to one frozen plan and SSH transport.
No training, inference, or competition upload is performed here.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    value['recorded_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def remote(config, command, timeout=75):
    return subprocess.run(config['transport'] + ['remote', command], capture_output=True,
                          text=True, timeout=timeout, check=True).stdout


def fetch_chunk(config, source, offset, length, output):
    command = shlex.join(['dd', 'if=' + source, 'bs=1M', 'iflag=skip_bytes,count_bytes',
                          'skip=' + str(offset), 'count=' + str(length), 'status=none'])
    with Path(output).open('wb') as stream:
        subprocess.run(config['transport'] + ['remote', command], stdout=stream,
                       stderr=subprocess.PIPE, timeout=75, check=True)


def download(config, item, output, progress=None):
    """Only append whole successful chunks; the final digest also protects resume data."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        require(output.stat().st_size == item['bytes'] and digest(output) == item['sha256'],
                'Existing local artifact differs: ' + str(output))
        return
    partial = output.with_name(output.name + '.receiving')
    identity = partial.with_suffix(partial.suffix + '.json')
    if partial.exists():
        require(identity.exists() and json.loads(identity.read_text()) == item,
                'Partial download belongs to another artifact')
        require(partial.stat().st_size <= item['bytes'], 'Partial download exceeds expected size')
    else:
        identity.write_text(json.dumps(item))
        partial.touch()
    chunk = output.with_name(output.name + '.chunk')
    offset = partial.stat().st_size
    while offset < item['bytes']:
        length = min(config.get('chunk_bytes', 32 * 1024 * 1024), item['bytes'] - offset)
        for attempt in range(3):
            try:
                fetch_chunk(config, item['remote'], offset, length, chunk)
                require(chunk.stat().st_size == length, 'Incomplete transport chunk')
                break
            except (OSError, ValueError, subprocess.SubprocessError):
                if attempt == 2:
                    raise
                time.sleep(5)
        with partial.open('ab') as target, chunk.open('rb') as source:
            shutil.copyfileobj(source, target)
        offset += length
        if progress:
            progress(item, offset)
    require(digest(partial) == item['sha256'], 'Downloaded artifact digest mismatch')
    partial.replace(output)
    identity.unlink()
    chunk.unlink(missing_ok=True)


# Executed remotely only after the monitor has verified the completed submission.
INVENTORY = r'''
import hashlib, json, pathlib, sys, tarfile
root = pathlib.Path(sys.argv[1]).parent
def read(name): return json.loads((root/name).read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4194304), b''): h.update(block)
    return h.hexdigest()
controller, plan, report = read('controller.json'), read('plan.json'), read('submission/report.json')
assert controller['status']=='completed_delivered' and controller['training_completed'] is True
assert sha(root/'plan.json')==sys.argv[2]==controller['plan_sha256']
assert report['status']=='package_ready' and report['rows']==37444 and report['weights']=='raw'
assert report['checkpoint']==str(root/'runs/full_576/last.pt')
weights=['runs/s2_448/best.pt','runs/s3_576/best.pt','runs/full_576/last.pt']
weights += [f'runs/full_576/epoch_{n:02d}_raw.pt' for n in (3,4,5)]
for stage in ('s2_448','s3_576','full_576'):
    status=read(f'runs/{stage}/status.json')
    assert status['status']=='complete' and status['updates']==plan['stages'][stage]['total_updates']
for name in weights[:3]:
    sidecar=read(str(pathlib.Path(name).with_suffix('.binding.json')))
    assert sidecar['binding']['complete'] is True
archive=root/'delivery_metadata.tar.gz'
with tarfile.open(archive.with_suffix('.tmp'), 'w:gz', dereference=True) as tar:
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if 'monitoring' in rel.parts or 'credentials' in rel.parts: continue
        if path.is_file() and path.suffix in ('.json','.csv','.yaml','.yml','.log','.npz','.py'):
            tar.add(path, arcname=str(rel), recursive=False)
    # Class maps, original configurations and frozen source live outside prepared/.
    for name, expected in sorted(plan['inputs'].items()):
        path=pathlib.Path(name)
        if path.suffix != '.pt':
            assert sha(path)==expected, f'Frozen metadata changed: {name}'
            tar.add(path, arcname='bound_inputs/'+path.as_posix().lstrip('/'), recursive=False)
archive.with_suffix('.tmp').replace(archive)
names=weights+['delivery_metadata.tar.gz','submission/pred_results.csv','submission/submission.zip','submission/report.json','submission_check.log']
files=[]
for name in names:
    path=root/name
    files.append(dict(relative=name, remote=str(path), bytes=path.stat().st_size, sha256=sha(path)))
assert files[2]['sha256']==report['checkpoint_sha256']==read('runs/full_576/last.binding.json')['sha256']
for item in files[:3]:
    assert item['sha256']==read(str(pathlib.Path(item['relative']).with_suffix('.binding.json')))['sha256']
print(json.dumps(dict(plan_sha256=sys.argv[2], controller=controller, report=report, files=files)))
'''


def validate_inventory(config, inventory):
    require(config.get('shutdown_after_verified_download') is True, 'No explicit shutdown authorization')
    require(inventory['plan_sha256'] == config['expected_plan_sha256'], 'Frozen plan mismatch')
    require(inventory['controller']['status'] == 'completed_delivered' and
            inventory['controller'].get('training_completed') is True, 'V2 has not completed')
    report = inventory['report']
    require(report['status'] == 'package_ready' and report['rows'] == 37444 and report['weights'] == 'raw',
            'Wrong submission policy or incomplete package')
    files = {item['relative']: item for item in inventory['files']}
    required = ['runs/s2_448/best.pt', 'runs/s3_576/best.pt', 'runs/full_576/last.pt',
                'delivery_metadata.tar.gz', 'submission/pred_results.csv', 'submission/submission.zip',
                'submission/report.json', 'submission_check.log']
    required += [f'runs/full_576/epoch_{n:02d}_raw.pt' for n in (3, 4, 5)]
    require(set(files) == set(required) and len(files) == len(inventory['files']), 'Delivery inventory is incomplete')
    for name, item in files.items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and item['bytes'] > 0, 'Unsafe or empty artifact')
        require(item['remote'] == str(PurePosixPath(config['remote_plan']).parent / path), 'Artifact outside bound workspace')
    for name, field in [('runs/full_576/last.pt', 'checkpoint_sha256'),
                        ('submission/pred_results.csv', 'csv_sha256'), ('submission/submission.zip', 'zip_sha256')]:
        require(files[name]['sha256'] == report[field], 'Submission and artifact digest disagree')


def verify_local(config, inventory, output):
    validate_inventory(config, inventory)
    for item in inventory['files']:
        path = output / item['relative']
        require(path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256'],
                'Local file verification failed: ' + item['relative'])
    require(json.loads((output / 'submission/report.json').read_text()) == inventory['report'],
            'Downloaded submission report differs')
    result = subprocess.run([config['local_python'], str(Path(__file__).with_name('check_submission.py')),
                             '--test_dir', config['local_test_root'], '--num-classes', '750',
                             '--csv', str(output / 'submission/pred_results.csv'),
                             '--zip', str(output / 'submission/submission.zip')],
                            capture_output=True, text=True, check=True)
    (output / 'local_submission_check.log').write_text(result.stdout + result.stderr)


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    state = Path(config['local_state'])
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / 'delivery.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    latest = state / 'latest.json'
    if latest.exists():
        previous = json.loads(latest.read_text())
        if previous.get('shutdown_issued'):
            require(previous['status'] == 'shutdown_endpoint_offline',
                    'Shutdown was already issued; diagnose its existing receipt instead of repeating it')
            return 0
    output = Path(config['local_delivery'])
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(status='collecting_completed_inventory', local_delivery=str(output), shutdown_issued=False)
    try:
        inventory_path = state / 'inventory.json'
        if inventory_path.exists():
            inventory = json.loads(inventory_path.read_text())
        else:
            command = shlex.join([config['remote_python'], '-c', INVENTORY,
                                  config['remote_plan'], config['expected_plan_sha256']])
            inventory = json.loads(remote(config, command))
            validate_inventory(config, inventory)
            inventory_path.write_text(json.dumps(inventory, indent=2) + '\n')
        validate_inventory(config, inventory)
        receipt.update(status='downloading', total_bytes=sum(item['bytes'] for item in inventory['files']))
        completed = 0

        def progress(item, offset):
            receipt.update(current_file=item['relative'], current_file_bytes=offset,
                           downloaded_bytes=completed + offset)
            write_json(latest, receipt)

        for item in inventory['files']:
            download(config, item, output / item['relative'], progress)
            completed += item['bytes']
        verify_local(config, inventory, output)
        receipt.update(status='verified_before_shutdown', downloaded_bytes=completed,
                       files=inventory['files'], plan_sha256=inventory['plan_sha256'], submission_check_passed=True)
        write_json(output / 'delivery_receipt.json', receipt)
        write_json(latest, receipt)
        # Re-read controller and GPU ownership immediately before the irreversible action.
        guard = shlex.join([config['remote_python'], '-c',
            "import json,pathlib,subprocess,sys; p=pathlib.Path(sys.argv[1]); "
            "s=json.loads((p.parent/'controller.json').read_text()); "
            "assert s['status']=='completed_delivered'; "
            "assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(); print('READY')",
            config['remote_plan']])
        require(remote(config, guard).strip() == 'READY', 'Shutdown readiness check failed')
        remote(config, 'sync && /usr/bin/shutdown')
        receipt.update(status='shutdown_issued', shutdown_issued=True)
        write_json(output / 'delivery_receipt.json', receipt)
        write_json(latest, receipt)
        checks = []
        for _ in range(3):
            time.sleep(10)
            try:
                remote(config, 'true', timeout=75)
                checks.append('reachable')
            except (subprocess.SubprocessError, OSError):
                checks.append('unreachable')
        receipt.update(status='shutdown_endpoint_offline' if checks == ['unreachable'] * 3 else 'shutdown_needs_check',
                       post_shutdown_ssh_checks=checks)
        write_json(output / 'delivery_receipt.json', receipt)
        write_json(latest, receipt)
        return 0 if receipt['status'] == 'shutdown_endpoint_offline' else 2
    except Exception as error:
        receipt.update(status='delivery_or_shutdown_needs_diagnosis', error=str(error))
        write_json(latest, receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    raise SystemExit(run(parser.parse_args().config))
