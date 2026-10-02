"""Irreversible shutdown must follow complete, intact, locally validated delivery."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


spec = importlib.util.spec_from_file_location('v2_delivery', Path(__file__).parents[1] / 'scripts/deliver_v2_39385.py')
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


def fixture(tmp_path):
    state, output = tmp_path / 'state', tmp_path / 'downloads'
    state.mkdir()
    config = dict(local_state=str(state), local_delivery=str(output), transport=['private'],
                  remote_plan='/run/plan.json', remote_python='python', expected_plan_sha256='plan-sha',
                  shutdown_after_verified_download=True, local_python='python', local_test_root='/test')
    names = ['runs/s2_448/best.pt', 'runs/s3_576/best.pt', 'runs/full_576/last.pt',
             'delivery_metadata.tar.gz', 'submission/pred_results.csv', 'submission/submission.zip',
             'submission/report.json', 'submission_check.log']
    names += [f'runs/full_576/epoch_{n:02d}_raw.pt' for n in (3, 4, 5)]
    report = dict(status='package_ready', rows=37444, weights='raw',
                  checkpoint_sha256=hashlib.sha256(b'weight').hexdigest(),
                  csv_sha256=hashlib.sha256(b'csv').hexdigest(), zip_sha256=hashlib.sha256(b'zip').hexdigest())
    files = []
    for name in names:
        data = {'runs/full_576/last.pt': b'weight', 'submission/pred_results.csv': b'csv',
                'submission/submission.zip': b'zip', 'submission/report.json': json.dumps(report).encode()}.get(name, b'other')
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files.append(dict(relative=name, remote='/run/' + name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    inventory = dict(plan_sha256='plan-sha', controller=dict(status='completed_delivered', training_completed=True),
                     report=report, files=files)
    (state / 'inventory.json').write_text(json.dumps(inventory))
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    return config, inventory, config_path, output


@pytest.mark.parametrize('fault', ['authorization', 'incomplete', 'plan', 'missing_weight', 'mismatched_weight', 'outside'])
def test_bad_inventory_refuses_shutdown(tmp_path, monkeypatch, fault):
    config, inventory, config_path, output = fixture(tmp_path)
    if fault == 'authorization': config['shutdown_after_verified_download'] = False
    if fault == 'incomplete': inventory['controller']['training_completed'] = False
    if fault == 'plan': inventory['plan_sha256'] = 'other'
    if fault == 'missing_weight': inventory['files'].pop()
    if fault == 'mismatched_weight': inventory['files'][2]['sha256'] = 'other'
    if fault == 'outside': inventory['files'][0]['remote'] = '/other/best.pt'
    config_path.write_text(json.dumps(config))
    (Path(config['local_state']) / 'inventory.json').write_text(json.dumps(inventory))
    commands = []
    monkeypatch.setattr(delivery, 'remote', lambda *args, **kwargs: commands.append(args[1]))
    with pytest.raises(ValueError): delivery.run(config_path)
    assert commands == []


@pytest.mark.parametrize('fault', ['corrupt_weight', 'checker_failure'])
def test_local_verification_failure_keeps_server_running(tmp_path, monkeypatch, fault):
    config, inventory, config_path, output = fixture(tmp_path)
    commands = []
    monkeypatch.setattr(delivery, 'remote', lambda *args, **kwargs: commands.append(args[1]))
    if fault == 'corrupt_weight': (output / 'runs/full_576/last.pt').write_bytes(b'broken')
    else:
        def fail(*args, **kwargs): raise subprocess.CalledProcessError(1, 'checker')
        monkeypatch.setattr(delivery.subprocess, 'run', fail)
    with pytest.raises((ValueError, subprocess.CalledProcessError)): delivery.run(config_path)
    assert commands == []


def test_verified_receipt_precedes_shutdown_and_offline_checks(tmp_path, monkeypatch):
    config, inventory, config_path, output = fixture(tmp_path)
    commands = []
    monkeypatch.setattr(delivery.subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, '9 passed', ''))
    monkeypatch.setattr(delivery.time, 'sleep', lambda _: None)
    def remote(config, command, **kwargs):
        commands.append(command)
        if command == 'sync && /usr/bin/shutdown':
            receipt = json.loads((output / 'delivery_receipt.json').read_text())
            assert receipt['status'] == 'verified_before_shutdown'
            assert receipt['submission_check_passed'] is True
            return ''
        if command == 'true': raise subprocess.CalledProcessError(255, 'ssh')
        return 'READY'
    monkeypatch.setattr(delivery, 'remote', remote)
    assert delivery.run(config_path) == 0
    assert commands[-4:] == ['sync && /usr/bin/shutdown', 'true', 'true', 'true']
    assert json.loads((output / 'delivery_receipt.json').read_text())['status'] == 'shutdown_endpoint_offline'
    previous_commands = list(commands)
    assert delivery.run(config_path) == 0
    assert commands == previous_commands


def test_interrupted_download_resumes_without_repeating_verified_chunks(tmp_path, monkeypatch):
    data = b'abcdefghijklmnop'
    item = dict(remote='/run/weight.pt', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    output = tmp_path / 'weight.pt'
    calls = []
    def fetch(config, source, offset, length, target):
        calls.append(offset)
        if offset == 8: raise subprocess.CalledProcessError(255, 'ssh')
        Path(target).write_bytes(data[offset:offset+length])
    monkeypatch.setattr(delivery, 'fetch_chunk', fetch)
    monkeypatch.setattr(delivery.time, 'sleep', lambda _: None)
    with pytest.raises(subprocess.CalledProcessError): delivery.download({'chunk_bytes': 4}, item, output)
    assert output.with_name('weight.pt.receiving').read_bytes() == data[:8]
    def resumed(config, source, offset, length, target):
        calls.append(offset)
        Path(target).write_bytes(data[offset:offset+length])
    monkeypatch.setattr(delivery, 'fetch_chunk', resumed)
    delivery.download({'chunk_bytes': 4}, item, output)
    assert calls[-2:] == [8, 12] and output.read_bytes() == data


def test_resume_identity_and_final_digest_are_enforced(tmp_path, monkeypatch):
    item = dict(remote='/run/weight.pt', bytes=4, sha256=hashlib.sha256(b'good').hexdigest())
    output = tmp_path / 'weight.pt'
    monkeypatch.setattr(delivery, 'fetch_chunk', lambda config, source, offset, length, target: Path(target).write_bytes(b'evil'))
    with pytest.raises(ValueError, match='digest mismatch'): delivery.download({}, item, output)
    changed = copy.deepcopy(item)
    changed['remote'] = '/other.pt'
    with pytest.raises(ValueError, match='another artifact'): delivery.download({}, changed, output)
