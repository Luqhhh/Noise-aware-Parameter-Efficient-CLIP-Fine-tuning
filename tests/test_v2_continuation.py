"""Protect parent import, cost bounds and actual progress monitoring."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproducibility/aegis_f1'))
from v2.controller import auth_from_cost
from v2.health import classify, log_progress, process_matches
from v2.import_parent import canonical_config, manifest_rows
from v2.plan import authorize, sha


def test_resource_migration_retains_all_training_hyperparameters():
    source = dict(paths=dict(project_root='Windows'), data=dict(train_dir='D:/train', num_workers=0, batch_size=96),
                  model=dict(official_checkpoint='D:/clip.pt'), local_replay=dict(micro_batch_size=16),
                  train=dict(lr_backbone=1e-5, seed=13), augment=dict(mixup=.2))
    migrated = copy.deepcopy(source)
    migrated['paths']['project_root'] = '/repo'
    migrated['data'].update(train_dir='/train', num_workers=2)
    migrated['model']['official_checkpoint'] = '/clip.pt'
    assert canonical_config(source) == canonical_config(migrated)
    migrated['train']['lr_backbone'] = 2e-5
    assert canonical_config(source) != canonical_config(migrated)


def test_manifest_portability_preserves_labels_and_content_groups(tmp_path):
    windows, linux = tmp_path / 'windows.csv', tmp_path / 'linux.csv'
    windows.write_text('relative_path,label,content_group\r\n0001\\x.jpg,1,g\r\n')
    linux.write_text('relative_path,label,content_group\n0001/x.jpg,1,g\n')
    assert manifest_rows(windows) == manifest_rows(linux)
    linux.write_text('relative_path,label,content_group\n0001/x.jpg,2,g\n')
    assert manifest_rows(windows) != manifest_rows(linux)


@pytest.mark.parametrize('operation', ['train', 'probe'])
def test_imported_s1_cannot_be_retrained(tmp_path, monkeypatch, operation):
    import v2.plan as module
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    auth = tmp_path / 'auth.json'
    auth.write_text(json.dumps(dict(authorized=True, operation=operation, stage='s1_384', plan_sha256=sha(plan), max_seconds=100)))
    monkeypatch.setattr(module, 'verify_prepared', lambda _: dict(imported_stages={'s1_384': {}}))
    with pytest.raises(ValueError, match='imported completed stage'):
        authorize(plan, auth, operation, 's1_384')


def test_final_budget_uses_same_resolution_cost_without_holdout(tmp_path, monkeypatch):
    import v2.controller as module
    path = tmp_path / 'plan.json'
    path.write_text('{}')
    monkeypatch.setattr(module, 'verify_prepared', lambda _: dict(stages={'full_576': dict(total_updates=9290, epochs=5)}))
    cost = dict(plan_sha256=sha(path), measured_seconds_per_update=2., measured_validation_seconds=500., measured_overhead_seconds=80.)
    auth = auth_from_cost(path, 'full_576', cost)
    assert auth['measured_validation_seconds'] == 0
    assert auth['estimated_seconds'] == 9290 * 2 + 100
    assert auth['estimated_seconds'] <= .8 * auth['max_seconds']
    cost['plan_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='another plan'):
        auth_from_cost(path, 'full_576', cost)


def test_monitor_distinguishes_dead_stale_and_finished_controller(tmp_path):
    assert not process_matches(999999999, tmp_path / 'plan.json')
    state = {'status': 'training'}
    assert classify(state, True, 5, 3 * 1024**3) == 'running'
    assert classify(state, False, 5, 3 * 1024**3) == 'needs_diagnosis'
    assert classify(state, True, 4000, 3 * 1024**3) == 'no_recent_log_needs_diagnosis'
    assert classify(state, True, 5, 1024**3) == 'low_disk_needs_diagnosis'
    assert classify({'status': 'completed_delivered'}, False, 4000, 1024**3) == 'completed_delivered'


def test_monitor_reads_actual_update_record_through_non_json_logs(tmp_path):
    log = tmp_path / 'controller.log'
    log.write_text('loading weights\n{"stage":"s2_448","updates":50,"loss":2.9}\nPillow warning\n')
    progress, tail = log_progress(log)
    assert progress['updates'] == 50
    assert tail[-1] == 'Pillow warning'
