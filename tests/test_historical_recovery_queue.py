"""Completion markers must bind real outputs before descendants may start."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('recovery_queue', Path(__file__).resolve().parents[1]/'scripts/run_historical_recovery_queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def fixture(tmp_path):
    directory = tmp_path/'models'/'parent'/'seed42'/'checkpoints'
    directory.mkdir(parents=True)
    model = directory/'best.pt'
    model.write_bytes(b'completed model')
    data = tmp_path/'train.csv'
    data.write_text('image_path,label\na.jpg,0\n')
    config = {'output': {'root': str(tmp_path/'models')}, 'data': {'train_csv': str(data), 'val_csv': str(data)}}
    manifest = {'experiment_id': 'parent', 'best_checkpoint': str(model), 'best_checkpoint_sha256': queue.sha(model), 'train_csv_sha256': queue.sha(data), 'val_csv_sha256': queue.sha(data)}
    return directory, model, data, config, manifest


def test_best_checkpoint_alone_is_not_completion(tmp_path):
    _, _, _, config, _ = fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        queue.verify_completion({'node_id': 'parent'}, config)


@pytest.mark.parametrize('changed', ['model', 'data', 'experiment'])
def test_changed_completion_binding_rejected(tmp_path, changed):
    directory, model, data, config, manifest = fixture(tmp_path)
    if changed == 'model': model.write_bytes(b'changed')
    if changed == 'data': data.write_text('changed')
    if changed == 'experiment': manifest['experiment_id'] = 'other'
    (directory/'artifact_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        queue.verify_completion({'node_id': 'parent'}, config)


def test_intact_completion_accepted(tmp_path):
    directory, _, _, config, manifest = fixture(tmp_path)
    (directory/'artifact_manifest.json').write_text(json.dumps(manifest))
    assert queue.verify_completion({'node_id': 'parent'}, config) == manifest
