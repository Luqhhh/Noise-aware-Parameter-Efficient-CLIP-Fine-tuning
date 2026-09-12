import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('resource_report',Path(__file__).resolve().parents[1]/'scripts/report_recovery_resources.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def setup_run(root):
    (root/'recovery_plan.json').write_text(json.dumps({'nodes':[{'node_id':'root','parent':None,'epochs':3}]}))
    (root/'e2_recovery.log').write_text('2026-09-12 10:00:00,000 | INFO | Start\n2026-09-12 10:00:10,000 | INFO | Epoch 1 | raw_micro=0.1\n')
    logs=root/'recovery_models/root/seed42/logs';logs.mkdir(parents=True)
    (logs/'longtail_epoch_1.json').write_text(json.dumps({'optimizer_steps':2,'actual_draws':5}))
    return logs


def test_counts_and_missing_peak_measurements(tmp_path):
    setup_run(tmp_path)
    r=module.summarize(tmp_path)
    assert r['total_logged_span_seconds']==10
    assert r['total_sample_draws']==5
    assert r['nodes'][0]['completed_epochs']==1
    assert r['nodes'][0]['peak_gpu_bytes'] is None


def test_missing_epoch_ledger_rejected(tmp_path):
    logs=setup_run(tmp_path)
    (logs/'longtail_epoch_1.json').unlink()
    with pytest.raises(ValueError,match='disagree'):module.summarize(tmp_path)
