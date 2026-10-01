"""Recovery must reproduce the real uninterrupted optimizer/EMA trajectory."""
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from strong_aug_pair import runtime

spec = importlib.util.spec_from_file_location('resume_engine', Path(__file__).resolve().parents[3] / 'scripts/strong_aug_resume.py')
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def test_real_frozen_loop_resume_equals_uninterrupted(tmp_path, monkeypatch):
    # Reuse the existing real-loop CPU harness, including its AMP-disabled
    # update path and non-divisible final logical batch; capture its source.
    test_spec = importlib.util.spec_from_file_location('original_cpu_harness',
        Path(__file__).with_name('test_strong_aug_pair.py'))
    harness = importlib.util.module_from_spec(test_spec)
    test_spec.loader.exec_module(harness)
    original = runtime.train_arm
    captured = {}
    def capture(*args, **kwargs):
        captured['args'] = args
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime, 'train_arm', capture)
    full = tmp_path / 'uninterrupted'
    full.mkdir()
    harness.test_real_four_epoch_loop_exports_mean_of_only_epoch2_to4(full, monkeypatch)
    monkeypatch.setattr(runtime, 'train_arm', original)
    plan, source, _, arm, masks, baseline = captured['args']
    recovered = tmp_path / 'recovered'
    recovered.mkdir()
    for name in ('plan.json', 'frozen_groups.csv'):
        shutil.copy2(full / name, recovered / name)
    arm_path = recovered / 'arms' / arm
    arm_path.mkdir(parents=True)
    full_arm = full / 'arms' / arm
    for epoch in (1, 2, 3):
        for path in full_arm.glob(f'epoch_{epoch:02d}.*'):
            shutil.copy2(path, arm_path / path.name)
        shutil.copy2(full_arm / f'val_epoch{epoch:02d}_ema.npz', arm_path)
    (arm_path / 'history.json').write_text(json.dumps(json.loads((full_arm / 'history.json').read_text())[:3]))
    # Simulate a failure after two fourth-epoch updates. Neither belongs to
    # the checkpoint boundary and neither may enter the recovered trajectory.
    aborted_draws = tmp_path / 'aborted_draws.jsonl'
    aborted_draws.write_text('\n'.join((full_arm / 'draws.jsonl').read_text().splitlines()[:11]) + '\n')
    prefix = resume.accepted_prefix(aborted_draws, arm_path / 'draws.jsonl', 3)
    assert prefix == dict(accepted_prefix_updates=9, discarded_partial_epoch4_updates=2)
    namespace = dict(vars(runtime), runtime=runtime, restore_epoch3=resume.restore_epoch3)
    resumed = resume.compiled(resume.resumed_train_source(runtime), 'train_arm', namespace)
    resumed(plan, source, recovered, arm, masks, baseline)
    for filename in ('epoch_04.pt', 'last_raw.pt', 'last_ema.pt', 'ema_swa_2_4.pt'):
        equal_tree(torch.load(full_arm / filename, weights_only=False),
                   torch.load(arm_path / filename, weights_only=False))
    assert (full_arm / 'draws.jsonl').read_text() == (arm_path / 'draws.jsonl').read_text()
    assert [row['epoch'] for row in json.loads((arm_path / 'history.json').read_text())] == [1, 2, 3, 4]
    # The original full-run finalization is reused with preflight replaced;
    # training, export, review, and checksum logic must still occur once.
    finalized = resume.continued_run_source(runtime)
    assert finalized.count('reused_preflight(runtime, plan, source, output)') == 1
    assert 'freeze_groups(plan, source, output)' not in finalized
    assert 'export_arm(source, output, arm, bindings[arm])' in finalized
    compile(finalized, '<test-frozen-finalization>', 'exec')


@pytest.mark.parametrize('field,value', [('epoch', 4), ('updates', 10), ('complete', True)])
def test_rejects_partial_or_wrong_boundary(field, value):
    payload = dict(epoch=3, updates=9, complete=False, classes=['0000'], bias=None, selected_policy='ema',
        ema=dict(count=9, decay=.999), average=dict(count=2, decay=None),
        scheduler=dict(last_epoch=9, total_steps=12))
    resume.validate_epoch3(payload, 3, ['0000'])
    payload[field] = value
    with pytest.raises(ValueError):
        resume.validate_epoch3(payload, 3, ['0000'])


def test_rejects_draw_gap_and_source_drift(tmp_path):
    old = tmp_path / 'old.jsonl'
    old.write_text(json.dumps(dict(update=2, epoch=1, batch=2)) + '\n')
    with pytest.raises(ValueError, match='discontinuity'):
        resume.accepted_prefix(old, tmp_path / 'new.jsonl', 3)
    with pytest.raises(ValueError, match='anchor'):
        resume.substitute_once('missing anchor', 'directory.mkdir()', 'restore()')


def test_display_exception_is_one_pid_and_keeps_workload_stop():
    adapted = resume.resource_source(runtime, dict(pid=18332, name='steamwebhelper.exe'))
    assert "pid == 18332 and Path(name).name.lower() == 'steamwebhelper.exe'" in adapted
    assert 'kind in ("G", "C+G")' in adapted
    assert 'Existing or unknown GPU workload; no preemption' in adapted
    compile(adapted, '<test-inspected-display-check>', 'exec')
