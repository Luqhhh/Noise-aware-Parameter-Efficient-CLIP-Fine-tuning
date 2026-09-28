"""CPU tests for evidence boundaries, exact gates, gradients, and safe dispatch."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'scripts/p75_runtime'))
import p75_supported_ce as pipeline
import p75_supported_ce_report as reporting
import run_p75_supported_ce as runner
from p75_hard_overlay import transform
from p75_hard_support import (classification_support, file_sha256, load_support,
    local_admission, record_classification, restore_original_label_loss)

RULES = json.loads(pipeline.FIXED.read_text())['rules']


def test_rules_do_not_require_original_top1_or_apply_distribution_threshold():
    first = np.array([[.2, .4, .1, .1, .1, .1]], dtype=np.float32)
    second = np.array([[.1, .6, .075, .075, .075, .075]], dtype=np.float32)
    metrics = pipeline.weak_metrics(first, second, [0])
    assert metrics['center_top1'].item() == 1
    assert metrics['distribution_l1'].item() > .3
    assert metrics['distribution_jsd'].item() > 0
    assert pipeline.low_confidence_mask(metrics, RULES).tolist() == [True]


@pytest.mark.parametrize('field,value', [
    ('center_label_probability', .3), ('flip_label_probability', .3),
    ('center_max_probability', .7), ('flip_max_probability', .7), ('flip_top1', 2)])
def test_strict_both_view_boundaries(field, value):
    metrics = dict(center_label_probability=np.array([.2]), flip_label_probability=np.array([.2]),
        center_max_probability=np.array([.4]), flip_max_probability=np.array([.4]),
        center_top1=np.array([1]), flip_top1=np.array([1]))
    metrics[field] = np.array([value])
    assert not pipeline.low_confidence_mask(metrics, RULES).item()


def rows(count=24):
    return [dict(image_path=f'train/{index}.jpg', label=0, content_group=f'g{index}')
            for index in range(count)]


def test_neighbor_votes_exclude_self_and_duplicate_groups_and_record_conflicts():
    training = rows()
    training[1]['content_group'] = training[0]['content_group']
    training[3]['content_group'] = training[2]['content_group']
    training[3]['label'] = 1  # conflicting group counts once, casts no support vote
    features = np.tile(np.array([[1., 0.]], dtype=np.float32), (len(training), 1))
    index, neighbors = next(pipeline.nearest_groups(features, training, [0]))
    assert index == 0 and len(neighbors) == 20
    assert len({row['content_group'] for row in neighbors}) == 20
    assert not any(row['content_group'] == 'g0' for row in neighbors)
    assert neighbors[0]['training_row'] == 2  # stable original row tie break
    assert neighbors[0]['original_labels'] == [0, 1]
    assert neighbors[0]['supports_original_label'] is False
    assert sum(row['supports_original_label'] for row in neighbors) == 19


def test_small_class_is_not_declared_noise_and_query_batch_is_bounded():
    training = rows(4)
    features = np.tile(np.array([[1., 0.]], dtype=np.float32), (4, 1))
    assert len(next(pipeline.nearest_groups(features, training, [0]))[1]) == 3
    with pytest.raises(ValueError, match='batch size'):
        next(pipeline.nearest_groups(features, training, [0], batch_size=17))


def test_content_boundary_and_evidence_row_alignment_fail_closed():
    training, validation = rows(2), [dict(image_path='train/v.jpg', label=0, content_group='v')]
    pipeline.validate_rows(training, validation, 2)
    validation[0]['content_group'] = 'g0'
    with pytest.raises(ValueError, match='overlap'):
        pipeline.validate_rows(training, validation, 2)
    with pytest.raises(ValueError, match='order'):
        pipeline.assert_alignment(training, training[::-1])
    changed = copy.deepcopy(training)
    changed[0]['label'] = 1
    with pytest.raises(ValueError, match='order'):
        pipeline.assert_alignment(training, changed)
    with pytest.raises(ValueError, match='relative'):
        pipeline.canonical('../test/x.jpg')


def test_distribution_jsd_zeros_and_invalid_raw_probabilities():
    p = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    metrics = pipeline.weak_metrics(p, p[::-1], [0, 1])
    assert np.allclose(metrics['distribution_jsd'], np.log(2))
    with pytest.raises(ValueError, match='Invalid'):
        pipeline.weak_metrics(p*2, p, [0, 1])


def options():
    return {'loss': {'ce_warmup_epochs': 2, 'hard_support': dict(enabled=True,
        ce_restore_enabled=True, local_restore_enabled=False, version=1, maximum_weight=.5)}}


def test_warmup_and_disabled_classification_are_exact_identity():
    config = options()
    support = torch.tensor([.5, 0.])
    base = torch.tensor([1., 2.], requires_grad=True)
    for epoch in (1, 2):
        active = classification_support(config, support, epoch)
        assert restore_original_label_loss(base, torch.zeros(2, 2), torch.eye(2), active) is base
    assert classification_support(config, support, 3) is support
    config['loss']['hard_support']['ce_restore_enabled'] = False
    assert classification_support(config, support, 16) is None


def test_ce_gradient_scale_recovers_original_label_signal_and_rejects_others():
    logits = torch.tensor([[np.log(.01), np.log(.99)], [np.log(.04), np.log(.96)]],
                          dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[1., 0.], [1., 0.]])
    p = logits.float().softmax(1)[:, 0]
    gce = (1-p.sqrt())/.5
    blend = restore_original_label_loss(gce, logits, target, torch.tensor([.5, 0.]))
    grad = torch.autograd.grad(blend.sum(), logits, retain_graph=True)[0]
    ce_grad = torch.autograd.grad(F.cross_entropy(logits.float(), torch.zeros(2, dtype=torch.long),
                                                reduction='sum'), logits)[0]
    assert torch.allclose(grad[0]/ce_grad[0], torch.tensor([.55, .55], dtype=logits.dtype), atol=1e-6)
    assert torch.allclose(grad[1]/ce_grad[1], torch.tensor([.2, .2], dtype=logits.dtype), atol=1e-6)


def test_local_gate_disabled_matches_original_at_boundary_and_ignores_crop_quality(monkeypatch):
    import p75_hard_support as runtime
    monkeypatch.setattr(runtime, 'local_quality', lambda _: pytest.fail('Crop quality must not run'))
    confidence = torch.tensor([.2, .69999, .7, .9])
    supported = torch.full((4,), .5)
    actual = local_admission(confidence, .7, supported, torch.zeros(4, 3, 8, 8))
    assert torch.equal(actual, confidence >= .7)


def test_support_loader_no_implicit_enable_and_exact_cardinality(tmp_path):
    config = options()
    path = tmp_path/'support.csv'
    pipeline.write_rows(path, ['image_path', 'label', 'support_weight'], [
        dict(image_path='train/a.jpg', label=0, support_weight=.5),
        dict(image_path='train/b.jpg', label=1, support_weight=0)])
    config['loss']['hard_support'].update(path=str(path), sha256=file_sha256(path))
    assert load_support(config, ['a.jpg', 'b.jpg'], [0, 1]).tolist() == [.5, 0.]
    with pytest.raises(ValueError, match='different lengths'):
        load_support(config, ['a.jpg', 'b.jpg'], [0])
    with pytest.raises(ValueError, match='row count'):
        load_support(config, ['a.jpg'], [0])
    with pytest.raises(ValueError, match='label mismatch'):
        load_support(config, ['a.jpg', 'b.jpg'], [1, 1])
    config['loss']['hard_support'].pop('ce_restore_enabled')
    assert load_support(config, ['a.jpg', 'b.jpg'], [0, 1]) is None


def test_training_fit_recording_does_not_change_gradients():
    logits = torch.tensor([[0., 1.], [2., 0.]], requires_grad=True)
    targets = torch.eye(2)
    base = torch.tensor([1., 2.], requires_grad=True)
    totals = {}
    record_classification(totals, logits, targets, base, base, torch.tensor([.5, 0.]))
    assert totals['supported_ce']['examples'] == 1
    assert totals['supported_ce']['correct'] == 0
    assert all(not isinstance(value, torch.Tensor) for value in totals['supported_ce'].values())
    assert logits.grad is None and base.grad is None


@pytest.mark.parametrize('phase', sorted(runner.GPU_PHASES))
def test_gpu_dispatch_guard_precedes_all_work(monkeypatch, phase):
    monkeypatch.setattr(runner, 'preflight', lambda **_: pytest.fail('GPU guard ran too late'))
    with pytest.raises(ValueError, match='explicit --execute-gpu'):
        runner.main(['--phase', phase])


def test_logs_do_not_create_trainer_owned_run():
    assert runner.RUN not in runner.phase_log('train').parents
    assert runner.phase_log('train').parent == runner.RUN.parent


def test_private_environment_excludes_model_hook_and_inherited_sources(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', str(ROOT/'scripts/p75_runtime'))
    paths = pipeline.environment()['PYTHONPATH'].split(':')
    assert paths == [str(pipeline.FRAMEWORK/'support_runtime'),
                     str(pipeline.FRAMEWORK/'reproducibility/aegis_f1')]
    assert str(ROOT/'scripts/p75_runtime') not in paths


def test_idle_device_check_honors_visibility_and_rejects_compute_processes(monkeypatch):
    monkeypatch.setattr(runner, 'read_json', lambda _: {'train': {'device': 'cuda:0'}})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    def output(command, **kwargs):
        if '--query-gpu=uuid' in command:
            assert command[command.index('-i')+1] == '2'
            return 'GPU-fixed\n'
        return 'GPU-fixed, 123\n'
    monkeypatch.setattr(runner.subprocess, 'check_output', output)
    with pytest.raises(RuntimeError, match='busy'):
        runner.ensure_idle_device()


@pytest.mark.parametrize('section,key,value', [
    ('train', 'amp', False), ('train', 'epochs', 2), ('train', 'backbone_lr', .001),
    ('loss', 'gce_q', .3), ('model', 'input_resolution', 224)])
def test_recipe_comparison_rejects_unregistered_training_changes(section, key, value):
    original = dict(project={}, output={}, train=dict(epochs=16, effective_batch_size=1024, amp=True,
        backbone_lr=1.2e-5), model=dict(input_resolution=384), trust=dict(enabled=False),
        loss=dict(name='gce', gce_q=.5, ce_warmup_epochs=2, mixup_probability=0,
                  attention_local_training=dict(confidence_gate=.7)))
    config = pipeline.candidate_config(original, {})
    pipeline.validate_recipe(config, original, bound=False)
    config[section][key] = value
    with pytest.raises(ValueError, match='drifted'):
        pipeline.validate_recipe(config, original, bound=False)


def make_selection(tmp_path, monkeypatch, *, admitted=True, votes=16):
    directory = tmp_path/'output'
    assets = tmp_path/'assets'
    assets.mkdir()
    fixed = tmp_path/'fixed.json'
    fixed.write_text(json.dumps(dict(assets=str(assets), rules=RULES)))
    monkeypatch.setattr(pipeline, 'OUT', directory)
    monkeypatch.setattr(pipeline, 'CONFIG', directory/'configs/P75_SUPPORTED_CE.yaml')
    monkeypatch.setattr(pipeline, 'FIXED', fixed)
    saved = dict(identity=dict(num_classes=6), rules=RULES)
    monkeypatch.setattr(pipeline, 'preflight', lambda: saved)
    training = rows(22)
    for index in range(1, len(training)):
        training[index]['label'] = 0 if index <= votes else 1
    pipeline.write_rows(assets/'train_dev.csv', pipeline.ROW_FIELDS, training)
    snapshot = directory/'snapshot'
    snapshot.mkdir(parents=True)
    pipeline.write_rows(snapshot/'rows.csv', pipeline.ROW_FIELDS, training)
    p = np.tile(np.array([[.8, .04, .04, .04, .04, .04]], dtype=np.float32), (22, 1))
    if admitted:
        p[0] = [.2, .4, .1, .1, .1, .1]
    np.save(snapshot/'center_probabilities.npy', p)
    np.save(snapshot/'flip_probabilities.npy', p)
    pipeline.write_json(snapshot/'manifest.json', dict(identity=saved['identity'], temperature=1.,
        prior_applied=False, tta_fusion_applied=False, views=['center_384', 'horizontal_flip_384'],
        files={name: pipeline.sha(snapshot/name) for name in
               ('rows.csv', 'center_probabilities.npy', 'flip_probabilities.npy')}))
    template = directory/'configs/template.yaml'
    template.parent.mkdir()
    template.write_text(yaml.safe_dump(dict(loss=dict(hard_support=dict(enabled=False, sha256=None)))))
    bank = np.tile(np.array([[1., 0.]], dtype=np.float32), (22, 1))
    monkeypatch.setattr(pipeline, 'train_features', lambda _: bank)
    return directory


@pytest.mark.parametrize('votes,expected', [(16, 'admitted'), (15, 'closed_empty_support')])
def test_selection_end_to_end_fixed_vote_boundary_and_bound_hash(tmp_path, monkeypatch, votes, expected):
    directory = make_selection(tmp_path, monkeypatch, votes=votes)
    result = pipeline.select()
    assert result['status'] == expected
    support = pipeline.read_rows(directory/'selection/support.csv')
    assert len(support) == 22
    assert float(support[0]['support_weight']) == (.5 if votes == 16 else 0.)
    assert result['rejected_samples_are_noise'] is False
    assert result['support_count'] == (1 if votes == 16 else 0)
    if votes == 16:
        config = yaml.safe_load((directory/'configs/P75_SUPPORTED_CE.yaml').read_text())
        assert config['loss']['hard_support']['sha256'] == pipeline.sha(directory/'selection/support.csv')
    else:
        assert not (directory/'configs/P75_SUPPORTED_CE.yaml').exists()
    with pytest.raises(FileExistsError):
        pipeline.select()


def test_empty_weak_subset_stops_without_loading_neighbor_bank(tmp_path, monkeypatch):
    directory = make_selection(tmp_path, monkeypatch, admitted=False)
    monkeypatch.setattr(pipeline, 'train_features', lambda _: pytest.fail('Empty queries must skip bank'))
    result = pipeline.select()
    assert result['status'] == 'closed_empty_support'
    assert result['support_count'] == 0
    assert not (directory/'configs/P75_SUPPORTED_CE.yaml').exists()


def test_report_uses_all_validation_and_groups_from_training_support_only():
    result = reporting.grouped_report([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 1], [0]*5, 3)
    assert result['corrections'] == 1 and result['regressions'] == 1
    assert result['net_correct'] == 0
    assert not result['package_gate_passed']
    assert sum(group['validation_samples'] for group in result['groups']) == 6
    assert result['groups'][0]['classes'] == [1, 2]
    assert result['groups'][2]['classes'] == [0]
    assert result['by_class'][0]['delta_recall'] == .5
    assert result['by_class'][2]['delta_recall'] == -.5


def test_overlay_on_actual_pinned_trainer_compiles_and_preserves_disabled_loss_and_gate():
    commit = json.loads(pipeline.FIXED.read_text())['framework_commit']
    source = subprocess.check_output(['git', 'show',
        f'{commit}:reproducibility/aegis_f1/aegis_clip/trainer.py'], cwd=ROOT, text=True)
    updated = transform(source)
    compile(updated, 'private_trainer.py', 'exec')
    start = updated.index('                supported_base_loss = per_sample\n')
    end = updated.index('                if cyclic_enabled:', start)
    block = textwrap.dedent(updated[start:end])
    confidence_start = updated.index('                    admitted = local_admission(')
    confidence_end = updated.index('                    totals["attention_local_fallback_examples"]', confidence_start)
    gate_block = textwrap.dedent(updated[confidence_start:confidence_end])
    logits = torch.tensor([[0., 1.], [1., 0.]], requires_grad=True)
    targets = torch.eye(2)
    support = torch.tensor([.5, 0.])
    for epoch, enabled in [(1, True), (2, True), (3, False), (3, True)]:
        config = options()
        config['loss']['hard_support']['ce_restore_enabled'] = enabled
        py = logits.softmax(1).gather(1, targets.argmax(1)[:, None]).squeeze(1)
        base = F.cross_entropy(logits, targets.argmax(1), reduction='none') if epoch <= 2 else (1-py.sqrt())/.5
        scope = dict(per_sample=base, training_logits=logits, mixed_targets=targets, batch_support=support,
            config=config, epoch=epoch, totals={}, hard_support=support, batch_indices=torch.arange(2),
            restore_original_label_loss=restore_original_label_loss, classification_support=classification_support,
            record_classification=record_classification)
        exec(block, scope)
        if epoch <= 2 or not enabled:
            assert scope['per_sample'] is base
        else:
            expected = .5*base[0]+.5*F.cross_entropy(logits[:1], torch.tensor([0]))
            assert torch.allclose(scope['per_sample'][0], expected)
        scope.update(global_confidence=torch.tensor([.2, .7]), attention_local_confidence_gate=.7,
            local_inputs=torch.randn(2, 3, 8, 8), local_per_sample=base+1,
            local_admission=local_admission, torch=torch)
        exec(gate_block, scope)
        assert scope['fallback_mask'].tolist() == [True, False]
        assert torch.equal(scope['local_per_sample'], torch.where(
            scope['global_confidence'] >= .7, base+1, scope['per_sample']))
    with pytest.raises(RuntimeError, match='insertion point'):
        transform(updated)
