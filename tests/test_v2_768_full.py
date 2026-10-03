"""Full-route architecture identity, gradients, legacy dispatch and stage handoff."""
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproducibility/aegis_f1'))
from v2.architecture import architecture_spec, check_plan_architecture
from v2.core import check_checkpoint, check_checkpoint_architecture
from v2.preprojection import V2FeatureClassifier, fold_projection


@pytest.mark.parametrize('value', [True, 256, '768', None, 768.0])
def test_reject_invalid_feature_dimension(value):
    with pytest.raises(ValueError, match='feature_dim'):
        architecture_spec({'feature_dim': value})


def test_factory_preserves_legacy_class_and_explicit_768(monkeypatch):
    import v2.preprojection as module
    monkeypatch.setattr(module, 'V2Classifier', lambda recipe: ('legacy', recipe))
    monkeypatch.setattr(module, 'V2FeatureClassifier', lambda recipe, dim: ('new', dim))
    assert module.build_classifier({}) == ('legacy', {})
    assert module.build_classifier({'feature_dim': 512})[0] == 'legacy'
    assert module.build_classifier({'feature_dim': 768}) == ('new', 768)


def explicit_checkpoint():
    from tests.test_v2 import checkpoint_fixture
    payload, plan = checkpoint_fixture()
    plan['recipe']['feature_dim'] = 768
    spec = architecture_spec(plan['recipe'])
    plan['architecture'] = spec
    payload.update(architecture=copy.deepcopy(spec), config={'model': {'architecture': copy.deepcopy(spec)}},
                   model={'head.weight': torch.zeros(750, 768), 'head.bias': torch.zeros(750)})
    payload['binding']['architecture'] = copy.deepcopy(spec)
    return payload, plan


@pytest.mark.parametrize('change', ['head', 'projection', 'binding', 'config', 'plan', 'ema'])
def test_checkpoint_rejects_cross_architecture_state(change):
    payload, plan = explicit_checkpoint()
    check_checkpoint(payload, plan, 's1_384')
    if change == 'head': payload['model']['head.weight'] = torch.zeros(750, 512)
    elif change == 'projection': payload['model']['visual.proj'] = torch.zeros(768, 512)
    elif change == 'binding': payload['binding']['architecture']['feature_dim'] = 512
    elif change == 'config': payload['config']['model']['architecture']['feature_dim'] = 512
    elif change == 'plan': plan['architecture'] = architecture_spec({})
    elif change == 'ema': payload['ema'] = {'head.weight': torch.zeros(750, 512), 'head.bias': torch.zeros(750)}
    with pytest.raises(ValueError):
        check_checkpoint(payload, plan, 's1_384')


def test_stage_config_must_keep_768_architecture():
    payload, plan = explicit_checkpoint()
    check_plan_architecture(plan, payload['config'])
    with pytest.raises(ValueError):
        check_plan_architecture(plan, {'model': {'architecture': architecture_spec({})}})
    legacy = copy.deepcopy(plan)
    del legacy['recipe']['feature_dim']
    with pytest.raises(ValueError):
        check_plan_architecture(legacy)


def tiny_preprojection(checkpointing):
    from clip.model import VisionTransformer
    model = V2FeatureClassifier.__new__(V2FeatureClassifier)
    torch.nn.Module.__init__(model)
    model.visual = VisionTransformer(224, 32, 64, 2, 4, 32).float()
    model.visual.proj = None
    model.head = torch.nn.Linear(64, 7)
    model.feature_dim = 768  # Route identity; the synthetic tower is intentionally small.
    model.gradient_checkpointing = checkpointing
    return model


def test_preprojection_forward_interpolation_and_checkpointed_gradients():
    torch.manual_seed(91)
    model = tiny_preprojection(True)
    plain = copy.deepcopy(model)
    plain.gradient_checkpointing = False
    images = torch.randn(2, 3, 384, 384)
    labels = torch.tensor([0, 1])
    assert model.embed(images).shape == (2, 64)
    torch.nn.functional.cross_entropy(model(images), labels).backward()
    torch.nn.functional.cross_entropy(plain(images), labels).backward()
    for (name, param), other in zip(model.named_parameters(), plain.parameters()):
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        torch.testing.assert_close(param.grad, other.grad, atol=1e-6, rtol=1e-5)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(images)
    assert output.dtype == torch.float32
    assert 'visual.proj' not in model.state_dict()


def test_projection_fold_preserves_old_function_but_new_head_has_extra_directions():
    generator = torch.Generator().manual_seed(712)
    state = {'visual.proj': torch.randn(16, 9, generator=generator, dtype=torch.float64),
             'head.weight': torch.randn(12, 9, generator=generator, dtype=torch.float64),
             'head.bias': torch.randn(12, generator=generator, dtype=torch.float64)}
    folded = fold_projection(state)
    features = torch.randn(7, 16, dtype=torch.float64)
    before = features @ state['visual.proj'] @ state['head.weight'].T + state['head.bias']
    after = features @ folded['head.weight'].T + folded['head.bias']
    torch.testing.assert_close(before, after, atol=1e-12, rtol=1e-12)
    outside = torch.linalg.qr(state['visual.proj'], mode='complete').Q[:, 9:]
    weight = folded['head.weight'].clone().requires_grad_()
    torch.nn.functional.cross_entropy(outside.T @ weight.T, torch.arange(7)).backward()
    assert (weight.grad @ outside).norm() > .1


def test_stage_handoff_uses_the_selected_768_parent(monkeypatch, tmp_path):
    import v2.runtime as runtime
    calls = []
    model = torch.nn.Linear(2, 2)
    parent_state = {name: torch.full_like(value, 3.) for name, value in model.state_dict().items()}
    parent = dict(model={name: torch.zeros_like(value) for name, value in model.state_dict().items()},
                  ema=parent_state, ema_enabled=True, metrics={'chosen': 'ema'})
    monkeypatch.setattr(runtime, 'build_classifier', lambda recipe: calls.append(recipe['feature_dim']) or model)
    monkeypatch.setattr(runtime, 'read_checkpoint', lambda *a: parent)
    monkeypatch.setattr(runtime, 'sha', lambda *a: 'verified-parent')
    plan = {'recipe': {'feature_dim': 768, 'official_sha256': 'official'}}
    actual, source = runtime.initialize_model(plan, tmp_path, 's2_448', torch.device('cpu'))
    assert calls == [768] and source['weights'] == 'ema'
    for value in actual.state_dict().values(): assert (value == 3.).all()


def test_explicit_controller_runs_fresh_s1_then_all_remaining_stages(monkeypatch, tmp_path):
    import v2.controller as controller
    from v2.plan import STAGES, dump, sha
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text('{}')
    plan = dict(recipe={'feature_dim': 768, 'initial_parent': 'official_openai',
                        'test_rows': 37444, 'test_root': str(tmp_path / 'test')},
                architecture=architecture_spec({'feature_dim': 768}),
                split={'val_rows': 14880},
                stages={s: {'total_updates': 2, 'epochs': 1} for s in STAGES})
    calls = []
    monkeypatch.setattr(controller, 'verify_prepared', lambda *a: plan)
    monkeypatch.setattr(controller, 'read_checkpoint', lambda *a: {})
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *a, **kw: '')
    monkeypatch.setattr(controller.subprocess, 'run', lambda *a, **kw: None)
    def train(plan_path, authorization, stage, probe_steps=0):
        calls.append(('probe' if probe_steps else 'train', stage))
        if probe_steps:
            dump(tmp_path / 'probes' / stage / 'cost.json', dict(
                status='cost_probe_only', no_candidate=True, plan_sha256=sha(plan_path),
                measured_seconds_per_update=1., measured_validation_seconds=1.,
                measured_overhead_seconds=1., peak_allocated_bytes=1))
        else:
            dump(tmp_path / 'runs' / stage / 'status.json', {'status': 'complete'})
    def infer(plan_path, authorization, output):
        output.mkdir()
        (output / 'pred_results.csv').write_bytes(b'synthetic\n')
        (output / 'submission.zip').write_bytes(b'synthetic')
        dump(output / 'report.json', dict(status='package_ready', rows=37444, weights='raw',
             csv_sha256=sha(output / 'pred_results.csv'), zip_sha256=sha(output / 'submission.zip')))
    monkeypatch.setattr(controller, 'train', train)
    monkeypatch.setattr(controller, 'infer', infer)
    controller.run(plan_path, from_official=True)
    assert [s for kind, s in calls if kind == 'train'] == list(STAGES)
    assert [s for kind, s in calls if kind == 'probe'] == list(STAGES[:3])


def test_controller_cannot_silently_reuse_old_s1(monkeypatch, tmp_path):
    import v2.controller as controller
    plan = dict(recipe={'feature_dim': 768, 'initial_parent': 'official_openai'},
                imported_stages={'s1_384': {}})
    monkeypatch.setattr(controller, 'verify_prepared', lambda *a: plan)
    with pytest.raises(ValueError, match='without imported'):
        controller.run(tmp_path / 'plan.json', from_official=True)


def test_768_inference_loads_raw_state_and_writes_checked_package(monkeypatch, tmp_path):
    """Two synthetic images exercise the actual inference/reduction/ZIP/report path."""
    from contextlib import nullcontext
    from PIL import Image
    from torchvision.transforms import ToTensor
    import v2.runtime as runtime
    from v2.plan import json_read
    root = tmp_path / 'images'
    root.mkdir()
    for name in ('a.png', 'b.png'):
        Image.new('RGB', (32, 32), (64, 32, 16)).save(root / name)
    (tmp_path / 'test_manifest.csv').write_text('image_path\na.png\nb.png\n')
    spec = architecture_spec({'feature_dim': 768})
    plan = dict(architecture=spec, recipe=dict(feature_dim=768, num_classes=2,
        test_root=str(root), stage_artifacts=str(tmp_path), test_rows=2, micro_batch_size=2,
        views=['resize576', 'center', 'flip', 'resize806.4']))

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(768, 2)
        def forward(self, images):
            return self.head(images.mean((2, 3)).repeat(1, 256))

    payload = dict(architecture=spec, binding={'architecture': spec},
        config={'model': {'architecture': spec}}, epoch=4, metrics={'chosen': 'raw'},
        model={'head.weight': torch.zeros(2, 768), 'head.bias': torch.tensor([0., 1.])})
    checkpoint = tmp_path / 'runs/full_576/last.pt'
    checkpoint.parent.mkdir(parents=True)
    runtime.save_checkpoint(checkpoint, payload)
    calls = []
    def read_checkpoint(path, actual_plan, stage):
        loaded = torch.load(path, weights_only=False)
        check_checkpoint_architecture(loaded, actual_plan)
        return loaded
    monkeypatch.setattr(runtime, 'read_checkpoint', read_checkpoint)
    monkeypatch.setattr(runtime, 'build_classifier', lambda recipe: calls.append(recipe['feature_dim']) or Tiny())
    monkeypatch.setattr(runtime, 'verify_prepared', lambda *a: plan)
    monkeypatch.setattr(runtime, 'authorize', lambda *a: {'max_seconds': 60})
    monkeypatch.setattr(runtime, 'device_after_authorization', lambda: torch.device('cpu'))
    monkeypatch.setattr(runtime, 'amp', nullcontext)
    monkeypatch.setattr(runtime, 'build_view', lambda *a: ToTensor())
    monkeypatch.setattr(runtime, 'loader', lambda dataset, cfg, batch_size:
                        torch.utils.data.DataLoader(dataset, batch_size=batch_size))
    output = tmp_path / 'submission'
    runtime.infer(tmp_path / 'plan.json', 'synthetic', output)
    assert calls == [768]
    assert (output / 'pred_results.csv').read_text() == 'a.png, 0001\nb.png, 0001\n'
    report = json_read(output / 'report.json')
    assert report['architecture'] == spec and report['weights'] == 'raw' and report['rows'] == 2
    assert report['platform_metrics'] is None
