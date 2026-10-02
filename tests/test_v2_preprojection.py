import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproducibility/aegis_f1'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from v2.preprojection import fold_projection, V2FeatureClassifier


def state():
    generator = torch.Generator().manual_seed(712)
    return {'visual.proj': torch.randn(16, 9, generator=generator),
            'head.weight': torch.randn(12, 9, generator=generator),
            'head.bias': torch.randn(12, generator=generator),
            'visual.example': torch.randn(2, generator=generator)}


def test_fold_preserves_function_and_feature_gradient():
    source = {k: v.double() for k, v in state().items()}
    target = fold_projection(source)
    x = torch.randn(7, 16, dtype=torch.double, requires_grad=True)
    before = (x @ source['visual.proj']) @ source['head.weight'].T + source['head.bias']
    after = x @ target['head.weight'].T + target['head.bias']
    torch.testing.assert_close(before, after, atol=1e-12, rtol=1e-12)
    a, = torch.autograd.grad(before.square().sum(), x, retain_graph=True)
    b, = torch.autograd.grad(after.square().sum(), x)
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-12)
    assert 'visual.proj' not in target
    assert torch.equal(target['visual.example'], source['visual.example'])
    target['visual.example'].add_(1)
    assert not torch.equal(target['visual.example'], source['visual.example'])


def test_direct_head_can_learn_outside_original_projection():
    source = state()
    target = fold_projection(source)
    q = torch.linalg.qr(source['visual.proj'], mode='complete').Q
    x = q[:, 9:].T
    old = (x @ source['visual.proj']) @ source['head.weight'].T
    assert old.abs().max() < 1e-5
    w = target['head.weight'].clone().requires_grad_()
    torch.nn.functional.cross_entropy(x @ w.T + target['head.bias'], torch.arange(7)).backward()
    assert (w.grad @ q[:, 9:]).norm() > 0.1


def test_invalid_or_nonfinite_projection_rejected():
    source = state()
    source['visual.proj'][0, 0] = float('nan')
    with pytest.raises(ValueError):
        fold_projection(source)
    source = state()
    source['head.weight'] = torch.zeros(12, 8)
    with pytest.raises(ValueError):
        fold_projection(source)


@pytest.mark.parametrize('dimension', [512, 768])
def test_common_readout_remains_fp32_under_autocast(dimension):
    model = object.__new__(V2FeatureClassifier)
    torch.nn.Module.__init__(model)
    model.visual = torch.nn.Module()
    model.visual.proj = torch.nn.Parameter(torch.randn(768, 512)) if dimension == 512 else None
    model.head = torch.nn.Linear(dimension, 5)
    features = torch.randn(3, 768)
    expected = model.readout(features)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = model.readout(features)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_schedule_only_selects_supplied_training_population_and_is_reproducible():
    from train_v2_preprojection_pair import schedule
    import numpy as np
    cfg = dict(seed=20260926, steps=5, logical_batch_size=8)
    labels = [0, 0, 0, 1, 1, 2]
    a, b = schedule(labels, 3, cfg), schedule(labels, 3, cfg)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert a['indices'].shape == a['image_seeds'].shape == (40,)
    assert a['mix_seeds'].shape == (5,)
    assert 0 <= a['indices'].min() <= a['indices'].max() < len(labels)


def test_review_requires_parent_control_and_mechanism_improvement():
    from train_v2_preprojection_pair import review
    gate = dict(net_vs_control_min=75, net_vs_parent_min=75, common_error_net_vs_control_min=25,
                macro_delta_vs_control_min=0., corrections_to_regressions_min=1.25)
    metric = {'candidate': {'all': {'macro': .8}}, 'control': {'all': {'macro': .79}}}
    comparisons = {'candidate_vs_control': {'all': {'net': 80, 'corrections': 120, 'regressions': 40},
                    'common_five_errors': {'net': 25}}, 'candidate_vs_parent': {'all': {'net': 80}}}
    assert review(metric, comparisons, gate)
    comparisons['candidate_vs_parent']['all']['net'] = -10
    assert not review(metric, comparisons, gate)
    comparisons['candidate_vs_parent']['all']['net'] = 80
    comparisons['candidate_vs_control']['common_five_errors']['net'] = 24
    assert not review(metric, comparisons, gate)
