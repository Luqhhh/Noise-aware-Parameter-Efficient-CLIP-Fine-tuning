"""Geometry and effective-batch invariants for global/local continuation."""
import numpy as np
import torch
import pytest
from aegis_clip.prelim75_joint import fixed_choices, crop_with_bound_boxes, weighted_micro_loss
from aegis_clip.cli.infer_prelim75 import infer
import json
from aegis_clip.cli.train_prelim75_combination import select_combination


def test_choices_do_not_change_with_microbatch_partition():
    paths = [f'official/{i}.jpg' for i in range(32)]
    flip, scale = fixed_choices(paths, 1)
    af, asc = fixed_choices(paths[:16], 1)
    bf, bsc = fixed_choices(paths[16:], 1)
    np.testing.assert_array_equal(flip, np.r_[af, bf])
    np.testing.assert_array_equal(scale, np.r_[asc, bsc])
    assert np.isin(scale, [0, 1, 2, 3]).all()
    assert not np.array_equal(flip, fixed_choices(paths, 2)[0])


def test_native_box_crop_is_exact_tensor_slice_resize_and_differentiable():
    image = torch.arange(3*224*224, dtype=torch.float32).reshape(1, 3, 224, 224).requires_grad_(True)
    actual = crop_with_bound_boxes(image, [[5, 7, 117, 119]])
    expected = torch.nn.functional.interpolate(image[:, :, 7:119, 5:117], (224,224),
                                               mode='bilinear', align_corners=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.mean().backward()
    assert image.grad[:, :, 7:119, 5:117].abs().sum() > 0
    assert image.grad[:, :, :7].abs().sum() == 0
    with pytest.raises(ValueError, match='Invalid native'):
        crop_with_bound_boxes(image, [[-1, 7, 117, 119]])


def test_accumulation_matches_full_weighted_group_even_with_unequal_mass():
    config = {'name':'gce', 'gce_q':.5, 'epsilon':1e-7}
    targets = torch.tensor([[1.,0.],[.2,.8],[0.,1.],[1.,0.]])
    weights = torch.tensor([1.,1.,.1,0.])
    a = torch.tensor([[1.,-1.],[-.5,2.],[.2,.4],[2.,1.]],requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    total = weighted_micro_loss(a,targets,weights,weights.sum(),config,4)
    total.backward()
    first = weighted_micro_loss(b[:2],targets[:2],weights[:2],weights.sum(),config,4)
    second = weighted_micro_loss(b[2:],targets[2:],weights[2:],weights.sum(),config,4)
    (first+second).backward()
    torch.testing.assert_close(total, first+second)
    torch.testing.assert_close(a.grad,b.grad)


def test_submission_stops_before_test_access_on_engineering_stop(tmp_path):
    diagnostic = tmp_path/'H0/diagnostic'
    diagnostic.mkdir(parents=True)
    (diagnostic/'result.json').write_text(json.dumps({'status':'complete','engineering_stop':True}))
    with pytest.raises(ValueError, match='stopped'):
        infer({'output':str(tmp_path)},'H0')
    assert not (tmp_path/'H0/submission').exists()


def test_combination_requires_real_scores_and_closes_at_target():
    assert select_combination({})['status'] == 'pending_real_platform_feedback'
    scores = {k:{'source':'user_reported_platform','accuracy_percent':67.1} for k in ('H0','H1','V0','V1')}
    assert select_combination({k:scores[k] for k in ('H0','V0')})['status'] == 'pending_real_platform_feedback'
    choice = select_combination(scores)
    assert choice['head_winner']=='H0' and choice['visual_winner']=='V0'
    scores['H0']['accuracy_percent'] = 75.
    assert select_combination(scores)['status']=='target_reached_no_new_training'
    scores['H0']['accuracy_percent'] = None
    with pytest.raises(ValueError, match='real user-reported'):
        select_combination(scores)


@pytest.mark.parametrize('route', [('H0', 'H1'), ('V0', 'V1')])
def test_scored_route_below_gate_closes_C_without_other_route_feedback(route):
    scores = {name: {'source': 'user_reported_platform', 'accuracy_percent': score}
              for name, score in zip(route, (65.8629, 66.1914))}
    assert select_combination(scores)['status'] == 'closed_below_combination_gate'
    scores[route[1]]['accuracy_percent'] = 67.0681
    assert select_combination(scores)['status'] == 'pending_real_platform_feedback'


def test_stopped_route_member_does_not_keep_C_waiting_or_win_selection():
    scores = {'H0': {'source': 'user_reported_platform', 'accuracy_percent': 66.5}}
    assert select_combination(scores, stopped=('H1',))['status'] == 'closed_below_combination_gate'
    assert select_combination({}, stopped=('H0', 'H1'))['status'] == 'closed_no_eligible_route_pair'
    scores = {name: {'source': 'user_reported_platform', 'accuracy_percent': score}
              for name, score in zip(('H0', 'H1', 'V0', 'V1'), (67.1, 70., 67.1, 67.1))}
    choice = select_combination(scores, stopped=('H1',))
    assert choice['status'] == 'eligible'
    assert choice['head_winner'] == 'H0'
