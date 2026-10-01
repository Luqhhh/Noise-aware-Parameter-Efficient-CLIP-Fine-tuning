import random

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from aegis_clip.v1_pipeline import image_transform
from aegis_clip.v1_strategy import weighted_mixup_loss
from v1_continuation.runtime import mixed_backward
from strong_aug_pair.core import (ARMS, training_transform, augmentation_rng, PairedImages,
    epoch_order, mixup_draw, group_excluded_maximum, metrics, paired, review_decision)
from strong_aug_pair.source import check_protocol, ROOT


def test_exact_recipes_and_deterministic_validation():
    control, candidate = [training_transform(arm).transforms for arm in ARMS]
    assert [type(op) for op in control] == [T.RandomResizedCrop, T.RandomHorizontalFlip,
                                           T.RandAugment, T.ToTensor, T.Normalize]
    assert [type(op) for op in candidate] == [T.RandomResizedCrop, T.RandomHorizontalFlip,
        T.ColorJitter, T.RandAugment, T.ToTensor, T.Normalize, T.RandomErasing]
    assert control[0].scale == (.8, 1.) and candidate[0].scale == (.35, 1.)
    assert candidate[0].ratio == (3/4, 4/3)
    assert candidate[0].interpolation == T.InterpolationMode.BICUBIC
    assert candidate[-1].p == .3 and candidate[-1].value == "random"
    image = Image.fromarray(np.random.default_rng(2).integers(0, 256, (120, 80, 3), dtype=np.uint8))
    torch.manual_seed(1)
    a = image_transform(448)(image)
    torch.manual_seed(999)
    assert torch.equal(a, image_transform(448)(image))


def test_augmentation_cannot_perturb_sampler_or_mixup():
    random.seed(13); np.random.seed(13); torch.manual_seed(13)
    state = torch.get_rng_state().clone()
    expected = (random.getstate(), np.random.get_state())
    for arm in ARMS:
        with augmentation_rng(53):
            training_transform(arm)(Image.new("RGB", (500, 300), (70, 160, 110)))
            np.random.rand(57); random.random()
        assert torch.equal(torch.get_rng_state(), state)
        assert random.getstate() == expected[0]
        assert np.array_equal(np.random.get_state()[1], expected[1][1])
    order = epoch_order(113, 42, 1)
    assert len(set(order)) == 113 and order != epoch_order(113, 42, 2)
    a, lam = mixup_draw(17, 42, 1, 4)
    torch.rand(151); np.random.rand(200)
    b, lam_b = mixup_draw(17, 42, 1, 4)
    assert torch.equal(a, b) and lam == lam_b


def test_dataset_reproducibility_independent_of_access_order(tmp_path):
    (tmp_path / "0000").mkdir()
    for i in range(3):
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (72, 90, 3), dtype=np.uint8)).save(tmp_path / "0000" / f"{i}.png")
    rows = [dict(image_path=f"train/0000/{i}.png") for i in range(3)]
    data = PairedImages(tmp_path, rows, ARMS[1], 1, 42)
    a = data[1][0]
    data[0]; data[2]
    assert torch.equal(a, data[1][0])
    assert not torch.equal(a, PairedImages(tmp_path, rows, ARMS[1], 2, 42)[1][0])


@pytest.mark.parametrize("micro", [1, 2, 4, 7])
def test_accumulation_matches_full_logical_weighted_mixup(micro):
    torch.manual_seed(7)
    full, accumulated = torch.nn.Linear(5, 3), torch.nn.Linear(5, 3)
    accumulated.load_state_dict(full.state_dict())
    images = torch.randn(7, 5)
    targets = torch.rand(7, 3); targets /= targets.sum(1, keepdim=True)
    weights = torch.tensor([0., .2, 1., .7, 0., 1., .9])
    permutation, lam = mixup_draw(7, 42, 2, 17)
    loss = weighted_mixup_loss(full(lam*images+(1-lam)*images[permutation]), targets, weights, permutation, lam)
    loss.backward()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    actual = mixed_backward(accumulated, images, targets, weights, permutation, lam, micro, scaler)
    assert actual == pytest.approx(loss.item(), abs=2e-7)
    for a, b in zip(full.parameters(), accumulated.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-7, rtol=1e-6)


def test_group_exclusion_and_validation_gallery_is_train_only():
    gallery = torch.tensor([[1., 0.], [1., 0.], [.6, .8], [0., 1.]])
    groups = torch.tensor([0, 0, 1, 2])
    actual = group_excluded_maximum(gallery, gallery, groups, groups, query_chunk=2, gallery_chunk=2)
    assert np.allclose(actual, [.6, .6, .8, .8])
    validation = torch.tensor([[-1., 0.]])
    assert group_excluded_maximum(validation, gallery, None, None)[0] == pytest.approx(0.)


def test_readouts_count_corrections_and_regressions():
    y = np.array([0, 0, 1, 1])
    a, b = np.array([1, 0, 1, 0]), np.array([0, 1, 1, 1])
    groups = dict(all=np.ones(4, dtype=bool), target=np.array([1, 0, 0, 1], dtype=bool))
    result = paired(y, a, b, 2, groups)
    assert result['all']['corrections'] == 2 and result['all']['regressions'] == 1
    assert result['all']['net'] == 1 and result['target']['net'] == 2
    assert metrics(y, b, 2)['macro'] == .75


def test_protocol_rejects_extra_epochs_and_time_limits():
    import json
    protocol = json.loads((ROOT / 'configs/team_exploration_20261001/machine_b.json').read_text())
    check_protocol(protocol)
    for key, value in [('epochs_per_arm', 5), ('wall_clock_time_limits', True), ('platform_upload', True)]:
        with pytest.raises(ValueError):
            check_protocol(dict(protocol, **{key: value}))


def test_review_requires_target_and_complement():
    good = dict(all=dict(net=80, corrections=400, regressions=320),
                target=dict(net=25), complement=dict(net=55), tail75=dict(net=0))
    assert review_decision(good, good)['status'] == 'supports_review'
    assert review_decision(dict(good, target=dict(net=24)), good)['status'] == 'closed_fixed_recipe'
    assert review_decision(dict(good, complement=dict(net=-1)), good)['status'] == 'closed_fixed_recipe'


def test_real_four_epoch_loop_exports_mean_of_only_epoch2_to4(tmp_path, monkeypatch):
    """Exercise the actual orchestration/optimizer with a small CPU classifier."""
    import json
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, TensorDataset
    import strong_aug_pair.runtime as runtime

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(5, 5)
            self.head = torch.nn.Linear(5, 750)
            with torch.no_grad():
                self.head.bias[0] = 8.

        def forward(self, x):
            return self.head(self.visual(x))

    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    as_tensor = torch.as_tensor
    def cpu_tensor(value, **kwargs):
        kwargs.pop('device', None)
        return as_tensor(value, **kwargs)
    monkeypatch.setattr(torch, 'as_tensor', cpu_tensor)
    scaler_class = torch.amp.GradScaler
    monkeypatch.setattr(torch.amp, 'GradScaler', lambda *args, **kwargs: scaler_class('cuda', enabled=False))
    original_update = runtime.logical_update
    def cpu_update(model, optimizer, images, probabilities, weights, permutation, lam,
                   micro_batch, scaler, amp, grad_clip):
        return original_update(model, optimizer, images, probabilities, weights, permutation, lam,
                               micro_batch, scaler, False, grad_clip)
    monkeypatch.setattr(runtime, 'logical_update', cpu_update)
    images = torch.arange(35, dtype=torch.float32).reshape(7, 5)/35
    data = DataLoader(TensorDataset(images, torch.arange(7)), batch_size=3, shuffle=False)
    monkeypatch.setattr(runtime, 'training_loader', lambda source, arm, epoch: data)
    def prediction(model, source, rows, output, phase, **kwargs):
        with torch.no_grad():
            logits = model(torch.ones(len(rows), 5)).numpy()
        return logits, None, .01
    monkeypatch.setattr(runtime, 'predict', prediction)
    source = SimpleNamespace(model=Tiny, binding=dict(test='cpu_actual_four_epoch_loop'), active=np.arange(7),
        classes=[f'{i:04d}' for i in range(750)], val=[dict(image_path=f'train/0000/{i}', label='0') for i in range(4)],
        supervision=dict(labels=np.zeros(7, dtype=np.int64), targets=np.zeros(7, dtype=np.int64),
                         weights=np.array([1., .2, .8, .3, 1., .5, .7], dtype=np.float32),
                         original_alpha=np.zeros(7, dtype=np.float32)))
    (tmp_path/'plan.json').write_text('{}')
    (tmp_path/'frozen_groups.csv').write_text('test')
    baseline = np.zeros(4, dtype=np.int64)
    masks = dict(all=np.ones(4, dtype=bool), target=np.ones(4, dtype=bool),
                 complement=np.zeros(4, dtype=bool), tail75=np.ones(4, dtype=bool))
    runtime.train_arm(dict(logical_batches_per_epoch=3), source, tmp_path, ARMS[0], masks, baseline)
    root = tmp_path/'arms'/ARMS[0]
    primary = torch.load(root/'ema_swa_2_4.pt', weights_only=False)
    assert primary['epoch'] == 4 and primary['optimizer_updates'] == 12
    assert primary['average_epochs'] == [2, 3, 4] and primary['complete'] is True
    states = [torch.load(root/f'epoch_{i:02d}.pt', weights_only=False)['selected_state'] for i in (2, 3, 4)]
    for name, value in primary['selected_state'].items():
        assert torch.allclose(value, sum(state[name] for state in states)/3, atol=1e-6, rtol=1e-6)
    draws = [json.loads(row) for row in (root/'draws.jsonl').read_text().splitlines()]
    assert len(draws) == 12 and all(len(row['indices'])==1 for row in draws if row['batch']==3)
