"""Check shared sampling/Mixup and that the intervention consumes no extra RNG."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_v1_detail_aug_pair import crop_tensor, pil_augmentation, schedule


def test_frozen_sample_and_mixup_schedule_are_shared_and_complete():
    cfg = json.loads((ROOT / 'configs/v1_detail_aug_pair_20261001.json').read_text())
    active = np.arange(0, 100000, 2)
    a, b = schedule(active, cfg), schedule(active, cfg)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert len(set(a['train_indices'])) == 32768
    assert all(i % 2 == 0 for i in a['train_indices'])
    assert np.all(a['detail_flags'].reshape(-1, 32).sum(1) == 16)
    assert np.all(np.sort(a['mixup_permutations'], axis=1) == np.arange(32))
    assert np.all((a['mixup_lambdas'] > 0) & (a['mixup_lambdas'] < 1))


def test_detail_reduction_keeps_later_shared_augmentation_rng():
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (650, 580, 3), dtype=np.uint8))
    following, crops, states = [], [], []
    for reduced in (False, True):
        torch.manual_seed(42)
        augment = pil_augmentation()
        crop = augment(image)
        crops.append(np.asarray(crop).copy())
        tensor = crop_tensor(crop, reduced)
        states.append(torch.get_rng_state())
        following.append(np.asarray(augment(image)).copy())
        assert tensor.shape == (3, 512, 512) and torch.isfinite(tensor).all()
    assert np.array_equal(crops[0], crops[1])
    assert torch.equal(states[0], states[1])
    assert np.array_equal(following[0], following[1])


def test_the_intervention_changes_information_without_changing_crop_geometry():
    image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (512, 512, 3), dtype=np.uint8))
    a, b = crop_tensor(image, False), crop_tensor(image, True)
    assert a.shape == b.shape == (3, 512, 512)
    assert not torch.equal(a, b)
