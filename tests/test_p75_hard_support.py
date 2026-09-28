"""CPU checks for original-label restoration and strict evidence alignment."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/p75_runtime'))
from p75_hard_support import (file_sha256, load_support, local_admission,
                              restore_original_label_loss)  # noqa: E402


def test_blend_recovers_ce_without_changing_other_samples():
    logits = torch.tensor([[0., 2.], [2., 0.]], requires_grad=True)
    targets = F.one_hot(torch.tensor([0, 0]), num_classes=2).float()
    probabilities = logits.softmax(1).gather(1, torch.tensor([[0], [0]])).squeeze(1)
    gce = (1. - probabilities.sqrt()) / .5
    support = torch.tensor([.5, 0.])
    blended = restore_original_label_loss(gce, logits, targets, support)
    ce = F.cross_entropy(logits, torch.tensor([0, 0]), reduction='none')
    assert torch.allclose(blended[0], .5*gce[0] + .5*ce[0])
    assert torch.allclose(blended[1], gce[1])
    blended.sum().backward()
    assert logits.grad.isfinite().all()


def test_local_gate_retains_quality_check():
    local = torch.randn(2, 3, 8, 8)
    local[0] = 0
    admitted = local_admission(torch.tensor([.2, .2]), .7,
                               torch.tensor([.5, .5]), local)
    assert admitted.tolist() == [False, True]
    baseline = local_admission(torch.tensor([.2, .8]), .7, None, local)
    assert baseline.tolist() == [False, True]


def test_support_csv_requires_exact_order_and_hash(tmp_path):
    path = tmp_path/'support.csv'
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_path', 'label', 'support_weight'])
        writer.writerow(['train/a.jpg', 1, .5])
        writer.writerow(['train/b.jpg', 0, 0.])
    config = {'loss': {'hard_support': {'enabled': True, 'version': 1,
              'maximum_weight': .5, 'path': str(path), 'sha256': file_sha256(path)}}}
    assert load_support(config, ['a.jpg', 'b.jpg'], [1, 0]).tolist() == [.5, 0.]
    with pytest.raises(ValueError, match='paths'):
        load_support(config, ['b.jpg', 'a.jpg'], [1, 0])
    config['loss']['hard_support']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='hash'):
        load_support(config, ['a.jpg', 'b.jpg'], [1, 0])
