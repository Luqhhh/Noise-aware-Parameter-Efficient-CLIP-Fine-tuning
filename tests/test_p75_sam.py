"""CPU contract for complete effective-batch L05 SAM with fixed local crops."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/p75_runtime'))
sys.path.insert(0, str(ROOT/'reproducibility/aegis_f1'))
import p75_sam  # noqa: E402


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.classifier = nn.Linear(4, 3)

    def forward(self, *, images, return_features=False):
        feature = F.normalize(self.encoder(images.mean(dim=(2, 3))), dim=1)
        logits = self.classifier(feature)
        return (logits, feature) if return_features else logits


def microbatch(model, image, label):
    feature = torch.zeros(image.shape[0], 4)
    return {'forward_key': 'images', 'forward_inputs': image,
            'mixed_gate': None, 'mixed_reference': feature,
            'mixed_targets': F.one_hot(label, 3).float(),
            'mixed_weights': torch.ones(len(label)),
            'class_counts': torch.ones(3), 'prior_tau': 0.,
            'loss_config': {'name': 'gce', 'gce_q': .5, 'ce_warmup_epochs': 0},
            'epoch': 5, 'batch_suspicious': None, 'distill_weight': 0.,
            'sample_count': len(label),
            'local_attention': torch.tensor([[[.1, .4], [.2, .3]]]).expand(len(label), -1, -1),
            'local_crop_size': 4, 'local_top_patches': 1,
            'local_weight': .25, 'local_gate': .7}


def make_first_pass(model, batches):
    model.zero_grad(set_to_none=True)
    state = p75_sam.trainer_rng_state(torch.device('cpu'))
    for batch in batches:
        loss = p75_sam._second_loss(model, batch, torch.device('cpu'))
        batch['first_loss'] = float(loss.detach())
        (loss * batch['sample_count']/sum(x['sample_count'] for x in batches)).backward()
    return state


def test_full_batch_local_sam_updates_once():
    torch.manual_seed(7)
    model = TinyModel()
    images = torch.randn(4, 3, 8, 8)
    batches = [microbatch(model, images[:2], torch.tensor([0, 1])),
               microbatch(model, images[2:], torch.tensor([2, 0]))]
    state = make_first_pass(model, batches)
    original = [x.detach().clone() for x in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    calls = []
    original_step = optimizer.step

    def step(*args, **kwargs):
        calls.append(1)
        return original_step(*args, **kwargs)

    optimizer.step = step
    receipt = p75_sam.full_l05_sam_step(
        model, optimizer, microbatches=batches, cycle_samples=4, rho=.05,
        clip_norm=1., gsam_alpha=None, rng_state=state, device=torch.device('cpu'))
    assert len(calls) == 1
    assert receipt['actual_radius'] == pytest.approx(.05, abs=1e-6)
    assert receipt['first_loss'] > 0 and receipt['second_loss'] > 0
    assert any(not torch.equal(x, y) for x, y in zip(model.parameters(), original))


def test_second_pass_exception_restores_parameters(monkeypatch):
    model = TinyModel()
    image = torch.randn(2, 3, 8, 8)
    batches = [microbatch(model, image, torch.tensor([0, 1]))]
    state = make_first_pass(model, batches)
    original = [x.detach().clone() for x in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    monkeypatch.setattr(p75_sam, '_second_loss', lambda *_args: (_ for _ in ()).throw(RuntimeError('boom')))
    with pytest.raises(RuntimeError, match='boom'):
        p75_sam.full_l05_sam_step(model, optimizer, microbatches=batches,
            cycle_samples=2, rho=.05, clip_norm=1., gsam_alpha=None,
            rng_state=state, device=torch.device('cpu'))
    assert all(torch.equal(x, y) for x, y in zip(model.parameters(), original))
