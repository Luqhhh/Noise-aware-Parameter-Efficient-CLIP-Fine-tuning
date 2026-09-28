"""CPU contract for the single-checkpoint patch readout."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/p75_runtime'))
from p75_patch_readout import install  # noqa: E402


class FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = FakeTransformer()
        self.ln_post = nn.LayerNorm(4)
        self.proj = nn.Parameter(torch.eye(4))


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, images):
        return self.linear(images).permute(1, 0, 2)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = FakeVisual()
        self.classifier = nn.Linear(4, 3)
        self.feature_dim = 4
        self.peft_mode = 'full_finetune'

    def encode_image(self, images):
        tokens = self.visual.transformer(images)
        cls = self.visual.ln_post(tokens[0]) @ self.visual.proj
        return F.normalize(cls, dim=-1)

    def forward(self, images):
        return self.classifier(self.encode_image(images))

    def parameter_groups(self, head_lr=1., head_weight_decay=0., **_kwargs):
        return [{'name': 'head', 'params': list(self.classifier.parameters()),
                 'lr': head_lr, 'weight_decay': head_weight_decay}]

    def effective_spec(self):
        return {'backbone': 'fake'}


def test_initial_path_and_single_head_gradient():
    torch.manual_seed(2)
    model = FakeModel()
    images = torch.randn(3, 5, 4)
    original = model(images).detach()
    parent = {key: value.clone() for key, value in model.state_dict().items()}
    install(model)
    assert torch.allclose(model(images), original, atol=1e-6, rtol=0)
    assert model.effective_spec()['patch_readout'] == 'single_head_zero_residual_v1'
    groups = model.parameter_groups()
    assert len(groups) == 1
    assert {id(p) for p in model.patch_readout.parameters()} <= {
        id(p) for p in groups[0]['params']}
    assert model.load_state_dict(parent, strict=False).missing_keys == []
    loss = model(images).square().sum()
    loss.backward()
    assert model.patch_readout.residual.weight.grad.abs().sum() > 0
    assert len(model.state_dict()) == len(parent) + 3


def test_partial_or_strict_parent_load_fails():
    model = FakeModel()
    parent = dict(model.state_dict())
    install(model)
    try:
        model.load_state_dict(parent, strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Strict load accepted missing readout')
    parent['patch_readout.score.weight'] = model.patch_readout.score.weight.detach().clone()
    try:
        model.load_state_dict(parent, strict=False)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Partial readout state was accepted')
