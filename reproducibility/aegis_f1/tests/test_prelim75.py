"""Hand-calculated supervision/prior checks for the fixed competition run."""
import math
import numpy as np
import pytest
import torch
from aegis_clip.prelim75 import effective_counts, head_loss, cached_logits
from aegis_clip.model import AegisCLIP


def test_effective_supervision_is_soft_weighted_not_argmax_counted():
    q = torch.tensor([[1., 0.], [.25, .75], [0., 1.]])
    w = torch.tensor([1., .5, 0.])
    torch.testing.assert_close(effective_counts(w, q), torch.tensor([1.125, .375], dtype=torch.float64))
    with pytest.raises(ValueError, match='sum to one'):
        effective_counts(w, q * .5)


def test_uniform_prior_preserves_loss_and_gradient():
    a = torch.tensor([[[1., 2.], [3., -1.]]], requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    q = torch.tensor([[.2, .8]])
    w = torch.tensor([.7])
    plain = head_loss(a, q, w, view_weights=[.6, .4])
    balanced = head_loss(b, q, w, torch.tensor([3., 3.]), view_weights=[.6, .4])
    plain.backward(); balanced.backward()
    torch.testing.assert_close(plain, balanced)
    torch.testing.assert_close(a.grad, b.grad)


def test_nonuniform_prior_positive_sign_and_no_extra_temperature():
    z = torch.zeros(1, 1, 2, requires_grad=True)
    loss = head_loss(z, torch.tensor([[0., 1.]]), torch.ones(1),
                     torch.tensor([8., 2.]), view_weights=[1.])
    assert float(loss.detach()) == pytest.approx(-math.log(.2))
    loss.backward()
    torch.testing.assert_close(z.grad, torch.tensor([[[.8/1.5, -.8/1.5]]]))
    with pytest.raises(ValueError, match='strictly positive'):
        head_loss(z, torch.tensor([[0., 1.]]), torch.ones(1),
                  torch.tensor([8., 0.]), view_weights=[1.])


def test_soft_targets_weights_and_view_reduction_hand_example():
    q = torch.tensor([[.25, .75], [1., 0.]])
    w = torch.tensor([.6, 0.])
    loss = head_loss(torch.zeros(2, 2, 2), q, w, view_weights=[.3, .7])
    assert float(loss) == pytest.approx(.3 * math.log(2))


def test_cached_residual_uses_updated_shared_weight_and_one_bias():
    arrays = {'global': np.ones((2, 2, 3), np.float32),
              'local_base': np.ones((2, 8, 3), np.float32),
              'local_residual': np.ones((2, 8, 3), np.float32) * .5}
    head = torch.nn.Linear(3, 2)
    with torch.no_grad():
        head.weight.fill_(2.); head.bias.fill_(4.)
    logits = cached_logits(head, arrays, np.array([0, 1]), torch.device('cpu'))
    torch.testing.assert_close(logits[:, :2], torch.full((2, 2, 2), 10.))
    torch.testing.assert_close(logits[:, 2:], torch.full((2, 8, 2), 13.))
    with torch.no_grad():
        head.weight.fill_(3.)
    updated = cached_logits(head, arrays, np.array([0, 1]), torch.device('cpu'))
    torch.testing.assert_close(updated[:, 2:], torch.full((2, 8, 2), 17.5))


def test_head_refit_frozen_scope_strictly_preserves_plain_fullft_structure():
    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, 1)
            self.positional_embedding = torch.nn.Parameter(torch.zeros(1, 4))
            self.proj = torch.nn.Parameter(torch.randn(4, 3))

        def forward(self, images):
            return self.conv1(images).mean((2,3)) @ self.proj

    parent = AegisCLIP(Visual(), num_classes=2, feature_dim=3, peft_mode='full_finetune')
    refit = AegisCLIP(Visual(), num_classes=2, feature_dim=3, peft_mode='frozen')
    refit.load_state_dict(parent.state_dict(), strict=True)
    image = torch.randn(3, 3, 4, 4)
    torch.testing.assert_close(parent(images=image), refit(images=image), rtol=0, atol=0)
    assert all(not p.requires_grad for p in refit.visual.parameters())
