"""Test that MixUp respects per-sample weights."""

import pytest
import torch

from common.mixup import mixup_batch
from experiments.baseline.train import _reduce_weighted_mixup


class TestWeightedMixup:
    """Verify that zero-weight samples don't contribute to MixUp loss."""

    batch_size = 8

    def make_batch(self):
        torch.manual_seed(42)
        images = torch.randn(self.batch_size, 3, 32, 32)
        labels = torch.randint(0, 10, (self.batch_size,))
        return images, labels

    def test_mixup_returns_five_outputs(self):
        """mixup_batch now returns 5-element tuple."""
        images, labels = self.make_batch()
        result = mixup_batch(images, labels, alpha=0.2, probability=1.0)
        assert len(result) == 5
        mixed, la, lb, lam, perm = result
        assert isinstance(perm, torch.Tensor)
        assert perm.shape == (self.batch_size,)

    def test_no_mixup_returns_identity_permutation(self):
        """Without MixUp, permutation is identity."""
        images, labels = self.make_batch()
        _, _, _, lam, perm = mixup_batch(images, labels, alpha=0.2, probability=0.0)
        assert lam == 1.0
        assert torch.equal(perm, torch.arange(self.batch_size))

    def test_unity_weights_equal_original(self):
        """All-ones weights should give same loss as original MixUp formula."""
        torch.manual_seed(42)
        loss_a = torch.rand(self.batch_size)
        loss_b = torch.rand(self.batch_size)
        weights = torch.ones(self.batch_size)
        perm = torch.arange(self.batch_size)

        weighted = _reduce_weighted_mixup(
            loss_a, loss_b, weights, perm, lam=0.7, normalize_by_weight_sum=True,
        )
        original = (0.7 * loss_a + 0.3 * loss_b).mean()
        assert torch.allclose(weighted, original)

    def test_zero_weight_no_gradient_contribution(self):
        """Sample with weight=0 contributes zero gradient."""
        loss_a = torch.tensor([1.0, 0.0, 1.0], requires_grad=True)
        loss_b = torch.tensor([0.0, 1.0, 1.0], requires_grad=True)
        weights = torch.tensor([0.0, 1.0, 1.0])
        perm = torch.tensor([0, 1, 2])

        result = _reduce_weighted_mixup(
            loss_a, loss_b, weights, perm, lam=0.5,
            normalize_by_weight_sum=True,
        )
        result.backward()

        # loss_a[0] has weight 0 → no gradient
        assert loss_a.grad[0] == 0.0
        # loss_a[1] and loss_a[2] have weight 1 → nonzero gradient
        assert loss_a.grad[1] != 0.0
        assert loss_a.grad[2] != 0.0

    def test_paired_zero_weight_no_contribution(self):
        """When partner sample has weight=0, it contributes zero gradient."""
        loss_a = torch.tensor([1.0], requires_grad=True)
        loss_b = torch.tensor([1.0], requires_grad=True)
        weights = torch.tensor([1.0, 0.0])
        # Sample 0 is paired with sample 1 (weight=0)
        perm = torch.tensor([1])

        result = _reduce_weighted_mixup(
            loss_a, loss_b, weights, perm, lam=0.5,
            normalize_by_weight_sum=True,
        )
        result.backward()

        # loss_b[0] corresponds to paired sample's weight=0 → no gradient
        assert loss_b.grad[0] == 0.0
        # loss_a[0] has weight 1 → nonzero gradient
        assert loss_a.grad[0] != 0.0

    def test_all_zero_weights_no_nan(self):
        """All-zero weights produce finite 0, not NaN."""
        loss_a = torch.tensor([1.0, 2.0])
        loss_b = torch.tensor([3.0, 4.0])
        weights = torch.tensor([0.0, 0.0])
        perm = torch.tensor([0, 1])

        result = _reduce_weighted_mixup(
            loss_a, loss_b, weights, perm, lam=0.5,
            normalize_by_weight_sum=True,
        )
        assert torch.isfinite(result)
        assert result.item() == 0.0
