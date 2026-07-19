import pytest
import torch
import torch.nn.functional as F

from aegis_clip.tta import fuse_paired_logits


def test_mean_logits_is_backward_compatible() -> None:
    first = torch.randn(4, 3)
    second = torch.randn(4, 3)
    assert torch.equal(
        fuse_paired_logits(first, second, mode="mean_logits"),
        (first + second) / 2.0,
    )


def test_mean_probabilities_returns_normalized_log_probabilities() -> None:
    result = fuse_paired_logits(
        torch.tensor([[4.0, 0.0]]),
        torch.tensor([[0.0, 2.0]]),
        mode="mean_probabilities",
        temperature=1.0,
    )
    assert torch.allclose(result.exp().sum(1), torch.ones(1))


def test_entropy_weighting_prefers_confident_view() -> None:
    confident = torch.tensor([[8.0, 0.0]])
    uncertain = torch.tensor([[0.0, 0.1]])
    fused = fuse_paired_logits(
        confident,
        uncertain,
        mode="entropy_weighted_probabilities",
    )
    plain = (
        F.softmax(confident, 1) + F.softmax(uncertain, 1)
    ) / 2.0
    assert fused.exp()[0, 0] > plain[0, 0]


def test_max_margin_selects_more_decisive_view() -> None:
    first = torch.tensor([[1.0, 0.9, 0.0]])
    second = torch.tensor([[0.0, 3.0, 0.1]])
    fused = fuse_paired_logits(first, second, mode="max_margin")
    assert torch.equal(fused, second)


def test_tta_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="positive"):
        fuse_paired_logits(
            torch.randn(2, 3), torch.randn(2, 3), temperature=0.0
        )
