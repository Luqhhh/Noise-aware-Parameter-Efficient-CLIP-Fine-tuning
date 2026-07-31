import pytest
import torch

from aegis_clip.local_prototype import (
    local_prototype_logits,
    mean_global_prototype_logits,
    trust_weighted_local_prototype_weight,
)


def test_trust_weighted_prototypes_match_base_weight_norms() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    labels = torch.tensor([0, 0, 1, 1])
    clean = torch.tensor([1.0, 0.5, 1.0, 0.5])
    base_weight = torch.tensor([[3.0, 4.0], [0.0, 2.0]])

    weight, mass = trust_weighted_local_prototype_weight(
        features, labels, clean, base_weight
    )

    assert torch.allclose(weight.norm(dim=1), base_weight.norm(dim=1))
    assert torch.allclose(mass, torch.tensor([1.5, 1.5]))


def test_prototype_logits_and_mean_fusion() -> None:
    features = torch.tensor([[1.0, 2.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bias = torch.tensor([0.5, -0.5])
    local = local_prototype_logits(features, weight, bias)
    fused = mean_global_prototype_logits(torch.tensor([[3.5, 0.5]]), local)

    assert torch.equal(local, torch.tensor([[1.5, 1.5]]))
    assert torch.equal(fused, torch.tensor([[2.5, 1.0]]))


def test_prototype_rejects_missing_class() -> None:
    with pytest.raises(ValueError, match="no samples"):
        trust_weighted_local_prototype_weight(
            torch.randn(2, 3),
            torch.tensor([0, 0]),
            torch.ones(2),
            torch.randn(2, 3),
        )
