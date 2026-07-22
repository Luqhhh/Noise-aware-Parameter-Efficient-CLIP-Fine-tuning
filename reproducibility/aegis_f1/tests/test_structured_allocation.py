import pytest
import torch

from aegis_clip.structured_allocation import (
    backfill_minimum_class_support,
    classwise_curriculum_selection,
    log_sinkhorn_allocation,
)


def test_log_sinkhorn_matches_row_and_column_marginals() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [0.0, 4.0],
        ]
    )
    allocation, diagnostics = log_sinkhorn_allocation(
        logits, torch.tensor([1.0, 3.0]), iterations=100
    )
    assert torch.allclose(allocation.sum(dim=1), torch.ones(4), atol=1.0e-5)
    assert torch.allclose(
        allocation.sum(dim=0), torch.tensor([1.0, 3.0]), atol=1.0e-5
    )
    assert diagnostics["maximum_row_absolute_error"] < 1.0e-5


def test_log_sinkhorn_rejects_non_positive_residual_capacity() -> None:
    with pytest.raises(ValueError, match="positive"):
        log_sinkhorn_allocation(torch.zeros(2, 2), torch.tensor([0.0, 2.0]))


def test_classwise_curriculum_selection_is_quota_bound_and_stable() -> None:
    assigned = torch.tensor([0, 0, 0, 1, 1, 1])
    reliability = torch.tensor([0.2, 0.9, 0.9, 0.7, 0.1, 0.8])
    eligible = torch.tensor([True, True, True, True, False, True])
    selected = classwise_curriculum_selection(
        assigned, reliability, eligible, torch.tensor([2, 1])
    )
    assert selected.tolist() == [False, True, True, False, False, True]


def test_classwise_curriculum_selection_does_not_backfill_ineligible_rows() -> None:
    selected = classwise_curriculum_selection(
        torch.tensor([0, 0, 1]),
        torch.tensor([0.9, 0.8, 0.7]),
        torch.tensor([False, True, False]),
        torch.tensor([2, 2]),
    )
    assert selected.tolist() == [False, True, False]


def test_topk_is_used_for_float16_tie_compatible_reconstruction() -> None:
    # FP16 serialization can collapse close float32 scores to an exact tie.
    # PyTorch's topk and argmax deliberately have different tie behavior; the
    # retained historical scalar must therefore be preferred as the local vote.
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float16).float()
    assert logits.argmax(dim=1).item() == 0
    assert logits.softmax(dim=1).topk(1, dim=1).indices.item() == 2


def test_minimum_class_support_backfill_is_original_label_only() -> None:
    selected = torch.tensor([True, False, False, True, False, False])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    reliability = torch.tensor([0.1, 0.8, 0.9, 0.1, 0.7, 0.6])
    result = backfill_minimum_class_support(
        selected, labels, reliability, minimum_per_class=2
    )
    assert result.tolist() == [True, False, True, True, True, False]
