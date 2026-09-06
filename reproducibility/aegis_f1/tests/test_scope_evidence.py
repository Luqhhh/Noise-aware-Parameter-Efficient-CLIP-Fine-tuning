from __future__ import annotations

import pytest
import torch
from torch import nn

from aegis_clip.scope_evidence import (
    aggregate_family_evidence,
    apply_scope_decision,
    family_eligibility,
    matched_pace_evidence,
    no_topology_view_evidence,
    pairwise_residual_grid,
    scope_energy,
    scope_view_evidence,
    validate_classifier_space_batch,
    validate_scope_parent_model,
)


class _TinyFullParent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.peft_mode = "full_finetune"
        self.classifier_mode = "linear"
        self.feature_adapter = nn.Identity()
        self.classifier = nn.Linear(3, 4)


def _checkpoint() -> dict:
    return {
        "config": {"model": {"peft_mode": "full_finetune"}},
        "local_feature_adapter": {"spec": {}, "state_dict": {}},
        "part_token_adapter": {"spec": {}, "state_dict": {}},
    }


def test_parent_gate_requires_fullft_linear_and_both_dual_adapters() -> None:
    audit = validate_scope_parent_model(_TinyFullParent(), _checkpoint())
    assert audit.peft_mode == "full_finetune"
    assert audit.classifier_mode == "linear"
    assert audit.has_local_feature_adapter
    assert audit.has_part_token_adapter

    payload = _checkpoint()
    del payload["local_feature_adapter"]
    with pytest.raises(ValueError, match="local feature adapter"):
        validate_scope_parent_model(_TinyFullParent(), payload)


def test_classifier_space_batch_rebuilds_base_and_dual_anchored_logits() -> None:
    model = _TinyFullParent()
    base_cls = torch.randn(5, 3)
    dual_cls = torch.randn(5, 3)
    base_logits = model.classifier(base_cls)
    dual_logits = base_logits + torch.nn.functional.linear(
        dual_cls - base_cls, model.classifier.weight
    )

    audit = validate_classifier_space_batch(
        model, base_logits, base_cls, dual_logits=dual_logits, dual_cls=dual_cls
    )

    assert audit.base_max_abs_error == 0.0
    assert audit.dual_max_abs_error == 0.0

    base_logits[0, 0] += 1.0e-3
    with pytest.raises(ValueError, match="base classifier-space logits"):
        validate_classifier_space_batch(model, base_logits, base_cls)


def test_scope_energy_matches_manual_node_and_edge_terms() -> None:
    grid = torch.zeros(1, 49, dtype=torch.float64)
    grid[0, 0] = 2.0
    grid[0, 1] = 1.0

    energy = scope_energy(grid)

    assert torch.equal(energy, torch.tensor([(3.0 / 49.0) + (1.0 / 84.0)], dtype=torch.float64))


def test_connected_positive_evidence_beats_isolated_with_same_nodes() -> None:
    connected = torch.zeros(1, 49, dtype=torch.float64)
    isolated = connected.clone()
    connected[0, [0, 1, 2]] = 1.0
    isolated[0, [0, 2, 4]] = 1.0

    assert scope_view_evidence(connected).item() > scope_view_evidence(isolated).item()


def test_patch_permutation_preserves_pace_and_no_topology_but_changes_scope() -> None:
    values = torch.zeros(1, 49, dtype=torch.float64)
    values[0, [0, 1, 2, 9, 20, 33, 48]] = torch.tensor(
        [4.0, 3.0, 2.0, -1.0, -2.0, -3.0, -4.0], dtype=torch.float64
    )
    permutation = torch.arange(49)
    permutation[[1, 2, 3, 4]] = torch.tensor([3, 1, 4, 2])
    shuffled = values[:, permutation]

    assert torch.equal(matched_pace_evidence(values), matched_pace_evidence(shuffled))
    assert torch.equal(no_topology_view_evidence(values), no_topology_view_evidence(shuffled))
    assert not torch.equal(scope_view_evidence(values), scope_view_evidence(shuffled))


def test_canonical_pair_residual_and_all_evidence_are_strictly_antisymmetric() -> None:
    generator = torch.Generator().manual_seed(13)
    cls = torch.randn(1, 8, generator=generator)
    patches = torch.randn(1, 49, 8, generator=generator)
    weight = torch.randn(4, 8, generator=generator)

    forward = pairwise_residual_grid(cls, patches, weight, torch.tensor([[0, 3]]))
    reverse = pairwise_residual_grid(cls, patches, weight, torch.tensor([[3, 0]]))

    assert torch.equal(reverse.residual, -forward.residual)
    assert torch.equal(scope_view_evidence(reverse.residual), -scope_view_evidence(forward.residual))
    assert torch.equal(matched_pace_evidence(reverse.residual), -matched_pace_evidence(forward.residual))
    assert torch.equal(no_topology_view_evidence(reverse.residual), -no_topology_view_evidence(forward.residual))


@pytest.mark.parametrize("bad", ["grid", "nan", "norm"])
def test_pairwise_residual_fails_closed_or_marks_invalid(bad: str) -> None:
    cls = torch.zeros(1, 2)
    patches = torch.ones(1, 49, 2)
    weight = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    if bad == "grid":
        patches = patches[:, :48]
        with pytest.raises(ValueError, match="49"):
            pairwise_residual_grid(cls, patches, weight, torch.tensor([[0, 1]]))
    elif bad == "nan":
        patches[0, 0, 0] = torch.nan
        with pytest.raises(ValueError, match="non-finite"):
            pairwise_residual_grid(cls, patches, weight, torch.tensor([[0, 1]]))
    else:
        weight[1] = weight[0]
        result = pairwise_residual_grid(cls, patches, weight, torch.tensor([[0, 1]]))
        assert not result.valid.item()
        assert torch.equal(result.residual, torch.zeros_like(result.residual))


def test_family_aggregation_and_eligibility_use_strict_frozen_gates() -> None:
    per_view = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 1.0, 1.0, 1.0, -0.75, 0.0]],
        dtype=torch.float64,
    )
    summary = aggregate_family_evidence(per_view)
    audit = family_eligibility(
        constituent_top1=torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]]),
        parent_corrupt=torch.tensor([False, False]),
        evidence_corrupt=torch.tensor([False, False]),
        weight_norm_valid=torch.tensor([True, True]),
        summary=summary,
    )

    assert torch.equal(summary.positive_view_count, torch.tensor([6, 4]))
    assert audit.eligible[0]
    assert not audit.eligible[1]
    assert not audit.orientation_positive[1]


def test_decision_uses_runnerup_minus_top1_and_strict_threshold() -> None:
    candidates = torch.tensor([[4, 7], [2, 9], [3, 8]])
    eligible = torch.tensor([True, True, False])
    margin = torch.tensor([-0.5, -0.5, -0.5], dtype=torch.float64)
    evidence = torch.tensor([1.0, 1.001, 20.0], dtype=torch.float64)

    predictions, eta = apply_scope_decision(
        candidates, margin, evidence, eligible, beta=1.0, threshold=0.5
    )

    assert torch.allclose(
        eta, torch.tensor([0.5, 0.501, 19.5], dtype=torch.float64), atol=1.0e-15, rtol=0.0
    )
    assert torch.equal(predictions, torch.tensor([4, 9, 3]))
