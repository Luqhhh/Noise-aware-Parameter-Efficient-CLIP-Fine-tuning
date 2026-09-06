from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from aegis_clip.scope_cache import (
    formal_row_binding_hash,
    pack_crop_boxes,
    replicate_semantic_sha256,
    scope_parent_batch_scores,
    semantic_sha256,
    stable_top2,
    tensor_sha256,
    validate_evidence_cache,
    validate_parent_cache,
)
from aegis_clip.scope_protocol import (
    EVIDENCE_VIEW_ORDER,
    EVIDENCE_VIEW_WEIGHTS,
    PARENT_BRANCH_ORDER,
    four_neighbor_edges,
    load_scope_protocol,
)


CONFIG = Path(__file__).parents[1] / "configs" / "scope_k2_fullft_dual_pa090.yaml"


def _digest(character: str) -> str:
    return character * 64


def _lineage() -> dict[str, str]:
    return {
        "checkpoint_sha256": _digest("a"), "split_sha256": _digest("b"),
        "class_to_idx_sha256": _digest("c"), "idx_to_class_sha256": _digest("d"),
        "trust_bundle_sha256": _digest("e"), "exact_group_sha256": _digest("f"),
        "protocol_sha256": _digest("1"), "code_sha256": _digest("2"),
        "dirty_diff_sha256": _digest("3"), "lockfile_sha256": _digest("4"),
    }


def _parent(split: str = "validation") -> dict:
    paths = ["0000/a.jpg", "0001/b.jpg", "0002/c.jpg"]
    rows = torch.arange(3, dtype=torch.int64)
    candidates = torch.tensor([[1, 2], [2, 4], [3, 5]], dtype=torch.int64)
    scores = torch.tensor([[-0.1, -0.3], [-0.2, -0.5], [-0.4, -0.9]], dtype=torch.float64)
    constituent = torch.zeros(3, 4, 500, dtype=torch.float32)
    constituent[:, :, 0] = 1.0
    payload = {
        "schema": "scope_parent_cache_v1", "split": split,
        "paths": paths, "formal_row_id": rows,
        "formal_row_binding_sha256": formal_row_binding_hash(rows, paths),
        "candidate_indices": candidates, "candidate_parent_log_scores": scores,
        "parent_margin": scores[:, 1] - scores[:, 0],
        "parent_predictions": candidates[:, 0].clone(),
        "constituent_scores": constituent,
        "constituent_top1": constituent.argmax(dim=2).to(torch.int64),
        "constituent_order": list(PARENT_BRANCH_ORDER),
        "constituent_scores_sha256": tensor_sha256(constituent),
        "crop_boxes": torch.tensor(
            [[[[0, 0, 112, 112], [0, 0, 128, 128], [0, 0, 144, 144], [0, 0, 160, 160]]] * 2] * 3,
            dtype=torch.int64,
        ).reshape(3, 2, 4, 4),
        "corrupt": torch.tensor([False, False, True]),
        "prior_bias": torch.zeros(500, dtype=torch.float64), "prior_iterations": 4,
        "prior_report_sha256": _digest("5"),
        "aligned_log_scores_shape": [3, 500], "aligned_log_scores_dtype": "float32",
        "aligned_log_scores_sha256": _digest("6"), "lineage": _lineage(),
    }
    if split == "validation":
        payload.update(
            label=torch.tensor([1, 4, 9]), clean_probability=torch.tensor([0.9, 0.8, 0.1]),
            pseudo_label=torch.tensor([1, 4, 9]), correction_alpha=torch.tensor([0.0, 0.2, 1.0]),
        )
    return payload


def _family(values: torch.Tensor) -> dict:
    return {
        "view_evidence": values, "aggregate": (values * torch.tensor(EVIDENCE_VIEW_WEIGHTS)).sum(1),
        "positive_count": (values > 0).sum(1).to(torch.int64),
        "orientation": torch.ones(values.shape[0], 2, dtype=torch.float64),
        "leave_one_scale": torch.ones(values.shape[0], 3, dtype=torch.float64),
        "eligibility": torch.ones(values.shape[0], dtype=torch.bool),
    }


def _evidence(parent: dict) -> dict:
    values = torch.ones(3, 6, dtype=torch.float64)
    edges = torch.tensor(four_neighbor_edges(), dtype=torch.int64)
    payload = {
        "schema": "scope_evidence_cache_v1", "split": parent["split"],
        "paths": list(parent["paths"]), "formal_row_id": parent["formal_row_id"].clone(),
        "formal_row_binding_sha256": parent["formal_row_binding_sha256"],
        "candidate_indices": parent["candidate_indices"].clone(),
        "crop_boxes": parent["crop_boxes"].clone(), "corrupt": parent["corrupt"].clone(),
        "parent_cache_sha256": _digest("7"), "parent_semantic_sha256": semantic_sha256(parent),
        "view_order": list(EVIDENCE_VIEW_ORDER), "view_weights": list(EVIDENCE_VIEW_WEIGHTS),
        "grid_shape": [7, 7], "adjacency": "four_neighbor_row_major_v1",
        "edges": edges, "edges_sha256": tensor_sha256(edges),
        "classifier_weight_sha256": _digest("8"),
        "weight_norm": torch.ones(3, dtype=torch.float64),
        "weight_norm_valid": torch.ones(3, dtype=torch.bool),
        "classifier_space_audit": {"base_max_abs_error": 0.0, "dual_max_abs_error": 0.0},
        "antisymmetry_audit": {"canonical_bitwise": True, "independent_max_abs_error": 0.0},
        "scope": _family(values), "pace": _family(values), "no_topology": _family(values),
        "lineage": dict(parent["lineage"]),
    }
    if parent["split"] == "validation":
        payload.update({field: parent[field].clone() for field in (
            "label", "clean_probability", "pseudo_label", "correction_alpha"
        )})
    return payload


def test_parent_and_evidence_cache_contracts_accept_valid_payloads() -> None:
    protocol = load_scope_protocol(CONFIG)
    parent = _parent()
    evidence = _evidence(parent)

    assert validate_parent_cache(parent, protocol, "validation") == 3
    assert validate_evidence_cache(evidence, parent, protocol, "validation") == 3


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["parent_margin"].add_(0.1), "parent_margin"),
        (lambda value: value["candidate_indices"].__setitem__((0, 1), 1), "distinct"),
        (lambda value: value.update(crop_boxes=value["crop_boxes"].float()), "crop_boxes"),
        (lambda value: value.update(raw_patch_features=torch.zeros(3, 49, 8)), "raw patch"),
    ],
)
def test_parent_cache_mutations_fail_closed(mutation, match: str) -> None:
    protocol = load_scope_protocol(CONFIG)
    payload = _parent()
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        validate_parent_cache(payload, protocol, "validation")


def test_test_parent_rejects_validation_only_fields() -> None:
    protocol = load_scope_protocol(CONFIG)
    payload = _parent("test")
    payload["label"] = torch.arange(3)
    with pytest.raises(ValueError, match="validation-only"):
        validate_parent_cache(payload, protocol, "test")


@pytest.mark.parametrize("field", ["formal_row_id", "candidate_indices", "crop_boxes", "corrupt"])
def test_evidence_parent_binding_mismatch_is_rejected(field: str) -> None:
    protocol = load_scope_protocol(CONFIG)
    parent = _parent()
    evidence = _evidence(parent)
    evidence[field] = evidence[field].clone()
    evidence[field].view(-1)[0] = ~evidence[field].view(-1)[0] if evidence[field].dtype == torch.bool else evidence[field].view(-1)[0] + 1
    with pytest.raises(ValueError, match=field):
        validate_evidence_cache(evidence, parent, protocol, "validation")


def test_semantic_hash_is_mapping_order_invariant_and_content_sensitive() -> None:
    first = _parent()
    second = dict(reversed(list(first.items())))
    assert semantic_sha256(first) == semantic_sha256(second)
    second["parent_margin"] = second["parent_margin"].clone()
    second["parent_margin"][0] += 0.01
    assert semantic_sha256(first) != semantic_sha256(second)


def test_replicate_semantic_hash_ignores_only_parent_file_instance_binding() -> None:
    parent = _parent()
    first = _evidence(parent)
    second = deepcopy(first)
    second["parent_cache_sha256"] = _digest("9")

    assert semantic_sha256(first) != semantic_sha256(second)
    assert replicate_semantic_sha256(first) == replicate_semantic_sha256(second)

    second["scope"]["aggregate"][0] += 0.01
    assert replicate_semantic_sha256(first) != replicate_semantic_sha256(second)


def test_stable_top2_prefers_lower_class_index_on_ties() -> None:
    candidates, scores = stable_top2(torch.tensor([[1.0, 2.0, 2.0, 0.0]]))
    assert torch.equal(candidates, torch.tensor([[1, 2]]))
    assert torch.equal(scores, torch.tensor([[2.0, 2.0]], dtype=torch.float64))


def test_parent_four_scale_fusion_and_box_pack_contract() -> None:
    logits = torch.tensor([[2.0, 0.0]])
    locals_o = [logits + index for index in range(4)]
    locals_f = [logits - index for index in range(4)]
    result = scope_parent_batch_scores(logits, locals_o, logits, locals_f)
    assert result.constituent_scores.shape == (1, 4, 2)
    assert result.constituent_top1.shape == (1, 4)
    assert result.fused_log_scores.shape == (1, 2)

    scales = [[(0, 0, size, size)] for size in (112, 128, 144, 160)]
    boxes = pack_crop_boxes(scales, scales)
    assert boxes.dtype == torch.int64
    assert boxes.shape == (1, 2, 4, 4)
