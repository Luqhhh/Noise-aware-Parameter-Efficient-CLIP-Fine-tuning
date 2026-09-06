from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aegis_clip.scope_protocol import (
    EVIDENCE_VIEW_ORDER,
    EVIDENCE_VIEW_WEIGHTS,
    PARENT_BRANCH_ORDER,
    PARENT_LOCAL_VIEW_ORDER,
    ScopePreflightError,
    four_neighbor_edges,
    load_scope_protocol,
    verify_scope_assets,
)


CONFIG = Path(__file__).parents[1] / "configs" / "scope_k2_fullft_dual_pa090.yaml"


def test_scope_protocol_freezes_parent_and_evidence_views() -> None:
    protocol = load_scope_protocol(CONFIG)

    assert protocol.protocol_id == "scope_k2_fullft_dual_pa090_v1"
    assert PARENT_LOCAL_VIEW_ORDER == (
        "original_112", "original_128", "original_144", "original_160",
        "flipped_112", "flipped_128", "flipped_144", "flipped_160",
    )
    assert EVIDENCE_VIEW_ORDER == (
        "original_128", "original_144", "original_160",
        "flipped_128", "flipped_144", "flipped_160",
    )
    assert EVIDENCE_VIEW_WEIGHTS == (0.1875, 0.25, 0.0625, 0.1875, 0.25, 0.0625)
    assert sum(EVIDENCE_VIEW_WEIGHTS) == 1.0
    assert PARENT_BRANCH_ORDER == (
        "original_global", "original_local", "flipped_global", "flipped_local",
    )
    assert protocol.fixed["parent"]["prior"]["strength"] == 0.90
    assert protocol.fixed["parent"]["peft_mode"] == "full_finetune"


def test_scope_four_neighbor_graph_is_exact_and_canonical() -> None:
    edges = four_neighbor_edges()

    assert len(edges) == 84
    assert len(set(edges)) == 84
    assert edges[:3] == ((0, 1), (1, 2), (2, 3))
    assert edges[41] == (47, 48)
    assert edges[42:45] == ((0, 7), (1, 8), (2, 9))
    for first, second in edges:
        assert first != second
        row_a, col_a = divmod(first, 7)
        row_b, col_b = divmod(second, 7)
        assert abs(row_a - row_b) + abs(col_a - col_b) == 1


def test_scope_asset_gate_verifies_every_frozen_asset() -> None:
    protocol = load_scope_protocol(CONFIG)

    audit = verify_scope_assets(protocol)

    assert audit.split_assets_verified
    assert audit.model_assets_verified
    assert audit.group_verified
    assert audit.fallback_verified


def test_scope_asset_gate_fails_closed_on_digest_mismatch() -> None:
    protocol = load_scope_protocol(CONFIG)
    bad_assets = replace(protocol.assets, checkpoint_sha256="0" * 64)

    with pytest.raises(ScopePreflightError, match="checkpoint SHA-256 mismatch"):
        verify_scope_assets(replace(protocol, assets=bad_assets))


def test_scope_test_schema_forbids_validation_diagnostics() -> None:
    protocol = load_scope_protocol(CONFIG)

    assert set(protocol.fixed["schemas"]["validation_only_fields"]) == {
        "label", "clean_probability", "pseudo_label", "correction_alpha",
    }
    assert protocol.fixed["schemas"]["test_forbid_validation_fields"] is True
