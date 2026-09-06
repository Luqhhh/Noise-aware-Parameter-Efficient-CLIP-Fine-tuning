from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from aegis_clip.cli.infer_scope_submission import (
    replay_scope_decisions,
    require_promoted_decision,
)


def test_rejected_decision_refuses_scope_test_inference(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"schema": "scope_decision_v1", "promoted": False}))
    with pytest.raises(ValueError, match="not promoted"):
        require_promoted_decision(decision)


def test_scope_decision_replay_uses_strict_greater_than() -> None:
    candidates = torch.tensor([[1, 2], [3, 4], [5, 6]])
    margin = torch.tensor([-0.25, -0.25, -0.5], dtype=torch.float64)
    evidence = torch.tensor([0.5, 0.75, 1.0], dtype=torch.float64)
    eligible = torch.tensor([True, True, False])
    prediction, switch = replay_scope_decisions(
        candidates, margin, evidence, eligible, beta=1.0,
        threshold={"mode": "finite", "gamma": 0.25},
    )
    assert torch.equal(switch, torch.tensor([False, True, False]))
    assert torch.equal(prediction, torch.tensor([1, 4, 5]))
