"""The v3 GPU handoff must wait for a checked v1 delivery."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_v3_after_v1 import check_v1


def test_v3_handoff_waits_for_complete_checked_v1(tmp_path):
    output = tmp_path / "v1"
    output.mkdir()
    status = output / "status.json"
    status.write_text(json.dumps(dict(status="running", output=str(output))))
    assert check_v1(status) is None
    status.write_text(json.dumps(dict(status="completed", output=str(output))))
    with pytest.raises(RuntimeError, match="lacks"):
        check_v1(status)
    for name in ("training/selected.pt", "validation_report.json", "submission/pred_results.csv",
                 "submission/submission.zip", "submission/submission_check.log"):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("All checks passed!" if name.endswith("submission_check.log") else "fixture")
    assert check_v1(status)["status"] == "completed"
