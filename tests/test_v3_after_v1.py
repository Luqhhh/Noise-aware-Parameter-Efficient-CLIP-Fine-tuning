"""The v3 GPU handoff must wait for a checked v1 delivery."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_v3_after_v1
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


def test_gpu_idle_check_uses_absolute_wsl_binary_without_service_path(monkeypatch):
    commands = []
    wsl_binary = Path("/usr/lib/wsl/lib/nvidia-smi")

    def run(command, **kwargs):
        commands.append(command)
        return type("Result", (), {"stdout": ""})()

    monkeypatch.setattr(run_v3_after_v1.subprocess, "run", run)
    monkeypatch.setattr(run_v3_after_v1.Path, "is_file", lambda path: path == wsl_binary)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert run_v3_after_v1.gpu_busy() is False
    assert commands == [[str(wsl_binary), "--query-compute-apps=pid",
                         "--format=csv,noheader,nounits"]]
