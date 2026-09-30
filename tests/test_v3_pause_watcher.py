import json
import os

import pytest

from scripts.pause_v3_after_epoch import boundary, process


def completed_arm(tmp_path):
    arm = tmp_path / "original"
    arm.mkdir()
    (arm / "epoch02.pt").write_bytes(b"durable checkpoint")
    (arm / "epoch02.binding.json").write_text(json.dumps(dict(
        sha256="test", binding=dict(plan_sha256="plan", arm="original", probe=False, complete=True))))
    (arm / "history.json").write_text(json.dumps([dict(epoch=1, updates=5), dict(epoch=2, updates=10)]))
    return arm


@pytest.mark.parametrize("missing", ["epoch02.pt", "epoch02.binding.json", "history.json"])
def test_never_pause_before_all_epoch_outputs_are_durable(tmp_path, missing):
    arm = completed_arm(tmp_path)
    (arm / missing).unlink()
    assert boundary(arm, 2, "plan", 10) is None


def test_epoch_history_and_plan_must_match(tmp_path):
    arm = completed_arm(tmp_path)
    assert boundary(arm, 2, "plan", 10)["history"][-1]["epoch"] == 2
    with pytest.raises(ValueError, match="binding"):
        boundary(arm, 2, "other plan", 10)
    with pytest.raises(ValueError, match="history"):
        boundary(arm, 2, "plan", 11)
    (arm / "history.json").write_text(json.dumps([dict(epoch=1, updates=5)]))
    assert boundary(arm, 2, "plan", 10) is None


def test_process_identity_includes_pid_lifetime_and_parent():
    value = process(os.getpid())
    assert value["pid"] == os.getpid() and value["ppid"] == os.getppid()
    assert value["start_ticks"] > 0 and value["argv"]
