"""CPU checks for the filtered LP -> L05 checkpoint lineage."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "outputs/codex/l05_oof_hard_filter/runner"


@pytest.fixture(scope="module")
def protocol():
    subprocess.run([sys.executable, "scripts/run_l05_oof_hard_filter.py", "prepare"],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    sys.path.insert(0, str(RUNNER / "reproducibility/aegis_f1"))
    from aegis_clip.config import load_config
    from aegis_clip import rematch_search_v4 as v4
    lp = load_config(RUNNER / "configs/LP_HF01.yaml")
    ft = load_config(RUNNER / "configs/L05_HF01.yaml")
    return v4, lp, ft


def test_filtered_lp_is_the_only_shared_parent(protocol, tmp_path, monkeypatch):
    v4, lp, ft = protocol
    stage = json.loads(Path(ft["data"]["dataset_manifest"]).read_text())
    monkeypatch.setattr(v4, "validate_dataset", lambda _: stage)
    lp_binding = v4.checkpoint_binding(lp)
    expected_parent = v4._selected_lp_binding(ft, stage)
    assert lp_binding == expected_parent
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint for lineage test")
    meta = {"checkpoint_sha256": v4.sha256_file(checkpoint),
            "binding": lp_binding, "experiment_id": "RM_LP_HF01"}
    sidecar = checkpoint.with_suffix(".binding.json")
    sidecar.write_text(json.dumps(meta))
    assert v4.validate_checkpoint(checkpoint, ft, parent=True) == meta

    old_id = {**meta, "experiment_id": "RM_LP"}
    sidecar.write_text(json.dumps(old_id))
    with pytest.raises(ValueError, match="freshly trained filtered LP"):
        v4.validate_checkpoint(checkpoint, ft, parent=True)

    wrong_subset = {**meta, "binding": {**lp_binding, "train_csv_sha256": "0" * 64}}
    sidecar.write_text(json.dumps(wrong_subset))
    with pytest.raises(ValueError, match="filtered LP parent lineage mismatch"):
        v4.validate_checkpoint(checkpoint, ft, parent=True)


def test_selected_csv_cannot_be_swapped(protocol, monkeypatch):
    v4, _, ft = protocol
    stage = json.loads(Path(ft["data"]["dataset_manifest"]).read_text())
    changed = {**ft, "data": {**ft["data"],
                               "train_csv": str(Path(ft["data"]["val_csv"]))}}
    with pytest.raises(ValueError, match="training CSV is not the bound filtered CSV"):
        v4.validate_train_selection(changed, stage)
