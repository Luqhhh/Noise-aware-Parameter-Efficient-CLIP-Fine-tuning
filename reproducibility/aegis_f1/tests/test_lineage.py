from pathlib import Path

import pandas as pd
import pytest
import torch

from aegis_clip.lineage import LineageAuditError, run_lineage_audit


def _write_split(path: Path, rows: list[tuple[str, int]]) -> None:
    pd.DataFrame(rows, columns=["image_path", "label"]).to_csv(path, index=False)


def _fullfit_fixture(tmp_path: Path) -> tuple[dict, Path, Path, Path, Path]:
    parent_train = tmp_path / "parent_train.csv"
    parent_val = tmp_path / "parent_val.csv"
    child_train = tmp_path / "child_train.csv"
    child_val = tmp_path / "child_val.csv"
    checkpoint = tmp_path / "parent.pt"
    train_rows = [("train/0000/a.jpg", 0), ("train/0001/b.jpg", 1)]
    val_rows = [("train/0000/c.jpg", 0), ("train/0001/d.jpg", 1)]
    _write_split(parent_train, train_rows)
    _write_split(parent_val, val_rows)
    _write_split(child_train, train_rows + val_rows)
    _write_split(child_val, val_rows)
    torch.save({"model_state_dict": {}}, checkpoint)
    config = {
        "lineage": {
            "enabled": True,
            "parent_experiment_id": "parent",
            "parent_train_csv": str(parent_train),
            "parent_val_csv": str(parent_val),
            "require_same_train": False,
            "require_same_val": True,
            "allow_parent_val_in_child_train": True,
        }
    }
    return config, child_train, child_val, checkpoint, parent_val


def test_exact_parent_train_val_fullfit_is_audited(tmp_path: Path) -> None:
    config, child_train, child_val, checkpoint, _ = _fullfit_fixture(tmp_path)
    audit = run_lineage_audit(
        config,
        child_train_csv=str(child_train),
        child_val_csv=str(child_val),
        checkpoint_path=str(checkpoint),
        output_path=tmp_path / "audit.json",
    )
    assert audit["protocol_valid"] is True
    assert audit["exact_parent_fullfit"] is True
    assert audit["child_train_in_parent_val"] == 2


def test_fullfit_permission_rejects_incomplete_parent_union(tmp_path: Path) -> None:
    config, child_train, child_val, checkpoint, _ = _fullfit_fixture(tmp_path)
    frame = pd.read_csv(child_train).iloc[:-1]
    frame.to_csv(child_train, index=False)
    with pytest.raises(LineageAuditError, match="full-fit child train must exactly"):
        run_lineage_audit(
            config,
            child_train_csv=str(child_train),
            child_val_csv=str(child_val),
            checkpoint_path=str(checkpoint),
            output_path=tmp_path / "audit.json",
        )


def test_parent_val_overlap_requires_explicit_fullfit_permission(tmp_path: Path) -> None:
    config, child_train, child_val, checkpoint, _ = _fullfit_fixture(tmp_path)
    config["lineage"]["allow_parent_val_in_child_train"] = False
    with pytest.raises(LineageAuditError, match="child training samples overlap"):
        run_lineage_audit(
            config,
            child_train_csv=str(child_train),
            child_val_csv=str(child_val),
            checkpoint_path=str(checkpoint),
            output_path=tmp_path / "audit.json",
        )
