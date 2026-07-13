"""Tests for parent-child split lineage audit."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from common.split_audit import (
    SplitAuditError,
    load_split_csv,
    run_split_audit,
)


def _make_csv(path: Path, image_paths: list[str]) -> None:
    """Write a minimal split CSV with the given image paths."""
    df = pd.DataFrame({
        "image_path": image_paths,
        "class_name": ["c"] * len(image_paths),
        "class_idx": [0] * len(image_paths),
    })
    df.to_csv(path, index=False)


class TestLoadSplitCsv:
    """Tests for load_split_csv()."""

    def test_loads_image_paths(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.csv"
            _make_csv(p, ["a.jpg", "b.jpg"])
            result = load_split_csv(p)
            assert result == {"a.jpg", "b.jpg"}

    def test_raises_on_missing_file(self):
        with pytest.raises(SplitAuditError, match="not found"):
            load_split_csv(Path("/nonexistent_xyz.csv"))

    def test_raises_on_missing_column(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.csv"
            p.write_text("other_column\nx.jpg\n")
            with pytest.raises(SplitAuditError, match="image_path"):
                load_split_csv(p)


class TestRunSplitAudit:
    """Tests for run_split_audit()."""

    def test_passes_when_valid(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _make_csv(d / "parent_train.csv", ["a.jpg", "b.jpg", "c.jpg"])
            _make_csv(d / "parent_val.csv", ["d.jpg", "e.jpg"])
            _make_csv(d / "child_train.csv", ["a.jpg", "b.jpg", "c.jpg"])
            _make_csv(d / "child_val.csv", ["d.jpg", "e.jpg"])

            audit = run_split_audit(
                "ref",
                "d3/best.pt",
                d / "parent_train.csv",
                d / "parent_val.csv",
                d / "child_train.csv",
                d / "child_val.csv",
                d,
            )
            assert audit["protocol_valid"] is True
            assert audit["child_val_equals_parent_val"] is True
            assert audit["child_val_in_parent_train"] == 0

    def test_detects_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _make_csv(
                d / "parent_train.csv",
                ["a.jpg", "b.jpg", "c.jpg", "LEAK.jpg"],
            )
            _make_csv(d / "parent_val.csv", ["d.jpg", "e.jpg"])
            _make_csv(d / "child_train.csv", ["a.jpg", "b.jpg", "c.jpg"])
            # LEAK.jpg was in parent_train, now appears in child_val
            _make_csv(
                d / "child_val.csv",
                ["d.jpg", "e.jpg", "LEAK.jpg"],
            )

            with pytest.raises(SplitAuditError, match="VALIDATION LEAK"):
                run_split_audit(
                    "ref",
                    "d3/best.pt",
                    d / "parent_train.csv",
                    d / "parent_val.csv",
                    d / "child_train.csv",
                    d / "child_val.csv",
                    d,
                )

    def test_detects_val_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _make_csv(d / "parent_train.csv", ["a.jpg", "b.jpg"])
            _make_csv(d / "parent_val.csv", ["d.jpg", "e.jpg"])
            _make_csv(d / "child_train.csv", ["a.jpg", "b.jpg"])
            _make_csv(
                d / "child_val.csv",
                ["d.jpg", "DIFFERENT.jpg"],
            )

            with pytest.raises(SplitAuditError, match="VALIDATION MISMATCH"):
                run_split_audit(
                    "ref",
                    "d3/best.pt",
                    d / "parent_train.csv",
                    d / "parent_val.csv",
                    d / "child_train.csv",
                    d / "child_val.csv",
                    d,
                )

    def test_writes_audit_json_on_success(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _make_csv(d / "parent_train.csv", ["a.jpg"])
            _make_csv(d / "parent_val.csv", ["b.jpg"])
            _make_csv(d / "child_train.csv", ["a.jpg"])
            _make_csv(d / "child_val.csv", ["b.jpg"])

            run_split_audit(
                "ref",
                "d3/best.pt",
                d / "parent_train.csv",
                d / "parent_val.csv",
                d / "child_train.csv",
                d / "child_val.csv",
                d,
            )

            audit_path = d / "split_lineage_audit.json"
            assert audit_path.exists()
            audit = json.loads(audit_path.read_text())
            assert audit["protocol_valid"] is True
            assert audit["parent_experiment"] == "ref"
