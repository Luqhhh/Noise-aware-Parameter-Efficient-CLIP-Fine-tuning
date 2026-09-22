"""Regression cover for the opt-in OOF sample-weight injection.

The injection must be inert for every config that does not set
``trust.sample_weight_path``: those runs have to stay numerically identical to
the pre-change trainer.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
import yaml

from aegis_clip.config import load_config
from aegis_clip.sample_weights import load_sample_weights


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_sidecar(path: Path, rows: list[tuple[str, float]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "weight"])
        writer.writerows(rows)
    return path


def test_weights_align_by_canonical_path_not_row_order(tmp_path: Path) -> None:
    sidecar = _write_sidecar(
        tmp_path / "weights.csv",
        [("train/0000/b.jpg", 0.4), ("train/0000/a.jpg", 0.9)],
    )
    weights = load_sample_weights(
        sidecar, ["train/0000/a.jpg", "train/0000/b.jpg"]
    )
    assert weights.tolist() == pytest.approx([0.9, 0.4])


def test_sidecar_key_matches_split_row_under_a_different_root(tmp_path: Path) -> None:
    """The sidecar is machine-independent: roots collapse to the same key."""
    sidecar = _write_sidecar(
        tmp_path / "weights.csv", [("0000/a.jpg", 0.25)]
    )
    weights = load_sample_weights(
        sidecar, ["/home/other/复赛数据集/train/0000/a.jpg"]
    )
    assert weights.tolist() == [0.25]


def test_identity_sidecar_leaves_weights_bitwise_unchanged(tmp_path: Path) -> None:
    """All-ones weights make the multiplicative injection an exact identity."""
    paths = [str(tmp_path / f"{index}.jpg") for index in range(8)]
    weights = load_sample_weights(
        _write_sidecar(tmp_path / "weights.csv", [(path, 1.0) for path in paths]),
        paths,
    )
    reference = torch.arange(1, 9, dtype=torch.float32) / 8.0
    assert torch.equal(reference, reference * weights)


def test_missing_coverage_fails_closed(tmp_path: Path) -> None:
    sidecar = _write_sidecar(tmp_path / "weights.csv", [("train/0000/a.jpg", 1.0)])
    with pytest.raises(ValueError, match="misses 1 of 2"):
        load_sample_weights(sidecar, ["train/0000/a.jpg", "train/0000/b.jpg"])


def test_duplicate_canonical_path_fails_closed(tmp_path: Path) -> None:
    sidecar = _write_sidecar(
        tmp_path / "weights.csv",
        [("train/0000/a.jpg", 1.0), ("/data/train/0000/a.jpg", 0.5)],
    )
    with pytest.raises(ValueError, match="duplicate sample path"):
        load_sample_weights(sidecar, ["train/0000/a.jpg"])


def test_out_of_range_weight_fails_closed(tmp_path: Path) -> None:
    sidecar = _write_sidecar(tmp_path / "weights.csv", [("train/0000/a.jpg", 1.5)])
    with pytest.raises(ValueError, match="outside"):
        load_sample_weights(sidecar, ["train/0000/a.jpg"])


def test_sample_weight_path_resolves_relative_to_config(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / "rematch750_ft.yaml").read_text(encoding="utf-8")
    )
    payload["trust"]["sample_weight_path"] = "sidecar/weights.csv"
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    resolved = load_config(config_path)
    assert resolved["trust"]["sample_weight_path"] == str(
        (tmp_path / "sidecar" / "weights.csv").resolve()
    )


def test_paired_control_config_has_no_sample_weight_key() -> None:
    """RM_FT is the unweighted control, so the injection must not fire for it."""
    config = load_config(REPO_ROOT / "configs" / "rematch750_ft.yaml")
    assert "sample_weight_path" not in config["trust"]
    assert config["trust"]["enabled"] is False
