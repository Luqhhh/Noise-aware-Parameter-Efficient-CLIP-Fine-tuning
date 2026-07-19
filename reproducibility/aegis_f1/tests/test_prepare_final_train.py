import csv

import pytest

from aegis_clip.cli.prepare_final_train import merge_splits


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("image_path", "class_name", "label")
        )
        writer.writeheader()
        writer.writerows(rows)


def test_merge_splits_is_complete_disjoint_and_sorted(tmp_path):
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    output = tmp_path / "full.csv"
    _write(train, [{"image_path": "b.jpg", "class_name": "0001", "label": "1"}])
    _write(val, [{"image_path": "a.jpg", "class_name": "0000", "label": "0"}])

    merge_splits(train, val, output, expected_samples=2)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["image_path"] for row in rows] == ["a.jpg", "b.jpg"]


def test_merge_splits_rejects_overlap(tmp_path):
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    output = tmp_path / "full.csv"
    row = {"image_path": "same.jpg", "class_name": "0000", "label": "0"}
    _write(train, [row])
    _write(val, [row])

    with pytest.raises(ValueError, match="overlap"):
        merge_splits(train, val, output, expected_samples=2)
