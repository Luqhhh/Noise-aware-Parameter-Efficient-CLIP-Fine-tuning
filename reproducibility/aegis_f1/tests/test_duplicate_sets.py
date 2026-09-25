"""Tests for cross-label duplicate candidate-label sets (NEW03)."""

import csv
import json
from pathlib import Path
from unittest import mock

import pytest

from aegis_clip.duplicate_sets import (
    build_candidate_label_sets,
    conflict_group_report,
    load_candidate_label_sets,
)
from aegis_clip.runtime import sha256_file


HEADER = [
    "image_path",
    "label",
    "bytes",
    "file_sha256",
    "width",
    "height",
    "content_group",
    "error",
]

# g1: labels {0,1} -> conflict (3 samples); g2: singleton; g3: same label twice
# (not a conflict); g4: labels {1,2,3} -> conflict (3 samples).
ROWS = [
    ("train/0000/a.jpg", 0, "g1"),
    ("train/0000/b.jpg", 1, "g1"),
    ("train/0000/c.jpg", 1, "g1"),
    ("train/0000/d.jpg", 2, "g2"),
    ("train/0000/e.jpg", 3, "g3"),
    ("train/0000/f.jpg", 3, "g3"),
    ("train/0000/g.jpg", 1, "g4"),
    ("train/0000/h.jpg", 2, "g4"),
    ("train/0000/i.jpg", 3, "g4"),
]


def _write_csv(path: Path, rows=ROWS) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for index, (image_path, label, group) in enumerate(rows):
            writer.writerow(
                [image_path, label, 100 + index, f"{index:064x}", 800, 800, group, ""]
            )
    return path


def _train(tmp_path: Path, rows=ROWS) -> Path:
    return _write_csv(tmp_path / "train_dev.csv", rows)


def _build_payload(train: Path, out: Path):
    """Build once, capturing the in-memory payload to check shared list objects."""
    with mock.patch("aegis_clip.duplicate_sets.atomic_json_dump") as dump:
        stats = build_candidate_label_sets(train, out)
    payload = dump.call_args.args[0]
    out.write_text(json.dumps(payload), encoding="utf-8")
    return stats, payload


def test_cross_label_group_becomes_union_and_every_copy_is_identical(tmp_path):
    train = _train(tmp_path)
    out = tmp_path / "sets.json"
    stats, payload = _build_payload(train, out)
    mapping = payload["candidate_labels"]

    assert mapping["train/0000/a.jpg"] == mapping["train/0000/b.jpg"] == mapping["train/0000/c.jpg"] == [0, 1]
    assert mapping["train/0000/a.jpg"] is mapping["train/0000/b.jpg"] is mapping["train/0000/c.jpg"]
    assert mapping["train/0000/g.jpg"] == mapping["train/0000/h.jpg"] == mapping["train/0000/i.jpg"] == [1, 2, 3]
    assert mapping["train/0000/g.jpg"] is mapping["train/0000/i.jpg"]
    assert stats["num_conflict_groups"] == 2
    assert stats["num_conflict_samples"] == 6
    # The serialized sidecar carries the same content.
    assert load_candidate_label_sets(out)["train/0000/a.jpg"] == [0, 1]


def test_singleton_and_same_label_groups_stay_singleton(tmp_path):
    train = _train(tmp_path)
    out = tmp_path / "sets.json"
    _, payload = _build_payload(train, out)
    mapping = payload["candidate_labels"]

    assert mapping["train/0000/d.jpg"] == [2]
    assert mapping["train/0000/e.jpg"] == mapping["train/0000/f.jpg"] == [3]
    assert mapping["train/0000/e.jpg"] is mapping["train/0000/f.jpg"]
    assert load_candidate_label_sets(out)["train/0000/e.jpg"] == [3]


def test_stats_and_histograms_match_hand_built_fixture(tmp_path):
    train = _train(tmp_path)
    stats = build_candidate_label_sets(train, tmp_path / "sets.json")
    report = conflict_group_report(train)

    assert stats["num_samples"] == 9
    assert stats["num_groups"] == 4
    assert stats["num_conflict_groups"] == 2
    assert stats["num_conflict_samples"] == 6
    assert stats["conflict_group_sizes"] == {"3": 2}
    assert stats["train_csv_sha256"] == sha256_file(train)
    assert stats["schema_version"] == 1
    assert stats["group_column"] == "content_group"

    assert report == {
        "num_groups": 4,
        "num_conflict_groups": 2,
        "num_conflict_samples": 6,
        "conflict_group_size_histogram": {"3": 2},
        "distinct_labels_per_group_histogram": {"2": 1, "3": 1},
    }


def test_sidecar_is_separate_and_does_not_touch_the_split(tmp_path):
    train = _train(tmp_path)
    before = train.read_bytes()
    sidecar = tmp_path / "sets.json"
    build_candidate_label_sets(train, sidecar)

    assert train.read_bytes() == before
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["num_samples"] == payload["num_conflict_samples"] + 3
    assert set(payload["candidate_labels"]) == {row[0] for row in ROWS}


def test_only_the_supplied_train_csv_is_opened(tmp_path):
    train = _train(tmp_path)
    # A sibling validation split whose labels must never leak into the result.
    _write_csv(
        tmp_path / "val_dev.csv",
        [("train/0000/a.jpg", 7, "g1"), ("train/0000/d.jpg", 5, "g2")],
    )
    opened: list[str] = []
    real_open = Path.open

    def recording_open(self, *args, **kwargs):
        opened.append(str(self))
        return real_open(self, *args, **kwargs)

    with mock.patch.object(Path, "open", recording_open):
        build_candidate_label_sets(train, tmp_path / "sets.json")

    assert set(opened) == {str(train)}
    mapping = load_candidate_label_sets(tmp_path / "sets.json")
    assert mapping["train/0000/a.jpg"] == [0, 1]
    assert mapping["train/0000/d.jpg"] == [2]


def test_conflict_report_writes_nothing(tmp_path):
    train = _train(tmp_path)
    before = sorted(path.name for path in tmp_path.iterdir())
    conflict_group_report(train)
    assert sorted(path.name for path in tmp_path.iterdir()) == before


@pytest.mark.parametrize("name", ["val_dev.csv", "test_manifest.csv", "full_train.csv"])
def test_non_train_split_paths_fail_closed(tmp_path, name):
    other = _write_csv(tmp_path / name)
    with pytest.raises(ValueError, match="training split"):
        build_candidate_label_sets(other, tmp_path / "sets.json")
    with pytest.raises(ValueError, match="training split"):
        conflict_group_report(other)


def test_duplicate_image_path_is_rejected(tmp_path):
    train = _train(
        tmp_path,
        [
            ("train/0000/a.jpg", 0, "g1"),
            ("train/0000/a.jpg", 1, "g2"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate image_path"):
        build_candidate_label_sets(train, tmp_path / "sets.json")


def test_label_outside_class_mapping_is_rejected(tmp_path):
    train = _train(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        build_candidate_label_sets(train, tmp_path / "sets.json", num_classes=3)
    # The same fixture is accepted once the range covers every recorded label.
    stats = build_candidate_label_sets(train, tmp_path / "ok.json", num_classes=4)
    assert stats["num_samples"] == len(ROWS)


def test_non_integer_label_is_rejected(tmp_path):
    train = _train(
        tmp_path,
        [
            ("train/0000/a.jpg", "0", "g1"),
            ("train/0000/b.jpg", "1.0", "g1"),
        ],
    )
    with pytest.raises(ValueError, match="not an integer"):
        build_candidate_label_sets(train, tmp_path / "sets.json")


def test_missing_group_column_is_rejected(tmp_path):
    train = _train(tmp_path)
    with pytest.raises(ValueError, match="missing columns"):
        build_candidate_label_sets(train, tmp_path / "sets.json", group_column="not_a_column")


def test_loader_rejects_malformed_sidecar(tmp_path):
    train = _train(tmp_path)
    sidecar = tmp_path / "sets.json"
    build_candidate_label_sets(train, sidecar)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))

    payload["candidate_labels"]["train/0000/a.jpg"] = [1, 0]
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sorted/unique"):
        load_candidate_label_sets(sidecar)

    payload["candidate_labels"]["train/0000/a.jpg"] = []
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_candidate_label_sets(sidecar)
