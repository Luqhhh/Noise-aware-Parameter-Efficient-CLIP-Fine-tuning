"""Cross-label duplicate groups become candidate-label sets (NEW03).

Some samples in the training split share a decoded-RGB+dimensions identity
(``content_group``) but carry different recorded labels.  The conservative
reading is that at least one label is wrong, so instead of rejecting the whole
group the NEW03 candidate keeps every sample and lets it supervise any label in
its group's candidate set.

This module is deliberately read-only with respect to the split: it only reads
the supplied training CSV and writes a separate JSON sidecar.  It never touches
the CSV, the recorded labels or the content groups, and it refuses any input that
is not the training split (``train_dev.csv``), because ``full_train.csv`` also
contains the validation rows and ``val_dev.csv`` / ``test_manifest.csv`` must
never contribute labels to a training-time candidate set.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from aegis_clip.runtime import atomic_json_dump, sha256_file


SCHEMA_VERSION = 1

#: Only the training split may feed a training-time candidate set.
TRAIN_SPLIT_NAME = "train_dev.csv"

_INTEGER = re.compile(r"[+-]?\d+")


def _require_train_split(path: Path) -> None:
    if path.name != TRAIN_SPLIT_NAME:
        raise ValueError(
            "duplicate_sets only accepts the training split "
            f"{TRAIN_SPLIT_NAME!r} (got {path.name!r}); validation and test "
            "labels must never enter a candidate set"
        )


def _parse_label(text: object) -> int:
    raw = str(text).strip()
    if not _INTEGER.fullmatch(raw):
        raise ValueError(f"duplicate_sets label is not an integer: {text!r}")
    return int(raw)


def _read_groups(
    train_csv: str | Path,
    group_column: str,
    num_classes: int | None,
) -> tuple[Path, dict[str, int], dict[str, str]]:
    """Read image_path/label/group rows, failing closed on any malformed input."""
    path = Path(train_csv).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    _require_train_split(path)
    if not str(group_column):
        raise ValueError("group_column must be a non-empty column name")
    if num_classes is not None:
        num_classes = int(num_classes)
        if num_classes <= 0:
            raise ValueError("num_classes must be positive when supplied")

    labels: dict[str, int] = {}
    groups: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = {"image_path", "label", str(group_column)} - fields
        if missing:
            raise ValueError(f"train csv is missing columns: {sorted(missing)}")
        for row in reader:
            sample_path = str(row["image_path"])
            if not sample_path:
                raise ValueError("duplicate_sets encountered an empty image_path")
            if sample_path in labels:
                raise ValueError(f"duplicate image_path in train csv: {sample_path}")
            label = _parse_label(row["label"])
            if num_classes is not None and not 0 <= label < num_classes:
                raise ValueError(
                    f"label {label} for {sample_path} is outside [0, {num_classes})"
                )
            labels[sample_path] = label
            groups[sample_path] = str(row[group_column])
    if not labels:
        raise ValueError("train csv contains no samples")
    return path, labels, groups


def _group_statistics(
    labels: dict[str, int], groups: dict[str, str]
) -> tuple[dict[str, set[int]], Counter, Counter, int, int]:
    labels_by_group: dict[str, set[int]] = {}
    for sample_path, group in groups.items():
        labels_by_group.setdefault(group, set()).add(labels[sample_path])
    conflict_groups = {
        group: members for group, members in labels_by_group.items() if len(members) > 1
    }
    conflict_samples = sum(
        1 for group in groups.values() if group in conflict_groups
    )
    size_histogram = Counter()
    labels_histogram = Counter()
    for members in conflict_groups.values():
        labels_histogram[len(members)] += 1
    group_member_counts = Counter(groups.values())
    for group in conflict_groups:
        size_histogram[group_member_counts[group]] += 1
    return labels_by_group, size_histogram, labels_histogram, len(conflict_groups), conflict_samples


def _string_keyed(counter: Counter) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def conflict_group_report(
    train_csv: str | Path, *, group_column: str = "content_group"
) -> dict[str, Any]:
    """Report cross-label duplicate statistics without writing any sidecar.

    Useful as a minimal read-only check of how many training samples a
    candidate-label-set mechanism would actually affect.
    """
    _, labels, groups = _read_groups(train_csv, group_column, None)
    _, size_histogram, labels_histogram, conflicts, conflict_samples = _group_statistics(
        labels, groups
    )
    return {
        "num_groups": len(set(groups.values())),
        "num_conflict_groups": conflicts,
        "num_conflict_samples": conflict_samples,
        "conflict_group_size_histogram": _string_keyed(size_histogram),
        "distinct_labels_per_group_histogram": _string_keyed(labels_histogram),
    }


def build_candidate_label_sets(
    train_csv: str | Path,
    output_json: str | Path,
    *,
    group_column: str = "content_group",
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Write a candidate-label-set sidecar for every training sample.

    A cross-label duplicate group (one ``group_column`` value carrying more than
    one distinct recorded label) maps every one of its samples to the sorted
    union of that group's labels.  Every other sample keeps its singleton
    ``[label]``.  The split CSV, its labels and its content groups are never
    modified.
    """
    path, labels, groups = _read_groups(train_csv, group_column, num_classes)
    labels_by_group, size_histogram, _, conflicts, conflict_samples = _group_statistics(
        labels, groups
    )
    conflict_groups = {
        group: members for group, members in labels_by_group.items() if len(members) > 1
    }
    shared = {group: sorted(members) for group, members in conflict_groups.items()}
    singleton: dict[int, list[int]] = {}
    candidate_labels: dict[str, list[int]] = {}
    for sample_path, label in labels.items():
        group = groups[sample_path]
        if group in shared:
            candidate_labels[sample_path] = shared[group]
        else:
            candidate_labels[sample_path] = singleton.setdefault(label, [label])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "train_csv": str(path.resolve()),
        "train_csv_sha256": sha256_file(path),
        "group_column": str(group_column),
        "num_samples": len(candidate_labels),
        "num_groups": len(set(groups.values())),
        "num_conflict_groups": conflicts,
        "num_conflict_samples": conflict_samples,
        "conflict_group_sizes": _string_keyed(size_histogram),
        "candidate_labels": candidate_labels,
    }
    atomic_json_dump(payload, output_json)
    return {key: value for key, value in payload.items() if key != "candidate_labels"}


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in candidate-label sidecar: {key!r}")
        result[key] = value
    return result


def load_candidate_label_sets(path: str | Path) -> dict[str, list[int]]:
    """Load and validate a sidecar; return ``{canonical image_path: [labels]}``."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(
        source.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys
    )
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported candidate-label-set sidecar schema version")
    if not str(payload.get("group_column", "")):
        raise ValueError("candidate-label-set sidecar does not declare group_column")
    mapping = payload.get("candidate_labels")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("candidate-label-set sidecar has no candidate_labels mapping")
    if payload.get("num_samples") != len(mapping):
        raise ValueError("candidate-label-set sidecar sample count mismatch")
    result: dict[str, list[int]] = {}
    for sample_path, values in mapping.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"candidate set for {sample_path!r} is not a non-empty list")
        parsed = [_parse_label(value) for value in values]
        if parsed != sorted(set(parsed)):
            raise ValueError(f"candidate set for {sample_path!r} is not sorted/unique")
        result[str(sample_path)] = parsed
    return result
