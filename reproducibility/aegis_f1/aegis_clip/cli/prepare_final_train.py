"""Build an auditable full-data training CSV after model selection is complete."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


REQUIRED_COLUMNS = ("image_path", "class_name", "label")


def merge_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    output_csv: str | Path,
    *,
    expected_samples: int,
) -> Path:
    """Merge disjoint selection splits into one deterministic final-training list."""
    rows: list[dict[str, str]] = []
    for source in (Path(train_csv), Path(val_csv)):
        with source.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
                raise ValueError(
                    f"Unexpected columns in {source}: {reader.fieldnames}; "
                    f"expected {list(REQUIRED_COLUMNS)}"
                )
            rows.extend(reader)

    paths = [row["image_path"] for row in rows]
    if len(rows) != expected_samples:
        raise ValueError(
            f"Merged sample count is {len(rows)}, expected {expected_samples}"
        )
    if len(set(paths)) != len(paths):
        raise ValueError("Train/validation CSVs overlap by image_path")

    rows.sort(key=lambda row: row["image_path"])
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    args = parser.parse_args()
    destination = merge_splits(
        args.train_csv,
        args.val_csv,
        args.output_csv,
        expected_samples=args.expected_samples,
    )
    print(destination)


if __name__ == "__main__":
    main()
