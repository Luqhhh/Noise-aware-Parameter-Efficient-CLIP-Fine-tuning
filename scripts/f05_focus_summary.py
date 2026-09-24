"""Shared writer for ``results/f05_focus_summary.csv``.

Only this module knows the column contract.  Rows are keyed by ``experiment_id``
and replaced when the same unit is re-evaluated, so repeated probe runs do not
duplicate history.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

SUMMARY_FIELDS = [
    "experiment_id",
    "parent_sha",
    "seed",
    "macro",
    "micro",
    "delta_macro",
    "delta_micro",
    "changed_predictions",
    "selected_epoch",
    "reject_rate",
    "platform_score",
    "status",
]


def append_summary_row(path: str | Path, row: dict[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if destination.is_file():
        with destination.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != SUMMARY_FIELDS:
                raise ValueError(
                    f"{destination} header differs from the focus summary contract"
                )
            existing = [
                {field: item.get(field, "") for field in SUMMARY_FIELDS}
                for item in reader
            ]
    rendered = {field: _format(row.get(field)) for field in SUMMARY_FIELDS}
    replaced = False
    for index, item in enumerate(existing):
        if item["experiment_id"] == rendered["experiment_id"]:
            existing[index] = rendered
            replaced = True
            break
    if not replaced:
        existing.append(rendered)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
    return destination


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.10f}".rstrip("0").rstrip(".") or "0"
    return str(value)
