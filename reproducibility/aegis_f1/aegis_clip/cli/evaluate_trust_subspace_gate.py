"""Persist the preregistered T0/T1 trust-subspace decision."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.trust_subspace_gate import evaluate_trust_subspace_gate


def _json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        values = list(csv.DictReader(handle))
    if not values:
        raise ValueError(f"Metrics CSV is empty: {path}")
    for row_index, row in enumerate(values, start=1):
        for name, raw_value in row.items():
            if raw_value in (None, ""):
                continue
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Metrics CSV has non-numeric {name} at row {row_index}"
                ) from exc
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"Metrics CSV has non-finite {name} at row {row_index}"
                )
    return values[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-m1", required=True)
    parser.add_argument("--t0-center", required=True)
    parser.add_argument("--t0-m1", required=True)
    parser.add_argument("--t1-center", required=True)
    parser.add_argument("--t1-m1", required=True)
    parser.add_argument("--t0-initial", required=True)
    parser.add_argument("--t1-initial", required=True)
    parser.add_argument("--t0-evaluation", required=True)
    parser.add_argument("--t1-evaluation", required=True)
    parser.add_argument("--t0-metrics", required=True)
    parser.add_argument("--t1-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in vars(args).items()
        if name != "output"
    }
    caches = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
        if name in {"original_m1", "t0_center", "t0_m1", "t1_center", "t1_m1"}
    }
    report = evaluate_trust_subspace_gate(
        original_m1=caches["original_m1"],
        t0_center=caches["t0_center"],
        t0_m1=caches["t0_m1"],
        t1_center=caches["t1_center"],
        t1_m1=caches["t1_m1"],
        t0_initial=_json(paths["t0_initial"]),
        t1_initial=_json(paths["t1_initial"]),
        t0_evaluation=_json(paths["t0_evaluation"]),
        t1_evaluation=_json(paths["t1_evaluation"]),
        t0_last_metrics=_last_csv_row(paths["t0_metrics"]),
        t1_last_metrics=_last_csv_row(paths["t1_metrics"]),
    )
    report["lineage"] = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    destination = Path(args.output).resolve()
    atomic_json_dump(report, destination)
    print(destination)


if __name__ == "__main__":
    main()
