"""Create a reproducible class-level A2/M1/M3 validation diagnostic."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import torch

from aegis_clip.class_diagnostic import diagnose_class_errors
from aegis_clip.runtime import atomic_json_dump, sha256_file


def _atomic_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_diagnostic(
    center_cache_path: str | Path,
    m1_cache_path: str | Path,
    m3_cache_path: str | Path,
    train_cache_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    paths = {
        "a2_center_cache": Path(center_cache_path).resolve(),
        "m1_cache": Path(m1_cache_path).resolve(),
        "m3_cache": Path(m3_cache_path).resolve(),
        "high_clean_train_cache": Path(train_cache_path).resolve(),
    }
    loaded = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    report, class_rows, confusion_rows = diagnose_class_errors(
        loaded["a2_center_cache"],
        loaded["m1_cache"],
        loaded["m3_cache"],
        loaded["high_clean_train_cache"],
    )
    report["lineage"] = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    report["protocol"] = {
        "test_data_used": False,
        "external_data_used": False,
        "parameter_scan": False,
        "purpose": "diagnosis and next-experiment selection only",
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "diagnostic.json"
    class_path = output_dir / "class_metrics.csv"
    confusion_path = output_dir / "top_confusions.csv"
    _atomic_csv(class_rows, class_path)
    _atomic_csv(confusion_rows, confusion_path)
    report["artifacts"] = {
        "class_metrics_csv": str(class_path),
        "top_confusions_csv": str(confusion_path),
    }
    atomic_json_dump(report, report_path)
    return report_path, class_path, confusion_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--center-cache", required=True)
    parser.add_argument("--m1-cache", required=True)
    parser.add_argument("--m3-cache", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for path in run_diagnostic(
        args.center_cache,
        args.m1_cache,
        args.m3_cache,
        args.train_cache,
        args.output_dir,
    ):
        print(path)


if __name__ == "__main__":
    main()
