#!/usr/bin/env python3
"""Freeze the current-stage ``clean_probability >= 0.70`` adapter subset.

D1 and D2 must share exactly the same training subset and must not be coupled
to N1's stricter reject sidecar.  This script takes an existing current-stage
quality asset (CSV or trust-bundle ``.pt``), filters the F05 train CSV, and
preserves its schema byte-for-byte except for row selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
for location in (str(ROOT), str(AEGIS_ROOT)):
    if location not in sys.path:
        sys.path.insert(0, location)

from aegis_clip.features import canonical_sample_path  # noqa: E402
from aegis_clip.runtime import sha256_file  # noqa: E402


def _quality_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if "image_path" not in frame.columns:
            raise ValueError("quality CSV must contain image_path")
        if "clean_probability" not in frame.columns:
            raise ValueError("quality CSV must contain clean_probability")
        return frame[["image_path", "clean_probability"]].copy()
    if path.suffix.lower() in {".pt", ".pth"}:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("quality payload must be a mapping")
        if "paths" not in payload or "clean_probability" not in payload:
            raise ValueError("quality payload must contain paths and clean_probability")
        return pd.DataFrame(
            {
                "image_path": [str(item) for item in payload["paths"]],
                "clean_probability": torch.as_tensor(
                    payload["clean_probability"]
                ).float().tolist(),
            }
        )
    raise ValueError("quality asset must be CSV, .pt, or .pth")


def build_subset(
    source_csv: str | Path,
    quality_path: str | Path,
    output_csv: str | Path,
    *,
    threshold: float = 0.70,
) -> dict[str, Any]:
    source = Path(source_csv).expanduser().resolve()
    quality = Path(quality_path).expanduser().resolve()
    destination = Path(output_csv).expanduser().resolve()
    if not source.is_file() or not quality.is_file():
        raise FileNotFoundError(f"{source} or {quality} is missing")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    source_frame = pd.read_csv(source)
    if "image_path" not in source_frame.columns or "label" not in source_frame.columns:
        raise ValueError("source split must contain image_path and label")
    qframe = _quality_frame(quality)
    qframe["_key"] = qframe["image_path"].map(canonical_sample_path)
    if qframe["_key"].duplicated().any():
        raise ValueError("quality asset contains duplicate canonical sample paths")
    qframe["clean_probability"] = pd.to_numeric(
        qframe["clean_probability"], errors="raise"
    )
    if ((qframe["clean_probability"] < 0.0) | (qframe["clean_probability"] > 1.0)).any():
        raise ValueError("clean_probability must lie in [0, 1]")

    source_frame["_key"] = source_frame["image_path"].map(canonical_sample_path)
    if source_frame["_key"].duplicated().any():
        raise ValueError("source split contains duplicate canonical sample paths")
    merged = source_frame.merge(
        qframe[["_key", "clean_probability"]], on="_key", how="left", validate="one_to_one"
    )
    if merged["clean_probability"].isna().any():
        missing = merged.loc[
            merged["clean_probability"].isna(), "image_path"
        ].head(5).tolist()
        raise ValueError(
            f"quality asset does not cover all source rows; first missing={missing}"
        )
    selected = merged[merged["clean_probability"] >= float(threshold)].copy()
    if selected.empty:
        raise ValueError("clean subset is empty")
    selected = selected.drop(columns=["_key", "clean_probability"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(destination, index=False)

    labels = selected["label"].astype(int)
    audit = {
        "source_csv": str(source),
        "source_csv_sha256": sha256_file(source),
        "quality_asset": str(quality),
        "quality_asset_sha256": sha256_file(quality),
        "threshold": float(threshold),
        "source_samples": int(len(source_frame)),
        "selected_samples": int(len(selected)),
        "selected_fraction": float(len(selected) / max(len(source_frame), 1)),
        "class_count": int(labels.nunique()),
        "min_samples_per_class": int(labels.value_counts().min()),
        "max_samples_per_class": int(labels.value_counts().max()),
        "output_csv": str(destination),
        "output_csv_sha256": sha256_file(destination),
        "labels_modified": False,
        "validation_data_used": False,
        "test_data_used": False,
    }
    audit_path = destination.with_name(destination.stem + "_audit.json")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--output", default="artifacts/f05_focus/clean070.csv")
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()
    audit = build_subset(
        args.source_csv,
        args.quality,
        args.output,
        threshold=float(args.threshold),
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
