#!/usr/bin/env python3
"""Build a current-split content-group-aware OOF quality asset for C group.

The heavy lifting is the existing ``analysis.oof.run_oof`` pipeline.  This
driver only creates the current 750-class fold assignments, then invokes the
pipeline against the official frozen feature cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))

from analysis.oof.build_folds import assign_group_stratified_folds  # noqa: E402


def _sample_id(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--flip-batch-size", type=int, default=128)
    parser.add_argument("--flip-workers", type=int, default=4)
    args = parser.parse_args()

    frame = pd.read_csv(args.train_csv, dtype={"image_path": str, "label": int})
    required = {"image_path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"train csv missing columns: {sorted(missing)}")
    group_column = "content_group" if "content_group" in frame.columns else "file_sha256"
    if group_column not in frame.columns:
        raise ValueError("train csv needs content_group or file_sha256 for grouping")
    frame = frame.sort_values("image_path").reset_index(drop=True)
    frame["sample_id"] = frame["image_path"].map(_sample_id)
    frame["sha256"] = frame[group_column].astype(str)
    assignments = assign_group_stratified_folds(
        frame[["sample_id", "image_path", "label", "sha256"]],
        n_splits=int(args.folds),
        seed=int(args.seed),
    )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assignment_path = output / "fold_assignments.csv"
    assignments.to_csv(assignment_path, index=False)

    groups: dict[str, list[str]] = {}
    for path, group in zip(assignments["image_path"].astype(str), assignments["sha256"].astype(str)):
        groups.setdefault(group, []).append(path)
    duplicate_scan = {
        "duplicates": [
            {"group": group, "paths": paths}
            for group, paths in sorted(groups.items())
            if len(paths) > 1
        ]
    }
    duplicate_path = output / "duplicate_scan.json"
    duplicate_path.write_text(
        json.dumps(duplicate_scan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cache_manifest = Path(args.cache_dir) / "manifest.json"
    if not cache_manifest.is_file():
        raise FileNotFoundError(f"feature cache manifest is missing: {cache_manifest}")

    command = [
        sys.executable,
        "-m",
        "analysis.oof.run_oof",
        "--assignments",
        str(assignment_path),
        "--cache-dir",
        args.cache_dir,
        "--duplicate-scan",
        str(duplicate_path),
        "--output-dir",
        str(output),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        "256",
        "--infer-batch-size",
        "2048",
        "--lr",
        "0.005",
        "--weight-decay",
        "0.0001",
        "--warmup-epochs",
        "2",
        "--q",
        "0.5",
        "--seed",
        str(args.seed),
        "--num-classes",
        "750",
        "--k-neighbors",
        "10",
        "--query-batch-size",
        "64",
        "--reference-chunk-size",
        "4096",
        "--flip-batch-size",
        str(args.flip_batch_size),
        "--flip-workers",
        str(args.flip_workers),
        "--device",
        args.device,
        "--protocol-name",
        "REMATCH750_F05_FOCUS_OOF_3FOLD",
        "--parent-artifact",
        "openai_clip_vit_b32_frozen_features",
        "--data-stage",
        "repechage_20260921",
        "--source-train-csv-sha256",
        _sha256_file(Path(args.train_csv)),
        "--feature-cache-manifest-sha256",
        _sha256_file(cache_manifest),
    ]
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(ROOT)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
