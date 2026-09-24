#!/usr/bin/env python3
"""Audit and hardlink-preserve one REMATCH750 V5 platform candidate.

This script never uploads.  It re-reads the protected artifacts, validates the
checkpoint lineage, runs the repository submission checker, and creates an
independent hardlink copy only after every check passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_and_names(csv_path: Path, expected_rows: int) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError("prediction CSV is empty")
    if rows[0] != ["image_name", "predicted_class"]:
        raise ValueError(f"unexpected CSV header: {rows[0]!r}")
    body = rows[1:]
    if len(body) != expected_rows:
        raise ValueError(f"expected {expected_rows} prediction rows, got {len(body)}")
    names = [row[0] for row in body]
    if len(set(names)) != len(names):
        raise ValueError("prediction CSV contains duplicate image names")
    if any(len(name) < 4 or not name[-4:].isdigit() for name in names):
        raise ValueError("prediction CSV image names must end in four digits")
    return names


def _zip_same_bytes(csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if names != ["pred_results.csv"]:
            raise ValueError(f"ZIP must contain exactly pred_results.csv, got {names!r}")
        inner = archive.read("pred_results.csv")
    outer = csv_path.read_bytes()
    if inner != outer:
        raise ValueError("ZIP pred_results.csv bytes differ from the audited CSV")


def _hardlink(src: Path, dst: Path) -> None:
    if dst.exists():
        if os.stat(src).st_ino == os.stat(dst).st_ino:
            return
        raise FileExistsError(f"preserve destination already exists: {dst}")
    os.link(src, dst)
    if os.stat(src).st_ino != os.stat(dst).st_ino:
        raise RuntimeError(f"hardlink failed for {src} -> {dst}")


def main() -> int:
    from aegis_clip.config import load_config
    from aegis_clip.rematch_protocol import validate_checkpoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--preserve-dir")
    parser.add_argument("--expected-rows", type=int, default=37444)
    parser.add_argument("--skip-binding", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = Path(args.checkpoint).resolve()
    csv_path = Path(args.csv).resolve()
    zip_path = Path(args.zip).resolve()
    if not args.skip_binding:
        validate_checkpoint(checkpoint, config)

    _count_and_names(csv_path, int(args.expected_rows))
    _zip_same_bytes(csv_path, zip_path)

    test_dir = Path(config["data"]["test_root"]).resolve()
    class_mapping = Path(config["data"]["class_mapping"]).resolve()
    check = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_submission.py"),
            "--test_dir",
            str(test_dir),
            "--class-mapping",
            str(class_mapping),
            "--csv",
            str(csv_path),
            "--zip",
            str(zip_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "submission checker failed:\n" + check.stdout + "\n" + check.stderr
        )

    record: dict[str, Any] = {
        "candidate": config["project"]["experiment_id"],
        "protocol": config["project"]["protocol"],
        "trial_id": config["project"].get("trial_id"),
        "expected_rows": int(args.expected_rows),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256(Path(args.config).resolve()),
        "csv": str(csv_path),
        "csv_sha256": sha256(csv_path),
        "zip": str(zip_path),
        "zip_sha256": sha256(zip_path),
        "submission_checker": "passed",
        "zip_inner_csv_byte_identical": True,
    }
    load_config_hash = None
    if (Path(args.config).resolve().parent / "training_code_manifest.json").is_file():
        record["training_code_manifest"] = str(
            (Path(args.config).resolve().parent / "training_code_manifest.json")
        )

    if args.preserve_dir:
        preserve = Path(args.preserve_dir).resolve()
        preserve.mkdir(parents=True, exist_ok=True)
        _hardlink(checkpoint, preserve / "best.pt")
        _hardlink(Path(args.config).resolve(), preserve / "config.yaml")
        _hardlink(csv_path, preserve / "pred_results.csv")
        _hardlink(zip_path, preserve / "submission.zip")
        record["preserve_dir"] = str(preserve)
        record["hardlink_inode_match"] = True
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
