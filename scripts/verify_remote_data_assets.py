"""Verify a relocated copy against the current stage's immutable file manifests.

Read-only: no decoding, label edits, feature fitting, or training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_row(item):
    root, row = item
    relative = Path(row["image_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    path = root / relative
    try:
        actual_size = path.stat().st_size
        actual_sha = sha(path)
    except OSError as error:
        return {"path": str(relative), "error": str(error)}, 0
    if actual_size != int(row["bytes"]) or actual_sha != row["file_sha256"]:
        return {"path": str(relative), "bytes": actual_size, "sha256": actual_sha}, actual_size
    return None, actual_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stage-artifacts", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--official-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    started = time.monotonic()
    metadata = json.loads((args.stage_artifacts / "dataset_manifest.json").read_text())
    assert metadata["data_version"] == "20260921" and metadata["stage"] == "repechage"
    assert metadata["external_data"] is False and metadata["test_usage"] == "inference_only"
    changed_audits = [name for name, digest in metadata["files"].items()
                      if sha(args.stage_artifacts / name) != digest]
    weight_sha = sha(args.official_checkpoint)
    report = {"data_version": metadata["data_version"], "data_root": str(args.data_root),
              "stage_artifacts": str(args.stage_artifacts), "changed_audit_files": changed_audits,
              "official_checkpoint_sha256": weight_sha, "datasets": {},
              "training_started": False}
    for kind, manifest in (("train", "full_train.csv"), ("test", "test_manifest.csv")):
        with (args.stage_artifacts / manifest).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected_paths = {row["image_path"] for row in rows}
        actual_paths = {str(path.relative_to(args.data_root))
                        for path in (args.data_root / kind).rglob("*") if path.is_file()}
        errors, total_bytes = [], 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, (error, size) in enumerate(pool.map(check_row, ((args.data_root, row) for row in rows)), 1):
                total_bytes += size
                if error:
                    errors.append(error)
                if index % 10000 == 0:
                    print(f"{kind}: checked {index}/{len(rows)} files", flush=True)
        fingerprint = hashlib.sha256()
        for row in rows:
            line = f"{row['image_path']}:{row['file_sha256']}"
            if kind == "train":
                line = f"{row['image_path']}:{row['label']}:{row['file_sha256']}"
            fingerprint.update((line + "\n").encode())
        entry = {"expected_files": len(rows), "actual_files": len(actual_paths),
                 "bytes": total_bytes, "mismatches": errors,
                 "missing_files": sorted(expected_paths - actual_paths),
                 "extra_files": sorted(actual_paths - expected_paths),
                 "fingerprint": fingerprint.hexdigest(),
                 "fingerprint_matches": fingerprint.hexdigest() == metadata[f"{kind}_fingerprint"]}
        entry["passed"] = (len(rows) == metadata[f"{kind}_samples"] and not errors
                           and not entry["missing_files"] and not entry["extra_files"]
                           and entry["fingerprint_matches"])
        report["datasets"][kind] = entry
    report["train_classes"] = len([p for p in (args.data_root / "train").iterdir() if p.is_dir()])
    report["passed"] = (not changed_audits and weight_sha == args.official_sha256
                        and report["train_classes"] == metadata["num_classes"]
                        and all(entry["passed"] for entry in report["datasets"].values()))
    report["elapsed_seconds"] = time.monotonic() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
