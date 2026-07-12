#!/usr/bin/env python3
"""Register a submission in the submission registry CSV.

Reads a submission manifest and appends (or validates) a row in
results/submission_registry.csv.  Rejects duplicate entries by
submission_zip_sha256.

Usage:
    python scripts/register_submission.py \
      --manifest outputs/d3_strict/seed42/submissions/submission_manifest.json \
      [--notes "First platform submission for D3_STRICT"]
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_PATH = Path("results/submission_registry.csv")

REGISTRY_FIELDS = [
    "submission_id",
    "experiment_id",
    "git_commit",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_epoch",
    "split_seed",
    "train_seed",
    "local_micro_accuracy",
    "local_macro_accuracy",
    "prediction_csv_sha256",
    "submission_zip_sha256",
    "online_accuracy",
    "local_online_gap",
    "submission_time",
    "notes",
]


def _load_existing_hashes(registry_path: Path) -> set:
    """Return the set of already-registered submission_zip_sha256 values."""
    if not registry_path.exists():
        return set()
    hashes = set()
    with open(registry_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = row.get("submission_zip_sha256", "").strip()
            if h:
                hashes.add(h)
    return hashes


def _generate_submission_id(experiment_id: str) -> str:
    """Generate a unique submission ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{experiment_id}_{ts}"


def main():
    parser = argparse.ArgumentParser(
        description="Register a submission from a manifest."
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to submission_manifest.json."
    )
    parser.add_argument(
        "--notes", default="",
        help="Optional free-text notes for the registry."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Allow re-registration even if hash exists (updates row)."
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # ── Check for duplicate ─────────────────────────────────────────
    zip_sha = manifest["submission_zip_sha256"]
    existing = _load_existing_hashes(REGISTRY_PATH)

    if zip_sha in existing and not args.force:
        print(
            f"ERROR: submission_zip_sha256 {zip_sha[:16]}... already exists "
            f"in {REGISTRY_PATH}.\n"
            f"Use --force to override, or this is a duplicate submission.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Build row ───────────────────────────────────────────────────
    submission_time = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    submission_id = _generate_submission_id(manifest["experiment_id"])

    row = {
        "submission_id": submission_id,
        "experiment_id": manifest["experiment_id"],
        "git_commit": manifest.get("git_commit", ""),
        "checkpoint_path": manifest["checkpoint_path"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_epoch": manifest["checkpoint_epoch"],
        "split_seed": manifest.get("split_seed", ""),
        "train_seed": manifest.get("train_seed", ""),
        "local_micro_accuracy": manifest["local_micro_accuracy"],
        "local_macro_accuracy": manifest["local_macro_accuracy"],
        "prediction_csv_sha256": manifest["prediction_csv_sha256"],
        "submission_zip_sha256": manifest["submission_zip_sha256"],
        "online_accuracy": manifest["online_accuracy"],
        "local_online_gap": manifest["local_online_gap"],
        "submission_time": submission_time,
        "notes": args.notes,
    }

    # ── Write CSV ───────────────────────────────────────────────────
    file_exists = REGISTRY_PATH.exists()

    # If force-overwriting, read all rows, replace matching hash row
    if args.force and file_exists:
        rows = []
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Remove row with matching hash
        rows = [r for r in rows if r.get("submission_zip_sha256", "").strip() != zip_sha]

        with open(REGISTRY_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(row)
    else:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    print(f"Registered submission: {submission_id}")
    print(f"  experiment:   {row['experiment_id']}")
    print(f"  online acc:   {row['online_accuracy']}")
    print(f"  local micro:  {row['local_micro_accuracy']:.8f}")
    print(f"  gap:          {row['local_online_gap']:+.6f}")
    print(f"  registry:     {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
