#!/usr/bin/env python3
"""Build a submission manifest for a platform submission.

The manifest captures every artifact hash, local metrics, and the
online score so the submission is fully reproducible and auditable.

Usage:
    python scripts/build_submission_manifest.py \
      --experiment-id D3_STRICT \
      --checkpoint outputs/d3_strict/seed42/checkpoints/best.pt \
      --eval-results outputs/d3_strict/seed42/checkpoints/reeval_best.json \
      --prediction-csv outputs/d3_strict/seed42/submissions/pred_results.csv \
      --submission-zip outputs/d3_strict/seed42/submissions/submission.zip \
      --online-accuracy 0.573397 \
      --output outputs/d3_strict/seed42/submissions/submission_manifest.json
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_hex(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 (1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return ""


def _validate_labels(csv_path: Path) -> None:
    """Every label must match ^\\d{4}$."""
    pattern = re.compile(r"^\d{4}$")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV has no header row")
        for lineno, row in enumerate(reader, start=2):
            if len(row) < 2:
                raise ValueError(
                    f"Line {lineno}: expected 2 fields, got {len(row)}"
                )
            label = row[1].strip()
            if not pattern.match(label):
                raise ValueError(
                    f"Line {lineno}: invalid label '{label}' "
                    f"(must match ^\\d{{4}}$)"
                )


def _count_predictions(csv_path: Path) -> int:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def _validate_zip(zip_path: Path, csv_path: Path) -> str:
    """Validate ZIP contents and return the internal CSV SHA-256."""
    csv_sha = _sha256_hex(csv_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if names != ["pred_results.csv"]:
            raise ValueError(
                f"ZIP must contain exactly ['pred_results.csv']; "
                f"found: {names}"
            )

        with zf.open("pred_results.csv") as zf_csv:
            h = hashlib.sha256()
            while True:
                chunk = zf_csv.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
            zip_hash = h.hexdigest()

    if zip_hash != csv_sha:
        raise ValueError(
            f"ZIP-internal CSV hash ({zip_hash[:16]}...) does not "
            f"match external CSV hash ({csv_sha[:16]}...)"
        )

    return zip_hash


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a submission manifest."
    )
    parser.add_argument(
        "--experiment-id", required=True, help="e.g. D3_STRICT"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to best.pt checkpoint."
    )
    parser.add_argument(
        "--eval-results", required=True,
        help="Path to reeval_best.json (from standalone evaluate)."
    )
    parser.add_argument(
        "--prediction-csv", required=True,
        help="Path to pred_results.csv."
    )
    parser.add_argument(
        "--submission-zip", required=True,
        help="Path to submission.zip."
    )
    parser.add_argument(
        "--online-accuracy", required=True, type=float,
        help="Platform-reported online accuracy (e.g. 0.573397)."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write submission_manifest.json."
    )
    args = parser.parse_args()

    # ── Resolve paths ───────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint).resolve()
    eval_path = Path(args.eval_results).resolve()
    csv_path = Path(args.prediction_csv).resolve()
    zip_path = Path(args.submission_zip).resolve()
    output_path = Path(args.output).resolve()

    # ── Validate existence ──────────────────────────────────────────
    for p, name in [
        (ckpt_path, "checkpoint"),
        (eval_path, "eval results"),
        (csv_path, "prediction CSV"),
        (zip_path, "submission ZIP"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    # ── Load eval results ───────────────────────────────────────────
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    local_micro = float(eval_data["micro_accuracy"])
    local_macro = float(eval_data["macro_accuracy"])

    # ── Load checkpoint metadata (streaming hash, no full load) ─────
    ckpt_sha = _sha256_hex(ckpt_path)

    # Lightweight checkpoint metadata peek: read only small header keys
    import torch
    # We must load the checkpoint to read epoch/best_val_acc, but we
    # already computed the streaming hash before this point.
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_epoch = ckpt.get("epoch")
    ckpt_best_val_acc = float(ckpt.get("best_val_acc", -1.0))

    # ── Consistency: checkpoint best_val_acc == reeval micro ─────────
    if abs(ckpt_best_val_acc - local_micro) > 1e-8:
        raise ValueError(
            "Checkpoint best_val_acc does not match reeval micro_accuracy: "
            f"ckpt_best_val_acc={ckpt_best_val_acc:.10f}, "
            f"reeval_micro={local_micro:.10f}"
        )

    # ── Validate prediction CSV ─────────────────────────────────────
    _validate_labels(csv_path)
    num_preds = _count_predictions(csv_path)
    # The official test set contains 24,967 images; one may be skipped
    # by the inference script if it is corrupted/unreadable, yielding
    # 24,966 predictions.  Both counts are valid for platform submission.
    if num_preds not in (24966, 24967):
        raise ValueError(
            f"Expected 24966 or 24967 predictions, got {num_preds}"
        )

    csv_sha = _sha256_hex(csv_path)

    # ── Validate ZIP ────────────────────────────────────────────────
    zip_sha = _validate_zip(zip_path, csv_path)

    # ── Compute gap ─────────────────────────────────────────────────
    online_acc = args.online_accuracy
    gap = local_micro - online_acc

    # ── Build manifest ──────────────────────────────────────────────
    git_commit = _get_git_commit()
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest = {
        "experiment_id": args.experiment_id,
        "git_commit": git_commit,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": ckpt_epoch,
        "local_micro_accuracy": local_micro,
        "local_macro_accuracy": local_macro,
        "prediction_csv_path": str(csv_path),
        "prediction_csv_sha256": csv_sha,
        "submission_zip_path": str(zip_path),
        "submission_zip_sha256": zip_sha,
        "online_accuracy": online_acc,
        "local_online_gap": gap,
        "num_predictions": num_preds,
        "created_at_utc": created_at,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Manifest written to: {output_path}")
    print(f"  experiment:    {manifest['experiment_id']}")
    print(f"  ckpt epoch:    {manifest['checkpoint_epoch']}")
    print(f"  local micro:   {manifest['local_micro_accuracy']:.8f}")
    print(f"  local macro:   {manifest['local_macro_accuracy']:.8f}")
    print(f"  online:        {manifest['online_accuracy']:.6f}")
    print(f"  gap:           {manifest['local_online_gap']:+.6f}")
    print(f"  predictions:   {manifest['num_predictions']}")


if __name__ == "__main__":
    main()
