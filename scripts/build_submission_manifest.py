#!/usr/bin/env python3
"""Build a submission manifest for a platform submission.

The manifest captures every artifact hash, local metrics, and the
online score so the submission is fully reproducible and auditable.

Usage:
    python scripts/build_submission_manifest.py \
      --experiment-id ref \
      --checkpoint outputs/baselines/ref/seed42/checkpoints/best.pt \
      --eval-results outputs/baselines/ref/seed42/checkpoints/reeval_best.json \
      --prediction-csv outputs/baselines/ref/seed42/submissions/pred_results.csv \
      --submission-zip outputs/baselines/ref/seed42/submissions/submission.zip \
      --online-accuracy 0.573397 \
      --output outputs/baselines/ref/seed42/submissions/submission_manifest.json
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
    """Every label must match ^\\d{4}$.

    Note: pred_results.csv has NO header row — it is the submission
    format ``image_name.jpg, 0001`` directly.
    """
    pattern = re.compile(r"^\d{4}$")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for lineno, row in enumerate(reader, start=1):
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
    """Count data rows in pred_results.csv (no header row)."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return sum(1 for _ in reader)


def _validate_zip(zip_path: Path, csv_path: Path) -> tuple:
    """Validate ZIP contents and return (internal_csv_sha256, zip_sha256).

    Returns:
        (internal_csv_sha256, actual_zip_file_sha256)
        — two *different* hashes.
    """
    external_csv_sha = _sha256_hex(csv_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if names != ["pred_results.csv"]:
            raise ValueError(
                f"ZIP must contain exactly ['pred_results.csv']; "
                f"found: {names}"
            )

        with zf.open("pred_results.csv") as f:
            h = hashlib.sha256()
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
            internal_csv_sha = h.hexdigest()

    if internal_csv_sha != external_csv_sha:
        raise ValueError(
            f"ZIP-internal CSV hash ({internal_csv_sha[:16]}...) does not "
            f"match external CSV hash ({external_csv_sha[:16]}...)"
        )

    # The actual ZIP file hash — NOT the internal CSV hash
    actual_zip_sha = _sha256_hex(zip_path)

    return internal_csv_sha, actual_zip_sha


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a submission manifest."
    )
    parser.add_argument(
        "--experiment-id", required=True, help="e.g. ref"
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

    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_epoch = ckpt.get("epoch")
    ckpt_best_val_acc = float(ckpt.get("best_val_acc", -1.0))

    # ── Extract seeds from checkpoint config ────────────────────────
    ckpt_config = ckpt.get("config", {})
    ckpt_data = ckpt_config.get("data", {})
    split_seed = ckpt_data.get("split_seed")
    train_seed = ckpt_data.get("train_seed")
    split_dir = ckpt_data.get("split_dir", "")

    # ── Consistency: checkpoint best_val_acc == reeval micro ─────────
    if abs(ckpt_best_val_acc - local_micro) > 1e-8:
        raise ValueError(
            "Checkpoint best_val_acc does not match reeval micro_accuracy: "
            f"ckpt_best_val_acc={ckpt_best_val_acc:.10f}, "
            f"reeval_micro={local_micro:.10f}"
        )

    # ── Validate prediction CSV ─────────────────────────────────────
    # pred_results.csv has NO header — it is the submission format directly.
    _validate_labels(csv_path)
    num_preds = _count_predictions(csv_path)

    EXPECTED = 24967
    if num_preds != EXPECTED:
        raise ValueError(
            f"Expected exactly {EXPECTED} predictions, got {num_preds}. "
            f"Every test-set image must have a prediction. "
            f"If an image fails to load, use a deterministic fallback "
            f"(e.g. zero tensor → most frequent class) rather than "
            f"dropping the row."
        )

    csv_sha = _sha256_hex(csv_path)

    # ── Validate ZIP ────────────────────────────────────────────────
    internal_csv_sha, zip_sha = _validate_zip(zip_path, csv_path)

    # ── Compute gap ─────────────────────────────────────────────────
    online_acc = args.online_accuracy
    gap = local_micro - online_acc

    # ── Val CSV hash from reeval results ─────────────────────────────
    val_csv_sha256 = eval_data.get("val_csv_sha256", "")

    # ── Build manifest ──────────────────────────────────────────────
    git_commit = _get_git_commit()
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest = {
        "experiment_id": args.experiment_id,
        "git_commit": git_commit,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": ckpt_epoch,
        "split_seed": split_seed,
        "train_seed": train_seed,
        "split_dir": split_dir,
        "val_csv_sha256": val_csv_sha256,
        "local_micro_accuracy": local_micro,
        "local_macro_accuracy": local_macro,
        "prediction_csv_path": str(csv_path),
        "prediction_csv_sha256": csv_sha,
        "zip_internal_csv_sha256": internal_csv_sha,
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
    print(f"  experiment:       {manifest['experiment_id']}")
    print(f"  ckpt epoch:       {manifest['checkpoint_epoch']}")
    print(f"  split_seed:       {manifest['split_seed']}")
    print(f"  train_seed:       {manifest['train_seed']}")
    print(f"  local micro:      {manifest['local_micro_accuracy']:.8f}")
    print(f"  local macro:      {manifest['local_macro_accuracy']:.8f}")
    print(f"  online:           {manifest['online_accuracy']:.6f}")
    print(f"  gap:              {manifest['local_online_gap']:+.6f}")
    print(f"  predictions:      {manifest['num_predictions']}")
    print(f"  zip sha256:       {zip_sha}")


if __name__ == "__main__":
    main()
