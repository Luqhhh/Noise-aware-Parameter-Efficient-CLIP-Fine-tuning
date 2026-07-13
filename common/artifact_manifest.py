"""Unified artifact manifest builder.

Produces ``artifact_manifest.json`` with SHA-256 hashes of all key artifacts
for a training experiment.  Written to the experiment output directory.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _sha256(file_path: str) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def build_artifact_manifest(
    experiment_id: str,
    parent_experiment: Optional[str],
    config: Dict[str, Any],
    checkpoint_path: str,
    train_csv: str,
    val_csv: str,
    pred_csv: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the artifact manifest dict.

    Args:
        experiment_id: Unique experiment identifier.
        parent_experiment: Parent experiment ID (or None for root).
        config: Full resolved config dict (written as config_snapshot).
        checkpoint_path: Path to best.pt (or last.pt).
        train_csv: Path to train split CSV.
        val_csv: Path to val split CSV.
        pred_csv: Path to pred_raw.csv (optional, for inference runs).
        extra: Additional fields to merge.

    Returns:
        Manifest dict ready for JSON serialisation.
    """
    checkpoint_path = Path(checkpoint_path)
    train_csv = Path(train_csv)
    val_csv = Path(val_csv)

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "parent_experiment": parent_experiment,
        "commit_sha": _git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Config hash (hash the resolved config dict)
    config_json = json.dumps(config, sort_keys=True, default=str)
    manifest["config_sha256"] = hashlib.sha256(
        config_json.encode()
    ).hexdigest()

    # Checkpoint
    if checkpoint_path.exists():
        manifest["checkpoint_path"] = str(checkpoint_path)
        manifest["checkpoint_sha256"] = _sha256(str(checkpoint_path))

    # Split CSVs
    if train_csv.exists():
        manifest["train_csv_path"] = str(train_csv)
        manifest["train_csv_sha256"] = _sha256(str(train_csv))
    if val_csv.exists():
        manifest["val_csv_path"] = str(val_csv)
        manifest["val_csv_sha256"] = _sha256(str(val_csv))

    # Predictions
    if pred_csv:
        pred_path = Path(pred_csv)
        if pred_path.exists():
            manifest["prediction_csv_path"] = str(pred_path)
            manifest["prediction_csv_sha256"] = _sha256(str(pred_path))

    if extra:
        manifest.update(extra)

    return manifest


def write_artifact_manifest(
    manifest: Dict[str, Any],
    output_dir: str,
) -> str:
    """Write artifact_manifest.json to the output directory.

    Args:
        manifest: Dict from ``build_artifact_manifest()``.
        output_dir: Directory to write to.

    Returns:
        Path to the written file.
    """
    out = Path(output_dir) / "artifact_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return str(out)
