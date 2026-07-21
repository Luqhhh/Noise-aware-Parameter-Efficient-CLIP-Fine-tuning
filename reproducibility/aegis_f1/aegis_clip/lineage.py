"""Fail-closed parent-child split lineage audit for AEGIS checkpoint swaps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aegis_clip.features import canonical_sample_path
from aegis_clip.runtime import sha256_file


class LineageAuditError(ValueError):
    """Raised when the parent-child split lineage is violated."""


def _load_split(path: str | Path) -> dict[str, int]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise LineageAuditError(f"Split CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"image_path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise LineageAuditError(f"Split CSV missing columns: {sorted(missing)}")

    records: dict[str, int] = {}
    for row in frame.itertuples(index=False):
        identity = canonical_sample_path(str(row.image_path))
        label = int(row.label)
        if identity in records:
            raise LineageAuditError(f"Duplicate canonical sample: {identity}")
        records[identity] = label
    return records


def run_lineage_audit(
    config: dict[str, Any],
    *,
    child_train_csv: str,
    child_val_csv: str,
    checkpoint_path: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run a fail-closed split lineage audit.

    All critical mismatches raise ``LineageAuditError``.
    The audit artifact is always written before raising.
    """
    lineage = config.get("lineage", {})
    if not lineage.get("enabled", False):
        raise LineageAuditError("lineage.enabled must be true")

    parent_train_path = Path(lineage["parent_train_csv"])
    parent_val_path = Path(lineage["parent_val_csv"])
    child_train_path = Path(child_train_csv)
    child_val_path = Path(child_val_csv)
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise LineageAuditError(f"Parent checkpoint not found: {checkpoint}")

    parent_train = _load_split(parent_train_path)
    parent_val = _load_split(parent_val_path)
    child_train = _load_split(child_train_path)
    child_val = _load_split(child_val_path)

    parent_train_keys = set(parent_train)
    parent_val_keys = set(parent_val)
    child_train_keys = set(child_train)
    child_val_keys = set(child_val)

    label_mismatches = sorted(
        key
        for key in (set(parent_train) | set(parent_val))
        & (set(child_train) | set(child_val))
        if (
            parent_train.get(key, parent_val.get(key))
            != child_train.get(key, child_val.get(key))
        )
    )

    audit = {
        "protocol_valid": True,
        "parent_experiment_id": lineage["parent_experiment_id"],
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": sha256_file(checkpoint),
        "parent_train_csv": str(parent_train_path),
        "parent_train_csv_sha256": sha256_file(parent_train_path),
        "parent_val_csv": str(parent_val_path),
        "parent_val_csv_sha256": sha256_file(parent_val_path),
        "child_train_csv": str(child_train_path),
        "child_train_csv_sha256": sha256_file(child_train_path),
        "child_val_csv": str(child_val_path),
        "child_val_csv_sha256": sha256_file(child_val_path),
        "parent_train_count": len(parent_train),
        "parent_val_count": len(parent_val),
        "child_train_count": len(child_train),
        "child_val_count": len(child_val),
        "child_val_in_parent_train": len(child_val_keys & parent_train_keys),
        "child_train_in_parent_val": len(child_train_keys & parent_val_keys),
        "child_train_equals_parent_train": child_train_keys == parent_train_keys,
        "child_val_equals_parent_val": child_val_keys == parent_val_keys,
        "label_mismatches": len(label_mismatches),
        "label_mismatch_examples": label_mismatches[:10],
    }

    errors: list[str] = []
    if audit["child_val_in_parent_train"]:
        errors.append(
            f"{audit['child_val_in_parent_train']} child validation samples "
            f"were seen by parent training"
        )
    if audit["child_train_in_parent_val"]:
        errors.append(
            f"{audit['child_train_in_parent_val']} child training samples "
            f"overlap parent validation"
        )
    if lineage.get("require_same_train", True) and not audit["child_train_equals_parent_train"]:
        errors.append("child train split differs from parent train split")
    if lineage.get("require_same_val", True) and not audit["child_val_equals_parent_val"]:
        errors.append("child val split differs from parent val split")
    if label_mismatches:
        errors.append(
            f"{len(label_mismatches)} labels differ for shared canonical samples"
        )

    audit["errors"] = errors
    audit["protocol_valid"] = not errors
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if errors:
        raise LineageAuditError("Lineage audit failed: " + "; ".join(errors))
    return audit
