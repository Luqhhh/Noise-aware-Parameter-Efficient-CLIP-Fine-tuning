"""
Parent-child split lineage audit.

Ensures experiments initialized from a parent checkpoint do not leak
validation data from the parent's training stage. Must be called before
any training occurs in the child experiment.

Rules enforced:
  1. child_val ∩ parent_train = ∅ (no leakage)
  2. child_val = parent_val (same comparison basis)
  3. Hard exit (SystemExit) on any violation

Usage:
    from common.split_audit import run_split_audit

    run_split_audit(
        parent_experiment_id="ref",
        parent_checkpoint_path="outputs/ref/seed42/checkpoints/best.pt",
        parent_train_csv=Path("outputs/master_splits/seed42/train.csv"),
        parent_val_csv=Path("outputs/master_splits/seed42/val.csv"),
        child_train_csv=Path("outputs/master_splits/seed42/train.csv"),
        child_val_csv=Path("outputs/master_splits/seed42/val.csv"),
        output_dir=Path("outputs/ft_lnpost/seed42"),
    )
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class SplitAuditError(ValueError):
    """Fatal split integrity violation — training must not proceed."""


def load_split_csv(csv_path: Path) -> set:
    """Load image paths from a split CSV.

    Returns:
        Set of ``image_path`` values.

    Raises:
        SplitAuditError: If the CSV is missing or lacks the expected column.
    """
    if not csv_path.exists():
        raise SplitAuditError(f"Split CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        raise SplitAuditError(
            f"No 'image_path' column in {csv_path}. "
            f"Columns: {list(df.columns)}"
        )
    return set(df["image_path"])


def run_split_audit(
    parent_experiment_id: str,
    parent_checkpoint_path: str,
    parent_train_csv: Path,
    parent_val_csv: Path,
    child_train_csv: Path,
    child_val_csv: Path,
    output_dir: Path,
) -> dict:
    """Run parent-child split lineage audit.

    Args:
        parent_experiment_id: Human-readable identifier for the parent.
        parent_checkpoint_path: Path to the checkpoint used for ``--init-checkpoint``.
        parent_train_csv: Path to parent's training split CSV.
        parent_val_csv: Path to parent's validation split CSV.
        child_train_csv: Path to child's training split CSV.
        child_val_csv: Path to child's validation split CSV.
        output_dir: Directory where ``split_lineage_audit.json`` will be written.

    Returns:
        Audit result dictionary.

    Raises:
        SplitAuditError: If **any** integrity rule is violated.
    """
    parent_train = load_split_csv(parent_train_csv)
    parent_val = load_split_csv(parent_val_csv)
    child_train = load_split_csv(child_train_csv)
    child_val = load_split_csv(child_val_csv)

    # ── Rule 1: no leakage ──────────────────────────────────────────
    child_val_in_parent_train = child_val & parent_train

    # ── Rule 2: same validation set ─────────────────────────────────
    child_val_in_parent_val = child_val & parent_val
    child_val_equals_parent_val = child_val == parent_val

    # ── Build audit record ──────────────────────────────────────────
    audit = {
        "parent_experiment": parent_experiment_id,
        "parent_checkpoint": parent_checkpoint_path,
        "parent_train_count": len(parent_train),
        "parent_val_count": len(parent_val),
        "child_train_count": len(child_train),
        "child_val_count": len(child_val),
        "child_val_in_parent_train": len(child_val_in_parent_train),
        "child_val_in_parent_val": len(child_val_in_parent_val),
        "child_val_equals_parent_val": child_val_equals_parent_val,
        "protocol_valid": True,
    }

    # ── Rule 1 enforcement ──────────────────────────────────────────
    if child_val_in_parent_train:
        audit["protocol_valid"] = False
        examples = sorted(child_val_in_parent_train)[:5]
        msg = (
            f"\n{'=' * 60}\n"
            f"VALIDATION LEAK DETECTED\n"
            f"{'=' * 60}\n"
            f"Parent experiment: {parent_experiment_id}\n"
            f"Parent checkpoint: {parent_checkpoint_path}\n"
            f"\n"
            f"{len(child_val_in_parent_train)} images in the child validation set\n"
            f"were already seen by the parent during training.\n"
            f"\n"
            f"This means the child's validation accuracy is INFLATED.\n"
            f"Training cannot proceed.\n"
            f"\n"
            f"Leaked examples (first 5):\n"
        )
        for p in examples:
            msg += f"  - {p}\n"
        msg += (
            f"\n"
            f"FIX: ensure child_val ∩ parent_train = ∅ and re-run.\n"
            f"{'=' * 60}\n"
        )
        logger.error(msg)
        raise SplitAuditError(msg)

    # ── Rule 2 enforcement ──────────────────────────────────────────
    if not child_val_equals_parent_val:
        audit["protocol_valid"] = False
        missing = parent_val - child_val
        extra = child_val - parent_val
        msg = (
            f"\n{'=' * 60}\n"
            f"VALIDATION MISMATCH\n"
            f"{'=' * 60}\n"
            f"Parent experiment: {parent_experiment_id}\n"
            f"Parent checkpoint: {parent_checkpoint_path}\n"
            f"\n"
            f"The child's validation set differs from the parent's.\n"
            f"  In parent but not child: {len(missing)} images\n"
            f"  In child but not parent: {len(extra)} images\n"
            f"\n"
            f"This means the two experiments are evaluated on DIFFERENT\n"
            f"data — their accuracies cannot be directly compared.\n"
            f"\n"
            f"FIX: ensure child_val == parent_val and re-run.\n"
            f"{'=' * 60}\n"
        )
        logger.error(msg)
        raise SplitAuditError(msg)

    # ── Success ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "split_lineage_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    logger.info("Split lineage audit PASSED.")
    logger.info("  child_val ∩ parent_train = ∅")
    logger.info("  child_val == parent_val")
    logger.info("Audit record: %s", audit_path)

    return audit
