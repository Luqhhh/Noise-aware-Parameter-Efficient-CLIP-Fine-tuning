"""Rebuild the exact A2-kept training split from read-only team artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from aegis_clip.features import canonical_sample_path
from aegis_clip.runtime import atomic_json_dump, sha256_file


def prepare_a2_kept_split(
    split_csv: str | Path,
    purification_csv: str | Path,
    validation_csv: str | Path,
    output_dir: str | Path,
    *,
    expected_classes: int = 500,
) -> dict[str, Any]:
    """Write only samples A2 actually trained on and audit disjoint validation."""
    split_path = Path(split_csv)
    purification_path = Path(purification_csv)
    validation_path = Path(validation_csv)
    split = pd.read_csv(split_path)
    purification = pd.read_csv(purification_path)
    validation = pd.read_csv(validation_path)
    required_split = {"image_path", "label"}
    required_purification = {
        "image_path",
        "original_label",
        "training_label",
        "sample_weight",
        "training_role",
    }
    if missing := required_split - set(split):
        raise ValueError(f"A2 split misses columns: {sorted(missing)}")
    if missing := required_purification - set(purification):
        raise ValueError(f"Purification manifest misses columns: {sorted(missing)}")
    if missing := required_split - set(validation):
        raise ValueError(f"Validation split misses columns: {sorted(missing)}")

    split = split.copy()
    purification = purification.copy()
    validation = validation.copy()
    split["canonical_path"] = split["image_path"].map(canonical_sample_path)
    purification["canonical_path"] = purification["image_path"].map(
        canonical_sample_path
    )
    validation["canonical_path"] = validation["image_path"].map(
        canonical_sample_path
    )
    for name, frame in (("split", split), ("purification", purification)):
        if frame["canonical_path"].duplicated().any():
            raise ValueError(f"{name} contains duplicate canonical paths")
    if set(split["canonical_path"]) != set(purification["canonical_path"]):
        raise ValueError("A2 split and purification manifest cover different paths")

    aligned = split.merge(
        purification[
            [
                "canonical_path",
                "original_label",
                "training_label",
                "sample_weight",
                "training_role",
            ]
        ],
        on="canonical_path",
        how="left",
        validate="one_to_one",
    )
    if not (
        aligned["label"].astype(int)
        == aligned["original_label"].astype(int)
    ).all():
        raise ValueError("A2 split labels differ from purification original labels")
    if not (
        aligned["original_label"].astype(int)
        == aligned["training_label"].astype(int)
    ).all():
        raise ValueError("A2 purification unexpectedly contains hard relabels")
    if not set(aligned["sample_weight"].astype(float).unique()) <= {0.0, 1.0}:
        raise ValueError("A2 purification weights must be binary")

    kept_mask = aligned["sample_weight"].astype(float) > 0.0
    kept = aligned.loc[kept_mask, list(split.columns)].drop(
        columns=["canonical_path"]
    )
    kept_keys = set(aligned.loc[kept_mask, "canonical_path"])
    val_keys = set(validation["canonical_path"])
    overlap = kept_keys & val_keys
    if overlap:
        raise ValueError(f"A2-kept train overlaps validation by {len(overlap)} paths")
    counts = kept["label"].astype(int).value_counts().reindex(
        range(int(expected_classes)), fill_value=0
    )
    if int((counts == 0).sum()) > 0:
        raise ValueError("A2-kept split is missing at least one class")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    kept_path = output / "a2_kept_train.csv"
    audit_path = output / "audit.json"
    kept.to_csv(kept_path, index=False)
    audit: dict[str, Any] = {
        "source_count": int(len(split)),
        "kept_count": int(len(kept)),
        "rejected_count": int((~kept_mask).sum()),
        "class_count": int((counts > 0).sum()),
        "minimum_class_count": int(counts.min()),
        "maximum_class_count": int(counts.max()),
        "validation_count": int(len(validation)),
        "train_validation_overlap": 0,
        "split_sha256": sha256_file(split_path),
        "purification_sha256": sha256_file(purification_path),
        "validation_sha256": sha256_file(validation_path),
        "kept_train_sha256": sha256_file(kept_path),
    }
    atomic_json_dump(audit, audit_path)
    return audit
