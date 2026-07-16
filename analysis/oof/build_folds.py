"""Build deterministic duplicate-group-aware OOF folds from the strict split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_COLUMNS = {"sample_id", "image_path", "label", "sha256"}


def assign_group_stratified_folds(
    samples: pd.DataFrame,
    n_splits: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign one fold per sample while keeping each SHA-256 group intact."""
    missing = REQUIRED_COLUMNS - set(samples.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if samples.empty:
        raise ValueError("Cannot build OOF folds from an empty dataset")
    if not samples["sample_id"].is_unique:
        raise ValueError("sample_id values must be unique")

    assignments = samples.sort_values("image_path").reset_index(drop=True).copy()
    assignments["fold"] = -1
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    placeholder = np.zeros((len(assignments), 1), dtype=np.uint8)
    for fold, (_, holdout_indices) in enumerate(
        splitter.split(
            placeholder,
            assignments["label"].astype(int).to_numpy(),
            groups=assignments["sha256"].astype(str).to_numpy(),
        )
    ):
        assignments.loc[holdout_indices, "fold"] = fold

    if (assignments["fold"] < 0).any():
        raise RuntimeError("At least one sample was not assigned to a fold")
    if assignments.groupby("sha256")["fold"].nunique().max() != 1:
        raise RuntimeError("A duplicate group was split across OOF folds")
    assignments["fold"] = assignments["fold"].astype(int)
    return assignments


def _canonical_image_key(path: str) -> str:
    parts = str(path).replace("\\", "/").lstrip("./").split("/")
    if parts and parts[0] in {"train", "train_dedup"}:
        parts = parts[1:]
    return "/".join(parts)


def audit_folds(
    assignments: pd.DataFrame,
    original_val_paths: Iterable[str],
    n_splits: int = 3,
) -> dict:
    """Validate OOF coverage, duplicate isolation, and strict-val exclusion."""
    missing = REQUIRED_COLUMNS.union({"fold"}) - set(assignments.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    invalid_folds = sorted(
        set(assignments["fold"].astype(int)) - set(range(n_splits))
    )
    duplicate_leaks = assignments.groupby("sha256")["fold"].nunique()
    duplicate_leak_count = int((duplicate_leaks > 1).sum())

    strict_keys = {_canonical_image_key(path) for path in assignments["image_path"]}
    val_keys = {_canonical_image_key(path) for path in original_val_paths}
    validation_overlap_count = len(strict_keys & val_keys)

    if invalid_folds:
        raise ValueError(f"Invalid fold ids: {invalid_folds}")
    if not assignments["sample_id"].is_unique:
        raise ValueError("OOF assignments contain duplicate sample_id values")
    if duplicate_leak_count:
        raise ValueError(
            f"Detected {duplicate_leak_count} duplicate groups split across folds"
        )
    if validation_overlap_count:
        raise ValueError(
            f"Detected {validation_overlap_count} original validation samples in OOF"
        )

    fold_counts = {
        str(fold): int((assignments["fold"] == fold).sum())
        for fold in range(n_splits)
    }
    per_class_fold_counts = {
        str(int(label)): {
            str(fold): int(((assignments["label"] == label) & (assignments["fold"] == fold)).sum())
            for fold in range(n_splits)
        }
        for label in sorted(assignments["label"].unique())
    }
    return {
        "sample_count": int(len(assignments)),
        "unique_sample_id_count": int(assignments["sample_id"].nunique()),
        "unique_sha256_group_count": int(assignments["sha256"].nunique()),
        "n_splits": int(n_splits),
        "fold_counts": fold_counts,
        "per_class_fold_counts": per_class_fold_counts,
        "duplicate_group_fold_leakage_count": duplicate_leak_count,
        "original_validation_overlap_count": validation_overlap_count,
        "all_samples_assigned_once": True,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-train", default="outputs/d3_strict/seed42/train.csv")
    parser.add_argument("--original-val", default="outputs/d3_strict/seed42/val.csv")
    parser.add_argument("--output-dir", default="outputs/phase3/oof")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-workers", type=int, default=16)
    args = parser.parse_args()

    strict_path = Path(args.strict_train)
    original_val_path = Path(args.original_val)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strict_df = pd.read_csv(
        strict_path,
        dtype={"image_path": str, "class_name": str, "label": int},
    ).sort_values("image_path").reset_index(drop=True)
    original_val_df = pd.read_csv(original_val_path, dtype={"image_path": str})
    strict_df["sample_id"] = strict_df["image_path"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )

    image_files = [Path(path) for path in strict_df["image_path"]]
    missing_files = [str(path) for path in image_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} strict-train images are missing; first: {missing_files[0]}"
        )
    with ThreadPoolExecutor(max_workers=max(1, args.hash_workers)) as executor:
        strict_df["sha256"] = list(executor.map(_sha256_file, image_files))

    assignments = assign_group_stratified_folds(
        strict_df,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    audit = audit_folds(
        assignments,
        original_val_paths=original_val_df["image_path"].tolist(),
        n_splits=args.n_splits,
    )

    assignment_path = output_dir / "fold_assignments.csv"
    assignments.to_csv(assignment_path, index=False)
    (output_dir / "fold_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    split_columns = ["image_path", "class_name", "label"]
    mapping_dir = strict_path.parent
    for fold in range(args.n_splits):
        fold_dir = output_dir / f"fold_{fold}" / "splits"
        fold_dir.mkdir(parents=True, exist_ok=True)
        assignments.loc[assignments["fold"] != fold, split_columns].to_csv(
            fold_dir / "train.csv", index=False
        )
        assignments.loc[assignments["fold"] == fold, split_columns].to_csv(
            fold_dir / "val.csv", index=False
        )
        for mapping_name in ("class_to_idx.json", "idx_to_class.json"):
            shutil.copy2(mapping_dir / mapping_name, fold_dir / mapping_name)

    manifest = {
        "protocol": "duplicate-group-aware-stratified-3-fold-oof",
        "seed": args.seed,
        "n_splits": args.n_splits,
        "strict_train_csv": str(strict_path),
        "strict_train_csv_sha256": _file_sha256(strict_path),
        "original_val_csv": str(original_val_path),
        "original_val_csv_sha256": _file_sha256(original_val_path),
        "fold_assignments_sha256": _file_sha256(assignment_path),
        "sample_count": len(assignments),
        "original_validation_used_for_training": False,
        "original_validation_used_for_epoch_selection": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
