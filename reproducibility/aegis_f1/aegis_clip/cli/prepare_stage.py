"""Create a reproducible SHA-256-grouped split from current-stage official data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from aegis_clip.data import IMAGE_EXTENSIONS
from aegis_clip.features import canonical_sample_path
from aegis_clip.runtime import atomic_json_dump, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output-root", default="artifacts/stages")
    parser.add_argument(
        "--stage", choices=["preliminary", "repechage", "semifinal"], required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--expected-classes", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--groups-path")
    parser.add_argument("--hash-workers", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output_root) / args.stage / f"seed{args.seed}"
    result = prepare_stage(
        train_root=args.train_root,
        output_dir=output,
        stage=args.stage,
        seed=args.seed,
        val_ratio=args.val_ratio,
        expected_classes=args.expected_classes,
        expected_samples=args.expected_samples,
        groups_path=args.groups_path,
        hash_workers=args.hash_workers,
    )
    print(json.dumps(result, indent=2))


def prepare_stage(
    *,
    train_root: str | Path,
    output_dir: str | Path,
    stage: str,
    seed: int,
    val_ratio: float,
    expected_classes: int,
    expected_samples: int,
    groups_path: str | Path | None = None,
    hash_workers: int = 8,
) -> dict:
    root = Path(train_root).resolve()
    destination = Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Official training root does not exist: {root}")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0,1)")
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if len(class_dirs) != expected_classes:
        raise ValueError(
            f"Found {len(class_dirs)} class directories, expected {expected_classes}"
        )
    class_to_idx = {directory.name: index for index, directory in enumerate(class_dirs)}
    records = []
    absolute_paths = []
    canonical_paths = []
    for directory in class_dirs:
        label = class_to_idx[directory.name]
        images = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"Class directory contains no images: {directory}")
        for path in images:
            canonical = f"{directory.name}/{path.name}"
            records.append(
                {
                    "image_path": f"{root.name}/{canonical}",
                    "class_name": directory.name,
                    "label": label,
                }
            )
            canonical_paths.append(canonical)
            absolute_paths.append(path)
    if len(records) != expected_samples:
        raise ValueError(
            f"Found {len(records)} official training images, expected {expected_samples}"
        )

    if groups_path:
        group_mapping = json.loads(Path(groups_path).read_text(encoding="utf-8"))
        missing = [path for path in canonical_paths if path not in group_mapping]
        if missing:
            raise ValueError(
                f"Reused content groups miss {len(missing)} images; first={missing[0]}"
            )
        hashes = [str(group_mapping[path]) for path in canonical_paths]
    else:
        with ThreadPoolExecutor(max_workers=hash_workers) as executor:
            hashes = list(executor.map(sha256_file, absolute_paths))
        group_mapping = dict(zip(canonical_paths, hashes))

    grouped: dict[str, list[int]] = {}
    for index, digest in enumerate(hashes):
        grouped.setdefault(digest, []).append(index)
    group_records = []
    for digest, indices in grouped.items():
        labels = [int(records[index]["label"]) for index in indices]
        group_records.append(
            {
                "sha256": digest,
                "majority_label": Counter(labels).most_common(1)[0][0],
            }
        )
    group_frame = pd.DataFrame(group_records)
    train_groups, val_groups = train_test_split(
        group_frame,
        test_size=val_ratio,
        random_state=seed,
        stratify=group_frame["majority_label"],
    )
    train_hashes = set(train_groups["sha256"])
    val_hashes = set(val_groups["sha256"])
    if train_hashes & val_hashes:
        raise RuntimeError("A content group was assigned to both train and validation")

    frame = pd.DataFrame(records)
    train_frame = frame[[digest in train_hashes for digest in hashes]].copy()
    val_frame = frame[[digest in val_hashes for digest in hashes]].copy()
    train_frame = train_frame.sort_values("image_path").reset_index(drop=True)
    val_frame = val_frame.sort_values("image_path").reset_index(drop=True)
    if len(train_frame) + len(val_frame) != expected_samples:
        raise RuntimeError("Prepared split does not cover every official image")

    destination.mkdir(parents=True, exist_ok=True)
    train_csv = destination / "train.csv"
    val_csv = destination / "val.csv"
    _atomic_csv(train_frame, train_csv)
    _atomic_csv(val_frame, val_csv)
    atomic_json_dump(class_to_idx, destination / "class_to_idx.json")
    atomic_json_dump(
        {str(index): name for name, index in class_to_idx.items()},
        destination / "idx_to_class.json",
    )
    atomic_json_dump(group_mapping, destination / "content_groups.json")
    manifest = {
        "format_version": 1,
        "stage": stage,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "source_root": str(root),
        "source_file_count": expected_samples,
        "num_classes": expected_classes,
        "train_count": len(train_frame),
        "val_count": len(val_frame),
        "unique_content_groups": len(grouped),
        "duplicate_samples": expected_samples - len(grouped),
        "duplicate_grouping_enabled": True,
        "external_data": False,
        "test_data_used": False,
        "train_csv_sha256": sha256_file(train_csv),
        "val_csv_sha256": sha256_file(val_csv),
        "content_groups_sha256": sha256_file(destination / "content_groups.json"),
    }
    atomic_json_dump(manifest, destination / "split_manifest.json")
    return manifest


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(
            temporary,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
