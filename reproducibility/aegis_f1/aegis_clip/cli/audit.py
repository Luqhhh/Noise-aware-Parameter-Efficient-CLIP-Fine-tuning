"""Read-only audit of configuration, data lineage, cache, trust, and checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from aegis_clip.config import load_config
from aegis_clip.data import (
    IMAGE_EXTENSIONS,
    TrustBundle,
    load_class_mapping,
    resolve_image_path,
)
from aegis_clip.features import FrozenFeatureStore, canonical_sample_path
from aegis_clip.runtime import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    train = pd.read_csv(data["train_csv"])
    val = pd.read_csv(data["val_csv"])
    train_paths = [canonical_sample_path(path) for path in train["image_path"].astype(str)]
    val_paths = [canonical_sample_path(path) for path in val["image_path"].astype(str)]
    overlap = set(train_paths) & set(val_paths)
    validation_overlap = bool(data.get("validation_overlap_with_training", False))
    if validation_overlap:
        missing_from_full_train = set(val_paths) - set(train_paths)
        if missing_from_full_train:
            raise ValueError(
                "Overlapping diagnostic validation must be a subset of full train: "
                f"missing={len(missing_from_full_train)}"
            )
    elif overlap:
        raise ValueError(f"Train/validation path overlap: {len(overlap)}")
    official_train_samples = int(data["expected_official_train_samples"])
    configured_official_samples = (
        len(train_paths) if validation_overlap else len(train_paths) + len(val_paths)
    )
    if configured_official_samples != official_train_samples:
        raise ValueError(
            "Configured split does not cover the complete official training set: "
            f"{configured_official_samples} != {official_train_samples}"
        )
    coverage_paths = train_paths if validation_overlap else train_paths + val_paths
    groups_path_value = config.get("trust", {}).get("groups_path")
    groups_path = (
        Path(groups_path_value)
        if groups_path_value
        else Path(data["train_csv"]).parent / "content_groups.json"
    )
    group_summary = None
    if groups_path.is_file():
        group_mapping = json.loads(groups_path.read_text(encoding="utf-8"))
        missing_groups = [
            path for path in coverage_paths if path not in group_mapping
        ]
        if missing_groups:
            raise ValueError(
                f"Content grouping misses {len(missing_groups)} split samples; "
                f"first={missing_groups[0]}"
            )
        train_groups = {group_mapping[path] for path in train_paths}
        val_groups = {group_mapping[path] for path in val_paths}
        group_overlap = train_groups & val_groups
        if group_overlap and not validation_overlap:
            raise ValueError(
                f"Train/validation content-group overlap: {len(group_overlap)}"
            )
        group_summary = {
            "path": str(groups_path),
            "unique_groups": len(set(group_mapping.values())),
            "train_val_overlap": len(group_overlap),
        }
    train_root = Path(data["train_root"]).resolve()
    test_root = Path(data["test_root"]).resolve()
    if train_root == test_root:
        raise ValueError("Training and test roots must be distinct")
    missing_images = [
        path
        for path in coverage_paths
        if not resolve_image_path(train_root, path).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"Official training split misses {len(missing_images)} files; "
            f"first={missing_images[0]}"
        )
    test_images = sorted(
        path
        for path in test_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    expected_test_samples = int(data["expected_test_samples"])
    if len(test_images) != expected_test_samples:
        raise ValueError(
            f"Official test image count mismatch: {len(test_images)} "
            f"!= {expected_test_samples}"
        )
    class_to_idx, _ = load_class_mapping(data["class_mapping"])
    if len(class_to_idx) != int(config["model"]["num_classes"]):
        raise ValueError("Class mapping size does not match model.num_classes")
    features = FrozenFeatureStore(
        config["features"]["tensor_path"],
        config["features"]["paths_path"],
        config["features"].get("manifest_path"),
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    features.verify_coverage(coverage_paths)
    trust_summary = None
    if config.get("trust", {}).get("enabled", False):
        trust = TrustBundle(config["trust"]["bundle_path"])
        trust.verify_coverage(coverage_paths)
        trust_summary = {
            "samples": len(trust),
            "mean_clean_probability": float(trust.clean_probability.mean()),
            "corrected_samples": int((trust.correction_alpha > 0).sum()),
        }
    checkpoint_summary = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint.get("config", {})
        checkpoint_project = checkpoint_config.get("project", {})
        if checkpoint_project.get("experiment_id") != config["project"]["experiment_id"]:
            raise ValueError("Checkpoint experiment_id does not match audit config")
        effective_spec = checkpoint.get("effective_model_spec", {})
        if effective_spec.get("backbone") != "ViT-B/32":
            raise ValueError("Checkpoint is not an OpenAI CLIP ViT-B/32 model")
        if int(effective_spec.get("num_classes", -1)) != len(class_to_idx):
            raise ValueError("Checkpoint class count does not match current stage")
        checkpoint_summary = {
            "sha256": sha256_file(args.checkpoint),
            "epoch": checkpoint.get("epoch"),
            "format_version": checkpoint.get("format_version"),
            "effective_model_spec": checkpoint.get("effective_model_spec"),
        }
    result = {
        "experiment_id": config["project"]["experiment_id"],
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "validation_overlap_with_training": validation_overlap,
        "classes": len(class_to_idx),
        "stage": config["project"]["stage"],
        "external_data": config["data"]["external_data"],
        "test_usage": config["data"]["test_usage"],
        "test_samples": len(test_images),
        "feature_cache_samples": len(features),
        "train_csv_sha256": sha256_file(data["train_csv"]),
        "val_csv_sha256": sha256_file(data["val_csv"]),
        "content_groups": group_summary,
        "trust": trust_summary,
        "checkpoint": checkpoint_summary,
        "status": "passed",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
