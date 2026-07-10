"""Test feature caching infrastructure."""
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch

from common.cache import (
    compute_quick_fingerprint,
    compute_full_fingerprint,
    CachedFeatureDataset,
)


def make_dummy_cache_dir(base_dir):
    """Create a minimal valid cache directory for testing."""
    cache_dir = Path(base_dir) / "cache" / "preliminary" / "clip_vit_b32_openai"
    cache_dir.mkdir(parents=True)

    # features.pt — 10 samples, 512-dim
    features = torch.randn(10, 512, dtype=torch.float32)
    features = features / features.norm(dim=1, keepdim=True)
    torch.save(features, cache_dir / "features.pt")

    # image_paths.json
    paths = [f"0000/img_{i:04d}.jpg" for i in range(5)] + \
            [f"0001/img_{i:04d}.jpg" for i in range(5)]
    with open(cache_dir / "image_paths.json", "w") as f:
        json.dump(paths, f)

    # labels.json
    labels = [0] * 5 + [1] * 5
    with open(cache_dir / "labels.json", "w") as f:
        json.dump(labels, f)

    # class_to_idx.json
    class_to_idx = {"0000": 0, "0001": 1}
    with open(cache_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f)

    # idx_to_class.json
    with open(cache_dir / "idx_to_class.json", "w") as f:
        json.dump({"0": "0000", "1": "0001"}, f)

    # manifest.json — valid hard fields for CachedFeatureDataset validation
    canon_str = json.dumps(
        {k: class_to_idx[k] for k in sorted(class_to_idx.keys())},
        sort_keys=True,
    )
    class_mapping_hash = hashlib.sha256(canon_str.encode()).hexdigest()
    manifest = {
        "backbone": "ViT-B/32",
        "pretrained_source": "openai",
        "feature_dim": 512,
        "normalized": True,
        "dtype": "float32",
        "preprocess": "clip_deterministic",
        "feature_encode_amp": False,
        "autocast_dtype": None,
        "class_mapping_hash": class_mapping_hash,
        "dataset_quick_fingerprint": "dummy_quick_fingerprint",
        "dataset_full_fingerprint": "dummy_full_fingerprint",
        "dataset_root": str(Path(base_dir).resolve()),
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return cache_dir, paths, labels, class_to_idx


def test_cached_dataset_rejects_missing_manifest():
    """CachedFeatureDataset should fail without manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="manifest"):
            CachedFeatureDataset(cache_dir, "fake.csv", "fake.json", tmpdir)


# Note: Full integration tests require actual cached features which
# need CLIP. These tests validate the validation logic.


def test_cached_dataset_hard_field_mismatch():
    """CachedFeatureDataset should raise ValueError on hard field mismatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir, _, _, _ = make_dummy_cache_dir(tmpdir)

        # Corrupt the backbone field in manifest
        manifest_path = cache_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        manifest["backbone"] = "ViT-L/14"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with pytest.raises(
            ValueError, match="Cache manifest field 'backbone' mismatch"
        ):
            CachedFeatureDataset(
                cache_dir, "dummy.csv", "dummy.json", tmpdir,
                verification="quick",
            )


def test_cached_dataset_successful_load():
    """CachedFeatureDataset should load successfully with valid cache and dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create dataset directory structure matching the cache paths
        dataset_root = tmp_path / "dataset"
        for cls_dir_name in ["0000", "0001"]:
            cls_dir = dataset_root / cls_dir_name
            cls_dir.mkdir(parents=True)
            for i in range(5):
                (cls_dir / f"img_{i:04d}.jpg").write_bytes(b"dummy_image_data")

        # Create cache
        cache_dir, paths, labels, class_to_idx = make_dummy_cache_dir(tmpdir)

        # Compute the actual quick fingerprint for the dataset root
        quick_fp = compute_quick_fingerprint(dataset_root)

        # Recompute class_mapping_hash to match what _validate_class_mapping_hash expects
        canon_str = json.dumps(
            {k: class_to_idx[k] for k in sorted(class_to_idx.keys())},
            sort_keys=True,
        )
        class_mapping_hash = hashlib.sha256(canon_str.encode()).hexdigest()

        # Update manifest with correct fingerprint and hash
        with open(cache_dir / "manifest.json", "r") as f:
            manifest = json.load(f)
        manifest["dataset_quick_fingerprint"] = quick_fp
        manifest["class_mapping_hash"] = class_mapping_hash
        with open(cache_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Create split CSV
        split_csv = tmp_path / "split.csv"
        df_data = []
        for i in range(5):
            df_data.append({"image_path": f"0000/img_{i:04d}.jpg", "label": 0})
        for i in range(5):
            df_data.append({"image_path": f"0001/img_{i:04d}.jpg", "label": 1})
        pd.DataFrame(df_data).to_csv(split_csv, index=False)

        # Load the dataset
        class_to_idx_path = cache_dir / "class_to_idx.json"
        dataset = CachedFeatureDataset(
            cache_dir, str(split_csv), str(class_to_idx_path), dataset_root,
            verification="quick",
        )

        # Verify
        assert len(dataset) == 10
        feat, label = dataset[0]
        assert feat.shape == (512,)
        assert label == 0
        feat, label = dataset[5]
        assert feat.shape == (512,)
        assert label == 1
