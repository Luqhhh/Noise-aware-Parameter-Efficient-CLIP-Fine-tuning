"""Test feature caching infrastructure."""
import json
import tempfile
from pathlib import Path

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
