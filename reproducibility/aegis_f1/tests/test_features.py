import json

import pytest
import torch

from aegis_clip.features import FrozenFeatureStore, canonical_sample_path
from aegis_clip.cli.cache_features import apply_feature_augmentation
from aegis_clip.runtime import sha256_lines


def test_canonical_paths_are_machine_independent() -> None:
    assert canonical_sample_path("/mnt/data/train/0001/a.jpg") == "0001/a.jpg"
    assert canonical_sample_path("train\\0001\\a.jpg") == "0001/a.jpg"


def test_feature_store_validates_and_normalizes(tmp_path) -> None:
    tensor = tmp_path / "features.pt"
    paths = tmp_path / "paths.json"
    manifest = tmp_path / "manifest.json"
    torch.save(torch.tensor([[3.0, 4.0], [0.0, 2.0]]), tensor)
    paths.write_text(json.dumps(["train/0000/a.jpg", "0001/b.jpg"]))
    manifest.write_text(
        json.dumps(
            {
                "backbone": "ViT-B/32",
                "pretrained": "openai",
                "normalized": True,
                "dataset_size": 2,
                "external_data": False,
                "test_data_used": False,
                "path_index_sha256": sha256_lines(
                    ["train/0000/a.jpg", "0001/b.jpg"]
                ),
            }
        )
    )
    store = FrozenFeatureStore(tensor, paths, manifest, expected_dim=2)
    assert torch.allclose(store.get("0000/a.jpg").norm(), torch.tensor(1.0))
    with pytest.raises(KeyError, match="missing"):
        store.get("0002/c.jpg")


def test_feature_augmentation_horizontal_flip_is_exact() -> None:
    images = torch.arange(12).reshape(1, 1, 3, 4)
    flipped = apply_feature_augmentation(images, "horizontal_flip")
    assert torch.equal(flipped, images.flip(3))
    assert torch.equal(apply_feature_augmentation(images, "none"), images)
