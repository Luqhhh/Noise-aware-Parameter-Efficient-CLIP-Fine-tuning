"""Test that hard_relabel actually changes the batch label in training."""

import pytest
import torch
import torch.nn as nn

from common.sample_weighting import (
    BaseWeightProvider,
    RelabelManifestProvider,
)


class TestRelabelProviderTrainingLabels:
    """Verify RelabelManifestProvider.get_training_labels behaviour."""

    def test_hard_relabel_true_returns_new_label(self, tmp_path):
        """When hard_relabel=True, training_label replaces original_label."""
        import pandas as pd
        csv = tmp_path / "manifest.csv"
        pd.DataFrame([{
            "image_path": "/a/b/img1.jpg",
            "sample_id": "s1",
            "original_label": 5,
            "training_label": 3,
            "sample_weight": 1.0,
            "quality_score": 0.9,
        }]).to_csv(csv, index=False)

        p = RelabelManifestProvider(
            manifest_path=str(csv),
            hard_relabel=True,
            missing_policy="error",
        )
        orig = torch.tensor([5])
        result = p.get_training_labels(["/a/b/img1.jpg"], orig)
        assert result.tolist() == [3]

    def test_hard_relabel_false_returns_original(self, tmp_path):
        """When hard_relabel=False, original label is returned unchanged."""
        import pandas as pd
        csv = tmp_path / "manifest.csv"
        pd.DataFrame([{
            "image_path": "/a/b/img1.jpg",
            "sample_id": "s1",
            "original_label": 5,
            "training_label": 3,
            "sample_weight": 1.0,
            "quality_score": 0.9,
        }]).to_csv(csv, index=False)

        p = RelabelManifestProvider(
            manifest_path=str(csv),
            hard_relabel=False,
            missing_policy="error",
        )
        orig = torch.tensor([5])
        result = p.get_training_labels(["/a/b/img1.jpg"], orig)
        assert result.tolist() == [5]

    def test_missing_sample_with_error_raises(self, tmp_path):
        """Missing sample with policy=error raises KeyError."""
        import pandas as pd
        csv = tmp_path / "manifest.csv"
        pd.DataFrame([{
            "image_path": "/a/b/img1.jpg",
            "sample_id": "s1",
            "original_label": 5,
            "training_label": 3,
            "sample_weight": 1.0,
            "quality_score": 0.9,
        }]).to_csv(csv, index=False)

        p = RelabelManifestProvider(
            manifest_path=str(csv),
            hard_relabel=True,
            missing_policy="error",
        )
        orig = torch.tensor([5])
        with pytest.raises(KeyError):
            p.get_training_labels(["/a/b/unknown.jpg"], orig)

    def test_get_training_labels_returns_long_tensor(self, tmp_path):
        """Return value is a LongTensor on the same device."""
        import pandas as pd
        csv = tmp_path / "manifest.csv"
        df = pd.DataFrame([
            {"image_path": f"/img{i}.jpg", "sample_id": f"s{i}",
             "original_label": i % 3, "training_label": (i + 1) % 3,
             "sample_weight": 1.0, "quality_score": 0.9}
            for i in range(4)
        ])
        df.to_csv(csv, index=False)

        p = RelabelManifestProvider(str(csv), hard_relabel=True, missing_policy="error")
        orig = torch.tensor([0, 0, 0, 3], dtype=torch.long, device="cpu")
        result = p.get_training_labels(
            ["/img0.jpg", "/img1.jpg", "/img2.jpg", "/img3.jpg"], orig
        )
        assert result.dtype == torch.long
        assert result.device == orig.device


class TestBaseWeightProviderDefaults:
    """Default interface methods always exist (tested via concrete subclass)."""

    def test_get_training_labels_default(self):
        from common.sample_weighting import NoneWeightProvider
        p = NoneWeightProvider()
        labels = torch.tensor([1, 2, 3])
        out = p.get_training_labels(["a", "b", "c"], labels)
        assert torch.equal(out, labels)

    def test_get_roles_default(self):
        from common.sample_weighting import NoneWeightProvider
        p = NoneWeightProvider()
        roles = p.get_roles(["a", "b", "c"])
        assert roles == ["clean", "clean", "clean"]
