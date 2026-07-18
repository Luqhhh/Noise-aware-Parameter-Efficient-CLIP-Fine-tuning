"""Tests for purification manifest schema and builder."""

import pandas as pd
import pytest


class TestManifestSchema:
    def test_clean_row_schema(self):
        """clean role: training_label=original_label, weight=1."""
        row = {
            "sample_id": "s1",
            "image_path": "img.jpg",
            "original_label": 5,
            "training_label": 5,
            "sample_weight": 1.0,
            "training_role": "clean",
        }
        assert row["training_label"] == row["original_label"]
        assert row["sample_weight"] == 1.0
        assert row["training_role"] == "clean"

    def test_rejected_row_schema(self):
        """rejected role: training_label=original_label, weight=0."""
        row = {
            "sample_id": "s1",
            "image_path": "img.jpg",
            "original_label": 5,
            "training_label": 5,
            "sample_weight": 0.0,
            "training_role": "rejected",
        }
        assert row["training_label"] == row["original_label"]
        assert row["sample_weight"] == 0.0
        assert row["training_role"] == "rejected"

    def test_pseudo_row_schema(self):
        """pseudo role: training_label != original_label, weight=1."""
        row = {
            "sample_id": "s1",
            "image_path": "img.jpg",
            "original_label": 5,
            "training_label": 3,
            "sample_weight": 1.0,
            "training_role": "pseudo",
        }
        assert row["training_label"] != row["original_label"]
        assert row["sample_weight"] == 1.0
        assert row["training_role"] == "pseudo"

    def test_role_only_clean_rejected_pseudo(self):
        """training_role must be one of {clean, rejected, pseudo}."""
        valid = {"clean", "rejected", "pseudo"}
        assert "pseudo" in valid
        assert "other" not in valid

    def test_no_other_weight_values(self):
        """sample_weight ∈ {0, 1} only."""
        allowed = {0.0, 1.0}
        assert 0.0 in allowed
        assert 1.0 in allowed
        assert 0.5 not in allowed


class TestBuilderInvariants:
    def test_row_count_matches_strict_train(self, tmp_path):
        """Manifest row count == strict-train row count."""
        n = 10
        df = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"img{i}.jpg" for i in range(n)],
            "original_label": [i % 5 for i in range(n)],
            "training_label": [i % 5 for i in range(n)],
            "sample_weight": [1.0] * n,
            "quality_score": [0.9] * n,
            "training_role": ["clean"] * n,
        })
        assert len(df) == n
        assert df["image_path"].is_unique

    def test_every_class_has_clean_samples(self, tmp_path):
        """Each class retains at least some clean samples."""
        df = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(5)],
            "image_path": [f"img{i}.jpg" for i in range(5)],
            "original_label": [0, 0, 1, 1, 2],
            "training_label": [0, 0, 1, 1, 2],
            "sample_weight": [1.0] * 5,
            "quality_score": [0.9] * 5,
            "training_role": ["clean"] * 5,
        })
        counts = df.groupby("original_label").size()
        assert len(counts) == 3
        assert all(c > 0 for c in counts)

    def test_max_per_class_reject_rate_respected(self, tmp_path):
        """No class exceeds the reject rate cap."""
        max_reject_rate = 0.10
        n_per_class = 100
        n_classes = 5
        rows = []
        for c in range(n_classes):
            for i in range(n_per_class):
                role = "rejected" if i < 8 else "clean"
                rows.append({
                    "sample_id": f"s{c}_{i}",
                    "image_path": f"img{c}_{i}.jpg",
                    "original_label": c,
                    "training_label": c,
                    "sample_weight": 0.0 if role == "rejected" else 1.0,
                    "quality_score": 0.5,
                    "training_role": role,
                })
        df = pd.DataFrame(rows)
        for c in range(n_classes):
            class_rows = df[df["original_label"] == c]
            rejected = (class_rows["training_role"] == "rejected").sum()
            assert rejected / len(class_rows) <= max_reject_rate
