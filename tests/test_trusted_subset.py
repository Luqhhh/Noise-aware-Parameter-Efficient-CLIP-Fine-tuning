"""Tests for common.trusted_subset module."""
import pandas as pd
import numpy as np
import pytest
from common.trusted_subset import TrustedSubsetConfig, build_trusted_subset


@pytest.fixture
def sample_df():
    """DataFrame with all required columns for trusted subset rules."""
    n = 100
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "sample_index": range(n),
        "image_path": [f"img_{i}.jpg" for i in range(n)],
        "image_sha256": [f"sha_{i}" for i in range(n)],
        "noisy_label": rng.integers(0, 10, n),
        "class_name": [f"{i%10:04d}" for i in range(n)],
        "knn_label_agreement": rng.uniform(0.3, 1.0, n),
        "prototype_supports_noisy_label": rng.choice([True, False], n),
        "prototype_margin": rng.uniform(-0.1, 0.3, n),
        "prototype_top1_label": rng.integers(0, 10, n),
        "clip_flip_cosine": rng.uniform(0.7, 1.0, n),
        "cross_class_duplicate_conflict": rng.choice([True, False], n, p=[0.1, 0.9]),
        # Model-specific columns — should be IGNORED by trusted subset
        "d3_pred": rng.integers(0, 10, n),
        "d3_correct": rng.choice([True, False], n),
        "d3_confidence": rng.uniform(0, 1, n),
        "d3_margin": rng.uniform(0, 1, n),
        "b2_pred": rng.integers(0, 10, n),
        "b2_correct": rng.choice([True, False], n),
        "b2_confidence": rng.uniform(0, 1, n),
    })
    return df


class TestTrustedSubsetConfig:
    def test_defaults(self):
        cfg = TrustedSubsetConfig()
        assert cfg.knn_label_agreement_min == 0.60
        assert cfg.prototype_margin_min == 0.02
        assert cfg.clip_flip_cosine_min == 0.90
        assert cfg.require_prototype_top1_matches_label is True
        assert cfg.reject_cross_class_duplicate_conflict is True


class TestBuildTrustedSubset:
    def test_all_rules_applied(self, sample_df):
        cfg = TrustedSubsetConfig()
        manifest, summary = build_trusted_subset(sample_df, cfg)
        assert "trusted_v1" in manifest.columns
        assert "rejection_reasons" in manifest.columns
        assert summary["total_samples"] == len(sample_df)

    def test_trusted_subset_is_subset(self, sample_df):
        cfg = TrustedSubsetConfig()
        manifest, summary = build_trusted_subset(sample_df, cfg)
        assert manifest["trusted_v1"].sum() <= len(sample_df)

    def test_rejection_reasons_nonempty_for_rejected(self, sample_df):
        cfg = TrustedSubsetConfig()
        manifest, summary = build_trusted_subset(sample_df, cfg)
        rejected = manifest[~manifest["trusted_v1"]]
        assert (rejected["rejection_reasons"].str.len() > 0).all()

    def test_all_trusted_no_rejection_reasons(self, sample_df):
        cfg = TrustedSubsetConfig()
        manifest, summary = build_trusted_subset(sample_df, cfg)
        trusted = manifest[manifest["trusted_v1"]]
        # Trusted samples should have empty rejection reasons
        assert (trusted["rejection_reasons"] == "").all()

    def test_coverage_reported(self, sample_df):
        cfg = TrustedSubsetConfig()
        manifest, summary = build_trusted_subset(sample_df, cfg)
        assert "coverage" in summary
        assert 0 <= summary["coverage"] <= 1

    def test_per_class_stats(self, sample_df):
        cfg = TrustedSubsetConfig()
        manifest, summary = build_trusted_subset(sample_df, cfg)
        assert "represented_classes" in summary
        assert "per_class_trusted" in summary

    def test_output_independent_of_model_columns(self, sample_df):
        """Trusted subset must NOT depend on D3/B2 columns."""
        cfg = TrustedSubsetConfig()
        manifest1, summary1 = build_trusted_subset(sample_df, cfg)

        # Remove all model-specific columns
        df_no_model = sample_df.drop(columns=[
            c for c in sample_df.columns
            if c.startswith("d3_") or c.startswith("b2_")
        ])
        manifest2, summary2 = build_trusted_subset(df_no_model, cfg)

        assert (manifest1["trusted_v1"] == manifest2["trusted_v1"]).all()
        assert summary1["trusted_count"] == summary2["trusted_count"]

    def test_missing_conflict_metadata_marks_partial(self, sample_df):
        cfg = TrustedSubsetConfig()
        df_no_conflict = sample_df.drop(columns=["cross_class_duplicate_conflict"])
        manifest, summary = build_trusted_subset(df_no_conflict, cfg)
        assert summary.get("conflict_metadata_available") is False

    def test_boundary_knn_agreement(self, sample_df):
        cfg = TrustedSubsetConfig(knn_label_agreement_min=0.60)
        sample_df["knn_label_agreement"] = 0.599
        sample_df["prototype_supports_noisy_label"] = True
        sample_df["prototype_margin"] = 0.1
        sample_df["clip_flip_cosine"] = 0.95
        sample_df["cross_class_duplicate_conflict"] = False
        manifest, _ = build_trusted_subset(sample_df, cfg)
        # knn at 0.599 < 0.60 → all rejected
        assert manifest["trusted_v1"].sum() == 0
        assert all("low_knn_agreement" in r for r in manifest["rejection_reasons"])

    def test_boundary_prototype_margin(self, sample_df):
        cfg = TrustedSubsetConfig(prototype_margin_min=0.02)
        sample_df["knn_label_agreement"] = 0.8
        sample_df["prototype_supports_noisy_label"] = True
        sample_df["prototype_margin"] = 0.0199
        sample_df["clip_flip_cosine"] = 0.95
        sample_df["cross_class_duplicate_conflict"] = False
        manifest, _ = build_trusted_subset(sample_df, cfg)
        assert manifest["trusted_v1"].sum() == 0
        assert all("low_prototype_margin" in r for r in manifest["rejection_reasons"])

    def test_all_conditions_pass(self, sample_df):
        cfg = TrustedSubsetConfig()
        sample_df["knn_label_agreement"] = 0.8
        sample_df["prototype_supports_noisy_label"] = True
        sample_df["prototype_margin"] = 0.1
        sample_df["clip_flip_cosine"] = 0.95
        sample_df["cross_class_duplicate_conflict"] = False
        manifest, _ = build_trusted_subset(sample_df, cfg)
        assert manifest["trusted_v1"].sum() == len(sample_df)
