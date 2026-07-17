"""Tests for common.trusted_subset module."""
import pandas as pd
import numpy as np
import pytest
from common.trusted_subset import (
    TrustedSubsetConfig,
    build_trusted_subset,
    build_trusted_subset_oof,
)


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


# ═══════════════════════════════════════════════════════════════════════
# V3: OOF single-threshold trusted subset
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_df_oof():
    """DataFrame with p_original_label for OOF-based trusted subset."""
    n = 100
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "sample_index": range(n),
        "image_path": [f"img_{i}.jpg" for i in range(n)],
        "noisy_label": rng.integers(0, 10, n),
        "p_original_label": rng.uniform(0.0, 1.0, n),
    })
    return df


class TestBuildTrustedSubsetOof:
    def test_p_oof_above_threshold_is_trusted(self, sample_df_oof):
        """Samples with p >= threshold should be trusted."""
        cfg = TrustedSubsetConfig(p_oof_label_min=0.60)
        sample_df_oof["p_original_label"] = 0.80
        manifest, summary = build_trusted_subset_oof(sample_df_oof, cfg)
        assert manifest["trusted_v1"].sum() == len(sample_df_oof)
        assert summary["trusted_count"] == len(sample_df_oof)
        assert summary["rejected_count"] == 0

    def test_p_oof_below_threshold_is_rejected(self, sample_df_oof):
        """Samples below threshold should be rejected with reason."""
        cfg = TrustedSubsetConfig(p_oof_label_min=0.60)
        sample_df_oof["p_original_label"] = 0.30
        manifest, summary = build_trusted_subset_oof(sample_df_oof, cfg)
        assert manifest["trusted_v1"].sum() == 0
        assert summary["rejected_count"] == len(sample_df_oof)
        assert all("low_oof_probability" in r for r in manifest["rejection_reasons"])

    def test_boundary_threshold(self, sample_df_oof):
        """p exactly at threshold should be trusted (≥ is inclusive)."""
        cfg = TrustedSubsetConfig(p_oof_label_min=0.60)
        sample_df_oof["p_original_label"] = 0.60
        manifest, _ = build_trusted_subset_oof(sample_df_oof, cfg)
        assert manifest["trusted_v1"].sum() == len(sample_df_oof)

    def test_mixed_trusted_and_rejected(self, sample_df_oof):
        """Half above, half below threshold."""
        cfg = TrustedSubsetConfig(p_oof_label_min=0.60)
        n = len(sample_df_oof)
        sample_df_oof["p_original_label"] = [0.80 if i < n//2 else 0.30 for i in range(n)]
        manifest, summary = build_trusted_subset_oof(sample_df_oof, cfg)
        assert summary["trusted_count"] == n // 2
        assert summary["rejected_count"] == n - n // 2

    def test_coverage_reported(self, sample_df_oof):
        cfg = TrustedSubsetConfig(p_oof_label_min=0.60)
        manifest, summary = build_trusted_subset_oof(sample_df_oof, cfg)
        assert "coverage" in summary
        assert 0 <= summary["coverage"] <= 1

    def test_per_class_stats(self, sample_df_oof):
        cfg = TrustedSubsetConfig(p_oof_label_min=0.60)
        manifest, summary = build_trusted_subset_oof(sample_df_oof, cfg)
        assert "represented_classes" in summary
        assert "per_class_trusted" in summary
        assert "missing_classes" in summary

    def test_summary_records_method_and_threshold(self, sample_df_oof):
        cfg = TrustedSubsetConfig(p_oof_label_min=0.45)
        manifest, summary = build_trusted_subset_oof(sample_df_oof, cfg)
        assert summary["method"] == "oof_single_threshold"
        assert summary["p_oof_label_min"] == 0.45

    def test_missing_column_raises(self, sample_df_oof):
        cfg = TrustedSubsetConfig()
        df_no_col = sample_df_oof.drop(columns=["p_original_label"])
        with pytest.raises(ValueError, match="p_original_label"):
            build_trusted_subset_oof(df_no_col, cfg)

    def test_default_threshold(self):
        cfg = TrustedSubsetConfig()
        assert cfg.p_oof_label_min == 0.60

    def test_override_threshold(self):
        cfg = TrustedSubsetConfig(p_oof_label_min=0.75)
        assert cfg.p_oof_label_min == 0.75


# ═══════════════════════════════════════════════════════════════════════
# V2: Continuous trust-weighted and class-balanced metrics
# ═══════════════════════════════════════════════════════════════════════


class TestTrustedSubsetConfigV2:
    def test_defaults(self):
        cfg = TrustedSubsetConfig()
        assert cfg.class_balanced_top_k == 5
        assert cfg.prototype_margin_ref == 0.05

    def test_override_class_balanced_k(self):
        cfg = TrustedSubsetConfig(class_balanced_top_k=10)
        assert cfg.class_balanced_top_k == 10

    def test_override_margin_ref(self):
        cfg = TrustedSubsetConfig(prototype_margin_ref=0.10)
        assert cfg.prototype_margin_ref == 0.10


class TestComputeCompositeTrustWeights:
    def test_weights_in_range(self, sample_df):
        from common.trusted_subset import _compute_composite_trust_weights
        w = _compute_composite_trust_weights(sample_df, margin_ref=0.05)
        assert w.shape == (len(sample_df),)
        assert w.min() >= 0.0
        assert w.max() <= 1.0

    def test_perfect_signals_give_high_weights(self):
        from common.trusted_subset import _compute_composite_trust_weights
        df = pd.DataFrame({
            "knn_label_agreement": [1.0] * 10,
            "prototype_margin": [1.0] * 10,
            "clip_flip_cosine": [1.0] * 10,
        })
        w = _compute_composite_trust_weights(df, margin_ref=0.05)
        # All signals at max → weights near 1.0
        assert (w > 0.99).all()

    def test_weak_signals_give_low_weights(self):
        from common.trusted_subset import _compute_composite_trust_weights
        df = pd.DataFrame({
            "knn_label_agreement": [0.0] * 10,
            "prototype_margin": [0.0] * 10,
            "clip_flip_cosine": [0.0] * 10,
        })
        w = _compute_composite_trust_weights(df, margin_ref=0.05)
        assert (w == 0.0).all()

    def test_margin_clamping(self):
        from common.trusted_subset import _compute_composite_trust_weights
        df = pd.DataFrame({
            "knn_label_agreement": [1.0, 1.0, 1.0],
            "prototype_margin": [0.0, 0.05, 0.25],
            "clip_flip_cosine": [1.0, 1.0, 1.0],
        })
        w = _compute_composite_trust_weights(df, margin_ref=0.05)
        # margin=0.0 → w_proto≈0; margin=0.05 → w_proto=1; margin=0.25 → w_proto=1
        assert w[0] == pytest.approx(0.0, abs=1e-9)
        assert w[1] == pytest.approx(1.0, abs=1e-9)
        assert w[2] == pytest.approx(1.0, abs=1e-9)


class TestTrustWeightedAccuracy:
    def test_perfect_correct_gives_one(self):
        from common.trusted_subset import compute_trust_weighted_accuracy
        df = pd.DataFrame({
            "noisy_label": [0] * 10,
            "knn_label_agreement": [0.8] * 10,
            "prototype_margin": [0.05] * 10,
            "clip_flip_cosine": [0.95] * 10,
        })
        correct = np.ones(10, dtype=bool)
        result = compute_trust_weighted_accuracy(df, correct)
        assert result["accuracy"] == pytest.approx(1.0, abs=1e-9)
        assert result["total_samples"] == 10

    def test_all_wrong_gives_zero(self):
        from common.trusted_subset import compute_trust_weighted_accuracy
        df = pd.DataFrame({
            "noisy_label": [0] * 10,
            "knn_label_agreement": [0.8] * 10,
            "prototype_margin": [0.05] * 10,
            "clip_flip_cosine": [0.95] * 10,
        })
        correct = np.zeros(10, dtype=bool)
        result = compute_trust_weighted_accuracy(df, correct)
        assert result["accuracy"] == pytest.approx(0.0, abs=1e-9)

    def test_between_raw_and_one(self, sample_df):
        from common.trusted_subset import compute_trust_weighted_accuracy
        correct = np.ones(len(sample_df), dtype=bool)
        # Make half correct
        correct[len(sample_df)//2:] = False
        raw = correct.mean()
        result = compute_trust_weighted_accuracy(sample_df, correct)
        # Trust-weighted should differ from raw if weights are non-uniform
        assert result["total_samples"] == len(sample_df)
        assert result["weight_sum"] > 0
        assert "per_class_accuracy" in result

    def test_effective_samples_bounded(self, sample_df):
        from common.trusted_subset import compute_trust_weighted_accuracy
        correct = np.ones(len(sample_df), dtype=bool)
        result = compute_trust_weighted_accuracy(sample_df, correct)
        assert 0 < result["effective_samples"] <= result["total_samples"]

    def test_empty_df(self):
        from common.trusted_subset import compute_trust_weighted_accuracy
        df = pd.DataFrame({
            "noisy_label": pd.Series([], dtype=int),
            "knn_label_agreement": pd.Series([], dtype=float),
            "prototype_margin": pd.Series([], dtype=float),
            "clip_flip_cosine": pd.Series([], dtype=float),
        })
        result = compute_trust_weighted_accuracy(df, np.array([], dtype=bool))
        assert result["total_samples"] == 0
        assert np.isnan(result["accuracy"])


class TestClassBalancedTrustedAccuracy:
    def test_all_classes_have_k(self):
        from common.trusted_subset import compute_class_balanced_trusted_accuracy
        n_per_class = 10
        n_classes = 5
        # Each class gets the same pattern: increasing trust across its 10 samples
        pattern = [0.3, 0.5, 0.7, 0.9, 1.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        pattern_margin = [0.01, 0.03, 0.05, 0.07, 0.10, 0.01, 0.03, 0.05, 0.07, 0.10]
        pattern_flip = [0.80, 0.85, 0.90, 0.95, 0.99, 0.80, 0.85, 0.90, 0.95, 0.99]
        df = pd.DataFrame({
            "noisy_label": np.repeat(np.arange(n_classes), n_per_class),
            "knn_label_agreement": np.tile(pattern, n_classes),
            "prototype_margin": np.tile(pattern_margin, n_classes),
            "clip_flip_cosine": np.tile(pattern_flip, n_classes),
        })
        correct = np.ones(n_per_class * n_classes, dtype=bool)
        # Make lowest-trust samples wrong, highest-trust correct
        # The top-2 per class will be the ones with highest trust weights
        result = compute_class_balanced_trusted_accuracy(df, correct, top_k=3)
        assert result["num_classes_with_k"] == n_classes
        assert result["top_k_per_class"] == 3
        # All correct → macro_accuracy = 1.0
        assert result["macro_accuracy"] == pytest.approx(1.0, abs=1e-9)

    def test_some_classes_below_k(self):
        from common.trusted_subset import compute_class_balanced_trusted_accuracy
        df = pd.DataFrame({
            "noisy_label": [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2],
            "knn_label_agreement": 0.8,
            "prototype_margin": 0.05,
            "clip_flip_cosine": 0.95,
        })
        correct = np.ones(len(df), dtype=bool)
        result = compute_class_balanced_trusted_accuracy(df, correct, top_k=5)
        # Class 0 has 4 (<5), class 1 has 2 (<5), class 2 has 6 (≥5)
        assert result["num_classes_with_k"] == 1  # only class 2
        assert result["num_classes_total"] == 3

    def test_samples_used_count(self):
        from common.trusted_subset import compute_class_balanced_trusted_accuracy
        n_per_class = 10
        n_classes = 10
        df = pd.DataFrame({
            "noisy_label": np.repeat(np.arange(n_classes), n_per_class),
            "knn_label_agreement": 0.8,
            "prototype_margin": 0.05,
            "clip_flip_cosine": 0.95,
        })
        correct = np.ones(n_per_class * n_classes, dtype=bool)
        result = compute_class_balanced_trusted_accuracy(df, correct, top_k=3)
        assert result["num_samples_used"] == n_classes * 3

    def test_empty_df(self):
        from common.trusted_subset import compute_class_balanced_trusted_accuracy
        df = pd.DataFrame({
            "noisy_label": pd.Series([], dtype=int),
            "knn_label_agreement": pd.Series([], dtype=float),
            "prototype_margin": pd.Series([], dtype=float),
            "clip_flip_cosine": pd.Series([], dtype=float),
        })
        result = compute_class_balanced_trusted_accuracy(
            df, np.array([], dtype=bool), top_k=5
        )
        assert result["num_classes_total"] == 0
        assert np.isnan(result["macro_accuracy"])
