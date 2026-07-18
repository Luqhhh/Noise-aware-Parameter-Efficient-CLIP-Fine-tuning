"""Tests for purification manifest audit fail-closed checks."""

import pandas as pd
import pytest

from scripts.audit_purification_manifest import audit_manifest


def _valid_manifest(n=10):
    rows = []
    for i in range(n):
        rows.append({
            "sample_id": f"s{i}",
            "image_path": f"img{i}.jpg",
            "original_label": i % 5,
            "training_label": i % 5,
            "sample_weight": 1.0,
            "quality_score": 0.9,
            "training_role": "clean",
            "selection_reason": "clean",
            "suggested_label": i % 5,
            "oof_top1": i % 5,
            "p_original_label": 0.8,
            "p_top1": 0.9,
            "top1_margin": 0.1,
            "prototype_top1": i % 5,
            "prototype_margin": 0.05,
            "knn_top1": i % 5,
            "knn_agreement": 0.7,
            "flip_consistency": 1.0,
            "duplicate_conflict_flag": False,
        })
    return pd.DataFrame(rows)


class TestManifestAudit:
    def test_valid_manifest_passes(self):
        df = _valid_manifest(10)
        # strict_train_df needs image_path + label columns
        strict = pd.DataFrame({
            "image_path": df["image_path"].tolist(),
            "label": df["original_label"].tolist(),
        })
        errors, warnings = audit_manifest(df, strict_train_df=strict, max_class_reject_rate=0.10)
        assert len(errors) == 0

    def test_missing_required_column_rejected(self):
        df = _valid_manifest(5).drop(columns=["training_role"])
        errors, _ = audit_manifest(df, max_class_reject_rate=0.10)
        assert any("Missing required column: training_role" in e for e in errors)

    def test_invalid_role_rejected(self):
        df = _valid_manifest(5)
        df.loc[0, "training_role"] = "other"
        errors, _ = audit_manifest(df, max_class_reject_rate=0.10)
        assert any("invalid training_role" in e for e in errors)

    def test_clean_with_weight_zero_rejected(self):
        df = _valid_manifest(5)
        df.loc[0, "sample_weight"] = 0.0
        errors, _ = audit_manifest(df, max_class_reject_rate=0.10)
        assert any("clean" in e and "weight" in e for e in errors)

    def test_rejected_with_weight_one_rejected(self):
        df = _valid_manifest(5)
        df.loc[0, "training_role"] = "rejected"
        df.loc[0, "sample_weight"] = 1.0
        errors, _ = audit_manifest(df, max_class_reject_rate=0.10)
        assert any("rejected" in e and "weight" in e for e in errors)

    def test_pseudo_label_unchanged_rejected(self):
        df = _valid_manifest(5)
        df.loc[0, "training_role"] = "pseudo"
        # training_label == original_label when role is pseudo → error
        errors, _ = audit_manifest(df, max_class_reject_rate=0.10)
        assert any("pseudo" in e for e in errors)

    def test_class_with_zero_clean_rejected(self):
        """All samples of a class being rejected should error."""
        rows = []
        for i in range(5):
            rows.append({
                "sample_id": f"s{i}",
                "image_path": f"img{i}.jpg",
                "original_label": 0,
                "training_label": 0,
                "sample_weight": 0.0,
                "quality_score": 0.5,
                "training_role": "rejected",
                "selection_reason": "",
                "suggested_label": 0,
                "oof_top1": 0,
                "p_original_label": 0.5,
                "p_top1": 0.5,
                "top1_margin": 0.0,
                "prototype_top1": 0,
                "prototype_margin": 0.0,
                "knn_top1": 0,
                "knn_agreement": 0.5,
                "flip_consistency": 1.0,
                "duplicate_conflict_flag": False,
            })
            rows.append({
                "sample_id": f"s{i+5}",
                "image_path": f"img{i+5}.jpg",
                "original_label": 1,
                "training_label": 1,
                "sample_weight": 1.0,
                "quality_score": 0.9,
                "training_role": "clean",
                "selection_reason": "",
                "suggested_label": 1,
                "oof_top1": 1,
                "p_original_label": 0.9,
                "p_top1": 0.9,
                "top1_margin": 0.0,
                "prototype_top1": 1,
                "prototype_margin": 0.0,
                "knn_top1": 1,
                "knn_agreement": 0.5,
                "flip_consistency": 1.0,
                "duplicate_conflict_flag": False,
            })
        df = pd.DataFrame(rows)
        errors, _ = audit_manifest(df, max_class_reject_rate=0.50)
        assert any("zero clean" in e or "class 0" in e.lower() for e in errors)
