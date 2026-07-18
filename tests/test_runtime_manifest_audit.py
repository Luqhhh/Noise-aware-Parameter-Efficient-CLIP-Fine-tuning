"""Tests for _runtime_manifest_audit fail-closed behavior."""
import logging
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock


class TestRuntimeManifestAudit:
    """Verify _runtime_manifest_audit correctly validates manifests."""

    def test_clean_manifest_passes_without_name_error(self, tmp_path):
        """A perfectly matching manifest returns cleanly (no NameError)."""
        from experiments.baseline.train import _runtime_manifest_audit

        # Build minimal dataset mock
        ds = MagicMock()
        ds.samples = ["img0.jpg", "img1.jpg"]
        ds.labels = [5, 3]

        # Build minimal manifest CSV that exactly matches
        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": ["img0.jpg", "img1.jpg"],
            "original_label": [5, 3],
            "training_label": [5, 3],
            "sample_weight": [1.0, 1.0],
            "training_role": ["clean", "clean"],
        }).to_csv(manifest_csv, index=False)

        # Build weight provider mock with correct _loader._path
        wp = MagicMock()
        wp._missing = "error"
        wp._loader._path = str(manifest_csv)

        save_dir = tmp_path / "save"
        audit_logger = logging.getLogger("test_audit")

        # This must NOT raise NameError or any other exception
        _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)

        # Verify audit file was written
        audit_path = save_dir / "manifest_runtime_audit.json"
        assert audit_path.exists()

    def test_missing_path_in_manifest_raises(self, tmp_path):
        """Dataset image missing from manifest raises ValueError."""
        from experiments.baseline.train import _runtime_manifest_audit

        ds = MagicMock()
        ds.samples = ["img0.jpg", "img1.jpg", "img2.jpg"]
        ds.labels = [5, 3, 7]

        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": ["img0.jpg", "img1.jpg"],
            "original_label": [5, 3],
            "training_label": [5, 3],
            "sample_weight": [1.0, 1.0],
            "training_role": ["clean", "clean"],
        }).to_csv(manifest_csv, index=False)

        wp = MagicMock()
        wp._missing = "error"
        wp._loader._path = str(manifest_csv)

        save_dir = tmp_path / "save"
        audit_logger = logging.getLogger("test_audit")

        with pytest.raises(ValueError, match="missing from manifest"):
            _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)

    def test_extra_path_in_manifest_raises(self, tmp_path):
        """Manifest image not in dataset raises ValueError."""
        from experiments.baseline.train import _runtime_manifest_audit

        ds = MagicMock()
        ds.samples = ["img0.jpg"]
        ds.labels = [5]

        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": ["img0.jpg", "EXTRA.jpg"],
            "original_label": [5, 3],
            "training_label": [5, 3],
            "sample_weight": [1.0, 1.0],
            "training_role": ["clean", "clean"],
        }).to_csv(manifest_csv, index=False)

        wp = MagicMock()
        wp._missing = "error"
        wp._loader._path = str(manifest_csv)

        save_dir = tmp_path / "save"
        audit_logger = logging.getLogger("test_audit")

        with pytest.raises(ValueError, match="not in dataset"):
            _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)

    def test_duplicate_paths_in_manifest_raises(self, tmp_path):
        """Duplicate image_path in manifest raises ValueError."""
        from experiments.baseline.train import _runtime_manifest_audit

        ds = MagicMock()
        ds.samples = ["img0.jpg"]
        ds.labels = [5]

        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": ["img0.jpg", "img0.jpg"],
            "original_label": [5, 5],
            "training_label": [5, 5],
            "sample_weight": [1.0, 1.0],
            "training_role": ["clean", "clean"],
        }).to_csv(manifest_csv, index=False)

        wp = MagicMock()
        wp._missing = "error"
        wp._loader._path = str(manifest_csv)

        save_dir = tmp_path / "save"
        audit_logger = logging.getLogger("test_audit")

        with pytest.raises(ValueError, match="duplicate"):
            _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)

    def test_original_label_mismatch_raises(self, tmp_path):
        """original_label differs between dataset and manifest raises ValueError."""
        from experiments.baseline.train import _runtime_manifest_audit

        ds = MagicMock()
        ds.samples = ["img0.jpg"]
        ds.labels = [5]

        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": ["img0.jpg"],
            "original_label": [3],
            "training_label": [3],
            "sample_weight": [1.0],
            "training_role": ["clean"],
        }).to_csv(manifest_csv, index=False)

        wp = MagicMock()
        wp._missing = "error"
        wp._loader._path = str(manifest_csv)

        save_dir = tmp_path / "save"
        audit_logger = logging.getLogger("test_audit")

        with pytest.raises(ValueError, match="original_label mismatch"):
            _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)

    def test_zero_clean_class_raises(self, tmp_path):
        """Class with zero clean samples raises ValueError."""
        from experiments.baseline.train import _runtime_manifest_audit

        ds = MagicMock()
        ds.samples = ["img0.jpg"]
        ds.labels = [5]

        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": ["img0.jpg"],
            "original_label": [5],
            "training_label": [5],
            "sample_weight": [0.0],
            "training_role": ["rejected"],
        }).to_csv(manifest_csv, index=False)

        wp = MagicMock()
        wp._missing = "error"
        wp._loader._path = str(manifest_csv)

        save_dir = tmp_path / "save"
        audit_logger = logging.getLogger("test_audit")

        with pytest.raises(ValueError, match="zero clean"):
            _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)

    def test_non_error_missing_policy_raises(self, tmp_path):
        """missing_policy != 'error' raises ValueError before any comparison."""
        from experiments.baseline.train import _runtime_manifest_audit

        ds = MagicMock()
        wp = MagicMock()
        wp._missing = "ignore"

        save_dir = tmp_path / "save"
        audit_logger = logging.getLogger("test_audit")

        with pytest.raises(ValueError, match="missing_weight_policy"):
            _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)

    def test_none_weight_provider_returns_early(self, tmp_path):
        """None weight_provider skips audit without error."""
        from experiments.baseline.train import _runtime_manifest_audit

        ds = MagicMock()
        save_dir = tmp_path / "save"
        audit_logger = logging.getLogger("test_audit")

        # Should not raise
        _runtime_manifest_audit(ds, None, "dev", save_dir, audit_logger)
