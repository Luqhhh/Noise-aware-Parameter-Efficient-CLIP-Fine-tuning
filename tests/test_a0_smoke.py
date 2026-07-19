"""A0 smoke tests: verify the critical acceptance path without GPU training."""

import logging
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock


class TestA0ManifestAuditPathCanonicalization:
    """Verify _runtime_manifest_audit handles path format differences."""

    def test_relative_manifest_matches_absolute_dataset(self, tmp_path):
        """Relative manifest path + absolute dataset path → PASS."""
        from experiments.baseline.train import _runtime_manifest_audit

        cwd = str(Path.cwd())
        ds = MagicMock()
        ds.samples = [
            Path(f"{cwd}/train/0000/a.jpg"),
            Path(f"{cwd}/train/0001/b.jpg"),
        ]
        ds.labels = [5, 3]

        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": ["train/0000/a.jpg", "train/0001/b.jpg"],
            "original_label": [5, 3],
            "training_label": [5, 3],
            "sample_weight": [1.0, 1.0],
            "training_role": ["clean", "clean"],
        }).to_csv(manifest_csv, index=False)

        wp = MagicMock()
        wp._missing = "error"
        wp._loader.path = str(manifest_csv)

        audit_logger = logging.getLogger("test_audit")
        save_dir = tmp_path / "save"

        _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)
        assert (save_dir / "manifest_runtime_audit.json").exists()

    def test_relative_dataset_matches_relative_manifest(self, tmp_path):
        """Both dataset and manifest use relative paths → PASS after canonicalization."""
        from experiments.baseline.train import _runtime_manifest_audit

        cwd = str(Path.cwd())
        ds = MagicMock()
        ds.samples = [
            Path(f"{cwd}/train/0000/a.jpg"),
            Path(f"{cwd}/train/0001/b.jpg"),
        ]
        ds.labels = [5, 3]

        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": ["train/0000/a.jpg", "train/0001/b.jpg"],
            "original_label": [5, 3],
            "training_label": [5, 3],
            "sample_weight": [1.0, 1.0],
            "training_role": ["clean", "clean"],
        }).to_csv(manifest_csv, index=False)

        wp = MagicMock()
        wp._missing = "error"
        wp._loader.path = str(manifest_csv)

        audit_logger = logging.getLogger("test_audit")
        save_dir = tmp_path / "save"

        _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)
        assert (save_dir / "manifest_runtime_audit.json").exists()

    def test_canonical_duplicate_detected(self, tmp_path):
        """train/0000/a.jpg and ./train/0000/a.jpg canonicalize to same path → duplicate."""
        from experiments.baseline.train import _runtime_manifest_audit

        cwd = str(Path.cwd())
        ds = MagicMock()
        ds.samples = [
            Path(f"{cwd}/train/0000/a.jpg"),
        ]
        ds.labels = [5]

        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": [
                "train/0000/a.jpg",
                f"./train/0000/a.jpg",  # canonical duplicate
            ],
            "original_label": [5, 5],
            "training_label": [5, 5],
            "sample_weight": [1.0, 1.0],
            "training_role": ["clean", "clean"],
        }).to_csv(manifest_csv, index=False)

        wp = MagicMock()
        wp._missing = "error"
        wp._loader.path = str(manifest_csv)

        audit_logger = logging.getLogger("test_audit")
        save_dir = tmp_path / "save"

        with pytest.raises(ValueError, match="duplicate"):
            _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)

    def test_missing_manifest_row_raises(self, tmp_path):
        """Dataset has 3 images, manifest has 2 → FAIL."""
        from experiments.baseline.train import _runtime_manifest_audit

        cwd = str(Path.cwd())
        ds = MagicMock()
        ds.samples = [
            Path(f"{cwd}/train/a.jpg"),
            Path(f"{cwd}/train/b.jpg"),
            Path(f"{cwd}/train/c.jpg"),
        ]
        ds.labels = [0, 1, 2]

        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": ["train/a.jpg", "train/b.jpg"],
            "original_label": [0, 1],
            "training_label": [0, 1],
            "sample_weight": [1.0, 1.0],
            "training_role": ["clean", "clean"],
        }).to_csv(manifest_csv, index=False)

        wp = MagicMock()
        wp._missing = "error"
        wp._loader.path = str(manifest_csv)

        audit_logger = logging.getLogger("test_audit")
        save_dir = tmp_path / "save"

        with pytest.raises(ValueError, match="missing from manifest"):
            _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)

    def test_extra_manifest_row_raises(self, tmp_path):
        """Manifest has extra row not in dataset → FAIL."""
        from experiments.baseline.train import _runtime_manifest_audit

        cwd = str(Path.cwd())
        ds = MagicMock()
        ds.samples = [Path(f"{cwd}/train/a.jpg")]
        ds.labels = [0]

        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": ["train/a.jpg", "train/EXTRA.jpg"],
            "original_label": [0, 1],
            "training_label": [0, 1],
            "sample_weight": [1.0, 1.0],
            "training_role": ["clean", "clean"],
        }).to_csv(manifest_csv, index=False)

        wp = MagicMock()
        wp._missing = "error"
        wp._loader.path = str(manifest_csv)

        audit_logger = logging.getLogger("test_audit")
        save_dir = tmp_path / "save"

        with pytest.raises(ValueError, match="not in dataset"):
            _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)

    def test_label_mismatch_raises(self, tmp_path):
        """Manifest original_label differs from dataset → FAIL."""
        from experiments.baseline.train import _runtime_manifest_audit

        cwd = str(Path.cwd())
        ds = MagicMock()
        ds.samples = [Path(f"{cwd}/train/a.jpg")]
        ds.labels = [5]

        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": ["train/a.jpg"],
            "original_label": [3],
            "training_label": [3],
            "sample_weight": [1.0],
            "training_role": ["clean"],
        }).to_csv(manifest_csv, index=False)

        wp = MagicMock()
        wp._missing = "error"
        wp._loader.path = str(manifest_csv)

        audit_logger = logging.getLogger("test_audit")
        save_dir = tmp_path / "save"

        with pytest.raises(ValueError, match="original_label mismatch"):
            _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)

    def test_non_error_policy_raises(self, tmp_path):
        """missing_policy != 'error' → FAIL."""
        from experiments.baseline.train import _runtime_manifest_audit

        ds = MagicMock()
        wp = MagicMock()
        wp._missing = "ignore"

        audit_logger = logging.getLogger("test_audit")
        save_dir = tmp_path / "save"

        with pytest.raises(ValueError, match="missing_weight_policy"):
            _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)

    def test_clean_manifest_no_name_error(self, tmp_path):
        """Legal manifest returns cleanly, no NameError for logger."""
        from experiments.baseline.train import _runtime_manifest_audit

        cwd = str(Path.cwd())
        ds = MagicMock()
        ds.samples = [
            Path(f"{cwd}/train/a.jpg"),
            Path(f"{cwd}/train/b.jpg"),
        ]
        ds.labels = [5, 3]

        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": ["train/a.jpg", "train/b.jpg"],
            "original_label": [5, 3],
            "training_label": [5, 3],
            "sample_weight": [1.0, 1.0],
            "training_role": ["clean", "clean"],
        }).to_csv(manifest_csv, index=False)

        wp = MagicMock()
        wp._missing = "error"
        wp._loader.path = str(manifest_csv)

        audit_logger = logging.getLogger("test_audit")
        save_dir = tmp_path / "save"

        _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)
        assert (save_dir / "manifest_runtime_audit.json").exists()

    def test_none_provider_skips(self, tmp_path):
        """None weight_provider → early return."""
        from experiments.baseline.train import _runtime_manifest_audit

        ds = MagicMock()
        audit_logger = logging.getLogger("test_audit")
        save_dir = tmp_path / "save"

        _runtime_manifest_audit(ds, None, "dev", save_dir, audit_logger)
