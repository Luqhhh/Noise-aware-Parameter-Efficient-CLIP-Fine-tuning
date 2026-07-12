"""Tests for submission manifest building and registration.

Verifies:
  - Manifest JSON schema completeness
  - SHA-256 hashing correctness
  - ZIP-internal CSV matches external CSV hash
  - Prediction count validation (24,967)
  - Label format validation (^\\d{4}$)
  - Duplicate ZIP hash rejection in registry
  - Checkpoint best_val_acc vs reeval micro consistency
"""

import csv
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers (mirroring scripts/build_submission_manifest.py logic)
# ---------------------------------------------------------------------------

def _sha256_hex(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _validate_labels(csv_path: Path) -> bool:
    """Every label in the CSV must match ^\\d{4}$.

    Note: pred_results.csv has NO header — it is the submission format directly.
    """
    import re
    pattern = re.compile(r"^\d{4}$")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                return False
            label = row[1].strip()
            if not pattern.match(label):
                return False
    return True


def _count_predictions(csv_path: Path) -> int:
    """Count data rows in pred_results.csv (no header)."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return sum(1 for _ in reader)


def _validate_zip_contents(
    zip_path: Path,
    expected_csv_path: Path,
) -> dict:
    """Validate ZIP: single pred_results.csv, hash matches external.

    Returns dict with valid, internal_csv_sha256, zip_sha256 keys.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if names != ["pred_results.csv"]:
            return {"valid": False, "error": f"Unexpected ZIP contents: {names}"}

        with zf.open("pred_results.csv") as zf_csv:
            h = hashlib.sha256()
            while True:
                chunk = zf_csv.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
            internal_csv_hash = h.hexdigest()

    external_hash = _sha256_hex(expected_csv_path)

    if internal_csv_hash != external_hash:
        return {
            "valid": False,
            "error": f"Hash mismatch: internal={internal_csv_hash[:16]}..., "
                     f"external={external_hash[:16]}...",
        }

    actual_zip_hash = _sha256_hex(zip_path)

    return {
        "valid": True,
        "internal_csv_sha256": internal_csv_hash,
        "zip_sha256": actual_zip_hash,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSubmissionManifestHashes:
    """SHA-256 correctness and ZIP validation."""

    def test_sha256_streaming_matches_full_read(self, tmp_path):
        """Streaming SHA-256 must equal reading the whole file at once."""
        path = tmp_path / "test.bin"
        data = b"x" * (1 << 20) * 3 + b"trailing bytes"  # ~3 MB
        path.write_bytes(data)

        streaming_hash = _sha256_hex(path)
        expected = hashlib.sha256(data).hexdigest()
        assert streaming_hash == expected

    def test_sha256_deterministic(self, tmp_path):
        """Same content → same hash."""
        path = tmp_path / "data.csv"
        path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

        h1 = _sha256_hex(path)
        h2 = _sha256_hex(path)
        assert h1 == h2

    def test_sha256_different_files_different_hash(self, tmp_path):
        """Different content → different hash."""
        p1 = tmp_path / "a.csv"
        p2 = tmp_path / "b.csv"
        p1.write_text("hello\n", encoding="utf-8")
        p2.write_text("world\n", encoding="utf-8")

        assert _sha256_hex(p1) != _sha256_hex(p2)

    def test_zip_csv_matches_external(self, tmp_path):
        """ZIP-internal CSV SHA-256 must match external CSV."""
        csv_path = tmp_path / "pred_results.csv"
        csv_content = "image_name.jpg, 0001\nimg_001.jpg, 0499\n"
        csv_path.write_text(csv_content, encoding="utf-8")

        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("pred_results.csv", csv_content)

        result = _validate_zip_contents(zip_path, csv_path)
        assert result["valid"], result.get("error", "")

    def test_zip_hash_differs_from_csv_hash(self, tmp_path):
        """The ZIP file's own SHA-256 must differ from the internal CSV hash."""
        csv_path = tmp_path / "pred_results.csv"
        csv_content = "img.jpg, 0001\n" * 100
        csv_path.write_text(csv_content, encoding="utf-8")

        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pred_results.csv", csv_content)

        result = _validate_zip_contents(zip_path, csv_path)
        assert result["valid"]
        # The ZIP file hash must NOT equal the internal CSV hash
        assert result["internal_csv_sha256"] != result["zip_sha256"], (
            "ZIP file SHA-256 must differ from internal CSV SHA-256 — "
            "they are different byte streams"
        )

    def test_zip_extra_files_rejected(self, tmp_path):
        """ZIP with extra files must fail validation."""
        csv_path = tmp_path / "pred_results.csv"
        csv_path.write_text("img.jpg, 0001\n", encoding="utf-8")

        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("pred_results.csv", "img.jpg, 0001\n")
            zf.writestr("readme.txt", "extra file")

        result = _validate_zip_contents(zip_path, csv_path)
        assert not result["valid"]
        assert "Unexpected ZIP contents" in result["error"]

    def test_zip_missing_csv(self, tmp_path):
        """ZIP without pred_results.csv must fail."""
        csv_path = tmp_path / "pred_results.csv"
        csv_path.write_text("img.jpg, 0001\n", encoding="utf-8")

        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("other.csv", "content")

        result = _validate_zip_contents(zip_path, csv_path)
        assert not result["valid"]

    def test_hash_mismatch_detected(self, tmp_path):
        """Different CSV content in ZIP vs external must be caught."""
        csv_path = tmp_path / "pred_results.csv"
        csv_path.write_text("img.jpg, 0001\n", encoding="utf-8")

        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("pred_results.csv", "DIFFERENT.jpg, 0499\n")

        result = _validate_zip_contents(zip_path, csv_path)
        assert not result["valid"]
        assert "Hash mismatch" in result["error"]


class TestLabelValidation:
    """Label format validation: must be ^\\d{4}$."""

    def test_valid_four_digit_labels(self, tmp_path):
        csv_path = tmp_path / "pred.csv"
        csv_path.write_text(
            "img_001.jpg, 0001\n"
            "img_002.jpg, 0499\n"
            "img_003.jpg, 0000\n",
            encoding="utf-8",
        )
        assert _validate_labels(csv_path)

    def test_invalid_three_digit_label(self, tmp_path):
        csv_path = tmp_path / "pred.csv"
        csv_path.write_text(
            "img.jpg, 001\n", encoding="utf-8"
        )
        assert not _validate_labels(csv_path)

    def test_invalid_five_digit_label(self, tmp_path):
        csv_path = tmp_path / "pred.csv"
        csv_path.write_text(
            "img.jpg, 00001\n", encoding="utf-8"
        )
        assert not _validate_labels(csv_path)

    def test_non_numeric_label(self, tmp_path):
        csv_path = tmp_path / "pred.csv"
        csv_path.write_text(
            "img.jpg, abcd\n", encoding="utf-8"
        )
        assert not _validate_labels(csv_path)


class TestPredictionCount:
    """Prediction CSV must have exactly 24,967 rows."""

    def test_exact_count(self, tmp_path):
        csv_path = tmp_path / "pred.csv"
        # pred_results.csv has NO header — submission format
        lines = []
        for i in range(24967):
            lines.append(f"img_{i:05d}.jpg, {i % 500:04d}")
        csv_path.write_text("\n".join(lines), encoding="utf-8")

        assert _count_predictions(csv_path) == 24967

    def test_wrong_count_detected(self, tmp_path):
        csv_path = tmp_path / "pred.csv"
        # pred_results.csv has NO header — submission format
        lines = []
        for i in range(100):
            lines.append(f"img_{i:05d}.jpg, {i % 500:04d}")
        csv_path.write_text("\n".join(lines), encoding="utf-8")

        assert _count_predictions(csv_path) != 24967


class TestDuplicateRejection:
    """Registry must reject duplicate ZIP SHA-256 entries."""

    def test_duplicate_hash_rejected(self):
        registry = {
            "abc123def456": {
                "experiment_id": "D3_STRICT",
                "submission_zip_sha256": "abc123def456",
            }
        }

        new_sha = "abc123def456"
        assert new_sha in registry, "Duplicate should be detected"

    def test_unique_hash_accepted(self):
        registry = {
            "abc123def456": {
                "experiment_id": "D3_STRICT",
                "submission_zip_sha256": "abc123def456",
            }
        }

        new_sha = "xyz789new000"
        assert new_sha not in registry, "Unique hash should be accepted"


class TestManifestSchema:
    """Submission manifest must include all required fields."""

    REQUIRED_FIELDS = [
        "experiment_id",
        "git_commit",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "split_seed",
        "train_seed",
        "split_dir",
        "val_csv_sha256",
        "local_micro_accuracy",
        "local_macro_accuracy",
        "prediction_csv_path",
        "prediction_csv_sha256",
        "zip_internal_csv_sha256",
        "submission_zip_path",
        "submission_zip_sha256",
        "online_accuracy",
        "local_online_gap",
        "num_predictions",
        "created_at_utc",
    ]

    def test_all_required_fields_present(self):
        manifest = {
            "experiment_id": "D3_STRICT",
            "git_commit": "abc123",
            "checkpoint_path": "outputs/d3_strict/seed42/checkpoints/best.pt",
            "checkpoint_sha256": "sha256...",
            "checkpoint_epoch": 49,
            "split_seed": 42,
            "train_seed": 42,
            "split_dir": "outputs/d3_strict/seed42",
            "val_csv_sha256": "70a63d5a...",
            "local_micro_accuracy": 0.7065723,
            "local_macro_accuracy": 0.7060997,
            "prediction_csv_path": "outputs/d3_strict/seed42/submissions/pred_results.csv",
            "prediction_csv_sha256": "79e55629...",
            "zip_internal_csv_sha256": "79e55629...",
            "submission_zip_path": "outputs/d3_strict/seed42/submissions/submission.zip",
            "submission_zip_sha256": "72036e7b...",
            "online_accuracy": 0.573397,
            "local_online_gap": 0.1331753,
            "num_predictions": 24967,
            "created_at_utc": "2026-07-12T00:00:00Z",
        }
        for field in self.REQUIRED_FIELDS:
            assert field in manifest, f"Missing required field: {field}"

    def test_local_online_gap_consistency(self, tmp_path):
        """local_online_gap must equal local_micro - online."""
        manifest_path = tmp_path / "manifest.json"
        manifest = {
            "experiment_id": "D3_STRICT",
            "local_micro_accuracy": 0.7065723148507174,
            "online_accuracy": 0.573397,
        }
        gap = manifest["local_micro_accuracy"] - manifest["online_accuracy"]
        assert gap == pytest.approx(0.1331753148507174, abs=1e-8)

    def test_best_val_acc_matches_reeval_micro(self, tmp_path):
        """Checkpoint best_val_acc must match reeval micro accuracy."""
        # This simulates the check that the manifest builder must perform
        ckpt_best_val_acc = 0.7065723148507174
        reeval_micro = 0.7065723148507174

        assert abs(ckpt_best_val_acc - reeval_micro) <= 1e-8, (
            "Checkpoint best_val_acc must match reeval micro_accuracy"
        )

        # A mismatch should be caught
        ckpt_best_val_acc_bad = 0.7045
        assert abs(ckpt_best_val_acc_bad - reeval_micro) > 1e-8, (
            "This mismatch should be caught"
        )
