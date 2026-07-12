"""Tests for the fresh-run artifact guard (_prepare_fresh_run_artifacts).

Verifies:
  - Fresh run with existing artifacts raises FileExistsError
  - --allow-overwrite removes stale artifacts
  - Resume keeps existing log intact
  - Resume with missing checkpoint raises FileNotFoundError
"""

import tempfile
from pathlib import Path

import pytest

from experiments.baseline.train import _prepare_fresh_run_artifacts


class TestFreshRunRefusesExistingArtifacts:
    """Non-resume run must refuse when generated files already exist."""

    def test_all_artifacts_present_raises(self, tmp_path):
        save_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"
        save_dir.mkdir()
        log_dir.mkdir()

        # Create all generated artifacts
        (save_dir / "best.pt").touch()
        (save_dir / "last.pt").touch()
        (save_dir / "eval_results.json").touch()
        (save_dir / "config_snapshot.yaml").touch()
        (log_dir / "train_log.csv").touch()

        with pytest.raises(FileExistsError, match="Fresh run refused"):
            _prepare_fresh_run_artifacts(
                save_dir=save_dir,
                log_dir=log_dir,
                resume_path=None,
                allow_overwrite=False,
            )

    def test_partial_artifacts_raises(self, tmp_path):
        save_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"
        save_dir.mkdir()
        log_dir.mkdir()

        # Only train_log.csv exists
        (log_dir / "train_log.csv").touch()

        with pytest.raises(FileExistsError, match="Fresh run refused"):
            _prepare_fresh_run_artifacts(
                save_dir=save_dir,
                log_dir=log_dir,
                resume_path=None,
                allow_overwrite=False,
            )

    def test_empty_directories_pass(self, tmp_path):
        save_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"
        save_dir.mkdir()
        log_dir.mkdir()

        # Should not raise - directories are empty
        _prepare_fresh_run_artifacts(
            save_dir=save_dir,
            log_dir=log_dir,
            resume_path=None,
            allow_overwrite=False,
        )

    def test_allow_overwrite_removes_artifacts(self, tmp_path):
        save_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"
        save_dir.mkdir()
        log_dir.mkdir()

        best_pt = save_dir / "best.pt"
        log_csv = log_dir / "train_log.csv"
        best_pt.touch()
        log_csv.touch()

        assert best_pt.exists()
        assert log_csv.exists()

        _prepare_fresh_run_artifacts(
            save_dir=save_dir,
            log_dir=log_dir,
            resume_path=None,
            allow_overwrite=True,
        )

        assert not best_pt.exists()
        assert not log_csv.exists()


class TestResumeKeepsExistingLog:
    """Resume path must bypass the artifact guard entirely."""

    def test_resume_with_existing_log_passes(self, tmp_path):
        save_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"
        save_dir.mkdir()
        log_dir.mkdir()

        # Create artifacts including a fake resume checkpoint
        resume_ckpt = save_dir / "last.pt"
        resume_ckpt.touch()
        (log_dir / "train_log.csv").touch()

        # Should not raise - resume bypasses the guard
        _prepare_fresh_run_artifacts(
            save_dir=save_dir,
            log_dir=log_dir,
            resume_path=str(resume_ckpt),
            allow_overwrite=False,
        )

        # train_log.csv must NOT have been deleted
        assert (log_dir / "train_log.csv").exists()

    def test_resume_missing_checkpoint_raises(self, tmp_path):
        save_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"
        save_dir.mkdir()
        log_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
            _prepare_fresh_run_artifacts(
                save_dir=save_dir,
                log_dir=log_dir,
                resume_path="/nonexistent/path/checkpoint.pt",
                allow_overwrite=False,
            )

    def test_resume_does_not_delete_artifacts(self, tmp_path):
        save_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"
        save_dir.mkdir()
        log_dir.mkdir()

        best_pt = save_dir / "best.pt"
        last_pt = save_dir / "last.pt"
        eval_json = save_dir / "eval_results.json"
        train_csv = log_dir / "train_log.csv"

        best_pt.touch()
        last_pt.touch()
        eval_json.touch()
        train_csv.touch()

        _prepare_fresh_run_artifacts(
            save_dir=save_dir,
            log_dir=log_dir,
            resume_path=str(last_pt),
            allow_overwrite=False,
        )

        assert best_pt.exists()
        assert last_pt.exists()
        assert eval_json.exists()
        assert train_csv.exists()
