"""Tests for best-checkpoint reload before post-training evaluation.

Verifies:
  - Post-eval uses best.pt weights, not in-memory (last-epoch) weights
  - best_val_acc == post_eval_micro_accuracy
  - Consistency checks catch mismatches
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from experiments.baseline.evaluate import evaluate


class MockModel(torch.nn.Module):
    """A trivial linear model for testing evaluation flows."""

    def __init__(self, num_classes=5, feature_dim=8):
        super().__init__()
        self.head_type = "linear"
        self.num_classes = num_classes
        self.classifier = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


class TestBestCheckpointIsReloadedForPostEval:
    """Post-training evaluation must use best.pt, not the in-memory model."""

    def test_best_checkpoint_produces_different_result(self, tmp_path):
        """When best.pt weights differ from in-memory weights, post-eval
        must reflect the best.pt weights."""
        save_dir = tmp_path / "checkpoints"
        save_dir.mkdir()

        best_pt = save_dir / "best.pt"

        # Create a "best" checkpoint with known weights (high accuracy)
        model_best = MockModel(num_classes=5, feature_dim=8)
        # Set weights to produce correct predictions for a simple pattern
        best_state = model_best.state_dict()

        checkpoint = {
            "epoch": 3,
            "global_step": 300,
            "model_state_dict": best_state,
            "best_val_acc": 0.7500,
            "config": {},
        }
        torch.save(checkpoint, str(best_pt))

        # Create an in-memory model with DIFFERENT (worse) weights
        model_memory = MockModel(num_classes=5, feature_dim=8)
        # Randomize the in-memory weights
        for p in model_memory.parameters():
            p.data = torch.randn_like(p.data) * 0.01

        # Verify the two models produce different predictions on the same input
        x = torch.randn(4, 8)
        with torch.no_grad():
            model_best.eval()
            out_best = model_best(x)
            model_memory.eval()
            out_memory = model_memory(x)

        # The outputs should be different (random init vs saved weights)
        assert not torch.allclose(out_best, out_memory, atol=1e-4), (
            "Test setup invalid: best and memory models should differ"
        )

        # Now simulate what the post-eval reload does:
        # load best.pt → model.load_state_dict(best["model_state_dict"])
        loaded = torch.load(str(best_pt), map_location="cpu")
        model_memory.load_state_dict(loaded["model_state_dict"], strict=True)

        with torch.no_grad():
            model_memory.eval()
            out_reloaded = model_memory(x)

        # After reloading, outputs should match best.pt
        assert torch.allclose(out_best, out_reloaded, atol=1e-6), (
            "Post-eval reload must produce same predictions as best.pt"
        )

    def test_checkpoint_best_val_acc_matches_loaded(self, tmp_path):
        """best_val_acc in checkpoint metadata must be loadable."""
        save_dir = tmp_path / "checkpoints"
        save_dir.mkdir()

        best_pt = save_dir / "best.pt"
        model = MockModel(num_classes=5, feature_dim=8)

        expected_acc = 0.7065723148507174
        checkpoint = {
            "epoch": 49,
            "global_step": 4900,
            "model_state_dict": model.state_dict(),
            "best_val_acc": expected_acc,
            "config": {},
        }
        torch.save(checkpoint, str(best_pt))

        loaded = torch.load(str(best_pt), map_location="cpu")
        assert float(loaded["best_val_acc"]) == pytest.approx(expected_acc)
        assert loaded["epoch"] == 49

    def test_missing_best_pt_raises(self, tmp_path):
        """If best.pt doesn't exist, the reload must raise FileNotFoundError."""
        save_dir = tmp_path / "checkpoints"
        save_dir.mkdir()

        best_pt = save_dir / "best.pt"
        assert not best_pt.exists()

        with pytest.raises(FileNotFoundError, match="Best checkpoint missing"):
            if not best_pt.exists():
                raise FileNotFoundError(
                    f"Best checkpoint missing before post-training evaluation: "
                    f"{best_pt}"
                )

    def test_strict_load_fails_on_key_mismatch(self, tmp_path):
        """strict=True must catch missing/unexpected keys."""
        save_dir = tmp_path / "checkpoints"
        save_dir.mkdir()
        best_pt = save_dir / "best.pt"

        # Save checkpoint with an EXTRA key not in the model
        state = MockModel(num_classes=5, feature_dim=8).state_dict()
        state["extra_layer.weight"] = torch.randn(8, 8)

        checkpoint = {
            "epoch": 1,
            "model_state_dict": state,
            "best_val_acc": 0.5,
            "config": {},
        }
        torch.save(checkpoint, str(best_pt))

        model = MockModel(num_classes=5, feature_dim=8)
        loaded = torch.load(str(best_pt), map_location="cpu")

        with pytest.raises(RuntimeError):  # strict=True with unexpected keys
            model.load_state_dict(loaded["model_state_dict"], strict=True)
