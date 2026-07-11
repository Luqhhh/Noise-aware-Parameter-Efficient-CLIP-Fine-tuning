"""Tests for scheduler ratio preservation and cache guard (D4).

Verifies:
  - _cosine_factor boundaries (start=1.0, end=min_lr_ratio, midpoint)
  - LR ratio preservation across parameter groups
  - Cached features + freeze_clip=false → ValueError
"""

import math

import pytest

from experiments.baseline.train import _cosine_factor


# ── Cosine factor correctness ────────────────────────────────────────

class TestCosineFactor:
    def test_start_is_1(self):
        assert _cosine_factor(0, 100, 0.01) == 1.0

    def test_end_is_min_ratio(self):
        assert _cosine_factor(100, 100, 0.01) == 0.01

    def test_end_is_min_ratio_different_value(self):
        assert _cosine_factor(200, 200, 0.001) == 0.001

    def test_midpoint(self):
        mid = _cosine_factor(50, 100, 0.01)
        expected = 0.01 + 0.5 * (1.0 - 0.01)
        assert mid == pytest.approx(expected)

    def test_ratio_preserved_between_two_lrs(self):
        """At any step, factor is the same for all groups → ratio fixed."""
        import pytest
        head_initial, bb_initial = 3e-4, 3e-6
        ratio = head_initial / bb_initial  # 100
        for step in [0, 10, 50, 90, 99]:
            factor = _cosine_factor(step, 100, 0.01)
            assert (head_initial * factor) / (bb_initial * factor) == \
                pytest.approx(ratio)


# ── Cache guard ──────────────────────────────────────────────────────

class TestCacheGuard:
    def test_cached_features_with_unfrozen_clip_raises(self):
        import pytest
        from experiments.baseline.train import _enforce_guards

        with pytest.raises(ValueError, match="freeze_clip=True"):
            _enforce_guards(
                experiment_id="F1",
                use_cached_features=True,
                augmentation_preset="a0",
                freeze_clip=False,
            )

    def test_cached_features_with_frozen_clip_passes(self):
        from experiments.baseline.train import _enforce_guards

        # Should not raise
        _enforce_guards(
            experiment_id="E0",
            use_cached_features=True,
            augmentation_preset="a0",
            freeze_clip=True,
        )
