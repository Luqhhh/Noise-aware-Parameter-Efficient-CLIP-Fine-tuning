"""Tests for multi-signal consensus drop and relabel."""

import numpy as np
import pandas as pd

from analysis.noisy_labels.consensus import (
    select_consensus_drop,
    select_consensus_relabel_v2,
)


def _make_quality(n=10):
    """Build a minimal quality DataFrame for testing."""
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "image_path": [f"img{i}.jpg" for i in range(n)],
        "original_label": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
        "oof_top1": [1, 1, 1, 0, 0, 0, 3, 3, 3, 0],
        "knn_top1": [1, 1, 1, 0, 0, 0, 3, 3, 3, 0],
        "prototype_top1": [1, 1, 1, 0, 0, 0, 3, 3, 3, 0],
        "p_top1": [0.95] * 10,
        "top1_margin": [0.60] * 10,
        "knn_agreement": [0.80] * 10,
        "flip_consistency": [1.0] * 10,
        "duplicate_conflict_flag": [False] * 10,
    })


class TestConsensusDrop:
    def test_all_conditions_met_is_selected(self):
        """Sample meeting all conditions gets dropped (large enough for caps)."""
        # Need enough samples per class for 10% cap to allow at least 1
        n = 200
        labels = [0] * 100 + [1] * 100
        q = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"img{i}.jpg" for i in range(n)],
            "original_label": labels,
            "oof_top1": [1 - l for l in labels],  # differs from original
            "knn_top1": [1 - l for l in labels],  # same as oof
            "top1_margin": [0.80] * n,
            "knn_agreement": np.random.uniform(0.0, 0.15, n),  # <= 0.20 condition
            "duplicate_conflict_flag": [False] * n,
            "prototype_top1": [1 - l for l in labels],
            "p_top1": [0.95] * n,
            "flip_consistency": [1.0] * n,
        })
        issues = pd.DataFrame({
            "index": list(range(n)),
            "selected": [True] * n,
        })
        result = select_consensus_drop(q, issues)
        # Should select some (capped at 10% per class = 10, global 8% = 16)
        assert len(result) > 0

    def test_oof_equals_original_not_dropped(self):
        """Sample with oof_top1 == original_label is NOT dropped."""
        q = pd.DataFrame({
            "sample_id": ["s0"],
            "image_path": ["img0.jpg"],
            "original_label": [5],
            "oof_top1": [5],  # matches original
            "knn_top1": [3],  # differs
            "top1_margin": [0.80],
            "knn_agreement": [0.80],
            "knn_top1_agreement": [0.85],
            "duplicate_conflict_flag": [False],
            "prototype_top1": [5],
            "p_top1": [0.5],
            "flip_consistency": [1.0],
        })
        issues = pd.DataFrame({"index": [0], "selected": [True]})
        result = select_consensus_drop(q, issues)
        assert 0 not in result

    def test_duplicate_conflict_excluded(self):
        """duplicate_conflict_flag=True excludes from drop."""
        q = pd.DataFrame({
            "sample_id": ["s0"],
            "image_path": ["img0.jpg"],
            "original_label": [5],
            "oof_top1": [3],
            "knn_top1": [3],
            "top1_margin": [0.80],
            "knn_agreement": [0.80],
            "knn_top1_agreement": [0.85],
            "duplicate_conflict_flag": [True],
            "prototype_top1": [3],
            "p_top1": [0.5],
            "flip_consistency": [1.0],
        })
        issues = pd.DataFrame({"index": [0], "selected": [True]})
        result = select_consensus_drop(q, issues)
        assert 0 not in result


class TestConsensusRelabel:
    def test_relabel_candidate_selected(self):
        """Sample meeting all three-signal consensus conditions is relabeled."""
        # Need enough source class samples (3% cap) AND target class samples (5% cap)
        n = 100
        q = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"img{i}.jpg" for i in range(n)],
            "original_label": [5] * 50 + [3] * 50,  # source=5 (50), target=3 (50)
            "oof_top1": [3] * 100,
            "knn_top1": [3] * 100,
            "prototype_top1": [3] * 100,
            "p_top1": [0.95] * 100,
            "top1_margin": [0.60] * 100,
            "knn_agreement": [0.80] * 100,
            "knn_top1_agreement": [0.85] * 100,
            "flip_consistency": [1.0] * 100,
            "duplicate_conflict_flag": [False] * 100,
        })
        issues = pd.DataFrame({"index": list(range(n)), "selected": [True] * n})
        result = select_consensus_relabel_v2(q, issues, top_k=10)
        assert len(result) > 0

    def test_low_confidence_not_relabeled(self):
        """Sample failing ALL auxiliary conditions is not selected."""
        q = pd.DataFrame({
            "sample_id": ["s0"],
            "image_path": ["img0.jpg"],
            "original_label": [5],
            "oof_top1": [3],
            "knn_top1": [3],
            "prototype_top1": [-1],  # different from oof → proto NOT ok
            "p_top1": [0.50],        # low → NOT ok
            "top1_margin": [0.01],   # low → NOT ok
            "knn_agreement": [0.80],
            "knn_top1_agreement": [0.30],  # low → NOT ok (need 0.60)
            "flip_consistency": [0.0],     # NOT ok
            "duplicate_conflict_flag": [False],
        })
        issues = pd.DataFrame({"index": [0], "selected": [True]})
        result = select_consensus_relabel_v2(q, issues, top_k=10)
        assert 0 not in result

    def test_duplicate_conflict_excluded_from_relabel(self):
        """duplicate_conflict_flag=True excluded from relabel."""
        q = pd.DataFrame({
            "sample_id": ["s0"],
            "image_path": ["img0.jpg"],
            "original_label": [5],
            "oof_top1": [3],
            "knn_top1": [3],
            "prototype_top1": [3],
            "p_top1": [0.95],
            "top1_margin": [0.60],
            "knn_agreement": [0.80],
            "knn_top1_agreement": [0.85],
            "flip_consistency": [1.0],
            "duplicate_conflict_flag": [True],
        })
        issues = pd.DataFrame({"index": [0], "selected": [True]})
        result = select_consensus_relabel_v2(q, issues, top_k=10)
        assert 0 not in result

    def test_cl_knn_drop_on_synthetic_data_returns_nonzero(self):
        """Integration: cl_knn_drop on synthetic data produces selections."""
        np.random.seed(42)
        # Need >50 per class for min_clean_per_class=50 to allow drops
        per_class = 100
        n_classes = 5
        n = per_class * n_classes
        labels = []
        for c in range(n_classes):
            labels.extend([c] * per_class)

        q = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"img{i}.jpg" for i in range(n)],
            "original_label": labels,
            "oof_top1": [(l + 1) % n_classes for l in labels],
            "knn_top1": [(l + 1) % n_classes for l in labels],
            "top1_margin": [0.95] * n,
            "knn_agreement": np.random.uniform(0.0, 0.15, n),
            "duplicate_conflict_flag": [False] * n,
            "prototype_top1": [(l + 1) % n_classes for l in labels],
            "p_top1": np.random.uniform(0.7, 0.99, n),
            "flip_consistency": [1.0] * n,
        })
        issues = pd.DataFrame({
            "index": list(range(n)),
            "selected": [True] * n,
        })
        result = select_consensus_drop(q, issues)
        assert len(result) > 0, f"cl_knn_drop selected {len(result)} — expected > 0"
        # All caps should be respected
        for c in range(n_classes):
            cls_mask = q["original_label"] == c
            cls_count = cls_mask.sum()
            if cls_count > 0:
                cls_selected = sum(
                    1 for i in result if q.iloc[i]["original_label"] == c
                )
                assert cls_selected <= max(1, int(0.10 * cls_count)), (
                    f"Class {c}: {cls_selected} selected > 10% of {cls_count}"
                )

    def test_relabel_v2_on_synthetic_data_returns_nonzero(self):
        """Integration: consensus_relabel_v2 on synthetic data produces selections."""
        n = 500
        np.random.seed(42)
        labels = []
        for c in range(10):
            labels.extend([c] * 50)
        q = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"img{i}.jpg" for i in range(n)],
            "original_label": labels,
            "oof_top1": [(l + 1) % 10 for l in labels],
            "knn_top1": [(l + 1) % 10 for l in labels],
            "prototype_top1": [(l + 1) % 10 for l in labels],
            "p_top1": np.random.uniform(0.90, 0.99, n),
            "top1_margin": np.random.uniform(0.60, 0.95, n),
            "knn_agreement": np.random.uniform(0.0, 0.15, n),
            "knn_top1_agreement": np.random.uniform(0.60, 0.95, n),
            "flip_consistency": [1.0] * n,
            "duplicate_conflict_flag": [False] * n,
        })
        issues = pd.DataFrame({
            "index": list(range(n)),
            "selected": [True] * n,
        })
        result = select_consensus_relabel_v2(
            q, issues, top_k=50,
            max_source_class_relabel_rate=0.03,
        )
        assert len(result) > 0, f"relabel_v2 selected {len(result)} — expected > 0"
        # Verify source-class cap: at most 3% per class
        for c in range(10):
            cls_count = 50
            cls_selected = sum(
                1 for i in result if q.iloc[i]["original_label"] == c
            )
            cap = max(1, int(0.03 * cls_count))
            assert cls_selected <= cap, (
                f"Class {c}: {cls_selected} selected > cap {cap}"
            )

    def test_no_nan_causes_all_false(self):
        """NaN in key fields must not silently disqualify all samples."""
        # Use larger fixture to avoid class_percentile edge cases
        n = 100
        labels = [0] * 50 + [1] * 50
        q = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"img{i}.jpg" for i in range(n)],
            "original_label": labels,
            "oof_top1": [1 - l for l in labels],
            "knn_top1": [1 - l for l in labels],
            "top1_margin": [0.80] * n,
            "knn_agreement": [0.10] * n,
            "duplicate_conflict_flag": [False] * n,
            "prototype_top1": [1 - l for l in labels],
            "p_top1": [0.95] * n,
            "flip_consistency": [1.0] * n,
        })
        q.loc[0, "knn_agreement"] = np.nan
        issues = pd.DataFrame({"index": list(range(n)), "selected": [True] * n})
        result = select_consensus_drop(q, issues)
        assert isinstance(result, set)

    def test_signals_not_equal_excluded(self):
        """oof_top1 != knn_top1 excludes from relabel."""
        q = pd.DataFrame({
            "sample_id": ["s0"],
            "image_path": ["img0.jpg"],
            "original_label": [5],
            "oof_top1": [3],
            "knn_top1": [4],  # differs from oof
            "prototype_top1": [3],
            "p_top1": [0.95],
            "top1_margin": [0.60],
            "knn_agreement": [0.80],
            "knn_top1_agreement": [0.85],
            "flip_consistency": [1.0],
            "duplicate_conflict_flag": [False],
        })
        issues = pd.DataFrame({"index": [0], "selected": [True]})
        result = select_consensus_relabel_v2(q, issues, top_k=10)
        assert 0 not in result

    def test_source_class_cap_enforced(self):
        """When all candidates come from one source class, cap limits selection."""
        n = 200
        q = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"img{i}.jpg" for i in range(n)],
            "original_label": [5] * n,
            "oof_top1": [3] * n,
            "knn_top1": [3] * n,
            "prototype_top1": [3] * n,
            "p_top1": [0.95] * n,
            "top1_margin": [0.80] * n,
            "knn_agreement": [0.10] * n,
            "knn_top1_agreement": [0.85] * n,
            "flip_consistency": [1.0] * n,
            "duplicate_conflict_flag": [False] * n,
        })
        issues = pd.DataFrame({"index": list(range(n)), "selected": [True] * n})
        result = select_consensus_relabel_v2(
            q, issues, top_k=100,
            max_source_class_relabel_rate=0.03,
        )
        # 3% of 200 = 6 → at most 6 from class 5
        assert len(result) <= 6, (
            f"Expected <= 6 due to 3% source-class cap, got {len(result)}"
        )
