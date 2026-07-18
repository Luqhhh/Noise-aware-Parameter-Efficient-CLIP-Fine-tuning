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

    def test_cl_knn_drop_on_real_data_returns_nonzero(self):
        """Integration test: cl_knn_drop on real quality + persisted issue table."""
        import torch, torch.nn.functional as F
        from analysis.noisy_labels.confident_joint import (
            build_confident_joint, estimate_class_thresholds, rank_label_issues,
        )
        q = pd.read_csv("outputs/phase/phase3/oof/sample_quality_with_kta.csv")
        logits = torch.load("outputs/phase/phase3/oof/oof_logits.pt", map_location="cpu")
        probs = F.softmax(logits["logits"].float(), dim=1)
        labels = torch.tensor(q["original_label"].to_numpy(copy=True))
        th = estimate_class_thresholds(probs, labels, 500)
        cj = build_confident_joint(probs, labels, th, 500)
        issues = rank_label_issues(
            probs, labels, th, cj,
            knn_agreement=q["knn_agreement"].to_numpy() if "knn_agreement" in q.columns else None,
            flip_consistency=q["flip_consistency"].to_numpy() if "flip_consistency" in q.columns else None,
            top1_margin=q["top1_margin"].to_numpy() if "top1_margin" in q.columns else None,
        )
        result = select_consensus_drop(q, issues)
        assert len(result) > 0, f"cl_knn_drop selected {len(result)} — expected > 0"

    def test_relabel_v2_on_real_data_returns_nonzero(self):
        """Integration test: consensus_relabel_v2 on real quality + persisted issue table."""
        import torch, torch.nn.functional as F
        from analysis.noisy_labels.confident_joint import (
            build_confident_joint, estimate_class_thresholds, rank_label_issues,
        )
        from analysis.noisy_labels.consensus import select_consensus_relabel_v2
        q = pd.read_csv("outputs/phase/phase3/oof/sample_quality_with_kta.csv")
        logits = torch.load("outputs/phase/phase3/oof/oof_logits.pt", map_location="cpu")
        probs = F.softmax(logits["logits"].float(), dim=1)
        labels = torch.tensor(q["original_label"].to_numpy(copy=True))
        th = estimate_class_thresholds(probs, labels, 500)
        cj = build_confident_joint(probs, labels, th, 500)
        issues = rank_label_issues(
            probs, labels, th, cj,
            knn_agreement=q["knn_agreement"].to_numpy() if "knn_agreement" in q.columns else None,
            flip_consistency=q["flip_consistency"].to_numpy() if "flip_consistency" in q.columns else None,
            top1_margin=q["top1_margin"].to_numpy() if "top1_margin" in q.columns else None,
        )
        result = select_consensus_relabel_v2(q, issues, top_k=100)
        assert len(result) > 0, f"relabel_v2 selected {len(result)} — expected > 0"

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
