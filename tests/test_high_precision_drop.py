from __future__ import annotations

import pandas as pd
import pytest

from analysis.noisy_labels.high_precision_drop import (
    select_high_precision_drop,
)


def _quality() -> pd.DataFrame:
    rows = []
    hashes = []
    for label in range(4):
        for index in range(4):
            rows.append(
                {
                    "sample_id": f"s{label}_{index}",
                    "image_path": f"class{label}/img{index}.jpg",
                    "original_label": label,
                    "oof_top1": label,
                    "knn_top1": label,
                    "top1_margin": 0.10 + 0.01 * index,
                    "knn_agreement": 0.80,
                    "file_sha256": f"hash-{label}-{index}",
                }
            )
            hashes.append(f"hash-{label}-{index}")
    # Strict-consensus suspect: issue + both wrong + agree + high margin.
    rows[0].update(
        {
            "oof_top1": 1,
            "knn_top1": 1,
            "top1_margin": 2.0,
            "knn_agreement": 0.05,
        }
    )
    # Cross-label exact duplicate conflict: same content, different labels.
    rows[5]["file_sha256"] = "duplicate-hash"
    rows[9]["file_sha256"] = "duplicate-hash"
    frame = pd.DataFrame(rows)
    frame.loc[0, "file_sha256"] = "suspect-hash"
    return frame


def _issues(quality: pd.DataFrame, selected_indices: set[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index": list(range(len(quality))),
            "selected": [index in selected_indices for index in range(len(quality))],
        }
    )


def test_strict_consensus_and_duplicate_conflict_are_rejected() -> None:
    quality = _quality()
    result = select_high_precision_drop(quality, _issues(quality, {0}))
    manifest = result["manifest"]
    audit = result["audit"]

    assert bool(manifest.loc[0, "cl_knn_suspect"]) is True
    assert bool(manifest.loc[0, "cross_label_duplicate_conflict"]) is False
    assert float(manifest.loc[0, "weight"]) == 0.0
    # Samples 5 and 9 form the duplicate conflict and both are zeroed.
    for index in (5, 9):
        assert bool(manifest.loc[index, "cross_label_duplicate_conflict"]) is True
        assert float(manifest.loc[index, "weight"]) == 0.0
    # Everything else keeps the noisy label and weight one.
    for index in (1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15):
        assert float(manifest.loc[index, "weight"]) == 1.0
    assert audit["reject_count"] == 3
    assert audit["suspect_count"] == 1
    assert audit["duplicate_conflict_count"] == 2
    assert audit["samples"] == 16
    assert set(audit).issuperset(
        {
            "reject_rate",
            "per_class_reject_rate_min",
            "per_class_reject_rate_median",
            "per_class_reject_rate_max",
            "head_reject_rate",
            "middle_reject_rate",
            "tail_reject_rate",
        }
    )


def test_explicit_duplicate_flag_is_honoured() -> None:
    quality = _quality()
    quality["duplicate_conflict_flag"] = False
    quality.loc[7, "duplicate_conflict_flag"] = True
    result = select_high_precision_drop(quality, _issues(quality, set()))
    assert float(result["manifest"].loc[7, "weight"]) == 0.0
    assert bool(result["manifest"].loc[7, "cross_label_duplicate_conflict"]) is True


def test_rule_does_not_select_clean_consensus() -> None:
    quality = _quality()
    # OOF and kNN agree with the noisy label: no strict consensus drop.
    result = select_high_precision_drop(quality, _issues(quality, set()))
    assert result["audit"]["suspect_count"] == 0
    assert result["audit"]["reject_count"] == 2  # only the duplicate pair


def test_empty_class_fails_closed() -> None:
    quality = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "image_path": ["class0/a.jpg", "class1/b.jpg"],
            "original_label": [0, 1],
            "oof_top1": [1, 1],
            "knn_top1": [1, 1],
            "top1_margin": [5.0, 0.5],
            "knn_agreement": [0.0, 0.9],
        }
    )
    issues = pd.DataFrame({"index": [0, 1], "selected": [True, False]})
    with pytest.raises(ValueError, match="removes every clean sample"):
        select_high_precision_drop(quality, issues)


def test_missing_columns_fail_closed() -> None:
    quality = _quality().drop(columns=["knn_agreement"])
    with pytest.raises(ValueError, match="missing columns"):
        select_high_precision_drop(quality, _issues(quality, set()))
