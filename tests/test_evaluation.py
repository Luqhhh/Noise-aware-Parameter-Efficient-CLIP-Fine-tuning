"""Tests for multi-split evaluation module."""

import json
import tempfile
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.evaluation import (
    load_eval_results,
    compute_paired_deltas,
    apply_candidate_rules,
)


# ---------------------------------------------------------------------------
# load_eval_results
# ---------------------------------------------------------------------------


def test_load_missing_file():
    """Missing file -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_eval_results("/nonexistent/path/eval_results.json")


def test_load_valid_json():
    """Valid JSON file returns parsed dict."""
    data = {"experiment_id": "E0", "best_val_acc": 0.75}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "eval_results.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = load_eval_results(str(path))
        assert result["experiment_id"] == "E0"
        assert result["best_val_acc"] == 0.75


def test_load_invalid_json():
    """Malformed JSON -> JSONDecodeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "eval_results.json"
        with open(path, "w") as f:
            f.write("not valid json{")
        with pytest.raises(json.JSONDecodeError):
            load_eval_results(str(path))


# ---------------------------------------------------------------------------
# compute_paired_deltas
# ---------------------------------------------------------------------------


def test_paired_deltas_basic():
    """Basic delta calculation."""
    e0 = {42: 0.75, 43: 0.73, 44: 0.74}
    candidate = {42: 0.78, 43: 0.72, 44: 0.76}
    result = compute_paired_deltas(e0, candidate)

    assert result["deltas"][42] == pytest.approx(0.03)
    assert result["deltas"][43] == pytest.approx(-0.01)
    assert result["deltas"][44] == pytest.approx(0.02)

    # mean = (0.03 - 0.01 + 0.02) / 3 = 0.013333...
    assert result["mean_delta"] == pytest.approx(0.013333, abs=1e-5)

    # std with ddof=1
    assert isinstance(result["std_delta"], float)

    # min = -0.01
    assert result["min_delta"] == pytest.approx(-0.01)

    # max = 0.03
    assert result["max_delta"] == pytest.approx(0.03)

    # wins: deltas > -0.002: 0.03 > -0.002, -0.01 > -0.002? YES, -0.01 > -0.002 is False
    # Actually -0.01 > -0.002 is False, so only 2 splits win
    assert result["confirmation_wins"] == "2/3"

    assert "pooled_delta" in result


def test_paired_deltas_all_positive():
    """All deltas positive."""
    e0 = {42: 0.70, 43: 0.71}
    candidate = {42: 0.72, 43: 0.73}
    result = compute_paired_deltas(e0, candidate)

    assert all(d > 0 for d in result["deltas"].values())
    assert result["confirmation_wins"] == "2/2"
    assert result["mean_delta"] > 0


def test_paired_deltas_no_shared_seeds():
    """No overlapping seeds -> ValueError."""
    e0 = {42: 0.75}
    candidate = {99: 0.78}
    with pytest.raises(ValueError, match="No shared split seeds"):
        compute_paired_deltas(e0, candidate)


def test_paired_deltas_single_split():
    """Single split: std=0."""
    e0 = {42: 0.75}
    candidate = {42: 0.78}
    result = compute_paired_deltas(e0, candidate)

    assert result["deltas"][42] == pytest.approx(0.03)
    assert result["mean_delta"] == pytest.approx(0.03)
    assert result["std_delta"] == 0.0
    assert result["min_delta"] == result["max_delta"]
    assert result["confirmation_wins"] == "1/1"


def test_paired_deltas_empty_e0():
    """Empty E0 results dict -> ValueError."""
    e0 = {}
    candidate = {42: 0.78}
    with pytest.raises(ValueError, match="No shared split seeds"):
        compute_paired_deltas(e0, candidate)


def test_paired_deltas_exact_wins():
    """Deltas exactly at -0.002 should count as wins (strictly greater)."""
    e0 = {42: 0.750}
    candidate = {42: 0.748}  # delta = -0.002, NOT > -0.002
    result = compute_paired_deltas(e0, candidate)
    assert result["confirmation_wins"] == "0/1"

    # delta = -0.001999 > -0.002 -> count as win
    e0 = {42: 0.750}
    candidate = {42: 0.74801}  # delta ≈ -0.00199
    result = compute_paired_deltas(e0, candidate)
    assert result["confirmation_wins"] == "1/1"


# ---------------------------------------------------------------------------
# apply_candidate_rules
# ---------------------------------------------------------------------------


def test_apply_no_candidates():
    """Empty candidates dict -> fallback to E0."""
    winner, report = apply_candidate_rules({})
    assert winner == "E0"
    assert report["reason"] == "no_candidates_provided"


def test_apply_all_eliminated():
    """All candidates eliminated -> fallback to E0."""
    candidates = {
        "C1": {
            "min_delta": -0.005,
            "mean_delta": 0.001,
        },
    }
    winner, report = apply_candidate_rules(candidates)
    assert winner == "E0"
    assert report["reason"] == "no_candidates_survived"


def test_apply_min_delta_eliminates():
    """Candidate with min_delta <= -0.002 gets eliminated."""
    candidates = {
        "C1": {
            "min_delta": -0.003,
            "mean_delta": 0.01,
        },
    }
    winner, report = apply_candidate_rules(candidates)
    assert winner == "E0"
    assert "min_delta" in report.get("elimination_log", {}).get("C1", "")


def test_apply_mean_delta_eliminates():
    """Candidate with mean_delta <= 0 gets eliminated even if min_delta ok."""
    candidates = {
        "C1": {
            "min_delta": -0.001,
            "mean_delta": 0.0,
        },
    }
    winner, report = apply_candidate_rules(candidates)
    assert winner == "E0"


def test_apply_sole_survivor():
    """Exactly one survivor selects it."""
    candidates = {
        "C1": {
            "min_delta": -0.001,
            "mean_delta": 0.01,
        },
    }
    winner, report = apply_candidate_rules(candidates)
    assert winner == "C1"
    assert report["reason"] == "sole_survivor"


def test_apply_sole_survivor_multiple_eliminated():
    """One survivor among multiple candidates."""
    candidates = {
        "C1": {
            "min_delta": -0.001,
            "mean_delta": 0.01,
        },
        "C2": {
            "min_delta": -0.003,
            "mean_delta": 0.02,
        },
    }
    winner, report = apply_candidate_rules(candidates)
    assert winner == "C1"
    assert report["reason"] == "sole_survivor"


def test_apply_tiebreaker():
    """Candidates within tie threshold use tiebreaker."""
    candidates = {
        "C2": {
            "min_delta": -0.001,
            "mean_delta": 0.0105,
            "std_delta": 0.002,
        },
        "C1": {
            "min_delta": -0.0005,
            "mean_delta": 0.0100,
            "std_delta": 0.001,
        },
    }
    winner, report = apply_candidate_rules(candidates, tie_threshold=0.001)
    assert report["reason"] == "tiebreaker"
    # C1 has higher min_delta and lower std_delta, so it should win
    assert winner == "C1"


def test_apply_higher_mean_delta():
    """Clear winner by higher mean delta."""
    candidates = {
        "C1": {
            "min_delta": -0.001,
            "mean_delta": 0.005,
            "std_delta": 0.002,
        },
        "C2": {
            "min_delta": -0.001,
            "mean_delta": 0.020,
            "std_delta": 0.002,
        },
    }
    winner, report = apply_candidate_rules(candidates, tie_threshold=0.001)
    assert winner == "C2"
    assert report["reason"] == "higher_mean_delta"


def test_apply_custom_thresholds():
    """Custom elimination and tie thresholds."""
    candidates = {
        "C1": {
            "min_delta": -0.001,
            "mean_delta": 0.01,
        },
        "C2": {
            "min_delta": -0.001,
            "mean_delta": 0.011,
        },
    }
    # Use very tight thresholds
    winner, report = apply_candidate_rules(
        candidates, elimination_threshold=-0.0005, tie_threshold=0.005
    )
    assert winner == "E0"  # Both eliminated by min_delta threshold


def test_apply_tie_diff_above_threshold():
    """Diff above tie threshold: higher mean delta wins."""
    candidates = {
        "C1": {
            "min_delta": -0.001,
            "mean_delta": 0.01,
            "std_delta": 0.002,
        },
        "C2": {
            "min_delta": -0.001,
            "mean_delta": 0.02,
            "std_delta": 0.002,
        },
    }
    winner, report = apply_candidate_rules(candidates, tie_threshold=0.005)
    assert winner == "C2"
    assert report["reason"] == "higher_mean_delta"


def test_apply_with_full_report_dicts():
    """Apply rules using full reports from compute_paired_deltas."""
    e0_results = {42: 0.75, 43: 0.73, 44: 0.74}

    c1_results = compute_paired_deltas(e0_results, {42: 0.76, 43: 0.74, 44: 0.75})
    c2_results = compute_paired_deltas(e0_results, {42: 0.73, 43: 0.74, 44: 0.72})

    candidates = {
        "C1": c1_results,
        "C2": c2_results,
    }

    winner, report = apply_candidate_rules(candidates)
    # C1 has all positive deltas, C2 has some negative -> C1 wins
    assert winner == "C1"
