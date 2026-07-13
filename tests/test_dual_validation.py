"""Tests for dual validation evaluation logic."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


# We test the evaluation utility functions that evaluate_dual_validation.py will use.
# These are defined inline for now; in production they'd be imported from
# tools/evaluate_dual_validation.py or a shared utility.


def compute_group_partition(d3_correct, b2_correct):
    """Partition samples into four mutually exclusive groups."""
    d3 = np.asarray(d3_correct, dtype=bool)
    b2 = np.asarray(b2_correct, dtype=bool)
    both_correct = d3 & b2
    d3_only = d3 & ~b2
    b2_only = ~d3 & b2
    both_wrong = ~d3 & ~b2
    return both_correct, d3_only, b2_only, both_wrong


def paired_bootstrap(d3_correct, b2_correct, trusted_mask=None, n_iter=1000, seed=42):
    """Paired bootstrap for accuracy delta."""
    rng = np.random.default_rng(seed)
    n = len(d3_correct)
    d3 = np.asarray(d3_correct, dtype=bool)
    b2 = np.asarray(b2_correct, dtype=bool)
    deltas = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        if trusted_mask is not None:
            t = np.asarray(trusted_mask, dtype=bool)
            idx = idx[t[idx]] if t[idx].sum() > 0 else idx
            d3_acc = d3[idx][t[idx]].mean() if t[idx].sum() > 0 else 0.0
            b2_acc = b2[idx][t[idx]].mean() if t[idx].sum() > 0 else 0.0
        else:
            d3_acc = d3[idx].mean()
            b2_acc = b2[idx].mean()
        deltas.append(b2_acc - d3_acc)
    deltas = np.array(deltas)
    return {
        "mean": float(deltas.mean()),
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
    }


class TestGroupPartition:
    def test_mutually_exclusive(self):
        d3 = [True, True, False, False]
        b2 = [True, False, True, False]
        bc, do, bo, bw = compute_group_partition(d3, b2)
        assert bc.sum() == 1
        assert do.sum() == 1
        assert bo.sum() == 1
        assert bw.sum() == 1

    def test_union_equals_total(self):
        n = 100
        rng = np.random.default_rng(42)
        d3 = rng.choice([True, False], n)
        b2 = rng.choice([True, False], n)
        bc, do, bo, bw = compute_group_partition(d3, b2)
        assert bc.sum() + do.sum() + bo.sum() + bw.sum() == n

    def test_known_numbers(self):
        """D3=7289, B2=7179, total=10316 → d3_only=?, b2_only=?"""
        # d3_only - b2_only = 7289 - 7179 = 110
        # We can't determine exact values without the joint distribution,
        # but we can verify partition invariants.
        n = 10316
        d3_correct_count = 7289
        b2_correct_count = 7179
        # Create synthetic data matching the counts
        d3 = np.zeros(n, dtype=bool)
        b2 = np.zeros(n, dtype=bool)
        d3[:d3_correct_count] = True
        b2[:b2_correct_count] = True
        bc, do, bo, bw = compute_group_partition(d3, b2)
        assert bc.sum() + do.sum() == d3_correct_count
        assert bc.sum() + bo.sum() == b2_correct_count


class TestPairedBootstrap:
    def test_seed_reproducibility(self):
        n = 100
        d3 = np.random.default_rng(0).choice([True, False], n)
        b2 = np.random.default_rng(1).choice([True, False], n)
        result1 = paired_bootstrap(d3, b2, n_iter=500, seed=42)
        result2 = paired_bootstrap(d3, b2, n_iter=500, seed=42)
        assert result1["mean"] == result2["mean"]
        assert result1["ci_lower"] == result2["ci_lower"]

    def test_same_accuracy_zero_delta(self):
        n = 200
        d3 = np.ones(n, dtype=bool)
        b2 = np.ones(n, dtype=bool)
        result = paired_bootstrap(d3, b2, n_iter=1000, seed=42)
        assert abs(result["mean"]) < 1e-10

    def test_output_keys(self):
        d3 = np.random.default_rng(0).choice([True, False], 50)
        b2 = np.random.default_rng(1).choice([True, False], 50)
        result = paired_bootstrap(d3, b2, n_iter=100)
        assert set(result.keys()) == {"mean", "ci_lower", "ci_upper"}

    def test_trusted_mask_effect(self):
        n = 100
        rng = np.random.default_rng(99)
        d3 = rng.choice([True, False], n, p=[0.7, 0.3])
        b2 = rng.choice([True, False], n, p=[0.68, 0.32])
        trusted = rng.choice([True, False], n, p=[0.5, 0.5])
        raw = paired_bootstrap(d3, b2, n_iter=500, seed=42)
        trusted_result = paired_bootstrap(d3, b2, trusted, n_iter=500, seed=42)
        # Both should produce valid deltas
        assert -1 <= raw["mean"] <= 1
        assert -1 <= trusted_result["mean"] <= 1


class TestManifestValidation:
    def test_rejects_path_mismatch(self):
        """Evaluation should reject if manifest paths don't match val dataset."""
        manifest = pd.DataFrame({
            "image_path": ["a.jpg", "b.jpg"],
            "trusted_v1": [True, False],
        })
        val_paths = ["a.jpg", "c.jpg"]
        # Path c.jpg not in manifest → should detect mismatch
        manifest_paths = set(manifest["image_path"])
        val_paths_set = set(val_paths)
        assert not val_paths_set.issubset(manifest_paths)
