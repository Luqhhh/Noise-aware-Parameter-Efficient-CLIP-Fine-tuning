"""Tests for common.logit_adjustment."""

import csv
import tempfile
from pathlib import Path

import torch

from common.logit_adjustment import (
    adjust_logits,
    compute_class_priors,
    sweep_logit_adjustment,
)

# ---------------------------------------------------------------------------
# 1.  tau=0  → identity
# ---------------------------------------------------------------------------


def test_tau_zero_identity():
    """adjust_logits(logits, priors, tau=0) returns unchanged logits."""
    logits = torch.randn(16, 500)
    priors = torch.ones(500) / 500
    result = adjust_logits(logits, priors, tau=0.0)
    assert isinstance(result, torch.Tensor)
    assert result.shape == logits.shape
    assert torch.equal(result, logits), "tau=0 should return unchanged logits"


# ---------------------------------------------------------------------------
# 2.  tau > 0 with non-uniform priors → high-prior classes reduced
# ---------------------------------------------------------------------------


def test_tau_positive_shifts():
    """With tau>0 and non-uniform priors, logits for high-prior classes are
    reduced more (or at least not increased) relative to low-prior classes."""
    n = 500
    logits = torch.zeros(1, n)  # all equal
    priors = torch.full((n,), 1e-6)
    priors[0] = 0.9  # class 0 is a head class
    # Normalise so they sum to 1 (the function expects proper priors).
    priors = priors / priors.sum()

    result = adjust_logits(logits, priors, tau=2.0)
    adjustment = 2.0 * torch.log(priors + 1e-12)
    expected = logits - adjustment.unsqueeze(0)
    assert torch.allclose(result, expected, atol=1e-6)

    # High-prior class 0 gets a larger negative adjustment.
    diff = result[0, 0] - result[0, 1]
    assert diff < 0, (
        f"Expected logit for high-prior class 0 to be lower than for "
        f"class 1, got diff={diff:.4f}"
    )


# ---------------------------------------------------------------------------
# 3.  compute_class_priors sums to 1
# ---------------------------------------------------------------------------


def test_compute_class_priors_sums_to_one():
    """Priors returned by compute_class_priors sum to 1.0."""
    num_classes = 500
    # Build a temp CSV with uniform distribution.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "class_name"])
        for i in range(1000):
            label = i % num_classes
            writer.writerow([f"img_{i}.jpg", str(label), f"class_{label}"])
        tmp = f.name

    try:
        priors = compute_class_priors(tmp, num_classes=num_classes)
        assert isinstance(priors, torch.Tensor)
        assert priors.shape == (num_classes,)
        assert priors.dtype == torch.float32
        assert abs(priors.sum().item() - 1.0) < 1e-6
    finally:
        Path(tmp).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 4.  compute_class_priors from temp CSV with known counts
# ---------------------------------------------------------------------------


def test_compute_class_priors_from_temp_csv():
    """Verify computed priors match expected proportions for known counts."""
    num_classes = 5
    counts = {0: 50, 1: 30, 2: 10, 3: 7, 4: 3}
    total = sum(counts.values())
    expected_priors = torch.tensor(
        [(c + 1e-12) / (total + 5 * 1e-12) for c in counts.values()],
        dtype=torch.float32,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "class_name"])
        for label, cnt in counts.items():
            for i in range(cnt):
                writer.writerow(
                    [f"img_{label}_{i}.jpg", str(label), f"class_{label}"]
                )
        tmp = f.name

    try:
        priors = compute_class_priors(tmp, num_classes=num_classes)
        assert torch.allclose(priors, expected_priors, atol=1e-6)
    finally:
        Path(tmp).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 5.  sweep returns all taus
# ---------------------------------------------------------------------------


def test_sweep_returns_all_taus():
    """sweep returns a result entry for every tau value provided."""
    num_classes = 500
    n = 32
    logits = torch.randn(n, num_classes)
    labels = torch.randint(0, num_classes, (n,))
    priors = torch.ones(num_classes) / num_classes
    taus = [0.0, 0.5, 1.0, 2.0, 5.0]

    results = sweep_logit_adjustment(logits, labels, priors, taus)
    assert isinstance(results, dict)
    assert set(results.keys()) == set(taus)

    for tau, metrics in results.items():
        assert isinstance(metrics, dict)
        for key in (
            "micro_accuracy",
            "macro_accuracy",
            "median_per_class_accuracy",
            "bottom_10_percent_accuracy",
            "micro_macro_gap",
        ):
            assert key in metrics, f"tau={tau} missing {key}"
            assert isinstance(metrics[key], float)


# ---------------------------------------------------------------------------
# 6.  adjust_logits preserves relative order
# ---------------------------------------------------------------------------


def test_adjust_logits_preserves_relative_order():
    """The per-class shift is identical across samples, so the within-class
    ordering of logits for any class c is preserved after adjustment
    (adjustment subtracts a class-specific constant)."""
    priors = torch.tensor([0.7, 0.2, 0.1])
    logits = torch.tensor([[5.0, 3.0, 1.0], [1.0, 4.0, 2.0], [2.0, 5.0, 0.5]])

    for tau in [0.5, 1.0, 2.0]:
        adjusted = adjust_logits(logits, priors, tau)
        # The adjustment applied is exactly τ·log(π_c+ε) per class.
        expected_adj = tau * torch.log(priors + 1e-12)
        expected = logits - expected_adj.unsqueeze(0)
        assert torch.allclose(adjusted, expected, atol=1e-6), (
            f"tau={tau}: adjustment formula not applied correctly"
        )
        # Within each class c, the relative ordering of logits across
        # samples is unchanged (same bias subtracted from all samples for
        # that class).
        for c in range(3):
            _, orig_idx = logits[:, c].sort()
            _, adj_idx = adjusted[:, c].sort()
            assert torch.equal(orig_idx, adj_idx), (
                f"tau={tau}, class {c}: within-class ordering changed"
            )


# ---------------------------------------------------------------------------
# 7.  epsilon added to priors
# ---------------------------------------------------------------------------


def test_epsilon_added_to_priors():
    """With epsilon > 0, all priors > 0 even for unseen classes."""
    num_classes = 10
    # Only classes 0..4 appear in the CSV.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "class_name"])
        for i in range(50):
            label = i % 5
            writer.writerow([f"img_{i}.jpg", str(label), f"class_{label}"])
        tmp = f.name

    try:
        eps = 1e-8
        priors = compute_class_priors(tmp, num_classes=num_classes, epsilon=eps)
        assert (priors > 0).all(), "All priors should be positive with epsilon"
        assert abs(priors.sum().item() - 1.0) < 1e-6

        # Unseen classes 5..9 should have tiny but nonzero priors.
        unseen_priors = priors[5:]
        assert (unseen_priors > 0).all()
        expected_unseen = eps / (50 + num_classes * eps)
        assert torch.allclose(
            unseen_priors,
            torch.full((5,), expected_unseen, dtype=torch.float32),
            atol=1e-12,
        ), "Unseen classes should all have the same epsilon-smoothed prior"
    finally:
        Path(tmp).unlink(missing_ok=True)
