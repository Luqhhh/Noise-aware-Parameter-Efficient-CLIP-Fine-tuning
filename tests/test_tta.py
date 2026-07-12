"""Tests for Test-Time Augmentation (TTA) logic using horizontal flips."""

import torch


def horizontal_flip(x: torch.Tensor) -> torch.Tensor:
    """Flip a batch of (C, H, W) images horizontally."""
    return x.flip(-1)


def fuse_logits(orig: torch.Tensor, aug: torch.Tensor) -> torch.Tensor:
    """Average logits from original and augmented passes."""
    return (orig + aug) / 2.0


def prediction_change_rate(
    baseline_logits: torch.Tensor, tta_logits: torch.Tensor
) -> float:
    """Fraction of samples whose argmax class changes after TTA fusion."""
    if baseline_logits.shape != tta_logits.shape:
        raise ValueError("Logit tensors must have the same shape")
    baseline_preds = baseline_logits.argmax(dim=1)
    tta_preds = tta_logits.argmax(dim=1)
    changed = (baseline_preds != tta_preds).sum().item()
    total = baseline_preds.size(0)
    return changed / total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_horizontal_flip_idempotent():
    """Flipping a tensor twice returns the original."""
    batch_size, channels, height, width = 4, 3, 224, 224
    x = torch.randn(batch_size, channels, height, width)
    flipped_once = horizontal_flip(x)
    flipped_twice = horizontal_flip(flipped_once)
    assert torch.equal(flipped_twice, x), "Double flip should return original"


def test_tta_fusion_shape():
    """Fused logits have the same shape as individual passes."""
    batch_size, num_classes = 8, 500
    logits_orig = torch.randn(batch_size, num_classes)
    logits_flip = torch.randn(batch_size, num_classes)
    fused = fuse_logits(logits_orig, logits_flip)

    assert logits_orig.shape == (batch_size, num_classes)
    assert logits_flip.shape == (batch_size, num_classes)
    assert fused.shape == (batch_size, num_classes)
    assert logits_orig.shape == fused.shape


def test_tta_zero_change_for_symmetric():
    """If model predictions are unchanged by flipping, change rate is 0."""
    batch_size, num_classes = 10, 500
    logits = torch.randn(batch_size, num_classes)
    baseline_preds = logits.argmax(dim=1)
    tta_preds = baseline_preds.clone()

    # Simulate constant classifier: TTA logits identical to baseline
    rate = prediction_change_rate(logits, logits)
    assert rate == 0.0, (
        f"Identical baseline and TTA logits should yield 0 change rate, "
        f"got {rate}"
    )


def test_prediction_change_rate_computation():
    """Known baseline and TTA predictions produce the correct change rate."""
    batch_size, num_classes = 5, 4
    # Each row: baseline_logits, then TTA logits where some rows differ.
    rng = torch.manual_seed(42)

    # Baseline: all samples have identical strength per class
    baseline_logits = torch.tensor(
        [
            [1.0, 2.0, 0.5, 0.1],  # argmax=1
            [3.0, 1.0, 2.0, 0.0],  # argmax=0
            [0.0, 0.5, 1.0, 2.0],  # argmax=3
            [4.0, 3.0, 2.0, 1.0],  # argmax=0
            [0.1, 0.2, 3.0, 0.3],  # argmax=2
        ]
    )
    # TTA logits: samples 0, 2, 4 kept the same argmax; 1 and 3 changed.
    tta_logits = torch.tensor(
        [
            [1.2, 2.5, 0.3, 0.1],  # argmax=1 (unchanged)
            [1.0, 2.0, 2.5, 0.0],  # argmax=2 (changed from 0)
            [0.0, 0.5, 1.0, 2.5],  # argmax=3 (unchanged)
            [2.0, 4.0, 2.0, 1.0],  # argmax=1 (changed from 0)
            [0.2, 0.3, 3.0, 0.4],  # argmax=2 (unchanged)
        ]
    )

    rate = prediction_change_rate(baseline_logits, tta_logits)
    expected = 2 / 5  # 2 out of 5 samples changed class
    assert rate == expected, (
        f"Expected change rate {expected}, got {rate}"
    )


def test_fuse_logits_value():
    """Fused logits are exactly the elementwise mean of originals."""
    logits_orig = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    logits_flip = torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
    expected = torch.tensor([[3.5, 3.5, 3.5], [3.5, 3.5, 3.5]])
    result = fuse_logits(logits_orig, logits_flip)
    assert torch.equal(result, expected), (
        f"Expected {expected}, got {result}"
    )


def test_prediction_change_rate_all_changed():
    """When all argmax classes differ, change rate is 1.0."""
    baseline = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    tta = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    rate = prediction_change_rate(baseline, tta)
    assert rate == 1.0, f"Expected 1.0, got {rate}"


def test_prediction_change_rate_shape_mismatch():
    """Mismatched logit shapes should raise ValueError."""
    a = torch.randn(3, 5)
    b = torch.randn(4, 5)
    import pytest

    with pytest.raises(ValueError, match="must have the same shape"):
        prediction_change_rate(a, b)


def test_horizontal_flip_pixel_values():
    """Horizontal flip mirrors pixel positions but preserves values."""
    x = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ]
    )
    flipped = horizontal_flip(x)
    expected = torch.tensor(
        [
            [
                [3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0],
                [9.0, 8.0, 7.0],
            ]
        ]
    )
    assert torch.equal(flipped, expected), (
        f"Expected {expected}, got {flipped}"
    )
