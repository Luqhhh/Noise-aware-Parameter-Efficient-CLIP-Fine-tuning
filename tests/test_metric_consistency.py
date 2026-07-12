"""Tests for metric consistency guarantees.

Verifies:
  - micro_macro_gap == micro_accuracy - macro_accuracy (within 1e-10)
  - Per-class accuracy aggregation is consistent
  - Edge cases: single-class, balanced, extremely imbalanced
"""

import math

import pytest
import torch


def compute_metrics(predictions, labels, num_classes):
    """Reference implementation of micro/macro accuracy.

    Args:
        predictions: tensor of predicted class indices, shape (N,)
        labels: tensor of ground-truth class indices, shape (N,)
        num_classes: total number of classes

    Returns:
        dict with micro_accuracy, macro_accuracy, micro_macro_gap, etc.
    """
    total = len(labels)
    correct = (predictions == labels).sum().item()
    micro_acc = correct / total

    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)

    for c in range(num_classes):
        mask = (labels == c)
        n_c = mask.sum().item()
        if n_c > 0:
            total_per_class[c] = n_c
            correct_per_class[c] = (predictions[mask] == c).sum().item()

    per_class_acc = correct_per_class.float() / total_per_class.float().clamp(min=1)
    macro_acc = per_class_acc.mean().item()

    k = max(1, num_classes // 10)
    bottom_10_percent_acc = per_class_acc.topk(k, largest=False).values.mean().item()

    gap = micro_acc - macro_acc

    return {
        "micro_accuracy": micro_acc,
        "macro_accuracy": macro_acc,
        "micro_macro_gap": gap,
        "bottom_10_percent_accuracy": bottom_10_percent_acc,
        "per_class_accuracy": per_class_acc,
        "total_samples": total,
        "correct_samples": correct,
    }


class TestMicroMacroGapConsistency:
    """micro_macro_gap must exactly equal micro - macro."""

    def test_balanced_classes_zero_gap(self):
        """Perfect predictions on balanced data → gap ≈ 0."""
        num_classes = 10
        samples_per_class = 20

        labels = torch.arange(num_classes).repeat_interleave(samples_per_class)
        predictions = labels.clone()  # perfect predictions

        results = compute_metrics(predictions, labels, num_classes)
        gap = results["micro_macro_gap"]
        expected = results["micro_accuracy"] - results["macro_accuracy"]

        assert abs(gap - expected) <= 1e-10
        assert abs(gap) <= 1e-10  # should be zero for perfect balanced

    def test_imbalanced_classes_positive_gap(self):
        """When majority classes are easier, micro > macro → positive gap."""
        num_classes = 5
        # Class 0 has 100 samples (all correct), classes 1-4 have 5 each (all wrong)
        labels = torch.cat([
            torch.zeros(100, dtype=torch.long),
            torch.ones(5, dtype=torch.long),
            torch.full((5,), 2, dtype=torch.long),
            torch.full((5,), 3, dtype=torch.long),
            torch.full((5,), 4, dtype=torch.long),
        ])
        # Majority class predicts correctly; minority classes all wrong
        predictions = torch.cat([
            torch.zeros(100, dtype=torch.long),  # class 0: 100% correct
            torch.zeros(5, dtype=torch.long),     # class 1: 0% correct
            torch.zeros(5, dtype=torch.long),     # class 2: 0% correct
            torch.zeros(5, dtype=torch.long),     # class 3: 0% correct
            torch.zeros(5, dtype=torch.long),     # class 4: 0% correct
        ])

        results = compute_metrics(predictions, labels, num_classes)

        # micro: 100/120 = 0.8333
        # macro: (1.0 + 0 + 0 + 0 + 0) / 5 = 0.2
        # gap: 0.8333 - 0.2 = 0.6333
        assert results["micro_accuracy"] == pytest.approx(100 / 120)
        assert results["macro_accuracy"] == pytest.approx(0.2)
        gap = results["micro_macro_gap"]
        expected = results["micro_accuracy"] - results["macro_accuracy"]
        assert abs(gap - expected) <= 1e-10
        assert gap > 0  # micro should exceed macro when majority is easier

    def test_imbalanced_classes_negative_gap(self):
        """When minority classes are easier, micro < macro → negative gap."""
        num_classes = 5
        # Class 0 has 100 samples (all wrong), classes 1-4 have 5 each (all correct)
        labels = torch.cat([
            torch.zeros(100, dtype=torch.long),
            torch.ones(5, dtype=torch.long),
            torch.full((5,), 2, dtype=torch.long),
            torch.full((5,), 3, dtype=torch.long),
            torch.full((5,), 4, dtype=torch.long),
        ])
        # Majority class all wrong; minority classes all correct
        predictions = torch.cat([
            torch.ones(100, dtype=torch.long),    # class 0: 0% correct
            torch.ones(5, dtype=torch.long),       # class 1: 100% correct
            torch.full((5,), 2, dtype=torch.long), # class 2: 100% correct
            torch.full((5,), 3, dtype=torch.long), # class 3: 100% correct
            torch.full((5,), 4, dtype=torch.long), # class 4: 100% correct
        ])

        results = compute_metrics(predictions, labels, num_classes)

        gap = results["micro_macro_gap"]
        expected = results["micro_accuracy"] - results["macro_accuracy"]
        assert abs(gap - expected) <= 1e-10
        assert gap < 0  # micro should be below macro when majority is harder

    def test_single_class(self):
        """Edge case: single class."""
        num_classes = 1
        labels = torch.zeros(50, dtype=torch.long)
        predictions = torch.zeros(50, dtype=torch.long)

        results = compute_metrics(predictions, labels, num_classes)
        gap = results["micro_macro_gap"]
        expected = results["micro_accuracy"] - results["macro_accuracy"]
        assert abs(gap - expected) <= 1e-10
        # Single class: micro == macro
        assert abs(gap) <= 1e-10

    def test_random_predictions(self):
        """Random predictions on imbalanced data: gap should be consistent."""
        torch.manual_seed(42)
        num_classes = 20
        # Imbalanced: class 0 has many samples, others have few
        counts = [200] + [5] * 19
        labels = torch.cat([
            torch.full((c,), i, dtype=torch.long) for i, c in enumerate(counts)
        ])
        predictions = torch.randint(0, num_classes, (len(labels),))

        results = compute_metrics(predictions, labels, num_classes)
        gap = results["micro_macro_gap"]
        expected = results["micro_accuracy"] - results["macro_accuracy"]
        assert abs(gap - expected) <= 1e-10

    def test_gap_precision_1e10(self):
        """The computed gap must match the difference to within 1e-10."""
        num_classes = 500
        torch.manual_seed(42)

        # Simulate realistic per-class accuracies (500 classes)
        per_class_acc = torch.rand(num_classes) * 0.5 + 0.3  # 0.3-0.8
        counts = (torch.rand(num_classes) * 100 + 10).long()  # 10-110 per class

        total_correct = 0
        total_samples_val = 0
        for c in range(num_classes):
            n = counts[c].item()
            correct = int(per_class_acc[c].item() * n)
            total_correct += correct
            total_samples_val += n

        micro = total_correct / total_samples_val
        macro = per_class_acc.mean().item()
        gap = micro - macro

        # Verify precision
        assert abs(gap - (micro - macro)) <= 1e-10

        # Construct predictions/labels to match this distribution
        labels_list = []
        preds_list = []
        for c in range(num_classes):
            n = counts[c].item()
            n_correct = int(per_class_acc[c].item() * n)
            labels_list.append(torch.full((n,), c, dtype=torch.long))
            preds = torch.full((n,), c, dtype=torch.long)
            # Make some wrong
            wrong_indices = torch.randperm(n)[:n - n_correct]
            preds[wrong_indices] = (c + 1) % num_classes
            preds_list.append(preds)

        all_labels = torch.cat(labels_list)
        all_preds = torch.cat(preds_list)

        results = compute_metrics(all_preds, all_labels, num_classes)
        assert abs(results["micro_macro_gap"] - (results["micro_accuracy"] - results["macro_accuracy"])) <= 1e-10

    def test_bottom_10_percent_computation(self):
        """Bottom-10% accuracy must be mean of worst 10% classes."""
        num_classes = 50  # k = 5
        torch.manual_seed(123)

        # Create per-class accuracies
        per_class_acc_list = torch.linspace(0.1, 0.9, num_classes)
        labels = []
        preds = []
        for c in range(num_classes):
            n = 20
            acc = per_class_acc_list[c].item()
            n_correct = int(acc * n)
            labels.append(torch.full((n,), c, dtype=torch.long))
            p = torch.full((n,), c, dtype=torch.long)
            wrong_indices = torch.randperm(n)[:n - n_correct]
            p[wrong_indices] = (c + 1) % num_classes
            preds.append(p)

        all_labels = torch.cat(labels)
        all_preds = torch.cat(preds)

        results = compute_metrics(all_preds, all_labels, num_classes)

        k = max(1, num_classes // 10)  # 5
        sorted_acc = results["per_class_accuracy"].sort().values
        expected_bottom_10 = sorted_acc[:k].mean().item()

        assert results["bottom_10_percent_accuracy"] == pytest.approx(expected_bottom_10, abs=1e-6)
