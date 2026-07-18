"""Tests for confident joint label issue estimation."""

import numpy as np
import torch

from analysis.noisy_labels.confident_joint import (
    estimate_class_thresholds,
    build_confident_joint,
    rank_label_issues,
)


class TestClassThresholds:
    def test_two_class_thresholds(self):
        """Manual 2-class case: t_c = mean p_i(c) for i with y_i=c."""
        probs = torch.tensor([
            [0.9, 0.1],  # y=0, p_i(0)=0.9
            [0.7, 0.3],  # y=0, p_i(0)=0.7
            [0.2, 0.8],  # y=1, p_i(1)=0.8
        ])
        labels = torch.tensor([0, 0, 1])
        t = estimate_class_thresholds(probs, labels, num_classes=2)
        assert t.shape == (2,)
        assert abs(t[0] - 0.8) < 1e-6  # mean of 0.9, 0.7
        assert abs(t[1] - 0.8) < 1e-6  # only one sample for class 1


class TestConfidentJoint:
    def test_joint_shape_and_sum(self):
        """Joint is CxC and sums to N."""
        probs = torch.tensor([
            [0.9, 0.1],
            [0.7, 0.3],
            [0.2, 0.8],
        ])
        labels = torch.tensor([0, 0, 1])
        t = estimate_class_thresholds(probs, labels, num_classes=2)
        cj = build_confident_joint(probs, labels, t, num_classes=2)
        assert cj.shape == (2, 2)
        assert cj.sum() == len(labels)
        assert (cj >= 0).all()

    def test_manual_cell_allocation(self):
        """Sample whose suggested_label differs from observed goes to off-diagonal."""
        probs = torch.tensor([
            [0.6, 0.4],  # y=0, t=[0.6,0.4]. p_0=0.6 >=0.6 → S={0} → stays 0
            [0.1, 0.9],  # y=0, t=[0.6,0.4]. p_0=0.1 < 0.6, p_1=0.9 >=0.4 → S={1} → goes to 1
        ])
        labels = torch.tensor([0, 0])
        t = torch.tensor([0.6, 0.4])
        cj = build_confident_joint(probs, labels, t, num_classes=2)
        # First sample: observed=0, suggested=0 → CJ[0,0]++
        # Second sample: observed=0, suggested=1 → CJ[0,1]++
        assert cj[0, 0] == 1
        assert cj[0, 1] == 1
        assert cj.sum() == 2

    def test_empty_si_fallback(self):
        """When S_i is empty, use argmax."""
        probs = torch.tensor([
            [0.1, 0.2, 0.7],  # y=0, t=[0.5,0.5,0.5] — none above threshold
        ])
        labels = torch.tensor([0])
        t = torch.tensor([0.5, 0.5, 0.5])
        cj = build_confident_joint(probs, labels, t, num_classes=3)
        # suggested=argmax=2, observed=0 → CJ[0,2]++
        assert cj[0, 2] == 1
        assert cj.sum() == 1


class TestClassCap:
    def test_ten_percent_cap(self):
        """Class with 30% estimated issues gets capped at 10%."""
        probs = torch.zeros(100, 2)
        labels = torch.zeros(100, dtype=torch.long)
        # All probs for class 0 are low → estimated_issues ~100
        probs[:, 1] = 1.0  # all suggest class 1
        t = estimate_class_thresholds(probs, labels, num_classes=2)
        cj = build_confident_joint(probs, labels, t, num_classes=2)
        issues = rank_label_issues(
            probs, labels, t, cj,
            max_class_reject_rate=0.10,
            max_global_reject_rate=0.10,
        )
        selected = issues[issues["selected"]]
        # At most 10% of class 0 (10 samples) can be selected
        assert len(selected) <= 10

    def test_global_cap_enforced(self):
        """Global reject rate cap prevents too many issues being selected."""
        probs = torch.eye(500).repeat(200, 1)  # 100000 samples, 500 classes
        probs = probs[:100000]
        # Make all samples misclassified
        probs = 1.0 - probs
        labels = torch.arange(500).repeat(200)[:100000]
        t = estimate_class_thresholds(probs, labels, num_classes=500)
        cj = build_confident_joint(probs, labels, t, num_classes=500)
        issues = rank_label_issues(
            probs, labels, t, cj,
            max_class_reject_rate=0.10,
            max_global_reject_rate=0.10,
        )
        selected = issues[issues["selected"]]
        assert len(selected) <= int(0.10 * len(labels))
