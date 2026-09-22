"""Tie handling in the feature-drift / accuracy-change rank correlation."""

from __future__ import annotations

import pytest
import torch

from aegis_clip.cli.analyze_feature_drift import average_ranks, spearman


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def test_average_ranks_are_zero_based_and_ordered():
    assert average_ranks(_t([3.0, 1.0, 2.0])).tolist() == [2.0, 0.0, 1.0]


def test_average_ranks_share_the_mean_rank_across_ties():
    # 10, 20, 20, 30 -> the two 20s take ranks 1 and 2, so both get 1.5.
    assert average_ranks(_t([10.0, 20.0, 20.0, 30.0])).tolist() == [
        0.0, 1.5, 1.5, 3.0,
    ]


def test_average_ranks_put_a_constant_column_at_one_value():
    # Without tie handling this would come back as 0,1,2,3 -- a fabricated
    # spread that then correlates with anything.
    assert average_ranks(_t([7.0, 7.0, 7.0, 7.0])).tolist() == [1.5] * 4


def test_spearman_is_plus_one_for_a_strict_increase():
    assert spearman(_t([1.0, 2.0, 3.0, 4.0]), _t([5.0, 9.0, 9.5, 40.0])) == pytest.approx(1.0)


def test_spearman_is_minus_one_for_a_strict_decrease():
    assert spearman(_t([1.0, 2.0, 3.0, 4.0]), _t([40.0, 9.5, 9.0, 5.0])) == pytest.approx(-1.0)


def test_spearman_is_zero_when_one_side_is_constant():
    # The tie fix is what makes this 0 rather than an arbitrary value.
    assert spearman(_t([1.0, 2.0, 3.0, 4.0]), _t([2.0, 2.0, 2.0, 2.0])) == 0.0
