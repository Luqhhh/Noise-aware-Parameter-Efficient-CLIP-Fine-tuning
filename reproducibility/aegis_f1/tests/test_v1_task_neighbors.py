"""Content isolation, deterministic ties, and feature validity for the fixed probe."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
from probe_v1_task_neighbors import gallery_indices, normalized64, stable_best, votes


def test_gallery_deduplicates_single_labels_and_excludes_all_conflicting_labels():
    rows = [dict(content_group='g1', label='1', image_path='z.jpg'),
            dict(content_group='g1', label='1', image_path='a.jpg'),
            dict(content_group='g2', label='0', image_path='b.jpg'),
            dict(content_group='g2', label='1', image_path='c.jpg'),
            dict(content_group='g3', label='2', image_path='d.jpg')]
    indices, counts = gallery_indices(rows)
    assert indices.tolist() == [1, 4]
    assert counts['excluded_conflict_groups'] == 1 and counts['excluded_conflict_rows'] == 2
    assert counts['excluded_redundant_same_label_rows'] == 1


def test_stable_best_returns_same_global_order_after_different_chunk_merges():
    scores = torch.tensor([[9, 9, 8, 10, 9, 10, 7]])
    indices = torch.tensor([[4, 2, 1, 8, 0, 3, 6]])
    expected_s, expected_i = stable_best(scores, indices, 3)
    assert expected_i.tolist() == [[3, 8, 0]]
    for split in [1, 2, 4, 6]:
        left_s, left_i = stable_best(scores[:, :split], indices[:, :split], 3)
        right_s, right_i = stable_best(scores[:, split:], indices[:, split:], 3)
        actual_s, actual_i = stable_best(torch.cat([left_s, right_s], 1), torch.cat([left_i, right_i], 1), 3)
        assert torch.equal(actual_s, expected_s) and torch.equal(actual_i, expected_i)


def test_votes_use_equal_content_group_mass_and_low_class_index_for_a_tie():
    prediction, strength, counts = votes(np.array([[0, 1, 2, 3], [0, 1, 2, 4]]), np.array([2, 1, 2, 1, 2]), 3)
    assert prediction.tolist() == [1, 2] and strength.tolist() == [2, 3]
    assert counts.tolist() == [[0, 2, 2], [0, 1, 3]]


@pytest.mark.parametrize('bad', [np.array([[0., 0.]]), np.array([[1., np.nan]])])
def test_invalid_cached_vectors_are_rejected(bad):
    with pytest.raises(ValueError, match='Invalid feature'):
        normalized64(bad)


def test_normalization_retains_float64_geometry():
    x = normalized64(np.array([[3., 4.], [0., 2.]], dtype=np.float32))
    assert x.dtype == np.float64 and np.array_equal(x, np.array([[.6, .8], [0., 1.]]))
