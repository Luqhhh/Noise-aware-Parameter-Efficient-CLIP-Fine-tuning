"""Protect alignment and whole-content-group uncertainty in the CPU diagnostic."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('candidate_errors',
    Path(__file__).resolve().parents[3] / 'scripts/diagnose_v1_candidate_errors.py')
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


@pytest.mark.parametrize('field', ['image_paths', 'labels'])
def test_rejects_misaligned_prediction_assets(tmp_path, field):
    paths, labels = np.array(['train/0/a', 'train/1/b']), np.array([0, 1])
    values = dict(image_paths=paths, labels=labels, predictions=labels)
    values[field] = values[field][::-1]
    path = tmp_path / 'predictions.npz'
    np.savez(path, **values)
    with pytest.raises(ValueError, match='Prediction'):
        diagnostic.load_aligned_npz(path, paths, labels)


def test_group_bootstrap_keeps_opposing_rows_together():
    labels = np.array([0, 1, 0, 1])
    predictions = dict(a=np.zeros(4, dtype=int), b=np.ones(4, dtype=int))
    result = diagnostic.bootstrap_intervals(labels, predictions, [('a', 'b')],
        np.array([0, 0, 1, 1]), dict(seed=42, resamples=50, chunk=7))
    # Every group contains one correction and one regression. Row bootstrap
    # would invent nonzero uncertainty; whole-group bootstrap must return zero.
    assert result['a__to__b']['delta_micro_pp_95_percentile_interval'] == [0., 0.]


def test_macro_weights_classes_instead_of_rows():
    result = diagnostic.metrics(np.array([0, 0, 0, 1]), np.array([0, 0, 0, 0]),
                                3, np.ones(4, dtype=bool))
    assert result['micro'] == .75
    assert result['macro'] == .5
    assert result['classes'] == 2


def test_wrong_to_wrong_changes_do_not_count_as_corrections():
    result = diagnostic.paired(np.array([0, 0, 0]), np.array([1, 0, 1]),
                               np.array([2, 1, 0]), np.ones(3, dtype=bool))
    assert result['corrections'] == result['regressions'] == 1
    assert result['both_wrong'] == 1
    assert result['changed'] == 3
    assert result['net'] == 0


def test_confusion_combines_both_directions():
    result = diagnostic.confusion(np.array([0, 1, 0, 2]), np.array([1, 0, 0, 1]),
                                 ['a', 'b', 'c'])
    assert result['errors'] == 3
    assert result['distinct_unordered_pairs'] == 2
    assert result['top100'][0] == dict(class_indices=[0, 1], class_names=['a', 'b'], errors=2)
