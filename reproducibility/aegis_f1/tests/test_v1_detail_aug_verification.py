"""Reject stale cached labels/predictions and nonfinite inference artifacts."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from verify_v1_detail_aug_pair import val_prediction


def test_replay_rejects_cached_prediction_disagreement(tmp_path):
    file = tmp_path/'val.npz'
    labels, paths = np.array([0, 1]), np.array(['a.jpg', 'b.jpg'])
    np.savez(file, labels=labels, image_paths=paths, predictions=[0, 0], logits=[[2., 0.], [0., 2.]])
    with pytest.raises(ValueError, match='argmax'):
        val_prediction(file, labels, paths, 2)


def test_replay_rejects_reordered_validation_rows(tmp_path):
    file = tmp_path/'val.npz'
    labels, paths = np.array([0, 1]), np.array(['a.jpg', 'b.jpg'])
    np.savez(file, labels=labels, image_paths=paths[::-1], predictions=labels, logits=[[2., 0.], [0., 2.]])
    with pytest.raises(ValueError, match='alignment'):
        val_prediction(file, labels, paths, 2)


def test_replay_rejects_nonfinite_logits(tmp_path):
    file = tmp_path/'val.npz'
    labels, paths = np.array([0, 1]), np.array(['a.jpg', 'b.jpg'])
    np.savez(file, labels=labels, image_paths=paths, predictions=labels, logits=[[np.inf, 0.], [0., 2.]])
    with pytest.raises(ValueError, match='Invalid validation logits'):
        val_prediction(file, labels, paths, 2)
