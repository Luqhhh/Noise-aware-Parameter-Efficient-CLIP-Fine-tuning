"""Guard against plausible silent corruption of the historical index join."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_v2_dev_errors import align_predictions


def fixture():
    manifest = [dict(relative_path='0000/a.jpg', label='0'),
                dict(relative_path='0001/b.jpg', label='1'),
                dict(relative_path='0001/c.jpg', label='1')]
    val = [dict(image_path='train/0001/c.jpg', label='1'),
           dict(image_path='train/0000/a.jpg', label='0')]
    data = dict(indices=np.array([0, 2]), labels=np.array([0, 1]),
                raw=np.array([1, 0]), ema=np.array([0, 1]))
    return data, manifest, val


def test_reorders_predictions_by_identity():
    data, manifest, val = fixture()
    result = align_predictions(data, manifest, val, [2, 0], 2)
    assert result['raw'].tolist() == [0, 1]
    assert result['ema'].tolist() == [1, 0]


@pytest.mark.parametrize('field,value', [('indices', [0, 0]), ('indices', [0, 1]),
    ('labels', [1, 0]), ('ema', [0, 2]), ('raw', [0., 1.])])
def test_rejects_misaligned_or_invalid_payload(field, value):
    data, manifest, val = fixture()
    data[field] = np.array(value)
    with pytest.raises(ValueError):
        align_predictions(data, manifest, val, [0, 2], 2)


def test_rejects_path_mapping_to_wrong_official_label():
    data, manifest, val = fixture()
    val[0]['label'] = '0'
    with pytest.raises(ValueError, match='Aligned label mismatch'):
        align_predictions(data, manifest, val, [0, 2], 2)
