import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_v2_task_teacher import score_logits, replay_check, check_teacher


def test_probability_margin_not_logit_margin_and_stable_shift():
    p = np.array([[.71, .20, .09], [.40, .40, .20]])
    result = score_logits(np.log(p) + 10000)
    assert result['prediction'].tolist() == [0, 0]
    np.testing.assert_allclose(result['confidence'], [.71, .40], atol=1e-12)
    np.testing.assert_allclose(result['margin'], [.51, 0.], atol=1e-12)


@pytest.mark.parametrize('maximum,margin,support,error', [
    (0, .005, False, 'Too many'), (1, .02, False, 'Confident'), (1, .005, True, 'Supported')])
def test_replay_rejects_each_frozen_violation(maximum, margin, support, error):
    with pytest.raises(ValueError, match=error):
        replay_check(np.array([1]), dict(prediction=np.array([0]), margin=np.array([margin])),
            np.array([support]), dict(max_top1_changes=maximum, changed_probability_margin_max=.01, supported_changes_allowed=0))


def test_full_checkpoint_cannot_enter_independent_teacher_diagnostic():
    with pytest.raises(ValueError, match='teacher stage'):
        check_teacher(dict(binding=dict(stage='full_576')), dict(teacher_stage='s3_576'), 750)
