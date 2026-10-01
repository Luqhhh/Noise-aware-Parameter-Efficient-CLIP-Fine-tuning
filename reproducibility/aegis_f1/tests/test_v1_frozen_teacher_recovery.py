"""Confidence support cannot replace net benefit or a common-error budget."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_v1_frozen_teacher_recovery import supported, gate_pass, replay_head


def test_support_requires_both_confidence_and_margin():
    scores = dict(confidence=np.array([.7,.9,.69,.9]),margin=np.array([.2,.19,.3,.3]))
    rule = dict(confidence_min=.7,probability_margin_min=.2)
    assert supported(scores,rule).tolist() == [True,False,False,True]


def test_gross_recoveries_do_not_satisfy_negative_net_gate():
    cfg=json.loads((ROOT/'configs/v1_frozen_teacher_recovery_20261001.json').read_text())
    assert not gate_pass(dict(corrections=400,regressions=550,net=-150),300,cfg['feasibility_gate'])


def test_large_supported_gain_does_not_replace_common_failure_budget():
    cfg=json.loads((ROOT/'configs/v1_frozen_teacher_recovery_20261001.json').read_text())
    assert not gate_pass(dict(corrections=300,regressions=100,net=200),74,cfg['feasibility_gate'])


def test_zero_norm_features_are_rejected_before_confidence_scoring():
    state=dict(weight=torch.eye(2),logit_scale=torch.tensor(2.))
    with pytest.raises(ValueError,match='features'):
        replay_head(np.zeros((1,2)),state)
