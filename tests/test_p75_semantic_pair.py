"""Budget cleanup and paired metric accounting independent of real training."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_p75_semantic_pair as campaign
from p75_semantic_pair_report import paired


def test_budget_timeout_reaps_owned_child_and_records_time(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'OUT',tmp_path)
    monkeypatch.setattr(campaign,'FRAMEWORK',tmp_path)
    monkeypatch.setattr(campaign,'idle_gpu',lambda:None)
    state=dict(used_seconds=0.,history=[],current=None)
    assert not campaign.execute(['-c','import time; time.sleep(60)'],'timeout',state,.1)
    assert state['current'] is None
    assert state['history'][0]['budget_timeout']
    assert state['history'][0]['returncode'] < 0
    assert 0 < state['used_seconds'] < 5


def test_child_failure_not_misreported_as_budget_or_success(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'OUT',tmp_path)
    monkeypatch.setattr(campaign,'FRAMEWORK',tmp_path)
    monkeypatch.setattr(campaign,'idle_gpu',lambda:None)
    state=dict(used_seconds=0.,history=[],current=None)
    with pytest.raises(RuntimeError,match='failed'):
        campaign.execute(['-c','raise SystemExit(3)'],'failure',state,5)
    assert state['history'][0]['returncode']==3
    assert 'budget_timeout' not in state['history'][0]


def test_pairs_keep_regressions_and_class_balanced_slice_denominators():
    y=np.array([0,0,1,2]); a=np.array([0,1,0,2]); b=np.array([0,0,1,1])
    full=paired(y,a,b)
    assert (full['corrected'],full['regressed'],full['net'])==(2,1,1)
    assert full['control_micro']==.5 and full['masked_micro']==.75
    assert full['control_macro']==.5 and full['masked_macro']==pytest.approx(2/3)
    subset=paired(y,a,b,np.array([True,True,False,False]))
    assert subset['classes_present']==1 and subset['masked_macro']==1
