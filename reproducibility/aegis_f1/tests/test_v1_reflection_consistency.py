"""Verify orientation membership, stable divergence, and oracle-budget semantics."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from probe_v1_reflection_consistency import reflection_statistics


def test_identical_distributions_have_no_orientation_instability_even_when_wrong():
    a=np.array([[0,4,0,0,0,0],[1000,0,-1000,-1000,-1000,-1000]],float)
    s=reflection_statistics(a,a,[0,0])
    assert np.allclose(s['js_nats'],0) and not s['high_instability'].any()
    assert s['either_view_top1'].tolist()==[False,True]


def test_js_is_symmetric_and_only_disagreeing_top1_can_enter_primary_group():
    a=np.array([[5,0,-5,-5,-5,-5],[5,4,0,0,0,0]],float)
    b=np.array([[0,5,-5,-5,-5,-5],[5,-4,0,0,0,0]],float)
    s=reflection_statistics(a,b,[0,1]);r=reflection_statistics(b,a,[0,1])
    assert np.allclose(s['js_nats'],r['js_nats']) and s['high_instability'].tolist()==[True,False]
    assert s['either_view_top1'].tolist()==[True,False] and s['either_view_top5'].all()


def test_nonfinite_view_is_rejected():
    with pytest.raises(ValueError,match='Invalid'):reflection_statistics(np.full((1,6),np.nan),np.zeros((1,6)),[0])
