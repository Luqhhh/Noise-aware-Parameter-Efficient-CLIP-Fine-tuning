"""Cover reliability/image dominance and frequency rather than exposure-count ranking."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_v1_mixup_mass import effective_fraction,select_classes


def test_equal_reliability_preserves_the_image_coefficient():
    assert np.allclose(effective_fraction(.73,np.array([.2,.7]),np.array([.2,.7])),.73)


def test_reliability_can_flip_dominance_without_a_change_in_image_fraction():
    effective=effective_fraction(.6,.2,.8)
    assert effective==pytest.approx(.12/.44) and effective<.5


def test_zero_reliability_is_rejected_for_this_active_only_population():
    with pytest.raises(ValueError,match='positive'):effective_fraction(.5,0.,.8)


def test_cohort_ranks_by_rate_and_uses_a_fixed_index_tie_break():
    owner=np.array([0]*10+[1,2,3]);event=np.array([True]*5+[False]*5+[True,True,False])
    exposure,events,rate,selected=select_classes(owner,event,5)
    assert exposure[4]==0 and selected.tolist()==[1] and rate[0]<rate[1]
