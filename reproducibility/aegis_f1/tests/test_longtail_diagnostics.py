import pytest
import torch
from aegis_clip.longtail_diagnostics import SupervisionLedger


def test_mixup_mass_draws_unique_and_zero_supervision():
    ledger=SupervisionLedger([0,1,1],3,frequency_segments=['head','tail','tail'])
    ledger.update([0,0,1],[[.5,.5,0],[1,0,0],[0,1,0]],[2,0,1],denominator=3)
    r=ledger.report();a,b,c=r['classes']
    assert a['actual_draws']==2 and a['actual_unique_samples']==1
    assert a['zero_weight_draws']==1 and a['weighted_supervision_mass']==1
    assert b['weighted_supervision_mass']==2 and b['corrected_target_mass_in']==.5
    assert a['corrected_target_mass_out']==.5 and c['weighted_supervision_mass']==0
    assert sum(x['actual_target_mass'] for x in r['classes'])==3
    assert sum(x['weighted_supervision_mass'] for x in r['classes'])==3
    assert r['loss_denominators']==[3] and c['frequency_segment']=='tail'


def test_unit_weights_and_missing_membership():
    l=SupervisionLedger([0,1],2);l.update([0,1],torch.eye(2),[1,1],denominator=2)
    assert all(x['weighted_supervision_mass']==1 for x in l.report()['classes'])
    assert all(x['frequency_segment'] is None for x in l.report()['classes'])


def test_invalid_input_does_not_modify_ledger():
    l=SupervisionLedger([0],2)
    with pytest.raises(ValueError):l.update([0],[[.2,.2]],[1],denominator=1)
    assert l.report()['actual_draws']==0
