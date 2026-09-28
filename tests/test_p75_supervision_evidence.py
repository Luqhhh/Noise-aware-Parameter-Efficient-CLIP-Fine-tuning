import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from p75_supervision_evidence import group_neighbors, channel_for, freeze_confusions, paired_counts

RULE=dict(original_support_votes=12,alternative_support_votes=16,confusion_min_each_direction=2,confusion_min_total=6)

def test_group_exclusion_conflict_and_boundary_tie():
    rows=[dict(content_group=g,label=y) for g,y in [('self',0),('dup',1),('dup',1),('bad',0),('bad',1),('a',0),('b',1)]]
    x=torch.ones(7,2)
    _,votes,neighbors=next(group_neighbors(x,rows,x[:1],rows[:1],k=4))
    assert [n['training_row'] for n in neighbors]==[1,3,5,6]
    assert votes=={0:1,1:2}
    assert all(n['content_group']!='self' for n in neighbors)

def test_val_not_in_bank_and_exhausted_groups():
    bank=torch.eye(2)
    rows=[dict(content_group='a',label=0),dict(content_group='b',label=1)]
    query=[dict(content_group='val',label=1)]
    _,votes,neighbors=next(group_neighbors(bank,rows,bank[:1],query,k=20))
    assert len(neighbors)==2 and votes=={0:1,1:1}

def test_model_agreement_never_grants_original_or_alternative():
    assert channel_for(0,{0:11},(0,0),(0,0),RULE)==('uncertain',None)
    assert channel_for(0,{1:16},(0,1),(1,1),RULE)==('uncertain',None)
    assert channel_for(0,{1:16},(1,1),(0,1),RULE)==('uncertain',None)
    assert channel_for(0,{1:16},(1,1),(1,1),RULE)==('soft',1)
    assert channel_for(0,{0:12},(0,0),(1,1),RULE)==('original',None)

def test_confusion_requires_both_directions():
    y=[0]*4+[1]*2+[2]*8
    p=[1]*4+[0]*2+[1]*8
    edges,groups,members=freeze_confusions(y,p,3,RULE)
    assert edges==[(0,1)] and groups==[[0,1]] and 2 not in members

def test_paired_corrections_and_regressions():
    result=paired_counts([0,0,1,1],[0,1,0,1],[1,0,1,1])
    assert result['corrections']==2 and result['regressions']==1 and result['net']==1
    assert result['baseline_micro']==.5 and result['candidate_micro']==.75

def test_deadline_refuses_partial_evidence():
    rows=[dict(content_group='a',label=0)]
    with pytest.raises(TimeoutError):
        list(group_neighbors(torch.ones(1,2),rows,torch.ones(1,2),rows,deadline=0))


def test_promotion_recomputed_from_bound_rows(tmp_path):
    from p75_supervision_report import verify_report
    from p75_supported_ce import write_rows,sha
    rows=[dict(original_label=y,l05_prediction=b,candidate_prediction=c,
        hard_supervision_associated=True) for y,b,c in [(0,1,0),(0,0,0),(1,0,1),(1,1,1)]]
    path=tmp_path/'paired.csv'
    write_rows(path,list(rows[0]),rows)
    metrics=paired_counts([0,0,1,1],[1,0,0,1],[0,0,1,1])
    report=dict(binding={},report_files={str(path):sha(path)},all_validation=metrics,
        slices={'hard_supervision_associated':metrics},route='R1',earlier_cache=None,
        priority_submission=True,full_training_signal=False)
    verify_report(report)
    report['full_training_signal']=True
    with pytest.raises(ValueError,match='flags'):
        verify_report(report)
