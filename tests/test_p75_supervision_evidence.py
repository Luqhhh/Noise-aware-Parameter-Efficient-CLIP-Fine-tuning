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


def test_end_to_end_diagnosis_keeps_missing_history_and_serializes_overlap(tmp_path,monkeypatch):
    import p75_supervision_evidence as evidence
    import p75_supported_ce_report as decoder
    import json
    rule=R=dict(original_support_votes=12,alternative_support_votes=16,
        neighbors=20,weak_label_probability=.3,local_confidence_gate=.7,
        minimum_trusted_groups=20,minimum_hard_groups=5,primary_error_scale=1000,
        max_undercovered_class_fraction=.05,minimum_groups_for_channel_coverage=5,
        confusion_min_each_direction=2,confusion_min_total=6)
    train=[dict(image_path=f'train/{i}.jpg',label=str(i//20),content_group=f'g{i}') for i in range(40)]
    val=[dict(image_path=f'train/v{i}.jpg',label=str(i//2),content_group=f'v{i}') for i in range(4)]
    oof=[dict(**r,centroid_top1=r['label'],ridge_top1=r['label']) for r in train]
    identity=dict(num_classes=3,split_hashes={'val_dev.csv':'val-sha'},control_sha256='control-sha')
    source=dict(reference_validation_cache='synthetic')
    monkeypatch.setattr(evidence,'OUT',tmp_path)
    monkeypatch.setattr(evidence,'context',lambda:(rule,source,identity,train,val,oof))
    monkeypatch.setattr(evidence,'lp_proxy',lambda *args:([.5]*40,[.5]*40))
    payload=dict(paths=[f'v{i}.jpg' for i in range(4)],labels=torch.tensor([0,0,1,1]),
        validation_csv_sha256='val-sha',checkpoint_sha256='control-sha',
        original_logits=torch.tensor([[1.,0.,0.]]*4),flip_logits=torch.tensor([[1.,0.,0.]]*4))
    monkeypatch.setattr(evidence.torch,'load',lambda *args,**kwargs:payload)
    monkeypatch.setattr(decoder,'predictions',lambda _:np.array([0,1,0,1]))
    directory=tmp_path/'neighbors'; directory.mkdir()
    path=directory/'groups.jsonl'
    with path.open('w') as f:
        for r in train+val:
            f.write(json.dumps(dict(**r,votes={r['label']:16},neighbors=[{}]*20))+'\n')
    evidence.write_json(directory/'manifest.json',dict(identity=identity,files={'groups.jsonl':evidence.sha(path)}))
    result=evidence.diagnose()
    assert result['decision']=='incomplete_P0_missing_L05_snapshot'
    assert result['baseline_errors']==2 and result['original_supported_classes']==2
    assert result['zero_original_support_classes']==[2]
    records=evidence.read_rows(tmp_path/'diagnosis_without_snapshot/sample_evidence.csv')
    assert len(records)==44 and records[0]['training_epoch_local_enabled']==''
    assert records[0]['center_label_probability']==''
    assert result['full_training_allowed'] is False
    assert json.loads((tmp_path/'diagnosis_without_snapshot/summary.json').read_text())['overlaps']==result['overlaps']
