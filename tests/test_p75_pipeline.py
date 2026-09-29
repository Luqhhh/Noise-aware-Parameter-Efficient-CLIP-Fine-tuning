"""Synthetic engineering checks only; no experiment training or image encoding."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from p75_pipeline import cost_gate, equal_state, frozen_groups, inspect_resume
from p75_pipeline_lp import make_heads, head_loss, train_pair
from p75_pipeline_report import summarize, decide
from p75_pipeline_overlay import transform
from run_p75_pipeline_a import authorize
from run_p75_semantic_pair import sha


def test_cost_gate_reserves_twenty_percent_and_rejects_unknown():
    assert cost_gate(20,1000,100)['fits']
    assert not cost_gate(40,1000,100)['fits']
    with pytest.raises(ValueError): cost_gate(float('nan'),1000,100)
    with pytest.raises(ValueError): cost_gate(0,1000,100)


def test_head_initialization_and_disabled_mask_match_loss_and_gradient():
    a,b=make_heads(3,4,42); x=torch.randn(5,3); y=torch.tensor([0,1,2,3,0]); mask=torch.zeros(5,dtype=torch.bool)
    la=head_loss(a,x,y,mask,True)
    lb=torch.nn.functional.cross_entropy(b(torch.nn.functional.normalize(x,dim=1)),y)
    la.backward(); lb.backward()
    assert torch.equal(la,lb)
    for p,q in zip(a.parameters(),b.parameters()): assert torch.equal(p.grad,q.grad)


def test_mask_keeps_original_denominator():
    a,b=make_heads(3,4,42); x=torch.randn(5,3,requires_grad=True); y=torch.tensor([0,1,2,3,0]); mask=torch.tensor([True,False,False,False,False])
    got=head_loss(a,x,y,mask,True)
    expected=torch.nn.functional.cross_entropy(b(torch.nn.functional.normalize(x[1:],dim=1)),y[1:],reduction='sum')/5
    assert torch.allclose(got,expected)
    got.backward(); assert torch.equal(x.grad[0],torch.zeros(3))


def test_pinned_overlay_compiles_and_only_extends_epochs():
    root=Path(__file__).resolve().parents[1]
    raw=subprocess.check_output(['git','show','f050ecb:reproducibility/aegis_f1/aegis_clip/trainer.py'],cwd=root,text=True)
    patched=transform(raw); compile(patched,'private_trainer','exec')
    assert 'schedule_epochs * steps_per_epoch' in patched
    assert 'pipeline_stop not in (5, 6)' in patched
    assert 'return checkpoint_dir / "last.pt"\n        interval' in patched
    assert '"resumable": False' in patched
    with pytest.raises(RuntimeError): transform(patched)


def test_authorization_cannot_come_from_proposal_or_other_manifest(tmp_path):
    (tmp_path/'A').mkdir(); (tmp_path/'manifest.json').write_text('{}')
    (tmp_path/'A/manifest.json').write_text(json.dumps(dict(experiment_id='P75_MASK_E6_EXTENSION',budget_seconds=21600)))
    auth=tmp_path/'approval.json'
    auth.write_text(json.dumps(dict(approved=False)))
    with pytest.raises(ValueError): authorize(tmp_path,auth,'A')
    auth.write_text(json.dumps(dict(approved=True,experiment_id='P75_MASK_E6_EXTENSION',budget_seconds=21600,manifest_sha256=sha(tmp_path/'manifest.json'),user_instruction='synthetic test')))
    assert authorize(tmp_path,auth,'A')['approved']
    (tmp_path/'manifest.json').write_text('{"changed":true}')
    with pytest.raises(ValueError): authorize(tmp_path,auth,'A')


def test_resume_rejects_missing_state_and_wrong_stage():
    with pytest.raises(ValueError,match='Incomplete'): inspect_resume({},'control')
    assert equal_state({'x':torch.tensor([1]),'r':np.array([2])},{'x':torch.tensor([1]),'r':np.array([2])})
    assert not equal_state({'x':torch.tensor([1])},{'x':torch.tensor([2])})


def test_paired_metrics_overlap_and_bad_identity():
    rows=[dict(label=y,selected_proxy=i==0,high_hit_classes=y==0,bio_dominant=i>0,bio_high_hit=i==1,bio_other_classes=i==2) for i,y in enumerate([0,0,1])]
    result=summarize(rows,[0,0,1],[1,0,0],[0,1,1])
    assert result['all']['net']==1
    assert result['selected_proxy']['net']==1
    assert result['remaining_proxy']['net']==0
    assert result['all']['per_class']['0']['net']==0
    with pytest.raises(ValueError): summarize(rows,[1,0,1],[1,0,0],[0,1,1])


def test_decision_does_not_auto_promote():
    base=dict(net=45,control_macro=.5,masked_macro=.51)
    g={k:copy.deepcopy(base) for k in ('all','remaining_proxy','bio_dominant')}
    assert decide(g)=='candidate_review_required'
    g['all'].update(net=-2,masked_macro=.49)
    assert decide(g)=='proxy_supported_review_required'
    g['remaining_proxy']['net']=44
    assert decide(g)=='bounded_signal_no_automatic_extension'
    g['remaining_proxy']['net']=0
    assert decide(g)=='close_recipe_no_full_training'


def test_frozen_real_groups_have_expected_counts():
    import gzip
    root=Path(__file__).resolve().parents[1]
    with gzip.open(root/'results/p75_semantic_content_probe_20260929/frozen_selection.jsonl.gz','rt') as f:
        groups,coverage=frozen_groups([json.loads(s) for s in f])
    assert len(groups)==14880 and len(coverage)==750
    assert sum(r['bio_high_hit'] for r in groups)==332


def tiny_config():
    return dict(project=dict(seed=42),model=dict(use_cached_training=True,peft_mode='frozen',feature_dim=3,num_classes=4),
        loss=dict(name='cross_entropy'),trust=dict(enabled=False),longtail=dict(sampler_mode='none',loss_reweighting='none',balanced_softmax_tau=0),
        train=dict(head_lr=.005,head_weight_decay=.0001,batch_size=4,schedule_epochs=2,epochs=2,max_grad_norm=1))


def test_synthetic_pair_zero_mask_same_endpoint_and_timeout(tmp_path):
    import time
    torch.set_num_threads(1)
    x=torch.randn(9,3); y=torch.arange(9)%4; mask=torch.zeros(9,dtype=torch.bool)
    heads,records=train_pair(x,y,mask,tiny_config(),tmp_path,time.monotonic()+10)
    assert equal_state(heads[0].state_dict(),heads[1].state_dict())
    assert records[-1]['updates']==6
    with pytest.raises(TimeoutError): train_pair(x,y,mask,tiny_config(),tmp_path,0)
    state=torch.load(tmp_path/'masked.pt',weights_only=False)
    assert state['status']=='incomplete' and not state['resumable']


def test_lp_schedule_matches_pinned_cosine_floor(tmp_path):
    import time
    torch.set_num_threads(1)
    heads,records=train_pair(torch.randn(8,3),torch.arange(8)%4,torch.zeros(8,dtype=torch.bool),tiny_config(),tmp_path,time.monotonic()+10)
    assert records[-1]['lr']==pytest.approx([.005*.01,.005*.01])


def test_continuation_selection_preserves_evaluated_file(tmp_path):
    import types
    from run_p75_pipeline_a import continuation_selection
    # Exercise the selection transaction on small saved tensors, no model construction.
    package=types.ModuleType('aegis_clip.checkpoint')
    package._atomic_torch_save=lambda state,path: torch.save(state,path)
    before=sys.modules.get('aegis_clip.checkpoint')
    sys.modules['aegis_clip.checkpoint']=package
    try:
        ck=tmp_path/'checkpoints'; ck.mkdir()
        old=dict(epoch=4,best_selector=.5,metrics={'raw_micro':.5})
        torch.save(old,ck/'best.pt')
        endpoint=dict(epoch=6,best_selector=.5,metrics={},model_state_dict={'w':torch.tensor([1.])},optimizer_state_dict={'step':786})
        torch.save(endpoint,ck/'epoch_6.pt'); original=sha(ck/'epoch_6.pt')
        (ck/'epoch_6.binding.json').write_text(json.dumps(dict(checkpoint_sha256=original,epoch=6)))
        config=tmp_path/'config.json'; config.write_text('{}')
        record=continuation_selection(tmp_path,dict(labels=[0,1],original_logits=torch.tensor([[2.,0.],[0.,2.]])),config)
        assert record['selected_epoch']==6 and sha(ck/'epoch_6.pt')==original
        saved=torch.load(ck/'continuation_E6.pt',weights_only=False)
        assert saved['best_selector']==1 and saved['optimizer_state_dict']['step']==786
        assert torch.equal(saved['model_state_dict']['w'],endpoint['model_state_dict']['w'])
    finally:
        if before is None: sys.modules.pop('aegis_clip.checkpoint',None)
        else: sys.modules['aegis_clip.checkpoint']=before
