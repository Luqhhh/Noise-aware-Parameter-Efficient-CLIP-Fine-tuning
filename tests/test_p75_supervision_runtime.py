import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from p75_supervision_runtime import (channel_loss, anchored_soft_targets, supervised_contrastive,
    ConfusionSampler, EvidenceState, digest)
from p75_supervision_overlay import transform


def test_unknown_classification_gradient_is_zero_and_hard_original_is_full_ce():
    x=torch.tensor([[-5.,5.],[-5.,5.],[-5.,5.]],requires_grad=True)
    y=torch.tensor([[1.,0.],[.2,.8],[1.,0.]])
    loss=channel_loss(x,torch.tensor([0,1,2]),y)
    grad=torch.autograd.grad(loss.sum(),x)[0]
    assert torch.equal(grad[2],torch.zeros(2))
    assert grad[0,0]<-.99
    assert torch.allclose(grad[:2],x[:2].softmax(1)-y[:2])


def test_teacher_cannot_expand_reference_support_or_channels():
    fixed=torch.tensor([[1.,0.,0.],[.2,.8,0.],[1.,0.,0.]])
    teacher=torch.tensor([[0.,0.,1.]]*3,requires_grad=True)
    result=anchored_soft_targets(fixed,teacher,torch.tensor([0,1,2]))
    assert torch.equal(result,fixed)
    assert not result.requires_grad


def test_supervised_contrast_requires_reliable_different_groups_and_class_edge():
    x=torch.randn(4,5,requires_grad=True)
    labels=torch.tensor([0,0,1,1])
    groups=torch.arange(4)
    trusted=torch.ones(4,dtype=torch.bool)
    value=supervised_contrastive(x,labels,groups,trusted,[(0,1)])
    assert torch.isfinite(value) and value>0
    value.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert supervised_contrastive(x,labels,groups,trusted,[])==0
    assert supervised_contrastive(x,labels,torch.tensor([0,0,1,1]),trusted,[(0,1)])==0
    assert supervised_contrastive(x,labels,groups,torch.zeros(4,dtype=torch.bool),[(0,1)])==0


def test_sampler_half_global_half_independent_confusions_is_reproducible():
    rows=[dict(channel='original',original_label=i//4,content_group=f'g{i}') for i in range(8)]
    sampler=ConfusionSampler(rows,[(0,1)])
    a=list(sampler)
    assert a==list(sampler) and len(a)==len(rows)
    paired=[rows[i] for i in a[4:8]]
    assert [r['original_label'] for r in paired]==[0,0,1,1]
    assert len({r['content_group'] for r in paired})==4
    sampler.set_epoch(1)
    assert a!=list(sampler)


def test_overlay_compiles_against_actual_pinned_trainer_and_refuses_drift():
    root=Path(__file__).resolve().parents[1]
    source=subprocess.check_output(['git','show','f050ecb:reproducibility/aegis_f1/aegis_clip/trainer.py'],cwd=root,text=True)
    patched=transform(source)
    compile(patched,'patched.py','exec')
    with pytest.raises(RuntimeError):
        transform(patched)


def test_local_admission_original_ignores_confidence_but_rejects_blank_and_unknown():
    state=object.__new__(EvidenceState)
    state.route='R1'
    state.batch_channels=torch.tensor([0,0,1,2])
    state.batch_targets=torch.tensor([[1.,0.]]*4)
    images=torch.randn(4,3,8,8)
    images[1]=0
    logits=torch.randn(4,2,requires_grad=True)
    global_loss=torch.tensor([1.,2.,3.,4.],requires_grad=True)
    loss,admitted=state.local_loss(logits,torch.ones(4),torch.tensor([.1,.9,.8,.9]),.7,images,global_loss)
    assert admitted.tolist()==[True,False,True,False]
    loss.sum().backward()
    assert logits.grad[3].abs().sum()==0 and global_loss.grad[3]==1


def test_lagged_teacher_is_frozen_and_updates_only_at_epoch_boundary():
    state=object.__new__(EvidenceState)
    state.route='R1'; state.cached=False; state.teacher=None
    model=torch.nn.Linear(2,2)
    state.begin_epoch(model,1)
    before=copy.deepcopy(state.teacher.state_dict())
    with torch.no_grad():
        model.weight.add_(1)
    assert torch.equal(state.teacher.weight,before['weight'])
    assert all(not p.requires_grad for p in state.teacher.parameters())
    state.begin_epoch(model,2)
    assert torch.equal(state.teacher.weight,model.weight)


def test_config_generation_uses_fresh_official_head_and_same_student_resume(tmp_path, monkeypatch):
    import run_p75_supervision_rebuild as runner
    import p75_supported_ce as common
    root=Path(__file__).resolve().parents[1]
    original_path=root/'outputs/codex/p75_supported_ce_20260929/configs/l05_original.json'
    if not original_path.exists():
        pytest.skip('Requires recorded original L05 config from CPU prepare')
    source=common.read_json(common.FIXED)
    real_read=runner.read_json
    identity={'test':'configuration only'}
    diagnosis=tmp_path/'diagnosis'
    diagnosis.mkdir()
    (diagnosis/'sample_evidence.csv').write_text('unbound configuration test fixture\n')
    (diagnosis/'frozen_slices.json').write_text('{}')
    summary=dict(identity=identity,snapshot_manifest_sha256='fixture',coverage_ok=True,
        decision='R1_bounded_pilot_first',files={},slices={'stable_trusted_confusion':{'errors':0}})
    monkeypatch.setattr(runner,'OUT',tmp_path)
    monkeypatch.setattr(runner,'CONFIGS',tmp_path/'configs')
    monkeypatch.setattr(runner,'context',lambda:(None,source,identity,None,None,None))
    monkeypatch.setattr(runner,'materialize',lambda _:None)
    monkeypatch.setattr(runner,'read_json',lambda path:summary if Path(path).name=='summary.json' else real_read(path))
    configs=runner.prepare('R1')
    head,visual=configs['R1_HEAD'],configs['R1']
    assert head['train']['init_checkpoint'] is None
    assert head['model']['peft_mode']=='frozen'
    assert head['project']['parent_kind']=='official_clip_head'
    assert head['train']['epochs']==20
    assert visual['train']['init_checkpoint'].endswith('P75_R1_HEAD/seed42/checkpoints/last.pt')
    assert visual['model']['input_resolution']==384
    assert visual['train']['epochs']==visual['train']['schedule_epochs']==16
    assert visual['loss']['feature_distillation_weight']==2
    assert visual['train']['effective_batch_size']==1024
    env=runner.environment()
    subprocess.run([sys.executable,'-c',
        'from aegis_clip.config import load_config; from aegis_clip.rematch_protocol import validate_dataset; '
        f'validate_dataset(load_config({str(tmp_path/"configs/R1_HEAD.yaml")!r})); '
        f'validate_dataset(load_config({str(tmp_path/"configs/R1.yaml")!r}))'],env=env,check=True)
    with pytest.raises(ValueError,match='1,000'):
        runner.prepare('R2')
    summary['coverage_ok']=False
    with pytest.raises(ValueError,match='coverage'):
        runner.prepare('R1')
    # R2 retains ordinary classification for all classes; its independent gate
    # is trustworthy confusion scale, not R1's three-channel class coverage.
    summary['slices']['stable_trusted_confusion']['errors']=1000
    r2=runner.prepare('R2')['R2']
    assert r2['loss']['name']=='gce'
    assert r2['train']['init_checkpoint']==source['parent_checkpoint']
    assert r2['loss']['supervision_evidence']['route']=='R2'


def test_evidence_table_load_and_one_training_step(tmp_path):
    table=tmp_path/'evidence.csv'
    rows=[dict(split='train',image_path=f'train/{i}.jpg',original_label=i%2,
        content_group=f'g{i}',channel=c,neighbor_votes='{"0": 4, "1": 16}')
        for i,c in enumerate(('original','soft','uncertain'))]
    with table.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    slices=tmp_path/'slices.json'; slices.write_text('{"trusted_edges": [[0,1]]}')
    config=dict(loss=dict(supervision_evidence=dict(route='R1',table=str(table),sha256=digest(table),
        slices=str(slices),slices_sha256=digest(slices))),model=dict(num_classes=2,use_cached_training=False),
        trust=dict(enabled=False),train=dict(batch_size=4))
    state=EvidenceState(config,[f'{i}.jpg' for i in range(3)],[0,1,0])
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.visual=torch.nn.Linear(12,4); self.classifier=torch.nn.Linear(4,2)
        def forward(self,*,images,return_features=False):
            z=self.visual(images.flatten(1)); logits=self.classifier(z)
            return (logits,z) if return_features else logits
    model=Toy(); state.begin_epoch(model,1)
    images=torch.randn(3,3,2,2)
    logits,features=model(images=images,return_features=True)
    logits.retain_grad()
    loss=state.global_loss(model,{'images':images},logits,features,torch.arange(3),torch.zeros(3),{})
    loss.mean().backward()
    assert logits.grad[0].abs().sum()>0 and logits.grad[1].abs().sum()>0
    assert logits.grad[2].abs().sum()==0
    assert model.visual.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in state.teacher.parameters())
    table.write_text(table.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):
        EvidenceState(config,[f'{i}.jpg' for i in range(3)],[0,1,0])
