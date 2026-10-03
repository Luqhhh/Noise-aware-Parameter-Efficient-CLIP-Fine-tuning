"""ND-CW correctness, attribution, coverage and V2 numerical equivalence."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reproducibility/aegis_f1'))
from ndcw import core, io, paired, perceptual, retrieval
from v2.core import logical_backward, sample_weights, training_utils
from v2.plan import sha


def rows(n=40):
    return [dict(image_path=f'train/{i // (n//2):04d}/{i}.jpg', label=i // (n//2),
                 content_group=f'g{i}') for i in range(n)]


def pair(i=0, j=20, s=.99, mutual=True, percept=True):
    return dict(i=i, j=j, cosine_similarity=s, mutual_nn=mutual, perceptual_pass=percept)


def test_exact_conflict_is_excluded_from_near_duplicate_weights_and_sweep():
    r = rows(); r[20]['content_group'] = r[0]['content_group']
    result = core.manifest(r, [pair()], .96, .98)
    assert all(w['nd_weight'] == 1 for w in result)
    assert all(s['candidate_pairs'] == 0 for s in core.sweep(r, [pair()]))


@pytest.mark.parametrize('p', [pair(0,1), pair(s=.95), pair(mutual=False,percept=False)])
def test_same_label_and_weak_edges_never_attenuate(p):
    assert all(r['nd_weight'] == 1 for r in core.manifest(rows(), [p], .96, .98))


def test_strong_endpoints_and_minimum_not_product():
    result = core.manifest(rows(), [pair(), pair(0,21,s=.97), pair(0,22,s=.97)], .96, .98)
    assert result[0]['nd_weight'] == .2 and result[20]['nd_weight'] == .2
    assert result[0]['num_conflict_neighbors'] == 3
    assert result[21]['nd_weight'] == result[22]['nd_weight'] == .5


def test_class_cap_retains_strongest_and_is_discrete_deterministic():
    ps = [pair(0,20,s=.99), pair(1,21,s=.985), pair(2,22,s=.97), pair(3,23,s=.96)]
    r = rows()
    result = core.manifest(r, ps, .96, .98)
    assert result[0]['nd_weight'] == result[1]['nd_weight'] == .2
    assert result[2]['nd_weight'] == result[3]['nd_weight'] == 1
    assert result[2]['raw_nd_weight'] == .5
    assert result == core.manifest(r, list(reversed(ps)), .96, .98)
    assert all(x['supervision_loss_rate'] <= .10 for x in core.impact(r, result))
    # Class with only two rows cannot afford even medium's 0.5 reduction.
    small = rows(4)
    tiny = core.manifest(small, [pair(0,2,s=.97)], .96, .98)
    assert all(x['nd_weight'] == 1 for x in tiny)


def test_dev_manifest_ignores_validation_labels_and_cross_split_edges():
    r = rows()
    ps = [pair(0,20), pair(1,21)]
    before = core.manifest(r, ps, .96, .98, eligible=set(range(40)) - {20})
    r[20]['label'] = 0; r[20]['content_group'] = 'changed'
    after = core.manifest(r, ps, .96, .98, eligible=set(range(40)) - {20})
    assert before[0] == after[0] and before[1] == after[1]
    assert before[0]['nd_weight'] == 1 and before[20]['nd_weight'] == 1


def write_weights(tmp_path, table):
    path = tmp_path/'weights.csv'
    io.write_csv(path, table, core.FIELDS)
    records = [dict(relative_path=r['image_path'].removeprefix('train/'), label=r['label'], content_group=r['content_group']) for r in table]
    return path, records


def test_manifest_full_coverage_order_and_checksum(tmp_path):
    table = core.manifest(rows(), [pair()], .96, .98)
    path, rec = write_weights(tmp_path, table)
    w = io.load_sidecar(path, list(reversed(rec)), expected_sha=sha(path))
    torch.testing.assert_close(w, torch.tensor([r['nd_weight'] for r in reversed(table)]))
    with pytest.raises(ValueError, match='checksum'):
        io.load_sidecar(path, rec, expected_sha='wrong')
    with pytest.raises(ValueError, match='cover all'):
        io.load_sidecar(path, rec[:-1])


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf'), 1.01])
def test_manifest_rejects_invalid_weights(tmp_path, bad):
    table = core.manifest(rows(), [], .96, .98); table[0]['nd_weight'] = bad
    path, rec = write_weights(tmp_path, table)
    with pytest.raises(ValueError, match='weights'):
        io.load_sidecar(path, rec)


@pytest.mark.parametrize('name', ['test/x.jpg','/tmp/test.jpg','train/0000/../x.jpg','train/0000/x.jpg/extra'])
def test_no_test_paths_are_read(name, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: pytest.fail('Unsafe path was opened'))
    with pytest.raises(ValueError, match='train paths'):
        io.image_file(tmp_path, dict(image_path=name, label=0))


def test_symlink_escape_rejected_before_image_decode(tmp_path):
    (tmp_path/'0000').mkdir(); (tmp_path/'0000/x.jpg').symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='symlink'):
        io.image_file(tmp_path, dict(image_path='train/0000/x.jpg',label=0))


def test_sampler_is_independent_of_nd_supervision():
    original = sample_weights([0]*4+[1]*16,2)
    torch.testing.assert_close(original[:4].sum(),torch.tensor(2.,dtype=torch.double))
    torch.testing.assert_close(original[4:].sum(),torch.tensor(4.,dtype=torch.double))


@pytest.mark.parametrize('mix', ['none','mixup','cutmix'])
def test_weighted_mix_and_micro_gradients_equal_source_weighted_reference(mix):
    training_utils.set_seed(42)
    model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*4*4,3))
    micro = copy.deepcopy(model)
    x = torch.randn(7,3,4,4)
    targets = training_utils.one_hot(torch.arange(7)%3,3,.15)
    weights = torch.tensor([.2,.5,1.,1.,.2,1.,.5])
    weighted = targets * weights[:,None]
    mx,my,_=training_utils.apply_mixup_cutmix(x,weighted, .2 if mix=='mixup' else 0,
                                               1. if mix=='cutmix' else 0, 0 if mix=='none' else 1,3)
    expected = -(my * model(mx).log_softmax(1)).sum()/weights.sum()
    expected.backward()
    loss = logical_backward(micro,mx,my,3,nullcontext,weights.sum())
    assert loss == pytest.approx(float(expected.detach()),abs=2e-7)
    for a,b in zip(model.parameters(),micro.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=2e-6,atol=1e-7)
    assert my.sum() == pytest.approx(float(weights.sum()),abs=1e-6)


def test_all_one_loaded_sidecar_loss_and_gradient_exactly_original(tmp_path):
    table=core.manifest(rows(),[],.96,.98)
    path,rec=write_weights(tmp_path,table)
    weights=io.load_sidecar(path,rec)[:7]
    torch.manual_seed(42)
    model=torch.nn.Linear(5,3); other=copy.deepcopy(model)
    x=torch.randn(7,5); y=training_utils.one_hot(torch.arange(7)%3,3,.15)
    a=logical_backward(model,x,y,3,nullcontext)
    mass=None
    if not bool((weights==1).all()):
        y=y*weights[:,None];mass=weights.sum()
    b=logical_backward(other,x,y,3,nullcontext,mass)
    assert a==b
    for p,q in zip(model.parameters(),other.parameters()):
        assert torch.equal(p.grad,q.grad)


def test_retrieval_exact_top10_self_excluded_mutual_and_unique():
    torch.manual_seed(2)
    f=torch.randn(17,8); f[1]=f[0]
    idx,sim=retrieval.neighbors(f,query_chunk=5,gallery_chunk=7)
    assert all(i not in row for i,row in enumerate(idx))
    normalized=torch.nn.functional.normalize(f,dim=1)
    expected=normalized@normalized.t();expected.fill_diagonal_(-torch.inf)
    order=expected.argsort(dim=1,descending=True,stable=True)[:,:10]
    assert np.array_equal(idx,order.numpy())
    r=rows(16)+[dict(image_path='train/0001/16.jpg',label=1,content_group='g16')]
    ps=list(retrieval.candidate_pairs(r,idx,sim,minimum=-1))
    assert len({(p['i'],p['j']) for p in ps})==len(ps)
    assert all(p['mutual_nn']==(p['i'] in idx[p['j']] and p['j'] in idx[p['i']]) for p in ps)


def test_perceptual_confirms_identical_textured_image_but_rejects_flat_and_different():
    policy=dict(max_phash_hamming=6,max_thumbnail_rmse=.10,max_log_aspect_ratio=.08,min_gray_std=.03)
    rng=np.random.default_rng(11)
    # High spatial contrast survives the full-image thumbnail.
    image=Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).resize((256,256))
    a=perceptual.fingerprint(image)
    assert perceptual.confirm(a,a,policy)['perceptual_pass']
    flat=perceptual.fingerprint(Image.new('RGB',(100,100),'white'))
    assert not perceptual.confirm(flat,flat,policy)['perceptual_pass']
    different=perceptual.fingerprint(Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).resize((256,256)))
    assert not perceptual.confirm(a,different,policy)['perceptual_pass']


def test_chaining_is_diagnostic_and_does_not_spread_weights():
    r=rows(60)
    ps=[pair(0,30),pair(1,30,s=.97),pair(1,31,s=.95)]
    weights=core.manifest(r,ps,.96,.98)
    assert weights[0]['nd_weight']==.2 and weights[1]['nd_weight']==.5 and weights[31]['nd_weight']==1
    assert core.clusters(r,ps,.96,.98)['largest_cluster']==3


def test_missing_trust_blocks_independent_information_gate():
    r=rows(200);w=core.manifest(r,[pair(0,100)],.96,.98)
    gate=core.gates(r,w,{})
    assert not gate['gate_c'] and gate['low_trust_given_nd'] is None


def test_unknown_val_trust_retained_in_conservative_independence_bounds():
    r=rows(200);w=core.manifest(r,[pair(i,100+i) for i in range(10)],.96,.98)
    known={r[i]['image_path']:dict(v1_low_trust=False) for i in range(9)}
    gate=core.gates(r,w,known)
    assert gate['gate_c'] and gate['low_trust_given_nd']==0
    assert gate['low_trust_given_nd_bounds']==[0,.55]


@pytest.mark.parametrize('operation,stage',[('train','s2_448'),('train','full_576'),('infer',None),('probe','s3_576')])
def test_ndcw_n2_and_inference_do_not_start_automatically(tmp_path,monkeypatch,operation,stage):
    import v2.plan as plan
    p=tmp_path/'plan.json';p.write_text('{}')
    a=tmp_path/'auth.json';a.write_text(json.dumps(dict(authorized=True,operation=operation,stage=stage,plan_sha256=sha(p),max_seconds=60)))
    monkeypatch.setattr(plan,'verify_prepared',lambda path:dict(recipe=dict(ndcw=dict(arm='control'))))
    with pytest.raises(ValueError,match='N1 only'):
        plan.authorize(p,a,operation,stage)


def test_ndcw_n1_training_requires_passed_bound_pair_probe(tmp_path,monkeypatch):
    import v2.plan as plan
    p=tmp_path/'plan.json';p.write_text('{}')
    a=tmp_path/'auth.json';a.write_text(json.dumps(dict(authorized=True,operation='train',stage='s1_384',plan_sha256=sha(p),max_seconds=60)))
    monkeypatch.setattr(plan,'verify_prepared',lambda path:dict(recipe=dict(ndcw=dict(arm='control'))))
    with pytest.raises(ValueError,match='probe report required'):
        plan.authorize(p,a,'train','s1_384')


def test_paired_metric_slices_and_strict_promotion_thresholds():
    y=np.repeat(np.arange(100),10)
    c=y.copy(); c[::10]=(c[::10]+1)%100
    t=c.copy(); t[0]=y[0];t[100]=y[100];t[200]=y[200]
    groups=dict(tail_classes=list(range(10)),affected_classes=list(range(30)))
    result=paired.compare(y,c,t,groups,np.ones(len(y),dtype=bool))
    assert result['passed'] and result['corrections']==3 and result['regressions']==0
    assert result['delta_pp']['macro']==pytest.approx(.30)
    assert not paired.compare(y,c,c,groups,np.ones(len(y),dtype=bool))['passed']
    assert not paired.compare(y,c,t,groups,np.zeros(len(y),dtype=bool))['passed']


def test_ndcw_stage_templates_keep_every_v2_hyperparameter():
    for stage in ('s1_384','s2_448','s3_576','full_576'):
        assert yaml.safe_load((ROOT/'configs/v2_ndcw'/(stage+'.yaml')).read_text()) == yaml.safe_load((ROOT/'configs/v2/stages'/(stage+'.yaml')).read_text())


def test_a0_complete_artifact_flow_and_threshold_freeze_lock(tmp_path, monkeypatch):
    from ndcw import audit, signals
    base=tmp_path/'stage';(base/'features').mkdir(parents=True)
    r=rows();f=torch.randn(40,512);f[20]=f[0]
    torch.save(f,base/'features/features.pt')
    split=dict(train=list(range(35)),val=list(range(35,40)))
    inputs={str(base/'features/features.pt'):sha(base/'features/features.pt')}
    cfg=yaml.safe_load((ROOT/'configs/v2_ndcw/audit.yaml').read_text());cfg['stage_artifacts']=str(base)
    cfg['query_chunk']=10;cfg['gallery_chunk']=15
    config=tmp_path/'audit.yaml';config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(io,'binding',lambda cfg:(r,split,dict(inputs)))
    def verify_pixels(rows,pairs,*args,**kwargs):
        for p in pairs:
            p.update(phash_hamming=0,thumbnail_rmse=0.,log_aspect_ratio=0.,perceptual_pass=not p['exact'],perceptual_error='')
        return pairs,dict(decoded_images=2,failed_images={})
    monkeypatch.setattr(perceptual,'verify',verify_pixels)
    table={x['image_path']:dict(knn_agreement=.8,prototype_agreement=True,oof_label_probability=None,
           oof_ridge_agreement=True,trusted_val_proxy=True,v1_low_trust=False,v1_weight=.9) for x in r}
    monkeypatch.setattr(signals,'load',lambda *args:(table,{}))
    directory=audit.build(config,tmp_path/'audit')
    audit.verify(directory)
    report=audit.audit(directory)
    assert report['total_images']==40 and report['exact_duplicate_images']==0
    manifest=audit.freeze(directory,tmp_path/'manifest',.96,.98,'Synthetic curve fixture only')
    assert not manifest['training_allowed']
    assert manifest['stats']['dev']['strong_conflict_images']==2
    weights=io.read_csv(tmp_path/'manifest/ndcw_weights.csv')
    assert len(weights)==40 and all(0<float(w['nd_weight'])<=1 for w in weights)
    assert (directory/'frozen_thresholds.json').is_file()
    with pytest.raises(ValueError,match='already frozen'):
        audit.freeze(directory,tmp_path/'retuned',.965,.985,'Retuning forbidden')
    with pytest.raises(ValueError,match='overwrite'):
        audit.audit(directory)
    with pytest.raises(FileExistsError):
        audit.build(config,directory)


@pytest.mark.parametrize('attenuate',[False,True])
def test_v2_real_runtime_sidecar_all_ones_parity_and_actual_effect(tmp_path,monkeypatch,attenuate):
    """CPU run through real dataset/mix/optimizer/EMA/save path for both arms."""
    import v2.runtime as rt
    from v2.plan import dump,stage_plan
    class Tiny(torch.nn.Module):
        def __init__(self,recipe):
            super().__init__();self.visual=torch.nn.Conv2d(3,4,3);self.head=torch.nn.Linear(4,2)
        def forward(self,x):
            return self.head(self.visual(x).mean((2,3)))
    monkeypatch.setattr(rt,'build_classifier',Tiny)
    monkeypatch.setattr(rt,'device_after_authorization',lambda:torch.device('cpu'))
    monkeypatch.setattr(rt,'amp',nullcontext)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(rt,'authorize',lambda *args:dict(max_seconds=60))
    monkeypatch.setattr(rt,'verify_prepared',lambda path:json.loads(Path(path).read_text()))
    monkeypatch.setattr(rt,'loader',lambda dataset,cfg,batch_size,sampler=None:torch.utils.data.DataLoader(dataset,batch_size=batch_size,sampler=sampler,drop_last=sampler is not None))
    image_root=tmp_path/'train'
    records=[]
    for i in range(12):
        name=f'{i%2:04d}/{i}.png';p=image_root/name;p.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(36,36),(i*18,30,50)).save(p)
        records.append(dict(index=i,relative_path=name,label=i%2,content_group=f'g{i}',source_sha256=sha(p)))
    table=core.manifest([dict(image_path='train/'+r['relative_path'],label=r['label'],content_group=r['content_group']) for r in records],[],.96,.98)
    if attenuate:
        table[0]['nd_weight']=.2;table[1]['nd_weight']=.5
    sidecar=tmp_path/'weights.csv';io.write_csv(sidecar,table,core.FIELDS)
    payloads=[]
    for arm in ('original','sidecar'):
        work=tmp_path/arm;work.mkdir()
        io.write_csv(work/'train_manifest.csv',records,list(records[0]))
        dump(work/'split.json',dict(train=list(range(10)),val=[10,11]))
        cfg=dict(paths=dict(project_root=str(work)),data=dict(train_dir=str(image_root),image_size=32,eval_size=32,eval_resize_ratio=1.14,
                  batch_size=4,decode_cap=0,num_workers=0,prefetch_factor=2,expected_train_images=12),
                  augment=dict(rrc_scale=[.35,1.],color_jitter=.5,rand_augment=True,rand_augment_ops=2,rand_augment_magnitude=7,
                               random_erase=.3,mixup=.2,cutmix=1.,mix_prob=.8,label_smoothing=.15),
                  train=dict(epochs=2,seed=13,lr_backbone=3e-5,lr_head=5e-4,llrd_gamma=1.,weight_decay=.15,warmup_epochs=.5,
                             min_lr_ratio=.02,ema_enabled=True,ema_decay=.9995,grad_clip=1.,log_every=200),
                  local_replay=dict(final_stage=False,micro_batch_size=2))
        if arm=='sidecar':cfg['ndcw']=dict(sidecar=str(sidecar),sha256=sha(sidecar))
        config=work/'stage.json';dump(config,cfg)
        spec=stage_plan(cfg,10,2,'official_openai');spec.update(config=config.name,config_sha256=sha(config))
        plan=dict(experiment_id=arm,data_version='20260921',strategy_revision='fixture',
                  recipe=dict(num_classes=2,official_sha256='official',ema_enabled=True,ema_decay=.9995),
                  inputs={str(work/'train_manifest.csv'):sha(work/'train_manifest.csv')},stages={'s1_384':spec})
        dump(work/'plan.json',plan)
        rt.train(work/'plan.json','synthetic_cpu','s1_384')
        payloads.append(torch.load(work/'runs/s1_384/last.pt',map_location='cpu',weights_only=False))
    a,b=payloads
    assert [h['sample_order_sha256'] for h in a['history']]==[h['sample_order_sha256'] for h in b['history']]
    if not attenuate:
        for key in a['model']:assert torch.equal(a['model'][key],b['model'][key])
        for key in a['ema']:assert torch.equal(a['ema'][key],b['ema'][key])
        assert a['history']==b['history'] or [h['train_loss'] for h in a['history']]==[h['train_loss'] for h in b['history']]
    else:
        assert any(not torch.equal(a['model'][key],b['model'][key]) for key in a['model'])
