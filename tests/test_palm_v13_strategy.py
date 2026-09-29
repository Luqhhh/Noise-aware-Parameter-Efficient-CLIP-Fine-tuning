"""Meaningful CPU checks for the transferred recipe, no CUDA operations."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))
sys.path.insert(0, str(ROOT / "scripts"))
from palm_v13.core import check_checkpoint, logical_backward, reference, sample_weights
from palm_v13.model import LocalFTClassifier
from palm_v13.plan import authorize, check_weight_policy, check_split, cost_estimate, json_read, stage_plan, verify_vendor
from palm_v13.runtime import Budget, BudgetExpired, WeightAverage, choose_parent, train, using_weights, write_submission
from check_submission import check_csv, check_zip


def test_logical_batch_gradients_equal_unsplit_even_with_short_last_microbatch():
    torch.manual_seed(13)
    full = torch.nn.Linear(7, 5)
    micro = copy.deepcopy(full)
    x = torch.randn(11, 7)
    y = reference.one_hot(torch.arange(11) % 5, 5, .15)
    reference.soft_cross_entropy(full(x), y).backward()
    logical_backward(micro, x, y, 3, nullcontext)
    for a, b in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-6, atol=1e-7)


def test_cutmix_targets_match_actual_area(monkeypatch):
    monkeypatch.setattr(reference, "rand_bbox", lambda *a: (1, 3, 0, 2))
    monkeypatch.setattr(torch, "randperm", lambda *a, **k: torch.tensor([1, 0]))
    x = torch.stack([torch.zeros(3, 4, 4), torch.ones(3, 4, 4)])
    targets = reference.one_hot(torch.tensor([0, 1]), 2, .15)
    mixed, y, _ = reference.apply_mixup_cutmix(x, targets, 0, 1, 1, 2)
    torch.testing.assert_close(y[0], .75 * targets[0] + .25 * targets[1])
    assert mixed[0].sum().item() == 12
    torch.testing.assert_close(y.sum(1), torch.ones(2))


def test_sampler_has_sqrt_class_exposure_not_uniform_prior():
    weights = sample_weights([0] * 4 + [1] * 16, 2)
    assert weights[:4].sum().item() == 2
    assert weights[4:].sum().item() == 4
    with pytest.raises(ValueError, match="empty a class"):
        sample_weights([0, 0], 2)


@pytest.mark.parametrize("split", [dict(train=[0, 1], val=[1, 2]), dict(train=[0], val=[1]), dict(train=[0, 0], val=[1, 2])])
def test_split_rejects_overlap_missing_and_duplicates(split):
    rows = [dict(content_group=str(i)) for i in range(3)]
    with pytest.raises(ValueError):
        check_split(rows, split, True)


def test_split_rejects_pixel_content_leakage():
    with pytest.raises(ValueError, match="groups overlap"):
        check_split([dict(content_group="same"), dict(content_group="same")], dict(train=[0], val=[1]), True)


def test_stage_update_budget_resets_each_stage_and_counts_drop_last():
    verify_vendor()
    expected = [(384,10,96,3e-5,5e-4),(448,6,96,1.5e-5,3e-4),(576,4,80,8e-6,2e-4),(576,5,80,4e-6,1.2e-4)]
    import yaml
    from palm_v13.plan import VENDOR
    files = ["v12/s1_384.yaml","v12/s2_448.yaml","v12/s3_576.yaml","v13_final.yaml"]
    for filename, (size, epochs, batch, bb, head) in zip(files, expected):
        cfg = yaml.safe_load((VENDOR / "configs" / filename).read_text())
        s = stage_plan(cfg, 148695 if size == 576 and epochs == 5 else 133815, 14880, "official")
        assert (s["image_size"],s["epochs"],s["logical_batch_size"],s["lr_backbone"],s["lr_head"]) == (size,epochs,batch,bb,head)
        assert s["total_updates"] == s["train_rows"] // batch * epochs
        assert s["optimizer_reset"] and s["scheduler_reset"] and s["ema_enabled"] and s["ema_reset"]
        assert s["ema_decay"] == .9995


def test_cosine_scheduler_uses_actual_logical_updates():
    p = torch.nn.Parameter(torch.tensor(1.))
    opt = torch.optim.SGD([p], lr=1.)
    sched = reference.build_scheduler(opt, 2, .5, 8, .02)
    assert opt.param_groups[0]["lr"] == .25
    for _ in range(16):
        opt.step(); sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(.02)


def test_cost_accounts_for_every_epoch_evaluation():
    s = dict(total_updates=100, epochs=4)
    assert cost_estimate(s, 2, 30, 10) == 330
    assert cost_estimate(s, 2, 0, 10) == 210  # final all-data stage has no holdout
    with pytest.raises(ValueError):
        cost_estimate(s, float("nan"), 30, 10)


def test_default_authorization_cannot_reach_cuda(tmp_path, monkeypatch):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps(dict(authorized=False)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA probed without authorization"))
    with pytest.raises(ValueError, match="not authorized"):
        train(tmp_path / "no_plan.json", auth, "s1_384")
    assert not torch.cuda.is_initialized()


def test_authorization_requires_exact_plan_and_budget(tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps(dict(authorized=True, operation="train", stage="s1_384", plan_sha256="wrong")))
    plan = tmp_path / "plan.json"; plan.write_text("{}")
    with pytest.raises(ValueError, match="bind this plan"):
        authorize(plan, auth, "train", "s1_384")


def checkpoint_fixture():
    stage = dict(epochs=10, steps_per_epoch=3, image_size=384, config_sha256="config")
    plan = dict(experiment_id="strategy",data_version="20260921",source_commit="source",
                recipe=dict(official_sha256="official",num_classes=750,ema_enabled=False,ema_decay=.9995),
                inputs={"/workspace/train_manifest.csv":"manifest"}, stages={"s1_384":stage})
    payload = dict(epoch=2,global_step=9,num_classes=750,image_size=384,metrics=dict(chosen="raw"),
                   ema_enabled=False,ema_updates=0,
                   binding=dict(experiment_id="strategy",data_version="20260921",source_commit="source",official_sha256="official",
                                manifest_sha256="manifest",stage="s1_384",complete=True,completed_epochs=10,stage_config_sha256="config"))
    return payload, plan


def test_complete_selected_earlier_epoch_is_valid_parent():
    payload, plan = checkpoint_fixture()
    check_checkpoint(payload, plan, "s1_384")


def test_ema_parent_requires_enabled_matching_recipe_and_state():
    payload, plan = checkpoint_fixture()
    payload["model"] = {"weight": torch.tensor([1.])}
    assert choose_parent(payload) is payload["model"]
    payload["ema"] = {"weight": torch.tensor([2.])}
    payload["metrics"]["chosen"] = "ema"
    with pytest.raises(ValueError, match="EMA"):
        choose_parent(payload)
    with pytest.raises(ValueError, match="EMA"):
        check_checkpoint(payload, plan, "s1_384")
    plan["recipe"]["ema_enabled"] = True
    payload.update(ema_enabled=True,ema_decay=.9995,ema_updates=9)
    assert choose_parent(payload) is payload["ema"]
    check_checkpoint(payload,plan,"s1_384")


@pytest.mark.parametrize("field,value", [("parent_policy","best_raw_on_holdout"),("final_weights","ema"),
    ("weight_averaging",True),("ema_decay",float('nan')),("ema_decay",1.),("ema_enabled","true")])
def test_weight_policy_rejects_inconsistent_or_invalid_ema_configuration(field, value):
    recipe = dict(parent_policy="best_chosen_on_holdout",final_weights="raw",ema_enabled=True,ema_decay=.9995)
    check_weight_policy(recipe)
    recipe[field] = value
    with pytest.raises(ValueError):
        check_weight_policy(recipe)


@pytest.mark.parametrize('field,value',[('ema_updates',8),('ema_decay',.9),('ema',{})])
def test_checkpoint_rejects_incomplete_or_mismatched_ema_state(field,value):
    payload,plan=checkpoint_fixture()
    plan['recipe']['ema_enabled']=True
    payload.update(model={'weight':torch.tensor([1.])},ema={'weight':torch.tensor([2.])},
                   ema_enabled=True,ema_decay=.9995,ema_updates=9)
    payload['metrics']['chosen']='ema'
    payload[field]=value
    with pytest.raises(ValueError,match='EMA'):
        check_checkpoint(payload,plan,'s1_384')


def test_ema_accumulator_is_an_independent_single_trajectory_state():
    state={'weight':torch.tensor([1.,3.])}
    ema=WeightAverage(state,decay=.5)
    state['weight'].copy_(torch.tensor([3.,7.]))
    ema.update(state)
    torch.testing.assert_close(ema.state['weight'],torch.tensor([2.,5.]))
    state['weight'].copy_(torch.tensor([5.,9.]))
    ema.update(state)
    torch.testing.assert_close(ema.state['weight'],torch.tensor([3.5,7.]))
    torch.testing.assert_close(state['weight'],torch.tensor([5.,9.]))
    assert ema.count==2


def test_ema_evaluation_restores_raw_weights_after_budget_failure():
    model=torch.nn.Linear(2,2).train()
    raw={k:v.detach().clone() for k,v in model.state_dict().items()}
    averaged={k:v+1 for k,v in raw.items()}
    with pytest.raises(BudgetExpired):
        with using_weights(model,averaged):
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,averaged[key])
            raise BudgetExpired('synthetic evaluation budget')
    assert model.training
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,raw[key],rtol=0,atol=0)


@pytest.mark.parametrize("field,value", [("data_version","old"),("complete",False),("manifest_sha256","other"),("completed_epochs",9)])
def test_checkpoint_rejects_wrong_phase_manifest_and_partial_parent(field, value):
    payload, plan = checkpoint_fixture();payload["binding"][field]=value
    with pytest.raises(ValueError):
        check_checkpoint(payload, plan, "s1_384")


def tiny_model(checkpointing):
    from clip.model import VisionTransformer
    m = LocalFTClassifier.__new__(LocalFTClassifier)
    torch.nn.Module.__init__(m)
    m.visual = VisionTransformer(224,32,64,2,4,32).float()
    m.head = torch.nn.Linear(32,7)
    m.gradient_checkpointing = checkpointing
    return m


def test_native_224_parity_and_interpolated_384_gradient_checkpointing():
    torch.manual_seed(13)
    m = tiny_model(True).eval()
    x = torch.randn(1,3,224,224)
    with torch.no_grad():
        torch.testing.assert_close(m.embed(x), m.visual(x))
    full = copy.deepcopy(m).train(); full.gradient_checkpointing=False
    m.train()
    x = torch.randn(2,3,384,384)
    y = torch.tensor([0,1])
    torch.nn.functional.cross_entropy(full(x),y).backward()
    torch.nn.functional.cross_entropy(m(x),y).backward()
    for a,b in zip(full.parameters(),m.parameters()):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-6)
    assert torch.isfinite(m.visual.positional_embedding.grad).all()


def test_csv_zip_delivers_exact_names_and_four_digit_labels(tmp_path):
    out=tmp_path/'submission';files=['a.jpg','b.png'];names=set(files)
    write_submission(out,files,np.array([0,749]),750)
    assert (out/'pred_results.csv').read_bytes() == b'a.jpg, 0000\nb.png, 0749\n'
    assert check_csv(out/'pred_results.csv',names,750)[0]
    assert check_zip(out/'submission.zip')[0]
    with zipfile.ZipFile(out/'submission.zip') as z:
        assert z.namelist()==['pred_results.csv']
        assert z.read('pred_results.csv')==(out/'pred_results.csv').read_bytes()


@pytest.mark.parametrize('predictions', [[750],[-1],[1.5],[True]])
def test_submission_rejects_invalid_label_types_and_range(tmp_path,predictions):
    with pytest.raises(ValueError):
        write_submission(tmp_path/'bad',['a.jpg'],predictions,750)


def test_budget_stop_is_explicit(monkeypatch):
    import palm_v13.runtime as runtime
    monkeypatch.setattr(runtime.time,'monotonic',lambda:0)
    b=Budget(1)
    monkeypatch.setattr(runtime.time,'monotonic',lambda:2)
    with pytest.raises(BudgetExpired):
        b.check()


def test_prepared_recipe_is_strategy_only_not_author_data_or_cleanup():
    r=json_read(ROOT/'configs/palm_v13_20260929/recipe.json')
    assert r['strategy_only'] and r['execution_authorized'] is False
    assert r['split_mode']=='frozen_grouped' and r['final_train_policy']=='all_official_rows'
    assert 'dedup_parent' not in r
    assert r['official_train_rows']==148695
    check_weight_policy(r)
    assert r['ema_enabled'] and r['ema_decay']==.9995


@pytest.mark.parametrize('ema_enabled',[False,True])
def test_cpu_synthetic_stage_chain_resets_and_final_never_evaluates_holdout(tmp_path, monkeypatch,ema_enabled):
    """Exercise real save/seal/load/initialize flow using ten synthetic images."""
    import csv
    from PIL import Image
    import palm_v13.runtime as rt
    from palm_v13.plan import dump, sha, stage_plan

    class TinyClassifier(torch.nn.Module):
        def __init__(self, recipe):
            super().__init__()
            self.visual = torch.nn.Conv2d(3, 4, 3)
            self.head = torch.nn.Linear(4, 2)
        def forward(self, x):
            return self.head(self.visual(x).mean((2,3)))

    stages=('s1_384','s2_448','v13_final')
    monkeypatch.setattr(rt,'STAGES',stages)
    monkeypatch.setattr(rt,'LocalFTClassifier',TinyClassifier)
    monkeypatch.setattr(rt,'device_after_authorization',lambda:torch.device('cpu'))
    monkeypatch.setattr(rt,'amp',nullcontext)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(rt,'authorize',lambda *a:dict(max_seconds=60))
    monkeypatch.setattr(rt,'verify_prepared',lambda p:json_read(p))
    monkeypatch.setattr(rt,'loader',lambda dataset,cfg,batch_size,sampler=None:torch.utils.data.DataLoader(
        dataset,batch_size=batch_size,sampler=sampler,drop_last=sampler is not None))
    images=tmp_path/'images';images.mkdir()
    manifest=tmp_path/'train_manifest.csv'
    rows=[]
    for i in range(10):
        p=images/f'{i}.png';Image.new('RGB',(36,36),(i*20,30,50)).save(p)
        rows.append(dict(index=i,relative_path=p.name,label=i%2,source_sha256=sha(p),content_group=str(i)))
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(tmp_path/'split.json',dict(train=list(range(8)),val=[8,9]))
    plan=dict(experiment_id='synthetic_cpu',data_version='20260921',source_commit='reference',
              recipe=dict(num_classes=2,official_sha256='official',seed=13,
                          parent_policy='best_chosen_on_holdout' if ema_enabled else 'best_raw_on_holdout',
                          final_weights='raw',ema_enabled=ema_enabled,ema_decay=.9995),
              inputs={str(manifest):sha(manifest)},stages={})
    for i,name in enumerate(stages):
        cfg=dict(data=dict(train_dir=str(images),manifest='train_manifest.csv',split='split.json',image_size=32,
                 eval_size=32,eval_resize_ratio=1.14,batch_size=4,decode_cap=0,num_workers=0,prefetch_factor=2,expected_train_images=10),
                 augment=dict(rrc_scale=[.35,1.],color_jitter=.5,rand_augment=True,rand_augment_ops=2,rand_augment_magnitude=7,
                              random_erase=.3,mixup=.2,cutmix=1.,mix_prob=.8,label_smoothing=.15),
                 train=dict(epochs=2 if i==0 else 1,seed=13,lr_backbone=3e-5,lr_head=5e-4,llrd_gamma=1.,
                            weight_decay=.15,warmup_epochs=.5,min_lr_ratio=.02,ema_enabled=ema_enabled,ema_decay=.9995,grad_clip=1.,log_every=200),
                 local_replay=dict(final_stage=i==2,micro_batch_size=2))
        config=tmp_path/(name+'.json');dump(config,cfg)
        spec=stage_plan(cfg,10 if i==2 else 8,0 if i==2 else 2,None)
        spec.update(config=config.name,config_sha256=sha(config));plan['stages'][name]=spec
    path=tmp_path/'plan.json';dump(path,plan)
    evaluations=[]
    evaluation_states=[]
    actual_evaluate=rt.evaluate
    def observed_evaluate(*args):
        evaluations.append(1)
        evaluation_states.append(rt.cpu_state(args[0]))
        metrics,predictions,truth=actual_evaluate(*args)
        # Controlled scores exercise EMA selection without claiming real accuracy.
        if ema_enabled:
            metrics=dict(metrics,macro_accuracy=1. if len(evaluations)%2==0 else 0.)
        return metrics,predictions,truth
    monkeypatch.setattr(rt,'evaluate',observed_evaluate)
    initialized=[]
    actual_initialize=rt.initialize_model
    def observed_initialize(*args):
        model,parent=actual_initialize(*args)
        initialized.append(rt.cpu_state(model))
        return model,parent
    monkeypatch.setattr(rt,'initialize_model',observed_initialize)
    averages=[]
    class ObservedAverage(WeightAverage):
        def __init__(self,*args):
            super().__init__(*args)
            assert self.count==0
            averages.append(self)
    monkeypatch.setattr(rt,'WeightAverage',ObservedAverage)
    for index,name in enumerate(stages):
        rt.train(path,'synthetic_cpu_only',name)
        payload=torch.load(tmp_path/'runs'/name/'last.pt',map_location='cpu',weights_only=False)
        check_checkpoint(payload,plan,name)
        assert payload['global_step']==plan['stages'][name]['total_updates']
        assert payload['scheduler']['last_epoch']==payload['global_step']
        assert payload['ema_enabled'] is ema_enabled
        assert ('ema' in payload) is ema_enabled
        assert payload['ema_updates']==(payload['global_step'] if ema_enabled else 0)
        if name!='v13_final':
            assert payload['metrics']['chosen']==('ema' if ema_enabled else 'raw')
            raw_state=evaluation_states[-2 if ema_enabled else -1]
            for key,value in payload['model'].items():
                torch.testing.assert_close(value,raw_state[key],rtol=0,atol=0)
        if index:
            parent=torch.load(tmp_path/'runs'/stages[index-1]/'best.pt',map_location='cpu',weights_only=False)
            for key,value in initialized[index].items():
                torch.testing.assert_close(value,choose_parent(parent)[key],rtol=0,atol=0)
    assert len(evaluations)==(6 if ema_enabled else 3)  # final all-data stage never evaluates
    assert len(averages)==(3 if ema_enabled else 0)
    final=torch.load(tmp_path/'runs/v13_final/last.pt',map_location='cpu',weights_only=False)
    assert final['metrics']['chosen']=='raw' and final['metrics']['val'] is None
    assert final['binding']['parent']['path'].endswith('s2_448/best.pt')
    assert final['binding']['parent']['weights']==('ema' if ema_enabled else 'raw')
    assert not torch.cuda.is_initialized()
