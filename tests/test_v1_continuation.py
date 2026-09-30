"""CPU checks of real conversion, weighted logical updates, and evidence boundaries."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from clip.model import VisionTransformer

from aegis_clip.v1_strategy import V1Classifier, trainable_state, weighted_mixup_loss
from aegis_clip.v1_pipeline import load_artifact
from v1_continuation.model import convert_parent
from v1_continuation.plan import FIXED, PARENT_FILES, compatible_parent_code, load_config, validate_support
from v1_continuation.runtime import fit, mixed_backward, optimizer_for, comparison, scheduler_for

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def parent(size=32, patch=8):
    torch.manual_seed(11)
    model = V1Classifier(VisionTransformer(size, patch, 32, 2, 4, 16), 3,
                         image_size=size, rank=2, alpha=4., blocks=2)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.03)
    return model


def test_selected_swa_loaded_before_merge_preserves_function_head_and_full_gradients():
    model = parent()
    head, head_state = model.head, copy.deepcopy(model.head.state_dict())
    images = torch.randn(2, 3, 32, 32)
    converted, report = convert_parent(model, 'WFT448', images, image_size=32)
    assert converted.head is head and report['merged_modules'] == 8 and report['passed']
    assert all(p.requires_grad for p in converted.parameters())
    assert not any('lora_' in name or 'parametrizations' in name for name, _ in converted.named_parameters())
    for key, value in head.state_dict().items():
        assert torch.equal(value, head_state[key])
    converted.train()
    converted(images).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in converted.parameters())
    with pytest.raises(ValueError, match='coverage'):
        convert_parent(model, 'WFT448', images, image_size=32)


def test_lr512_is_real_512_input_with_257_tokens_and_only_lora_head_trainable():
    model = parent(size=448, patch=32)
    converted, report = convert_parent(model, 'LR512', torch.randn(1,3,448,448))
    assert report['tokens'] == 257 and converted.visual.positional_embedding.shape == (257,32)
    assert report['merged_modules'] == 0
    assert all('lora_' in key or key.startswith('head.') for key in trainable_state(converted))
    assert converted(torch.randn(1,3,512,512)).shape == (1,3)
    with pytest.raises(ValueError, match='Actual input'):
        converted(torch.randn(1,3,448,448))


def test_global_weighted_mixup_gradients_equal_micro_accumulation_including_short_micro():
    torch.manual_seed(5)
    model = nn.Linear(7,3)
    micro = copy.deepcopy(model)
    images, probabilities = torch.randn(11,7), torch.randn(11,3).softmax(1)
    weights = torch.tensor([0.,1.,.2,.3,0.,.5,.8,.7,.1,.9,.4])
    permutation = torch.tensor([10,9,8,7,6,5,4,3,2,1,0])
    lam = .61
    expected = weighted_mixup_loss(model(lam*images+(1-lam)*images[permutation]),
                                   probabilities, weights, permutation, lam)
    expected.backward()
    scaler = torch.amp.GradScaler('cuda',enabled=False)
    actual = mixed_backward(micro, images, probabilities, weights, permutation, lam, 3, scaler)
    assert actual == pytest.approx(float(expected.detach()),abs=1e-6)
    for a,b in zip(model.parameters(),micro.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-7)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(7,8), nn.LayerNorm(8))
        self.head = nn.Linear(8,3)
    def forward(self,x):
        return self.head(self.visual(x))


def tiny_plan():
    return dict(config=dict(route='WFT448', recipe=copy.deepcopy(FIXED['WFT448']), seed=42,
                            micro_batch_size=2, partition='train_dev',parent_checkpoint='parent.pt'),
                inherited_train=dict(amp=False,label_smoothing=.1,mixup_alpha=.2), classes=['a','b','c'])


def test_wft_optimizer_preserves_groups_and_no_decay_on_vectors():
    model = TinyClassifier()
    opt = optimizer_for(model,FIXED['WFT448'])
    for group in opt.param_groups:
        for parameter in group['params']:
            assert group['weight_decay'] == (.05 if parameter.ndim>=2 else 0.)
            assert group['lr'] == (5e-5 if any(parameter is x for x in model.head.parameters()) else 5e-6)
    scheduler = scheduler_for(opt,FIXED['WFT448'],8)
    assert opt.param_groups[0]['lr'] == pytest.approx(2.5e-6)
    for _ in range(32):
        opt.step(); scheduler.step()
    assert opt.param_groups[0]['lr'] == pytest.approx(0.)


def test_fixed_four_epochs_save_raw_ema_and_mean_only_ema2_to_4(tmp_path):
    plan, binding = tiny_plan(), dict(test='single_trajectory')
    model = TinyClassifier()
    dataset = TensorDataset(torch.randn(7,7),torch.arange(7))
    loader = DataLoader(dataset,batch_size=3)
    labels = np.arange(7)%3
    targets = dict(targets=labels,labels=labels,weights=np.array([1,.2,.4,.5,.3,.8,.9]),original_alpha=np.zeros(7))
    evaluated = []
    def evaluation(model,epoch,policy):
        evaluated.append((epoch,policy))
        return dict(score=0.)
    history,_ = fit(model,loader,targets,plan,tmp_path/'run',evaluation=evaluation,binding=binding)
    assert len(history) == 4 and history[-1]['optimizer_updates'] == 12
    assert evaluated == [(e,p) for e in range(1,5) for p in ('raw','ema')]+[(4,'ema_swa_2_4')]
    swa = load_artifact(tmp_path/'run/ema_swa.pt',binding)
    last = load_artifact(tmp_path/'run/selected.pt',binding)
    assert swa['average_epochs'] == [2,3,4] and last['selected_policy'] == 'last_ema'
    for name, value in swa['selected_state'].items():
        expected = torch.stack([load_artifact(tmp_path/f'run/epoch_{e:02d}_ema.pt',binding)['selected_state'][name]
                                for e in (2,3,4)]).mean(0)
        torch.testing.assert_close(value,expected)
    with pytest.raises(FileExistsError):
        fit(model,loader,targets,plan,tmp_path/'run',binding=binding)


def test_full_rejects_overlapping_validation_callback(tmp_path):
    plan=tiny_plan(); plan['config']['partition']='full_train'
    with pytest.raises(ValueError,match='overlapping'):
        fit(TinyClassifier(),DataLoader(TensorDataset(torch.randn(7,7),torch.arange(7)),batch_size=3),
            {},plan,tmp_path/'run',evaluation=lambda *a: {},binding={})


@pytest.mark.parametrize('route',['wft448','lr512'])
@pytest.mark.parametrize('partition',['dev','full'])
def test_recipes_are_fixed(route,partition,tmp_path):
    path=ROOT/f'configs/v1_continuation/{route}_{partition}.yaml'
    cfg=load_config(path)
    assert cfg['recipe']['epochs'] == 4 and cfg['recipe']['average_epochs'] == [2,3,4]
    import yaml
    cfg['recipe']['backbone_lr']*=2
    changed=tmp_path/'recipe.yaml';changed.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='frozen'):
        load_config(changed)


def test_incomplete_probe_or_wrong_route_cannot_promote_full(tmp_path):
    path=tmp_path/'report.json'
    path.write_text(json.dumps(dict(status='probe',route='WFT448')))
    with pytest.raises(ValueError,match='complete matching DEV'):
        validate_support(path,'WFT448','some note')


def test_small_local_loss_is_reported_without_automatic_veto_or_promotion():
    plan=dict(classes=['a','b','c'],groups=dict(tail_classes=[2],other_classes=[0,1],
                                              few_support_classes=[2],head_classes=[0]))
    result=comparison(plan,[0,1,2],[0,1,1],[1,1,2])
    assert result['slices']['all']['net'] == 0
    assert result['slices']['tail']['net'] == 1
    assert result['decision']=='evidence_requires_review' and result['automatic_full'] is False


def test_archived_parent_code_allows_only_unused_calibration_changes(tmp_path):
    current, archived = tmp_path/'current',tmp_path/'archive'
    for relative in PARENT_FILES:
        for root in (current,archived):
            path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('def same():\n    return 1\n')
    pipeline=PARENT_FILES[1]
    (current/pipeline).write_text('def same():\n    return 1\ndef calibrate():\n    return "new reporting"\n')
    (archived/pipeline).write_text('def same():\n    return 1\ndef calibrate():\n    return "old reporting"\n')
    assert compatible_parent_code(current,archived)
    (archived/pipeline).write_text('def same():\n    return 2\ndef calibrate():\n    return "old reporting"\n')
    with pytest.raises(ValueError,match='effective pipeline'):
        compatible_parent_code(current,archived)


def test_execution_refuses_busy_cuda_before_initializing_it(monkeypatch):
    from types import SimpleNamespace
    import v1_continuation.runtime as rt
    monkeypatch.setattr(rt.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='8916\n'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA touched while another job is active'))
    with pytest.raises(RuntimeError,match='occupied'):
        rt.require_idle_cuda()
