import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from l05_mixstyle_support import PatchStemMixStyle


def test_eval_and_control_are_exact_identity():
    module=torch.nn.Identity();x=torch.randn(4,7,5,5)
    mixer=PatchStemMixStyle(1.)
    module.eval();assert mixer(module,(),x) is x
    module.train();assert PatchStemMixStyle(0.)(module,(),x) is x
    assert mixer.calls==0


def test_mixing_preserves_shape_has_finite_gradient_and_does_not_advance_torch_rng():
    module=torch.nn.Identity().train();x=torch.randn(4,7,5,5,requires_grad=True)
    x.data[1].mul_(3).add_(4)
    state=torch.get_rng_state().clone()
    mixer=PatchStemMixStyle(1.);y=mixer(module,(),x)
    assert torch.equal(state,torch.get_rng_state())
    assert y.shape==x.shape and not torch.equal(y,x)
    y.square().mean().backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert mixer.applied==1 and mixer.examples==4


def test_resume_reproduces_next_training_forward():
    module=torch.nn.Identity().train();x=torch.randn(4,7,5,5)
    a=PatchStemMixStyle(1.);a(module,(),x)
    b=PatchStemMixStyle(1.);b.load_state_dict(a.state_dict())
    assert torch.equal(a(module,(),x),b(module,(),x))


def test_hook_adds_no_parameters_and_is_removable():
    conv=torch.nn.Conv2d(3,8,4,4);keys=list(conv.state_dict())
    x=torch.randn(4,3,16,16);expected=conv(x)
    handle=conv.register_forward_hook(PatchStemMixStyle(1.))
    assert not torch.equal(conv(x),expected)
    assert list(conv.state_dict())==keys
    handle.remove();assert torch.equal(conv(x),expected)


def test_parent_training_flag_controls_frozen_eval_stem():
    from l05_mixstyle_support import install_mixstyle
    model=torch.nn.Module();model.visual=torch.nn.Module()
    model.visual.conv1=torch.nn.Conv2d(3,8,4,4)
    model.train();model.visual.conv1.eval().requires_grad_(False)
    x=torch.randn(4,3,16,16);expected=model.visual.conv1(x)
    mixer=PatchStemMixStyle(1.);handle=install_mixstyle(model,mixer)
    assert not torch.equal(model.visual.conv1(x),expected) and mixer.applied==1
    model.eval();assert torch.equal(model.visual.conv1(x),expected)
    assert mixer.applied==1
    handle.remove()
