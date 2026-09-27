import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from l05_stochastic_depth_support import StochasticDepth


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(.2))
    def forward(self,x):return x+self.weight*x


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.visual=nn.Module();self.visual.transformer=nn.Module()
        self.visual.transformer.resblocks=nn.Sequential(*[Block() for _ in range(12)])
    def forward(self,*,images):
        x=images.flatten(2).permute(2,0,1)
        return self.visual.transformer.resblocks(x)


def test_formula_and_gradient_with_independent_numpy_mask():
    depth=StochasticDepth(max_drop=.5);depth.begin(8,'cpu')
    rng=np.random.default_rng(np.random.SeedSequence([42,0]));p=.5*(np.arange(12)+1)/12
    keep=rng.random((12,8))>=p[:,None]
    x=torch.randn(5,8,3,requires_grad=True);w=torch.tensor(.3,requires_grad=True)
    y=depth.apply(11,x,x+w*x)
    expected=x.detach().numpy()+(w.detach().numpy()*x.detach().numpy())*(keep[11]/(1-p[11]))[None,:,None]
    np.testing.assert_allclose(y.detach().numpy(),expected,atol=3e-7)
    y.sum().backward()
    expected_grad=np.broadcast_to(1+.3*(keep[11]/(1-p[11]))[None,:,None],x.shape)
    np.testing.assert_allclose(x.grad.numpy(),expected_grad,atol=1e-7)
    assert torch.isfinite(w.grad) and np.array_equal(depth.dropped,(~keep).sum(1))


def test_eval_and_no_grad_are_bitwise_inert_and_weights_unchanged():
    model=Tiny();x=torch.randn(4,3,2,2);expected=model(images=x);state={k:v.clone() for k,v in model.state_dict().items()}
    depth=StochasticDepth();depth.install(model)
    model.eval();assert torch.equal(model(images=x),expected)
    model.train()
    with torch.no_grad():assert torch.equal(model(images=x),expected)
    assert depth.calls==0 and all(torch.equal(state[k],v) for k,v in model.state_dict().items())
    changed=model(images=x);assert not torch.equal(changed,expected)
    assert depth.calls==1 and np.all(depth.block_calls==1)


def test_resume_preserves_masks_and_shared_rng():
    depth=StochasticDepth();before=torch.random.get_rng_state().clone();depth.begin(16,'cpu')
    restored=StochasticDepth();restored.load_state_dict(depth.state_dict())
    depth.begin(16,'cpu');restored.begin(16,'cpu')
    assert torch.equal(depth.mask,restored.mask) and depth.state_dict()==restored.state_dict()
    assert torch.equal(before,torch.random.get_rng_state())


def test_disabled_identity():
    model=Tiny();x=torch.randn(2,3,2,2);expected=model(images=x)
    depth=StochasticDepth(max_drop=0);depth.install(model)
    assert torch.equal(model(images=x),expected) and depth.calls==0
