"""Cover control equivalence, mixing endpoints, gradients, and microbatch mass."""
import sys
from pathlib import Path
import numpy as np
import torch
import pytest

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from v1_cosine_margin_loss import component_loss,margin_mixup_loss,mixed_backward,logical_update


def tensors():
    torch.manual_seed(42)
    z=torch.randn(4,7,dtype=torch.float64,requires_grad=True)
    q=torch.nn.functional.one_hot(torch.tensor([1,3,5,2]),7).double()*.9+.1/7
    w=torch.tensor([.2,.8,.4,.7],dtype=torch.float64);perm=torch.tensor([3,2,0,1])
    return z,q,w,perm


def test_zero_margin_is_original_weighted_mixup_loss_and_gradient():
    z,q,w,p=tensors();mass=.63*w[:,None]*q+.37*w[p,None]*q[p]
    control=-(mass*z.log_softmax(1)).sum()/w.sum()
    candidate=margin_mixup_loss(z,q,w,p,.63,torch.tensor(20.),0.)
    assert torch.allclose(control,candidate,atol=1e-12,rtol=0)
    assert torch.allclose(torch.autograd.grad(control,z,retain_graph=True)[0],torch.autograd.grad(candidate,z)[0],atol=1e-12,rtol=0)


@pytest.mark.parametrize('lam',[0.,1.])
def test_unmixed_endpoint_matches_single_source_margin(lam):
    z,q,w,p=tensors();scale=torch.tensor(10.,dtype=torch.float64)
    expected=(w*component_loss(z if lam else z[torch.argsort(p)],q,scale,.05)).sum()/w.sum()
    assert torch.allclose(margin_mixup_loss(z,q,w,p,lam,scale,.05),expected,atol=1e-12,rtol=0)


def test_margin_increases_unsmoothed_positive_loss_and_preserves_scale_gradient():
    z,q,w,p=tensors();hard=torch.nn.functional.one_hot(q.argmax(1),7).double()
    scale=torch.tensor(10.,dtype=torch.float64,requires_grad=True)
    loss=component_loss(z,hard,scale,.05)
    assert (loss>component_loss(z,hard,scale,0.)).all()
    assert torch.autograd.grad(loss.sum(),scale)[0]>0


def test_microbatch_reduction_preserves_the_logical_weighted_gradient():
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.randn(7,3));self.logit_scale=torch.nn.Parameter(torch.tensor(2.))
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.head=Head()
        def forward(self,x):return x@self.head.weight.T
    z,q,w,p=tensors();q=q.float();w=w.float();x=torch.randn(4,3);m=Model();scaler=torch.amp.GradScaler('cpu',enabled=False)
    full=margin_mixup_loss(m(.63*x+.37*x[p]),q,w,p,.63,m.head.logit_scale.exp(),.05)
    full.backward();expected={n:v.grad.clone() for n,v in m.named_parameters()};m.zero_grad()
    mixed_backward(m,x,q,w,p,.63,2,scaler,.05,False)
    for n,v in m.named_parameters():assert torch.allclose(v.grad,expected[n],rtol=1e-6,atol=1e-7)


@pytest.mark.parametrize('bad_gradient', [False, True])
def test_logical_update_steps_once_only_for_finite_gradients(bad_gradient):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3, 7, bias=False)
            self.head.logit_scale = torch.nn.Parameter(torch.tensor(2.))
        def forward(self, x):
            return self.head(x)
    class CountedSGD(torch.optim.SGD):
        calls = 0
        def step(self, *args, **kwargs):
            self.calls += 1
            return super().step(*args, **kwargs)
    _, q, w, permutation = tensors()
    model = Model()
    optimizer = CountedSGD(model.parameters(), lr=.01)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    if bad_gradient:
        model.head.weight.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
        with pytest.raises(ValueError, match='Persistent nonfinite'):
            logical_update(model, optimizer, torch.randn(4, 3), q.float(), w.float(),
                           permutation, .63, 2, scaler, .05, 1., amp=False)
        assert optimizer.calls == 0
        assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
    else:
        loss, numeric = logical_update(model, optimizer, torch.randn(4, 3), q.float(), w.float(),
                                       permutation, .63, 2, scaler, .05, 1., amp=False)
        assert optimizer.calls == 1 and np.isfinite(loss) and np.isfinite(numeric['gradient_norm'])
        assert numeric['amp_retries'] == []
        assert not torch.equal(model.head.weight, initial['head.weight'])
