import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from l05_rsc_support import RepresentationChallenge, challenge_mask


def fixture():
    torch.manual_seed(73)
    model = nn.Module()
    model.classifier = nn.Linear(12, 5)
    x = torch.randn(8, 12, requires_grad=True)
    y = torch.arange(8) % 5
    return model, x, y


def test_mask_matches_independent_autograd_numpy_and_backward():
    model, x, y = fixture()
    head = model.classifier
    logits = head(x)
    gradient = torch.autograd.grad(logits.gather(1, y[:, None]).sum(), x)[0].numpy()
    mask_f = gradient < np.percentile(gradient, 100*(1-1/3), axis=1)[:, None]
    with torch.no_grad():
        changes = (head(x).softmax(1)-head(x*torch.from_numpy(mask_f)).softmax(1)).gather(1,y[:,None]).squeeze(1).numpy()
    expected = mask_f | (changes < np.percentile(changes,100*(1-1/3)))[:,None]
    mask = challenge_mask(x, head, y)
    assert np.array_equal(mask.numpy(), expected)
    rsc = RepresentationChallenge(); handle = rsc.install(model)
    output = rsc.apply(head(x), F.one_hot(y,5).float())
    expected_logits = F.linear(x*torch.from_numpy(expected), head.weight, head.bias)
    assert torch.equal(output, expected_logits)
    F.cross_entropy(output,y).backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(head.weight.grad).all()
    assert (x.grad[~mask] == 0).all() and x.grad[mask].abs().sum() > 0
    assert rsc.muted_channels == int((~mask).sum())
    handle.remove()


def test_eval_no_grad_and_resume_are_inert():
    model, x, y = fixture(); rsc=RepresentationChallenge(); rsc.install(model)
    rsc.apply(model.classifier(x),F.one_hot(y,5).float())
    state=rsc.state_dict(); restored=RepresentationChallenge(); restored.load_state_dict(state)
    assert restored.state_dict()==state
    model.eval(); original=model.classifier(x)
    assert rsc.apply(original,F.one_hot(y,5).float()) is original
    model.train()
    with torch.no_grad():
        original=model.classifier(x)
        assert rsc.apply(original,F.one_hot(y,5).float()) is original
    assert rsc.state_dict()==state


def test_reject_wrong_forward_or_soft_targets():
    model,x,y=fixture(); rsc=RepresentationChallenge(); rsc.install(model)
    logits=model.classifier(x)
    with pytest.raises(RuntimeError): rsc.apply(logits.clone(),F.one_hot(y,5).float())
    with pytest.raises(ValueError): rsc.apply(logits,torch.full((8,5),.2))


def test_small_batch_ties_and_singleton_finite():
    model,x,y=fixture()
    for n in (1,3,4):
        mask=challenge_mask(x[:n],model.classifier,y[:n])
        assert mask.dtype==torch.bool and mask.shape==x[:n].shape
        assert (~mask).any()
    with torch.no_grad(): model.classifier.weight.fill_(0)
    mask=challenge_mask(x,model.classifier,y)
    assert not mask.any()  # strict-percentile tie behavior is deliberate
