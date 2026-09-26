import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from l05_logit_penalty_support import LogitPenalty


def test_analytic_gradient_and_per_example_reduction():
    logits = torch.tensor([[1., 2., -3.], [0., -1., 4.]], requires_grad=True)
    p = LogitPenalty(.2)
    loss = p.apply(torch.zeros(2), logits)
    loss.sum().backward()
    torch.testing.assert_close(logits.grad, .2 * logits.detach() / 3)
    assert p.calls == 1 and p.examples == 2
    assert p.penalty_sum == pytest.approx(float(loss.detach().sum()))


def test_zero_and_evaluation_identity_no_rng_change():
    logits = torch.randn(4, 7, requires_grad=True)
    original = torch.randn(4)
    state = torch.get_rng_state().clone()
    assert LogitPenalty(0).apply(original, logits) is original
    p = LogitPenalty(.3)
    with torch.no_grad():
        assert p.apply(original, logits) is original
    assert p.apply(original, logits.detach()) is original
    assert p.calls == 0
    assert torch.equal(state, torch.get_rng_state())


def test_resume_counters_and_mismatch():
    p = LogitPenalty(.1)
    p.apply(torch.zeros(2), torch.ones(2, 4, requires_grad=True))
    q = LogitPenalty(.1)
    q.load_state_dict(p.state_dict())
    assert q.state_dict() == p.state_dict()
    with pytest.raises(ValueError, match='coefficient'):
        LogitPenalty(.2).load_state_dict(p.state_dict())
