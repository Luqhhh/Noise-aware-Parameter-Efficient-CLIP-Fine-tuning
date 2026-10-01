"""Boundaries and independent head computation for the fixed queued probe."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('post768_job',ROOT/'scripts/run_v1_post768_job.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_balancing_equalizes_reliable_mass_and_keeps_zeros_zero():
    weights=torch.tensor([.2,.6,1.,0.,.3])
    labels=torch.tensor([0,0,1,1,2])
    corrected,mass=module.equal_class_mass(weights,labels,3)
    totals=torch.zeros(3).scatter_add_(0,labels,corrected)
    torch.testing.assert_close(totals,torch.full((3,),weights.sum()/3))
    assert corrected[3]==0
    torch.testing.assert_close(corrected.sum(),weights.sum())
    assert mass.tolist()==pytest.approx([.8,1.,.3])


@pytest.mark.parametrize('weights',[torch.tensor([0.,1.]),torch.tensor([-1.,1.]),torch.tensor([float('nan'),1.])])
def test_balancing_rejects_missing_or_invalid_supervision(weights):
    with pytest.raises(ValueError):module.equal_class_mass(weights,torch.tensor([0,1]),2)


def test_paired_forward_is_exactly_two_independent_classifiers():
    torch.manual_seed(7)
    body=nn.Linear(5,8)
    heads=[nn.Linear(8,3),nn.Linear(8,3)]
    model=module.IndependentHeads(body,heads)
    value=torch.randn(4,5)
    result=model(value)
    for i,head in enumerate(heads):
        torch.testing.assert_close(result[:,i*3:(i+1)*3],head(body(value)),rtol=0,atol=0)
    assert result.shape==(4,6)
