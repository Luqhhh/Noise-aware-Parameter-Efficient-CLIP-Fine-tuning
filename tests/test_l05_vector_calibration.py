import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from l05_vector_calibration_support import objective,fit,apply,group_folds


def test_analytic_gradient_matches_independent_torch():
    rng=np.random.default_rng(3);x=rng.normal(size=(13,4));y=np.arange(13)%4
    w=1/(4*np.bincount(y)[y]);theta=rng.normal(size=8)
    value,grad=objective(theta,x,y,w)
    t=torch.tensor(theta,requires_grad=True);z=torch.tensor(x)*t[:4]+t[4:]
    loss=(torch.nn.functional.cross_entropy(z,torch.tensor(y),reduction='none')*torch.tensor(w)).sum()+((t[:4]-1).square().sum()+t[4:].square().sum())/8
    assert abs(value-loss.item())<1e-12
    np.testing.assert_allclose(grad,torch.autograd.grad(loss,t)[0].numpy(),atol=1e-12)


def test_fit_absent_class_is_identity_and_objective_decreases():
    rng=np.random.default_rng(44);x=rng.normal(size=(70,4));y=np.arange(70)%3
    a,b,report=fit(x,y)
    assert a[3]==1 and b[3]==0
    assert report['objective_final']<report['objective_initial']
    assert report['missing_classes_frozen']==[3] and report['projected_gradient_inf_norm']<1e-6


def test_apply_is_invariant_to_row_offsets_and_chunking():
    rng=np.random.default_rng(2);x=rng.normal(size=(12,5));a=rng.uniform(.5,2,5);b=rng.normal(size=5)
    np.testing.assert_allclose(apply(x,a,b),apply(x+rng.normal(size=(12,1)),a,b),atol=1e-14)
    np.testing.assert_array_equal(apply(x,a,b),np.concatenate([apply(x[:4],a,b),apply(x[4:],a,b)]))


def test_content_groups_never_cross_folds():
    groups=['g1','g2','g1','g3','g2','g4'];ids=group_folds(groups)
    assert ids[0]==ids[2] and ids[1]==ids[4]
    assert set(ids)<=set(range(3))
    np.testing.assert_array_equal(ids,group_folds(groups))
