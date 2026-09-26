"""Convex bounded class-wise affine calibration with a fixed identity prior."""
import hashlib
import math
import numpy as np
import torch
from scipy.optimize import minimize


def group_folds(groups, folds=3, seed=42):
    return np.array([int.from_bytes(hashlib.sha256(f'{seed}:{g}'.encode()).digest()[:8], 'big') % folds for g in groups])


def centered(scores):
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite scores [N,C]')
    return x - x.mean(1, keepdims=True)


def objective(theta, x, labels, weights, regularization=1.):
    classes=x.shape[1]; a,b=theta[:classes],theta[classes:]
    value=0.; ga=np.zeros(classes); gb=np.zeros(classes)
    for start in range(0,len(x),2048):
        block=x[start:start+2048]; y=labels[start:start+len(block)]; w=weights[start:start+len(block)]
        z=block*a+b; maximum=z.max(1); exponential=np.exp(z-maximum[:,None]); denom=exponential.sum(1)
        value+=np.dot(w,maximum+np.log(denom)-z[np.arange(len(y)),y])
        residual=exponential/denom[:,None]
        residual[np.arange(len(y)),y]-=1
        residual*=w[:,None]
        ga+=(residual*block).sum(0);gb+=residual.sum(0)
    value+=regularization*((a-1)@(a-1)+b@b)/(2*classes)
    ga+=regularization*(a-1)/classes;gb+=regularization*b/classes
    return float(value),np.concatenate([ga,gb])


def fit(scores, labels, regularization=1.):
    x=centered(scores); y=np.asarray(labels,dtype=np.int64); classes=x.shape[1]
    if y.shape!=(len(x),) or len(y)==0 or y.min()<0 or y.max()>=classes: raise ValueError('Invalid labels')
    counts=np.bincount(y,minlength=classes);present=counts>0
    weights=1/(present.sum()*counts[y])
    bound=math.log(4.)
    bounds=[(.5,2.) if p else (1.,1.) for p in present]+[(-bound,bound) if p else (0.,0.) for p in present]
    initial=np.concatenate([np.ones(classes),np.zeros(classes)])
    first,_=objective(initial,x,y,weights,regularization)
    result=minimize(objective,initial,args=(x,y,weights,regularization),method='L-BFGS-B',jac=True,bounds=bounds,
                    options={'maxiter':100,'maxfun':300,'maxls':30,'maxcor':10,'ftol':1e-12,'gtol':1e-8})
    last,gradient=objective(result.x,x,y,weights,regularization)
    projected=gradient.copy()
    for i,(lo,hi) in enumerate(bounds):
        if lo==hi or (result.x[i]<=lo+1e-10 and gradient[i]>0) or (result.x[i]>=hi-1e-10 and gradient[i]<0):projected[i]=0
    if not result.success or last>first+1e-12 or np.max(np.abs(projected))>1e-6:
        raise RuntimeError(f'Calibration did not converge: {result.message}, grad={max(abs(projected))}')
    # Independent torch/autograd check uses explicit weighted cross entropy.
    tx=torch.from_numpy(x);ty=torch.from_numpy(y);tw=torch.from_numpy(weights)
    theta=torch.tensor(result.x,requires_grad=True)
    loss=(torch.nn.functional.cross_entropy(tx*theta[:classes]+theta[classes:],ty,reduction='none')*tw).sum()
    loss=loss+regularization*((theta[:classes]-1).square().sum()+theta[classes:].square().sum())/(2*classes)
    torch_grad=torch.autograd.grad(loss,theta)[0].numpy()
    error=float(np.max(np.abs(torch_grad-gradient)))
    if abs(float(loss.detach())-last)>1e-10 or error>1e-10:raise AssertionError('Independent objective/gradient mismatch')
    return result.x[:classes],result.x[classes:],{'samples':len(y),'classes':classes,'missing_classes_frozen':np.where(~present)[0].tolist(),
        'iterations':int(result.nit),'function_evaluations':int(result.nfev),'converged':bool(result.success),
        'objective_initial':first,'objective_final':last,'projected_gradient_inf_norm':float(max(abs(projected))),
        'independent_autograd_max_abs':error,'regularization':regularization,
        'scale_min':float(min(result.x[:classes])),'scale_max':float(max(result.x[:classes])),
        'bias_min':float(min(result.x[classes:])),'bias_max':float(max(result.x[classes:]))}


def apply(scores, scale, bias):
    return centered(scores)*np.asarray(scale)+np.asarray(bias)


def metrics(prediction, labels, classes):
    y=np.asarray(labels);p=np.asarray(prediction);count=np.bincount(y,minlength=classes)
    if not np.all(count>0):raise ValueError('Aggregate metric requires all classes')
    correct=np.bincount(y[p==y],minlength=classes)
    return {'macro':float((correct/count).mean()),'micro':float((p==y).mean())}
