import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from l05_fourier_mix_support import amplitude_mix,FourierMix


def test_fft_matches_independent_full_numpy_transform():
    rng=np.random.default_rng(30);x=rng.uniform(0,1,(3,3,40,48)).astype('float32');d=np.roll(x,1,0);lam=np.array([.2,.5,.8])
    z=np.fft.fft2(x.astype(np.float64));other=np.fft.fft2(d.astype(np.float64));h,w=x.shape[-2:]
    ky=np.minimum(np.arange(h),h-np.arange(h));kx=np.minimum(np.arange(w),w-np.arange(w))
    mask=(ky<=int(h*.05))[:,None] & (kx<=int(w*.05))[None,:]
    amplitude=np.where(mask,(1-lam[:,None,None,None])*abs(z)+lam[:,None,None,None]*abs(other),abs(z))
    expected=np.fft.ifft2(amplitude*np.exp(1j*np.angle(z))).real
    actual=amplitude_mix(torch.from_numpy(x),torch.from_numpy(d),lam).numpy()
    np.testing.assert_allclose(actual,expected,atol=1e-6)
    np.testing.assert_allclose(amplitude_mix(torch.from_numpy(x),torch.from_numpy(d),np.zeros(3)).numpy(),x,atol=1e-6)


def test_resume_no_shared_rng_mutation_and_valid_range():
    x=torch.linspace(0,1,4*3*32*32).reshape(4,3,32,32)
    mixer=FourierMix(probability=1);state=torch.random.get_rng_state().clone()
    _=mixer.transform(x,(0,0,0),(1,1,1));restored=FourierMix(probability=1);restored.load_state_dict(mixer.state_dict())
    result=mixer.transform(x,(0,0,0),(1,1,1));repeat=restored.transform(x,(0,0,0),(1,1,1))
    assert torch.equal(result,repeat) and torch.equal(state,torch.random.get_rng_state())
    assert result.min()>=0 and result.max()<=1 and mixer.pixel_l1_sum>0
    assert FourierMix(probability=0).transform(x,(0,0,0),(1,1,1)) is x


class Dummy(nn.Module):
    def forward(self,*,images,return_features=False):return images


def test_hook_only_global_training_and_no_mutation():
    model=Dummy();mixer=FourierMix(probability=1)
    preprocess=SimpleNamespace(transforms=[SimpleNamespace(mean=(0,0,0),std=(1,1,1))]);mixer.install(model,preprocess)
    images=torch.stack([torch.zeros(3,384,384),torch.ones(3,384,384)])
    original=images.clone()
    assert model(images=images) is images
    assert mixer.calls==0
    with torch.no_grad():assert model(images=images,return_features=True) is images
    model.eval();assert model(images=images,return_features=True) is images
    model.train();changed=model(images=images,return_features=True)
    assert not torch.equal(changed,images) and torch.equal(images,original)
    assert mixer.calls==1 and mixer.applied==1


def test_zero_spectrum_and_singleton_are_finite():
    x=torch.zeros(2,3,32,32);donor=torch.ones_like(x)
    actual=amplitude_mix(x,donor,torch.ones(2))
    assert torch.isfinite(actual).all() and torch.allclose(actual,donor)
    one=donor[:1];mixer=FourierMix(probability=1)
    assert mixer.transform(one,(0,0,0),(1,1,1)) is one and mixer.applied==0
