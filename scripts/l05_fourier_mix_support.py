"""Training-only low-frequency amplitude mixing in RGB space."""
import numpy as np
import torch


def amplitude_mix(rgb, donor, coefficient, radius=.05):
    """Mix a symmetric low-frequency rectangle; return unclipped RGB."""
    if rgb.shape!=donor.shape or rgb.ndim!=4 or rgb.shape[1]!=3:
        raise ValueError('Expected matching RGB batches')
    if not 0<=radius<.5:raise ValueError('Invalid frequency radius')
    with torch.no_grad(),torch.autocast(device_type=rgb.device.type,enabled=False):
        x=rgb.float(); d=donor.float();h,w=x.shape[-2:]
        original=torch.fft.rfft2(x);other=torch.fft.rfft2(d)
        amplitude=original.abs(); donor_amplitude=other.abs()
        ky=torch.arange(h,device=x.device);kx=torch.arange(w//2+1,device=x.device)
        mask=(torch.minimum(ky,h-ky)<=int(h*radius))[:,None] & (kx<=int(w*radius))[None,:]
        lam=torch.as_tensor(coefficient,device=x.device,dtype=torch.float32).reshape(-1,1,1,1)
        if lam.shape[0]!=len(x) or bool(((lam<0)|(lam>1)).any()):raise ValueError('Invalid mixing coefficients')
        mixed=torch.where(mask,(1-lam)*amplitude+lam*donor_amplitude,amplitude)
        phase=torch.where(amplitude>0,original/amplitude.clamp_min(torch.finfo(torch.float32).tiny),torch.ones_like(original))
        return torch.fft.irfft2(mixed*phase,s=(h,w))


class FourierMix:
    def __init__(self, probability=.5, radius=.05, seed=42):
        if not 0<=probability<=1 or radius!=.05:raise ValueError('Unsupported fixed recipe')
        self.probability,self.radius,self.seed=probability,radius,seed
        self.calls=self.applied=self.examples=self.clipped_values=self.total_values=0
        self.pixel_l1_sum=0.

    def transform(self, images, mean, std):
        if self.probability==0:return images
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.calls]));self.calls+=1
        if len(images)<2 or rng.random()>=self.probability:return images
        shift=int(rng.integers(1,len(images)))
        coefficients=rng.uniform(0,1,len(images))
        with torch.no_grad(),torch.autocast(device_type=images.device.type,enabled=False):
            mean=torch.tensor(mean,device=images.device).reshape(1,3,1,1)
            std=torch.tensor(std,device=images.device).reshape(1,3,1,1)
            rgb=images.float()*std+mean
            mixed=amplitude_mix(rgb,rgb.roll(shift,0),coefficients,self.radius)
            self.clipped_values+=int(((mixed<0)|(mixed>1)).sum())
            self.total_values+=mixed.numel()
            clipped=mixed.clamp(0,1)
            self.pixel_l1_sum+=float((clipped-rgb).abs().sum())
            self.applied+=1;self.examples+=len(images)
            return ((clipped-mean)/std).to(images.dtype)

    def install(self, model, preprocess):
        normalizer=preprocess.transforms[-1]
        mean,std=tuple(normalizer.mean),tuple(normalizer.std)
        def before(module,args,kwargs):
            # The pinned trainer marks only the global classification/anchor branch
            # with return_features=True. Local crops and no-grad attention stay raw.
            if not module.training or not torch.is_grad_enabled() or not kwargs.get('return_features',False):return None
            if 'images' not in kwargs:return None
            assert kwargs['images'].shape[-2:]==(384,384)
            updated=dict(kwargs);updated['images']=self.transform(kwargs['images'],mean,std)
            return args,updated
        return model.register_forward_pre_hook(before,with_kwargs=True)

    def state_dict(self):
        return {key:getattr(self,key) for key in ('probability','radius','seed','calls','applied','examples','clipped_values','total_values','pixel_l1_sum')}

    def load_state_dict(self,state):
        for key in ('probability','radius','seed'):
            if state[key]!=getattr(self,key):raise ValueError('FourierMix recipe changed on resume')
        for key in ('calls','applied','examples','clipped_values','total_values'):
            if not isinstance(state[key],int) or state[key]<0:raise ValueError('Invalid counter')
            setattr(self,key,state[key])
        if not np.isfinite(state['pixel_l1_sum']) or state['pixel_l1_sum']<0:raise ValueError('Invalid pixel counter')
        self.pixel_l1_sum=float(state['pixel_l1_sum'])
