"""Training-only stochastic depth on whole OpenAI CLIP residual blocks."""
import numpy as np
import torch


class StochasticDepth:
    def __init__(self, max_drop=.1, seed=42, depth=12):
        if not 0<=max_drop<1 or depth<1:raise ValueError('Invalid stochastic depth recipe')
        self.max_drop,self.seed,self.depth=max_drop,seed,depth
        self.calls=self.examples=0
        self.block_calls=np.zeros(depth,dtype=np.int64)
        self.dropped=np.zeros(depth,dtype=np.int64)
        self.mask=None

    def begin(self,batch,device):
        probability=self.max_drop*(np.arange(self.depth)+1)/self.depth
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.calls]))
        keep=rng.random((self.depth,batch))>=probability[:,None]
        self.calls+=1;self.examples+=batch;self.dropped+=(~keep).sum(1)
        # Transfer all masks once per image forward; never synchronize each block.
        self.mask=torch.tensor(keep/(1-probability[:,None]),device=device,dtype=torch.float32)[:,None,:,None]

    def apply(self,index,inputs,output):
        if self.mask is None or inputs.shape!=output.shape or inputs.ndim!=3:
            raise RuntimeError('Expected sequence,batch,width residual-block tensors')
        if inputs.shape[1]!=self.mask.shape[2]:raise RuntimeError('Batch mismatch')
        self.block_calls[index]+=1
        return (inputs.float()+(output.float()-inputs.float())*self.mask[index]).to(output.dtype)

    def install(self,model):
        blocks=list(model.visual.transformer.resblocks)
        if len(blocks)!=self.depth:raise ValueError('Wrong visual block count')
        def before(module,args,kwargs):
            if module.training and torch.is_grad_enabled() and kwargs.get('images') is not None and self.max_drop>0:
                images=kwargs['images'];self.begin(len(images),images.device)
        handles=[model.register_forward_pre_hook(before,with_kwargs=True)]
        for index,block in enumerate(blocks):
            def hook(module,args,output,index=index):
                if not model.training or not torch.is_grad_enabled() or self.max_drop==0:return output
                return self.apply(index,args[0],output)
            handles.append(block.register_forward_hook(hook))
        return handles

    def state_dict(self):
        return {'max_drop':self.max_drop,'seed':self.seed,'depth':self.depth,'calls':self.calls,'examples':self.examples,
                'block_calls':self.block_calls.tolist(),'dropped':self.dropped.tolist()}

    def load_state_dict(self,state):
        for key in ('max_drop','seed','depth'):
            if state[key]!=getattr(self,key):raise ValueError('Stochastic depth recipe changed')
        for key in ('calls','examples'):
            if not isinstance(state[key],int) or state[key]<0:raise ValueError('Invalid counter')
            setattr(self,key,state[key])
        for key in ('block_calls','dropped'):
            value=np.array(state[key],dtype=np.int64)
            if value.shape!=(self.depth,) or (value<0).any():raise ValueError('Invalid block counters')
            setattr(self,key,value)
        self.mask=None
