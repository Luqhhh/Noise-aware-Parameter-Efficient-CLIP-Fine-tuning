"""Training-only patch-stem statistics mixing; no persistent model parameters.

Formula reference: Zhou et al., Domain Generalization with MixStyle (ICLR 2021).
This CLIP patch-stem placement and independent NumPy RNG are experiment choices.
"""
import numpy as np
import torch


class PatchStemMixStyle:
    def __init__(self, probability=.5, alpha=.1, seed=42):
        if not 0 <= probability <= 1 or alpha <= 0:
            raise ValueError('Invalid mixing probability or beta concentration')
        self.probability, self.alpha, self.seed = probability, alpha, seed
        self.calls = self.applied = self.examples = 0

    def __call__(self, module, inputs, output):
        if not module.training or self.probability == 0:
            return output
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.calls]))
        self.calls += 1
        if len(output) < 2 or rng.random() >= self.probability:
            return output
        if output.ndim != 4:
            raise ValueError('MixStyle hook requires [N,C,H,W] patch-stem output')
        # Rotation excludes self-pairing without advancing the trainer's torch RNG.
        shift = int(rng.integers(1, len(output)))
        coefficient = torch.as_tensor(rng.beta(self.alpha, self.alpha, len(output)),
                                      device=output.device, dtype=torch.float32)[:, None, None, None]
        work = output.float()
        location = work.mean((2,3), keepdim=True).detach()
        scale = (work.var((2,3), unbiased=False, keepdim=True)+1e-6).sqrt().detach()
        mixed_location = coefficient*location+(1-coefficient)*location.roll(shift,0)
        mixed_scale = coefficient*scale+(1-coefficient)*scale.roll(shift,0)
        result = ((work-location)/scale)*mixed_scale+mixed_location
        self.applied += 1; self.examples += len(output)
        return result.to(output.dtype)

    def state_dict(self):
        return dict(calls=self.calls, applied=self.applied, examples=self.examples,
                    probability=self.probability, alpha=self.alpha, seed=self.seed)

    def load_state_dict(self, state):
        for key in ('probability','alpha','seed'):
            if state[key] != getattr(self,key):
                raise ValueError('MixStyle resume recipe mismatch')
        self.calls, self.applied, self.examples = (int(state[k]) for k in ('calls','applied','examples'))


def install_mixstyle(model, mixer):
    # Aegis intentionally keeps its frozen conv1 in eval even during full FT.
    # Gate on the enclosing classifier's mode, not the frozen stem's mode.
    return model.visual.conv1.register_forward_hook(
        lambda _stem, inputs, output: mixer(model, inputs, output))
