"""Single-checkpoint V2 512/768 feature paths with a common FP32 readout."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from aegis_clip.model import interpolate_visual_positional_embedding
from .model import V2Classifier
from .plan import require
from .architecture import architecture_spec


def fold_projection(state):
    """Fuse consecutive linear maps; keep every other tensor unchanged."""
    p, w = state['visual.proj'], state['head.weight']
    require(p.ndim == w.ndim == 2 and p.shape[1] == w.shape[1], 'Incompatible linear maps')
    require(all(torch.isfinite(t).all() for t in (p, w, state['head.bias'])), 'Nonfinite readout')
    result = {k: v.detach().clone() for k, v in state.items() if k != 'visual.proj'}
    result['head.weight'] = (w.double() @ p.double().T).to(w.dtype)
    return result


class V2FeatureClassifier(V2Classifier):
    def __init__(self, recipe, feature_dim):
        require(feature_dim in (512, 768), 'Unsupported V2 feature dimension')
        super().__init__(recipe)
        self.feature_dim = feature_dim
        if feature_dim == 768:
            self.visual.proj = None
            self.head = nn.Linear(768, recipe['num_classes'])

    def preprojection(self, images):
        # Keep the original V2 visual computation, including its checkpointing.
        visual = self.visual
        x = visual.conv1(images.to(visual.conv1.weight.dtype))
        grid = tuple(x.shape[-2:])
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        token = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat((token, x), dim=1)
        x = x + interpolate_visual_positional_embedding(visual.positional_embedding, grid).to(x.dtype)
        x = visual.ln_pre(x).permute(1, 0, 2)
        for block in visual.transformer.resblocks:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return visual.ln_post(x.permute(1, 0, 2)[:, 0])

    def readout(self, features):
        # Both arms share this boundary; BF16 matrix reassociation is not a treatment.
        with torch.autocast(device_type=features.device.type, enabled=False):
            x = features.float()
            if self.visual.proj is not None:
                x = x @ self.visual.proj
            return self.head(x)

    def forward(self, images):
        return self.readout(self.preprojection(images))

    def embed(self, images):
        features = self.preprojection(images)
        if self.feature_dim == 768:
            return features
        with torch.autocast(device_type=features.device.type, enabled=False):
            return features.float() @ self.visual.proj


def build_classifier(recipe):
    """Preserve legacy 512 numerics; opt in to the official-initialized 768 route."""
    spec = architecture_spec(recipe)
    if spec["feature_dim"] == 512:
        return V2Classifier(recipe)
    return V2FeatureClassifier(recipe, 768)
