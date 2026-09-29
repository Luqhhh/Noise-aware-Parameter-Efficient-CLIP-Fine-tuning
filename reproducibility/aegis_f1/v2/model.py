"""Project-native official OpenAI CLIP, full visual FT and linear head."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from aegis_clip.model import interpolate_visual_positional_embedding
from .plan import require, sha


class V2Classifier(nn.Module):
    def __init__(self, recipe):
        super().__init__()
        import clip

        path = recipe["official_checkpoint"]
        require(sha(path) == recipe["official_sha256"], "Not the official OpenAI checkpoint")
        official, _ = clip.load(path, device="cpu", jit=False)
        self.visual = official.visual.float()
        require(tuple(self.visual.conv1.weight.shape) == (768, 3, 32, 32) and
                tuple(self.visual.positional_embedding.shape) == (50, 768) and
                tuple(self.visual.proj.shape) == (768, 512), "Not CLIP ViT-B/32")
        del official  # Text tower is not part of this classification model.
        self.head = nn.Linear(512, recipe["num_classes"])
        self.gradient_checkpointing = recipe.get("gradient_checkpointing", True)

    def embed(self, images):
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
        x = visual.ln_post(x.permute(1, 0, 2)[:, 0])
        return x @ visual.proj

    def forward(self, images):
        return self.head(self.embed(images))
