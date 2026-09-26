"""Stateless rectangular inference using the same CLIP visual parameters."""
import math
import torch
from torch.nn import functional as F


def rectangle_size(width, height, short_edge=384, long_cap=576, patch=32):
    if min(width, height) <= 0 or short_edge % patch or long_cap % patch:
        raise ValueError('Invalid image or patch geometry')
    scale = min(short_edge / min(width, height), long_cap / max(width, height))
    return tuple(max(patch, min(long_cap, int(math.floor(v * scale / patch + .5)) * patch))
                 for v in (width, height))


def rectangular_position(position, height, width):
    side = math.isqrt(len(position) - 1)
    if side * side != len(position) - 1 or min(height, width) <= 0:
        raise ValueError('Expected square source positional grid')
    if (height, width) == (side, side):
        return position
    patches = position[1:].float().reshape(side, side, -1).permute(2, 0, 1)[None]
    patches = F.interpolate(patches, (height, width), mode='bicubic', align_corners=False, antialias=False)
    patches = patches[0].permute(1, 2, 0).reshape(height * width, -1).to(position.dtype)
    return torch.cat((position[:1], patches), dim=0)


def rectangular_visual(visual, images):
    if images.shape[-2] % 32 or images.shape[-1] % 32:
        raise ValueError('Image dimensions must align with the B/32 patch grid')
    values = visual.conv1(images)
    height, width = values.shape[-2:]
    values = values.reshape(values.shape[0], values.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(values.dtype) + torch.zeros(
        values.shape[0], 1, values.shape[-1], dtype=values.dtype, device=values.device)
    values = torch.cat((cls, values), dim=1)
    position = rectangular_position(visual.positional_embedding, height, width)
    values = visual.ln_pre(values + position.to(values.dtype))
    values = visual.transformer(values.permute(1, 0, 2)).permute(1, 0, 2)
    values = visual.ln_post(values[:, 0, :])
    return values @ visual.proj if visual.proj is not None else values


def rectangular_logits(model, images):
    features = F.normalize(rectangular_visual(model.visual, images).float(), dim=-1)
    return model.classifier(features)
