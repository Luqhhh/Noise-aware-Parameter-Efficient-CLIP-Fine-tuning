"""Load the selected adapter average first; then merge or resize, retaining the head."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint

from aegis_clip.model import AdditiveLowRankParametrization
from aegis_clip.v1_strategy import aligned_positions


def merge_visual_lora(model):
    locations = []
    for module in model.visual.modules():
        for name, chain in list(getattr(module, "parametrizations", {}).items()):
            if len(chain) != 1 or not isinstance(chain[0], AdditiveLowRankParametrization):
                raise ValueError("Unexpected visual parametrization; refusing a partial merge")
            locations.append((module, name))
    if not locations or len(locations) != len(model.adapted_modules):
        raise ValueError("LoRA injection/merge coverage differs")
    for module, name in locations:
        # Uses the actual registered B@A orientation and alpha/r scaling once.
        parametrize.remove_parametrizations(module, name, leave_parametrized=True)
    for parameter in model.visual.parameters():
        parameter.requires_grad_(True)
    return len(locations)


class ContinuationClassifier(nn.Module):
    def __init__(self, parent, image_size, gradient_checkpointing=True):
        super().__init__()
        self.visual, self.head = parent.visual, parent.head
        self.image_size = image_size
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, images):
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError("Actual input differs from the frozen training/inference resolution")
        v = self.visual
        x = v.conv1(images.to(v.conv1.weight.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat((v.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x), 1)
        x = v.ln_pre(x + v.positional_embedding.to(x.dtype)).permute(1, 0, 2)
        for block in v.transformer.resblocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training and self.gradient_checkpointing else block(x)
        return self.head(v.ln_post(x.permute(1, 0, 2)[:, 0]) @ v.proj)


@torch.no_grad()
def convert_parent(parent, route, images, *, image_size=None):
    """Mandatory FP32/eval, zero-update function check before training."""
    parent.float().eval()
    before = parent(images.float()).clone()
    head_before = {k: v.detach().clone() for k, v in parent.head.state_dict().items()}
    merged = merge_visual_lora(parent) if route == "WFT448" else 0
    if route not in ("WFT448", "LR512"):
        raise ValueError("Unknown continuation route")
    source_size = int(parent.visual.input_resolution)
    model = ContinuationClassifier(parent, source_size).float().eval()
    after = model(images.float())
    torch.testing.assert_close(after, before, rtol=2e-5, atol=2e-5)
    for key, value in parent.head.state_dict().items():
        torch.testing.assert_close(value, head_before[key], rtol=0, atol=0)
    report = dict(passed=True, fp32=True, eval_mode=True, optimizer_updates=0,
                  merged_modules=merged, max_abs_logits_error=float((after-before).abs().max()),
                  atol=2e-5, rtol=2e-5, head_preserved=True, class_order_preserved=True)
    target_size = image_size or (448 if route == "WFT448" else 512)
    patch = int(parent.visual.conv1.kernel_size[0])
    if target_size % patch:
        raise ValueError("Resolution must be divisible by the patch size")
    if target_size != source_size:
        position = parent.visual.positional_embedding
        parent.visual.positional_embedding = nn.Parameter(
            aligned_positions(position.detach(), target_size // patch), requires_grad=position.requires_grad)
        parent.visual.input_resolution = target_size
    model.image_size = target_size
    report.update(image_size=target_size, tokens=1+(target_size//patch)**2)
    return model, report
