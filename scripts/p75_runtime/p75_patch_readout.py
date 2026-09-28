"""One-head patch-token readout for the pinned L05 CLIP runtime.

The module is installed only when model.patch_readout is enabled. It observes
the same visual transformer call as the ordinary CLS path and adds no image
view, backbone, or independent classifier.
"""
from __future__ import annotations

import types

import torch
from torch import nn
from torch.nn import functional as F


class PatchReadout(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.score = nn.Linear(width, 1)
        self.residual = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.residual.weight)

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3 or patches.shape[0] != cls.shape[0]:
            raise ValueError("Patch features must have [batch,patch,dimension] shape")
        attention = self.score(patches.float()).squeeze(-1).softmax(dim=1)
        local = torch.sum(attention.unsqueeze(-1) * patches.float(), dim=1)
        return F.normalize(cls.float() + self.residual(local), dim=-1)


def install(model: nn.Module) -> nn.Module:
    if hasattr(model, "patch_readout"):
        raise ValueError("Patch readout already installed")
    visual = model.visual
    if model.peft_mode == "visual_prompt" or not all(
        hasattr(visual, key) for key in ("transformer", "ln_post", "proj")
    ):
        raise ValueError("Patch readout requires the native CLIP ViT visual path")
    width = int(model.feature_dim)
    model.add_module("patch_readout", PatchReadout(width).to(
        device=model.classifier.weight.device, dtype=torch.float32
    ))
    original_encode = model.encode_image
    original_groups = model.parameter_groups
    original_load = model.load_state_dict
    original_spec = model.effective_spec
    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise RuntimeError("CLIP transformer did not emit [token,batch,width]")
        captured.append(output)

    visual.transformer.register_forward_hook(capture)

    def encode(self, images):
        captured.clear()
        cls = original_encode(images)
        if len(captured) != 1:
            raise RuntimeError(f"Expected one visual call, got {len(captured)}")
        tokens = captured.pop()
        if tokens.shape[0] < 2 or tokens.shape[1] != images.shape[0]:
            raise RuntimeError("Visual patch tokens do not match input")
        patches = visual.ln_post(tokens[1:].permute(1, 0, 2))
        if visual.proj is not None:
            patches = patches @ visual.proj
        return self.patch_readout(cls, patches)

    def groups(self, *args, **kwargs):
        result = original_groups(*args, **kwargs)
        head = [group for group in result if group.get("name") == "head"]
        if len(head) != 1:
            raise RuntimeError("Expected one original classifier optimizer group")
        head[0]["params"] = list(head[0]["params"]) + list(self.patch_readout.parameters())
        return result

    def load(self, state, strict=True, **kwargs):
        present = {key.removeprefix("patch_readout.") for key in state
                   if key.startswith("patch_readout.")}
        expected = set(self.patch_readout.state_dict())
        if present and present != expected:
            raise RuntimeError("Checkpoint has partial patch readout state")
        if strict or present:
            return original_load(state, strict=strict, **kwargs)
        result = original_load(state, strict=False, **kwargs)
        missing = {key.removeprefix("patch_readout.") for key in result.missing_keys
                   if key.startswith("patch_readout.")}
        if missing != expected:
            raise RuntimeError("Parent checkpoint has partial patch readout state")
        return type(result)([key for key in result.missing_keys
                             if not key.startswith("patch_readout.")], result.unexpected_keys)

    def spec(self):
        result = original_spec()
        result["patch_readout"] = "single_head_zero_residual_v1"
        return result

    model.encode_image = types.MethodType(encode, model)
    model.parameter_groups = types.MethodType(groups, model)
    model.load_state_dict = types.MethodType(load, model)
    model.effective_spec = types.MethodType(spec, model)
    return model


def patch_runtime() -> None:
    """Patch each import-time builder binding used by train and inference."""
    from aegis_clip import checkpoint, model, trainer

    if getattr(model.build_model, "_p75_patched", False):
        return
    original = model.build_model

    def build(config, device):
        built, preprocess = original(config, device)
        options = config["model"].get("patch_readout", {})
        if options.get("enabled", False):
            if options != {"enabled": True, "version": 1}:
                raise ValueError("Only the fixed P75 patch readout v1 is supported")
            install(built)
        return built, preprocess

    build._p75_patched = True
    model.build_model = build
    checkpoint.build_model = build
    trainer.build_model = build
