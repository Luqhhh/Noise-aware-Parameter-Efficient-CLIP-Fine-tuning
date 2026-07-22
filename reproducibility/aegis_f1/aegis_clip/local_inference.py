"""Training-free local/global inference from CLIP's own visual attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from aegis_clip.local_feature_adapter import (
    BottleneckLocalFeatureAdapter,
    fuse_global_local_log_probabilities,
)
from aegis_clip.model import AegisCLIP
from aegis_clip.part_token_adapter import (
    PartTokenResidualAdapter,
    anchored_classifier_residual_logits,
    pool_cls_aligned_patch_features,
)


def logits_with_last_block_attention(
    model: AegisCLIP, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the native 224px visual path and expose CLS-to-patch attention."""
    visual = model.visual
    if model.peft_mode == "visual_prompt":
        raise ValueError(
            "Attention-guided inference is not validated for visual prompts"
        )
    required = {
        "conv1",
        "class_embedding",
        "positional_embedding",
        "ln_pre",
        "transformer",
        "ln_post",
    }
    if not all(hasattr(visual, name) for name in required):
        raise ValueError("Attention-guided inference requires CLIP ViT-B/32 internals")
    if images.ndim != 4 or tuple(images.shape[-2:]) != (224, 224):
        raise ValueError("Attention-guided inference requires [N,C,224,224] images")
    dtype = visual.conv1.weight.dtype
    values = visual.conv1(images.to(dtype=dtype))
    grid_height, grid_width = int(values.shape[-2]), int(values.shape[-1])
    if grid_height * grid_width + 1 != visual.positional_embedding.shape[0]:
        raise ValueError("Attention path requires the native CLIP patch grid")
    values = values.reshape(values.shape[0], values.shape[1], -1).permute(0, 2, 1)
    class_token = visual.class_embedding.to(values.dtype).reshape(1, 1, -1)
    class_token = class_token.expand(values.shape[0], 1, values.shape[-1])
    values = torch.cat([class_token, values], dim=1)
    values = values + visual.positional_embedding.to(values.dtype).unsqueeze(0)
    values = visual.ln_pre(values).permute(1, 0, 2)

    blocks = visual.transformer.resblocks
    if len(blocks) == 0:
        raise ValueError("CLIP visual transformer has no residual blocks")
    for block in blocks[:-1]:
        values = block(values)
    final_block = blocks[-1]
    normalized = final_block.ln_1(values)
    attention_mask = final_block.attn_mask
    if attention_mask is not None:
        attention_mask = attention_mask.to(
            dtype=normalized.dtype, device=normalized.device
        )
    attention_output, attention_weights = final_block.attn(
        normalized,
        normalized,
        normalized,
        need_weights=True,
        attn_mask=attention_mask,
        average_attn_weights=False,
    )
    values = values + attention_output
    values = values + final_block.mlp(final_block.ln_2(values))
    values = values.permute(1, 0, 2)
    features = visual.ln_post(values[:, 0, :])
    if visual.proj is not None:
        features = features @ visual.proj
    encoded = model.adapt_features(features)
    logits = model.classifier(encoded)
    patch_attention = attention_weights.float().mean(dim=1)[:, 0, 1:]
    patch_attention = patch_attention.reshape(
        images.shape[0], grid_height, grid_width
    )
    return logits, encoded, patch_attention


def native_visual_forward_with_patch_features(
    model: AegisCLIP,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the native model once and expose its final projected patch tokens.

    A forward hook observes the exact transformer invocation used by the native
    CLIP path.  Consequently the returned logits and CLS feature retain the
    same numerical operation order as ``model(images=...)``; patch processing
    is read-only and cannot perturb the scored branch.
    """
    if model.peft_mode == "visual_prompt":
        raise ValueError("Part-token inference is not validated for visual prompts")
    visual = model.visual
    required = {"transformer", "ln_post", "proj"}
    if not all(hasattr(visual, name) for name in required):
        raise ValueError("Part-token inference requires CLIP ViT internals")
    captured: list[torch.Tensor] = []

    def capture_tokens(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise ValueError("CLIP visual transformer returned a non-tensor")
        captured.append(output)

    handle = visual.transformer.register_forward_hook(capture_tokens)
    try:
        logits, local_features = model(images=images, return_features=True)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            "Expected exactly one native visual transformer invocation, "
            f"captured {len(captured)}"
        )
    tokens = captured[0]
    if tokens.ndim != 3 or tokens.shape[1] != images.shape[0]:
        raise ValueError("Captured CLIP tokens must have [sequence,batch,width] shape")
    if tokens.shape[0] <= 1:
        raise ValueError("Captured CLIP sequence does not contain patch tokens")
    patch_values = tokens[1:].permute(1, 0, 2)
    patch_values = visual.ln_post(patch_values)
    if visual.proj is not None:
        patch_values = patch_values @ visual.proj
    patch_features = F.normalize(patch_values.float(), dim=2)
    return logits, local_features, patch_features


def attention_guided_crop(
    images: torch.Tensor,
    patch_attention: torch.Tensor,
    *,
    crop_size: int = 160,
    top_patches: int = 5,
) -> torch.Tensor:
    """Zoom around the weighted centroid of the most attended CLIP patches."""
    if images.ndim != 4:
        raise ValueError("Images must be rank-4")
    if patch_attention.ndim != 3 or patch_attention.shape[0] != images.shape[0]:
        raise ValueError("Patch attention must align with the image batch")
    height, width = int(images.shape[-2]), int(images.shape[-1])
    if height != width:
        raise ValueError("Attention crop currently requires square model inputs")
    if not 0 < int(crop_size) < height:
        raise ValueError("crop_size must be smaller than the model input")
    patch_count = patch_attention.shape[1] * patch_attention.shape[2]
    if not 1 <= int(top_patches) <= patch_count:
        raise ValueError("top_patches is out of range")

    flat = patch_attention.float().flatten(1)
    weights, indices = torch.topk(flat, k=int(top_patches), dim=1)
    weights = weights.clamp_min(0.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    grid_width = patch_attention.shape[2]
    rows = torch.div(indices, grid_width, rounding_mode="floor").float()
    columns = (indices % grid_width).float()
    patch_height = height / patch_attention.shape[1]
    patch_width = width / patch_attention.shape[2]
    center_y = ((rows + 0.5) * patch_height * weights).sum(dim=1)
    center_x = ((columns + 0.5) * patch_width * weights).sum(dim=1)
    half = float(crop_size) / 2.0
    center_y = center_y.clamp(half, height - half)
    center_x = center_x.clamp(half, width - half)

    scale = float(crop_size) / float(height)
    theta = torch.zeros(
        images.shape[0], 2, 3, device=images.device, dtype=torch.float32
    )
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = 2.0 * center_x / float(width) - 1.0
    theta[:, 1, 2] = 2.0 * center_y / float(height) - 1.0
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).to(dtype=images.dtype)


def attention_local_global_logits(
    model: AegisCLIP,
    images: torch.Tensor,
    *,
    crop_size: int = 160,
    top_patches: int = 5,
) -> dict[str, torch.Tensor]:
    # Keep the scored global branch bit-for-bit on the model's native forward
    # path.  The manual transformer replay is used only to expose attention;
    # its half-precision operation order can otherwise perturb a few logits.
    global_logits = model(images=images)
    _, _, attention = logits_with_last_block_attention(model, images)
    local_images = attention_guided_crop(
        images,
        attention,
        crop_size=crop_size,
        top_patches=top_patches,
    )
    local_logits = model(images=local_images)
    fused_log_probabilities = torch.logaddexp(
        F.log_softmax(global_logits.float(), dim=1),
        F.log_softmax(local_logits.float(), dim=1),
    ) - math.log(2.0)
    return {
        "logits": fused_log_probabilities,
        "global_logits": global_logits.float(),
        "local_logits": local_logits.float(),
        "attention": attention,
    }


def attention_local_adapter_global_logits(
    model: AegisCLIP,
    adapter: BottleneckLocalFeatureAdapter,
    images: torch.Tensor,
    *,
    crop_size: int = 160,
    top_patches: int = 5,
) -> dict[str, torch.Tensor]:
    """Apply O3 only to the local view while preserving native F1 globally."""
    global_logits = model(images=images)
    _, _, attention = logits_with_last_block_attention(model, images)
    local_images = attention_guided_crop(
        images,
        attention,
        crop_size=crop_size,
        top_patches=top_patches,
    )
    base_local_logits, local_features = model(
        images=local_images, return_features=True
    )
    adapted_local_features = adapter(local_features)
    adapted_local_logits = model.classifier(adapted_local_features)
    fused = fuse_global_local_log_probabilities(
        global_logits, adapted_local_logits
    )
    return {
        "logits": fused,
        "global_logits": global_logits.float(),
        "base_local_logits": base_local_logits.float(),
        "adapted_local_logits": adapted_local_logits.float(),
        "local_features": local_features.float(),
        "adapted_local_features": adapted_local_features.float(),
        "attention": attention,
    }


def attention_part_token_adapter_global_logits(
    model: AegisCLIP,
    adapter: PartTokenResidualAdapter,
    images: torch.Tensor,
    *,
    crop_size: int = 160,
    top_patches: int = 5,
    part_top_patches: int = 8,
    part_temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    """Fuse native F1 globally with an R1-adapted M1 part-token local view."""
    global_logits = model(images=images)
    _, _, attention = logits_with_last_block_attention(model, images)
    local_images = attention_guided_crop(
        images,
        attention,
        crop_size=crop_size,
        top_patches=top_patches,
    )
    base_local_logits, local_features, patch_features = (
        native_visual_forward_with_patch_features(model, local_images)
    )
    part_features = pool_cls_aligned_patch_features(
        local_features,
        patch_features,
        top_patches=part_top_patches,
        temperature=part_temperature,
    )
    adapted_local_features = adapter(local_features, part_features)
    # Anchor on the exact native/AMP local logits and apply only the shared
    # classifier weight to the learned feature residual.  The classifier bias
    # cancels, and a zero feature residual is therefore bit-exact regardless of
    # CPU/GPU matmul precision differences.
    adapted_local_logits = anchored_classifier_residual_logits(
        base_local_logits,
        local_features,
        adapted_local_features,
        model.classifier.weight,
    )
    fused = fuse_global_local_log_probabilities(
        global_logits, adapted_local_logits
    )
    return {
        "logits": fused,
        "global_logits": global_logits.float(),
        "base_local_logits": base_local_logits.float(),
        "adapted_local_logits": adapted_local_logits.float(),
        "local_features": local_features.float(),
        "part_features": part_features.float(),
        "adapted_local_features": adapted_local_features.float(),
        "attention": attention,
    }


def complementary_flip_local_fusion(
    global_logits: torch.Tensor,
    flip_logits: torch.Tensor,
    local_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not (
        global_logits.ndim == flip_logits.ndim == local_logits.ndim == 2
        and global_logits.shape == flip_logits.shape == local_logits.shape
    ):
        raise ValueError("Global, flip, and local logits must have equal [N,C] shape")
    global_values = global_logits.float()
    flip_values = flip_logits.float()
    local_values = local_logits.float()
    m1_log_probability = torch.logaddexp(
        F.log_softmax(global_values, dim=1),
        F.log_softmax(local_values, dim=1),
    ) - math.log(2.0)
    flip_fused_logits = (global_values + flip_values) / 2.0
    flip_log_probability = F.log_softmax(flip_fused_logits, dim=1)
    fused_log_probability = torch.logaddexp(
        m1_log_probability, flip_log_probability
    ) - math.log(2.0)
    return {
        "logits": fused_log_probability,
        "m1_logits": m1_log_probability,
        "flip_fused_logits": flip_fused_logits,
    }


def complementary_flip_local_global_logits(
    model: AegisCLIP,
    images: torch.Tensor,
    *,
    crop_size: int = 160,
    top_patches: int = 5,
) -> dict[str, torch.Tensor]:
    """Fuse the frozen A2 flip branch with the frozen M1 branch."""
    global_logits = model(images=images)
    _, _, attention = logits_with_last_block_attention(model, images)
    local_images = attention_guided_crop(
        images,
        attention,
        crop_size=crop_size,
        top_patches=top_patches,
    )
    local_logits = model(images=local_images)
    flip_logits = model(images=torch.flip(images, dims=(3,)))
    fused = complementary_flip_local_fusion(
        global_logits, flip_logits, local_logits
    )
    return {
        **fused,
        "global_logits": global_logits.float(),
        "local_logits": local_logits.float(),
        "flip_logits": flip_logits.float(),
        "attention": attention,
    }
