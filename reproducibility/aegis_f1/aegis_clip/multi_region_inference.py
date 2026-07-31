"""Deterministic discriminative multi-region inference for fine-grained CLIP."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from aegis_clip.local_inference import (
    attention_guided_crop,
    logits_with_last_block_attention,
)
from aegis_clip.model import AegisCLIP


FROZEN_GRID_CENTERS: tuple[tuple[float, float], ...] = (
    (80.0, 80.0),
    (80.0, 112.0),
    (80.0, 144.0),
    (112.0, 80.0),
    (112.0, 144.0),
    (144.0, 80.0),
    (144.0, 112.0),
    (144.0, 144.0),
)


def crop_at_center(
    images: torch.Tensor,
    *,
    center_y: float,
    center_x: float,
    crop_size: int,
) -> torch.Tensor:
    if images.ndim != 4 or tuple(images.shape[-2:]) != (224, 224):
        raise ValueError("Fixed multi-region crops require [N,C,224,224] images")
    height, width = int(images.shape[-2]), int(images.shape[-1])
    half = float(crop_size) / 2.0
    if not half <= float(center_y) <= height - half:
        raise ValueError("center_y places the crop outside the image")
    if not half <= float(center_x) <= width - half:
        raise ValueError("center_x places the crop outside the image")
    theta = torch.zeros(
        images.shape[0], 2, 3, device=images.device, dtype=torch.float32
    )
    scale = float(crop_size) / float(height)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = 2.0 * float(center_x) / float(width) - 1.0
    theta[:, 1, 2] = 2.0 * float(center_y) / float(height) - 1.0
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).to(dtype=images.dtype)


def discriminative_region_fusion(
    global_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    top_regions: int = 5,
) -> dict[str, torch.Tensor]:
    """Select candidate regions by global-pseudo-label soft-negative margin."""
    if global_logits.ndim != 2 or candidate_logits.ndim != 3:
        raise ValueError("Expected global [N,C] and candidate [N,R,C] logits")
    if (
        candidate_logits.shape[0] != global_logits.shape[0]
        or candidate_logits.shape[2] != global_logits.shape[1]
    ):
        raise ValueError("Global and candidate logits do not align")
    regions = int(candidate_logits.shape[1])
    if not 1 <= int(top_regions) <= regions:
        raise ValueError("top_regions is out of range")

    global_values = global_logits.float()
    candidate_values = candidate_logits.float()
    pseudo_label = global_values.argmax(dim=1)
    gather_index = pseudo_label[:, None, None].expand(-1, regions, 1)
    target_value = candidate_values.gather(2, gather_index).squeeze(2)
    negative_values = candidate_values.clone()
    negative_values.scatter_(2, gather_index, float("-inf"))
    negative_weight = F.softmax(negative_values, dim=2)
    negative_average = (negative_weight * candidate_values).sum(dim=2)
    region_score = target_value - negative_average

    selected_score, selected_index = torch.topk(
        region_score, k=int(top_regions), dim=1
    )
    candidate_probability = F.softmax(candidate_values, dim=2)
    probability_index = selected_index[:, :, None].expand(
        -1, -1, candidate_values.shape[2]
    )
    selected_probability = candidate_probability.gather(1, probability_index)
    region_weight = F.softmax(selected_score, dim=1)
    local_probability = (
        region_weight[:, :, None] * selected_probability
    ).sum(dim=1)
    global_probability = F.softmax(global_values, dim=1)
    fused_log_probability = torch.logaddexp(
        global_probability.clamp_min(1.0e-12).log(),
        local_probability.clamp_min(1.0e-12).log(),
    ) - math.log(2.0)
    return {
        "logits": fused_log_probability,
        "local_logits": local_probability.clamp_min(1.0e-12).log(),
        "region_score": region_score,
        "selected_region_indices": selected_index,
        "selected_region_weights": region_weight,
        "pseudo_label": pseudo_label,
    }


def discriminative_multi_region_logits(
    model: AegisCLIP,
    images: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run the single frozen M2 candidate with no search parameters."""
    global_logits = model(images=images)
    _, _, attention = logits_with_last_block_attention(model, images)
    attention_images = attention_guided_crop(
        images, attention, crop_size=160, top_patches=5
    )
    attention_local_logits = model(images=attention_images)
    candidates = [attention_local_logits]
    for center_y, center_x in FROZEN_GRID_CENTERS:
        region_images = crop_at_center(
            images,
            center_y=center_y,
            center_x=center_x,
            crop_size=160,
        )
        candidates.append(model(images=region_images))
    candidate_logits = torch.stack(candidates, dim=1)
    result = discriminative_region_fusion(
        global_logits, candidate_logits, top_regions=5
    )
    return {
        **result,
        "global_logits": global_logits.float(),
        "attention_local_logits": attention_local_logits.float(),
    }
