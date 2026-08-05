"""Attention-guided local views for object-centric CLIP inference."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


@torch.no_grad()
def forward_with_last_block_attention(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global logits and per-head CLS-to-patch attention.

    OpenAI CLIP requests no attention weights in its ordinary forward pass.
    We capture the input to the final residual block and replay only that
    block's attention operation with ``need_weights=True``.  The model forward
    itself is unchanged, including any parametrized LoRA weights.
    """
    logits, _, attention = _forward_with_last_block_attention(
        model, images, return_features=False
    )
    return logits, attention


@torch.no_grad()
def forward_features_with_last_block_attention(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return global logits, normalized features, and CLS-patch attention."""
    logits, features, attention = _forward_with_last_block_attention(
        model, images, return_features=True
    )
    if features is None:
        raise RuntimeError("Model did not return image features")
    return logits, features, attention


def _forward_with_last_block_attention(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    return_features: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    try:
        block = model.visual.transformer.resblocks[-1]
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError(
            "Attention localization requires OpenAI CLIP ViT residual blocks"
        ) from exc

    captured: dict[str, torch.Tensor] = {}

    def capture_input(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        if not inputs:
            raise RuntimeError("Final visual block received no token input")
        captured["tokens"] = inputs[0]

    handle = block.register_forward_pre_hook(capture_input)
    try:
        output = model(images=images, return_features=return_features)
    finally:
        handle.remove()
    if return_features:
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("Model did not return the requested image features")
        logits, features = output
    else:
        if isinstance(output, tuple):
            raise RuntimeError("Model returned unexpected auxiliary outputs")
        logits = output
        features = None

    if "tokens" not in captured:
        raise RuntimeError("Failed to capture final visual-block tokens")
    tokens = block.ln_1(captured["tokens"])
    attention_mask = getattr(block, "attn_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=tokens.device, dtype=tokens.dtype)
    _, weights = block.attn(
        tokens,
        tokens,
        tokens,
        need_weights=True,
        average_attn_weights=False,
        attn_mask=attention_mask,
    )
    if weights.ndim != 4 or weights.shape[-1] < 2:
        raise RuntimeError(
            f"Unexpected final-block attention shape: {tuple(weights.shape)}"
        )
    # MultiheadAttention returns [batch, heads, target, source].
    return logits, features, weights[:, :, 0, 1:].float()


def attention_weighted_centers(
    cls_patch_attention: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    top_k: int,
) -> torch.Tensor:
    """Map top-k mean-head patch attention to image-space ``(x, y)`` centers."""
    if cls_patch_attention.ndim != 3:
        raise ValueError(
            "CLS-patch attention must have shape [batch, heads, patches]"
        )
    if image_height <= 0 or image_width <= 0:
        raise ValueError("Image dimensions must be positive")
    patch_count = int(cls_patch_attention.shape[-1])
    grid_size = math.isqrt(patch_count)
    if grid_size * grid_size != patch_count:
        raise ValueError(
            f"Patch count must form a square grid, got {patch_count}"
        )
    if not 1 <= top_k <= patch_count:
        raise ValueError(f"top_k must be in [1, {patch_count}], got {top_k}")

    mean_attention = cls_patch_attention.float().mean(dim=1)
    values, indices = mean_attention.topk(top_k, dim=1)
    values = values.clamp_min(0.0)
    values = values / values.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    rows = torch.div(indices, grid_size, rounding_mode="floor").float()
    columns = indices.remainder(grid_size).float()
    center_x = ((columns + 0.5) * (float(image_width) / grid_size) * values).sum(
        dim=1
    )
    center_y = ((rows + 0.5) * (float(image_height) / grid_size) * values).sum(
        dim=1
    )
    return torch.stack((center_x, center_y), dim=1)


def extract_attention_crops(
    images: torch.Tensor,
    cls_patch_attention: torch.Tensor,
    *,
    crop_size: int = 160,
    top_k: int = 5,
) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
    """Extract fixed-size crops around attention centers and resize to input size."""
    if images.ndim != 4:
        raise ValueError("Images must have shape [batch, channels, height, width]")
    batch_size, _, height, width = images.shape
    if cls_patch_attention.shape[0] != batch_size:
        raise ValueError("Image and attention batch sizes do not match")
    if not 1 <= crop_size <= min(height, width):
        raise ValueError(
            f"crop_size must be in [1, {min(height, width)}], got {crop_size}"
        )
    centers = attention_weighted_centers(
        cls_patch_attention,
        image_height=height,
        image_width=width,
        top_k=top_k,
    )
    boxes: list[tuple[int, int, int, int]] = []
    crops: list[torch.Tensor] = []
    max_x = width - crop_size
    max_y = height - crop_size
    for image, center in zip(images, centers):
        x0 = int(round(float(center[0]) - crop_size / 2.0))
        y0 = int(round(float(center[1]) - crop_size / 2.0))
        x0 = min(max(x0, 0), max_x)
        y0 = min(max(y0, 0), max_y)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        boxes.append((x0, y0, x1, y1))
        crops.append(image[:, y0:y1, x0:x1])
    resized = F.interpolate(
        torch.stack(crops),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized, boxes


def fuse_global_local_probabilities(
    global_logits: torch.Tensor,
    local_logits: torch.Tensor,
    *,
    local_weight: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Fuse global and local class probabilities and return stable log scores."""
    if global_logits.shape != local_logits.shape:
        raise ValueError("Global and local logits must have identical shapes")
    if not 0.0 <= local_weight <= 1.0:
        raise ValueError("local_weight must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    global_probabilities = F.softmax(global_logits.float() / temperature, dim=1)
    local_probabilities = F.softmax(local_logits.float() / temperature, dim=1)
    fused = (
        (1.0 - local_weight) * global_probabilities
        + local_weight * local_probabilities
    )
    return fused.clamp_min(torch.finfo(fused.dtype).tiny).log()


def fuse_global_local_flip_probabilities(
    global_logits: torch.Tensor,
    local_logits: torch.Tensor,
    flipped_global_logits: torch.Tensor,
    flipped_local_logits: torch.Tensor,
    *,
    local_weight: float = 0.5,
    flip_weight: float = 0.5,
    temperature: float = 1.0,
    global_temperature: float | None = None,
    local_temperature: float | None = None,
) -> torch.Tensor:
    """Fuse deterministic global/local streams across original and flipped views.

    ``temperature`` is the backward-compatible single-value sharpness for all
    four views.  When ``global_temperature``/``local_temperature`` are supplied,
    the original+flipped global views use the global temperature and the
    original+flipped local views use the local temperature, enabling
    per-branch confidence calibration.
    """
    views = (
        global_logits,
        local_logits,
        flipped_global_logits,
        flipped_local_logits,
    )
    if global_logits.ndim != 2 or any(
        value.shape != global_logits.shape for value in views[1:]
    ):
        raise ValueError("All view logits must have identical [N,C] shapes")
    if not 0.0 <= local_weight <= 1.0:
        raise ValueError("local_weight must be in [0, 1]")
    if not 0.0 <= flip_weight <= 1.0:
        raise ValueError("flip_weight must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if global_temperature is None:
        global_temperature = temperature
    if local_temperature is None:
        local_temperature = temperature
    if global_temperature <= 0.0:
        raise ValueError("global_temperature must be positive")
    if local_temperature <= 0.0:
        raise ValueError("local_temperature must be positive")

    global_probabilities = (
        (1.0 - flip_weight)
        * F.softmax(global_logits.float() / global_temperature, dim=1)
        + flip_weight
        * F.softmax(flipped_global_logits.float() / global_temperature, dim=1)
    )
    local_probabilities = (
        (1.0 - flip_weight)
        * F.softmax(local_logits.float() / local_temperature, dim=1)
        + flip_weight
        * F.softmax(flipped_local_logits.float() / local_temperature, dim=1)
    )
    fused = (
        (1.0 - local_weight) * global_probabilities
        + local_weight * local_probabilities
    )
    return fused.clamp_min(torch.finfo(fused.dtype).tiny).log()


def fuse_global_multilocal_probabilities(
    global_logits: torch.Tensor,
    local_logits: Sequence[torch.Tensor],
    *,
    local_weight: float = 0.5,
    temperature: float = 1.0,
    local_scale_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Fuse a global view with weighted deterministic local probabilities."""
    if not local_logits:
        raise ValueError("At least one local-view tensor is required")
    if any(value.shape != global_logits.shape for value in local_logits):
        raise ValueError("Global and all local logits must have identical shapes")
    if not 0.0 <= local_weight <= 1.0:
        raise ValueError("local_weight must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    global_probabilities = F.softmax(global_logits.float() / temperature, dim=1)
    weights = normalized_probability_weights(
        local_scale_weights, len(local_logits), name="local_scale_weights"
    )
    stacked_local = torch.stack(
        [
            F.softmax(value.float() / temperature, dim=1)
            for value in local_logits
        ],
        dim=0,
    )
    mean_local = (
        stacked_local
        * torch.tensor(weights, device=stacked_local.device)[:, None, None]
    ).sum(dim=0)
    fused = (
        (1.0 - local_weight) * global_probabilities
        + local_weight * mean_local
    )
    return fused.clamp_min(torch.finfo(fused.dtype).tiny).log()


def fuse_global_multilocal_flip_probabilities(
    global_logits: torch.Tensor,
    local_logits: Sequence[torch.Tensor],
    flipped_global_logits: torch.Tensor,
    flipped_local_logits: Sequence[torch.Tensor],
    *,
    local_weight: float = 0.5,
    flip_weight: float = 0.5,
    temperature: float = 1.0,
    global_temperature: float | None = None,
    local_temperature: float | None = None,
    local_scale_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Fuse global/flip views with weighted paired multi-scale local views."""
    if not local_logits:
        raise ValueError("At least one local-view tensor is required")
    if len(local_logits) != len(flipped_local_logits):
        raise ValueError("Original and flipped local-view counts must match")
    views = [*local_logits, flipped_global_logits, *flipped_local_logits]
    if global_logits.ndim != 2 or any(
        value.shape != global_logits.shape for value in views
    ):
        raise ValueError("All view logits must have identical [N,C] shapes")
    if not 0.0 <= local_weight <= 1.0:
        raise ValueError("local_weight must be in [0, 1]")
    if not 0.0 <= flip_weight <= 1.0:
        raise ValueError("flip_weight must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if global_temperature is None:
        global_temperature = temperature
    if local_temperature is None:
        local_temperature = temperature
    if global_temperature <= 0.0:
        raise ValueError("global_temperature must be positive")
    if local_temperature <= 0.0:
        raise ValueError("local_temperature must be positive")

    global_probabilities = (
        (1.0 - flip_weight)
        * F.softmax(global_logits.float() / global_temperature, dim=1)
        + flip_weight
        * F.softmax(flipped_global_logits.float() / global_temperature, dim=1)
    )
    paired_local_probabilities = [
        (1.0 - flip_weight)
        * F.softmax(original.float() / local_temperature, dim=1)
        + flip_weight
        * F.softmax(flipped.float() / local_temperature, dim=1)
        for original, flipped in zip(local_logits, flipped_local_logits)
    ]
    weights = normalized_probability_weights(
        local_scale_weights,
        len(paired_local_probabilities),
        name="local_scale_weights",
    )
    stacked_local = torch.stack(paired_local_probabilities, dim=0)
    mean_local_probabilities = (
        stacked_local
        * torch.tensor(weights, device=stacked_local.device)[:, None, None]
    ).sum(dim=0)
    fused = (
        (1.0 - local_weight) * global_probabilities
        + local_weight * mean_local_probabilities
    )
    return fused.clamp_min(torch.finfo(fused.dtype).tiny).log()


def parse_int_sequence(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse a comma-separated CLI sequence while rejecting duplicates."""
    if isinstance(value, str):
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        parsed = tuple(int(item) for item in value)
    if not parsed:
        raise ValueError("At least one integer value is required")
    if len(parsed) != len(set(parsed)):
        raise ValueError("Sequence values must be unique")
    return parsed


def normalized_probability_weights(
    values: str | Sequence[float] | None,
    expected_count: int,
    *,
    name: str = "weights",
) -> tuple[float, ...]:
    """Parse a non-negative probability vector or return equal weights."""
    if int(expected_count) <= 0:
        raise ValueError("expected_count must be positive")
    if values is None:
        return tuple(1.0 / int(expected_count) for _ in range(int(expected_count)))
    if isinstance(values, str):
        parsed = tuple(
            float(item.strip()) for item in values.split(",") if item.strip()
        )
    else:
        parsed = tuple(float(item) for item in values)
    if len(parsed) != int(expected_count):
        raise ValueError(f"{name} count must match local crop count")
    if any(value < 0.0 for value in parsed):
        raise ValueError(f"{name} must be non-negative")
    if abs(sum(parsed) - 1.0) > 1.0e-8:
        raise ValueError(f"{name} must sum to one")
    return parsed
