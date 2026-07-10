"""
CLIP model loading and frozen-feature encoding.

Centralizes CLIP loading so model_name and pretrained_source are validated
in exactly one place. Also provides the single canonical encode path used
by both online training and feature caching.
"""

import logging
from typing import Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ALLOWED_MODEL_NAME = "ViT-B/32"
ALLOWED_PRETRAINED_SOURCE = "openai"


def load_openai_clip(device, model_name=ALLOWED_MODEL_NAME,
                     pretrained_source=ALLOWED_PRETRAINED_SOURCE):
    """Load CLIP model from OpenAI, validating model name and pretrained source.

    Args:
        device: torch device.
        model_name: Must be "ViT-B/32".
        pretrained_source: Must be "openai".

    Returns:
        Tuple of (clip_model, preprocess_fn).

    Raises:
        ValueError: If model_name or pretrained_source is not the allowed value.
    """
    if model_name != ALLOWED_MODEL_NAME:
        raise ValueError(
            f"This project requires {ALLOWED_MODEL_NAME}, got {model_name}"
        )
    if pretrained_source != ALLOWED_PRETRAINED_SOURCE:
        raise ValueError(
            f"Only OpenAI weights are allowed, got {pretrained_source}"
        )

    try:
        import clip
    except ImportError:
        raise ImportError(
            "The 'clip' package is required. Install it with:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    model, preprocess = clip.load(ALLOWED_MODEL_NAME, device=device, jit=False)
    model.visual = model.visual.float()  # convert fp16 conv1 to fp32 to avoid dtype mismatch
    return model, preprocess


@torch.no_grad()
def encode_frozen_clip_features(clip_model, images, device, use_amp=False):
    """Encode images through a frozen CLIP visual encoder.

    Features are L2-normalized. This is the single canonical encoding path
    shared by online training (B0) and offline cache building.

    Args:
        clip_model: CLIP model (from load_openai_clip).
        images: Image batch on the correct device, shape (B, 3, H, W).
        device: torch device (used for autocast device_type).
        use_amp: Whether to use torch.autocast during encoding.

    Returns:
        L2-normalized feature tensor of shape (B, feature_dim), float32.
    """
    with torch.autocast(device_type=device.type, enabled=use_amp):
        features = clip_model.encode_image(images)
    return F.normalize(features.float(), dim=-1)
