"""
CLIP loading and feature encoding utilities.

Provides a single validated entry point for loading OpenAI CLIP ViT-B/32
and a uniform feature encoding function used by both online training and
the caching pipeline.

Usage:
    from common.clip_utils import load_openai_clip, encode_frozen_clip_features

    model, preprocess = load_openai_clip(device)
    features = encode_frozen_clip_features(model, images, device)
"""

import logging
from typing import Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ALLOWED_MODEL_NAME = "ViT-B/32"
ALLOWED_PRETRAINED_SOURCE = "openai"


def load_openai_clip(
    device: torch.device,
    model_name: str = ALLOWED_MODEL_NAME,
    pretrained_source: str = ALLOWED_PRETRAINED_SOURCE,
) -> Tuple[torch.nn.Module, callable]:
    """Load OpenAI CLIP with hard validation of model name and source.

    Args:
        device: torch device to load the model onto.
        model_name: Must be "ViT-B/32".
        pretrained_source: Must be "openai".

    Returns:
        Tuple of (clip_model, preprocess_fn).

    Raises:
        ValueError: If model_name or pretrained_source are not allowed.
        ImportError: If the 'clip' package is not installed.
    """
    if model_name != ALLOWED_MODEL_NAME:
        raise ValueError(
            f"Requires {ALLOWED_MODEL_NAME}, got {model_name!r}"
        )
    if pretrained_source != ALLOWED_PRETRAINED_SOURCE:
        raise ValueError(
            f"Only OpenAI weights allowed, got {pretrained_source!r}"
        )

    try:
        import clip
    except ImportError:
        raise ImportError(
            "The 'clip' package is required. Install it with:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    logger.info(
        f"Loading CLIP {model_name} (source={pretrained_source}) on {device}"
    )
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model = model.float()
    return model, preprocess


@torch.no_grad()
def encode_frozen_clip_features(
    clip_model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    """Encode images through a frozen CLIP model with L2 normalization.

    This is the SINGLE canonical encoding path used by all parts of the
    pipeline (online training, cached feature generation, etc.) to ensure
    numerical consistency.

    Args:
        clip_model: CLIP model (frozen, in eval mode).
        images: Batch of preprocessed images (B, 3, H, W).
        device: torch device.
        use_amp: Whether to use AMP autocast.

    Returns:
        L2-normalized float32 feature tensor (B, feature_dim).
    """
    images = images.to(device, non_blocking=True)

    # Match input dtype to conv1 weight dtype for compatibility
    conv1_dtype = clip_model.visual.conv1.weight.dtype
    images = images.to(dtype=conv1_dtype)

    with torch.autocast(device_type=device.type, enabled=use_amp):
        features = clip_model.encode_image(images)

    # Pool spatial features if CLIP version returns non-vector format
    if features.dim() > 2:
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])
        else:
            features = features[:, 0]

    features = features.float()
    features = F.normalize(features, p=2, dim=-1)
    return features
