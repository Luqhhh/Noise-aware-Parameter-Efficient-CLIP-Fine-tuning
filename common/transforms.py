"""
Transform construction for training and validation.

Provides a validated entry point for building augmentation pipelines per preset.
The CLIP evaluation transform is loaded separately (by the caller) and passed
as an argument -- this module does NOT load CLIP internally.

Validation (deterministic) transforms always use the standard CLIP preprocess
(via the CLIP library). Training transforms vary by preset.

Presets:
    a0: Deterministic CLIP preprocess only (no augmentation).
    a1: RandomResizedCrop + RandomHorizontalFlip.
    a2: A1 + ColorJitter.
    a3: Heavy A2 + Normalize then add RandomErasing.
"""

import logging
from typing import Callable, Optional

import torchvision.transforms as T

logger = logging.getLogger(__name__)

VALID_PRESETS = {"a0", "a1", "a2", "a3"}

# CLIP-specific normalization constants (ViT-B/32, OpenAI)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_train_transform(
    preset: str,
    clip_eval_transform: Optional[Callable] = None,
    image_size: int = 224,
) -> Callable:
    """Build a training transform for the given augmentation preset.

    Args:
        preset: One of "a0", "a1", "a2", "a3".
        clip_eval_transform: The CLIP evaluation transform (used for a0).
                              May be None for a1-a3 (which build their own).
        image_size: Target spatial size (default 224 for CLIP ViT-B/32).

    Returns:
        A torchvision transform (Compose or single transform).

    Raises:
        ValueError: If preset is not in VALID_PRESETS.
    """
    if preset not in VALID_PRESETS:
        raise ValueError(
            f"Unknown augmentation preset: {preset!r}. "
            f"Valid presets: {sorted(VALID_PRESETS)}"
        )

    if preset == "a0":
        # A0: No augmentation -- use the CLIP evaluation transform directly
        if clip_eval_transform is None:
            raise ValueError(
                "clip_eval_transform is required for preset='a0'"
            )
        return clip_eval_transform

    elif preset == "a1":
        # A1: Light -- RandomResizedCrop + RandomHorizontalFlip
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    elif preset == "a2":
        # A2: Medium -- A1 + ColorJitter
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    elif preset == "a3":
        # A3: Heavy -- A2 + RandomErasing (after Normalize)
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])


def build_val_transform(clip_eval_transform: Callable) -> Callable:
    """Build a validation/evaluation transform.

    Always uses the deterministic CLIP preprocess (no augmentation).

    Args:
        clip_eval_transform: The standard CLIP preprocessing function.

    Returns:
        The CLIP evaluation transform unchanged.
    """
    return clip_eval_transform
