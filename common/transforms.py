"""
Training transform construction.

Provides build_train_transform() which composes CLIP's deterministic eval
transform with augmentation presets (A0-A3). The CLIP preprocess is NOT
loaded inside this module — it's passed in from the caller.
"""

from typing import Callable, Set

import torchvision.transforms as T

VALID_PRESETS: Set[str] = {"a0", "a1", "a2", "a3"}


def build_train_transform(preset: str, clip_eval_transform: Callable):
    """Build a training transform by composing augmentation presets.

    All presets start from CLIP's eval transform (Resize(224) + CenterCrop(224) +
    ToTensor + Normalize). A0 returns it unchanged.

    Args:
        preset: One of "a0", "a1", "a2", "a3".
        clip_eval_transform: CLIP's deterministic eval preprocess (torchvision
            Compose, typically Resize+CenterCrop+ToTensor+Normalize).

    Returns:
        A torchvision transform (Compose).

    Raises:
        ValueError: If preset is not in VALID_PRESETS.
    """
    if preset not in VALID_PRESETS:
        raise ValueError(
            f"Unknown augmentation preset: {preset!r}. "
            f"Valid presets: {sorted(VALID_PRESETS)}"
        )

    # Extract components from CLIP's eval transform.
    # CLIP's preprocess is typically: Compose([
    #   Resize(224, interpolation=BICUBIC),
    #   CenterCrop(224),
    #   ToTensor(),
    #   Normalize(mean, std),
    # ])
    # We need to replace the Resize+CenterCrop with augmentation equivalents
    # while keeping the Normalize at the end.
    #
    # Strategy: Extract the Normalize transform from the CLIP eval pipeline,
    # then build our own Compose that ends with it.

    # Find the Normalize transform in the CLIP eval pipeline
    normalize_transform = None
    clip_size = 224

    if isinstance(clip_eval_transform, T.Compose):
        transforms_list = list(clip_eval_transform.transforms)
    else:
        transforms_list = [clip_eval_transform]

    for t in transforms_list:
        if isinstance(t, T.Normalize):
            normalize_transform = t
        if isinstance(t, T.Resize):
            clip_size = t.size if isinstance(t.size, int) else t.size[0]

    if normalize_transform is None:
        raise ValueError(
            "Could not find T.Normalize in clip_eval_transform. "
            "The CLIP eval transform must contain Normalize."
        )

    if preset == "a0":
        # A0: Deterministic — same as eval
        return clip_eval_transform

    elif preset == "a1":
        # A1: RandomResizedCrop + RandomHorizontalFlip
        return T.Compose([
            T.RandomResizedCrop(clip_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize_transform,
        ])

    elif preset == "a2":
        # A2: A1 + ColorJitter
        return T.Compose([
            T.RandomResizedCrop(clip_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            normalize_transform,
        ])

    elif preset == "a3":
        # A3: A2 + RandomErasing (applied AFTER Normalize)
        return T.Compose([
            T.RandomResizedCrop(clip_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            normalize_transform,
            T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])
