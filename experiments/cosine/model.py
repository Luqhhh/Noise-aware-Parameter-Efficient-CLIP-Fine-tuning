"""
Cosine classifier model: CLIP ViT-B/32 with a cosine-similarity classification head.

Architecture:
    CLIP ViT-B/32 (frozen) -> L2 Normalize -> Linear(512, 500) with no bias,
    optionally scaled by a learnable logit_scale parameter.

The cosine head normalizes both features and classifier weights, computing
logits = scale * (w * x). This removes magnitude differences between features
and can improve calibration.

Usage:
    from experiments.cosine.model import build_model
    model, preprocess = build_model(config, device)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CosineClassifier(nn.Module):
    """CLIP ViT-B/32 + cosine similarity classification head.

    The classifier has no bias and normalizes both inputs and weights,
    producing logits = scale * cosine_similarity(features, weight).

    Args:
        clip_model: The CLIP model (from clip.load).
        num_classes: Number of output classes.
        feature_dim: Dimensionality of CLIP image features (512 for ViT-B/32).
        freeze_clip: Whether to freeze the CLIP backbone.
        init_scale: Initial value for the learnable logit_scale parameter.
        learnable_scale: Whether logit_scale is learnable.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        num_classes: int = 500,
        feature_dim: int = 512,
        freeze_clip: bool = True,
        init_scale: float = 10.0,
        learnable_scale: bool = True,
    ):
        super().__init__()

        self.visual = clip_model.visual
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.freeze_clip = freeze_clip
        self.init_scale = init_scale
        self.learnable_scale = learnable_scale

        # Validate init_scale
        if init_scale <= 0:
            raise ValueError(
                f"init_scale must be positive, got {init_scale}"
            )

        # Freeze CLIP backbone
        if freeze_clip:
            for param in self.visual.parameters():
                param.requires_grad = False
            logger.info("CLIP image encoder frozen.")

        # Cosine classifier: Linear with no bias
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

        # Initialize weights uniformly
        nn.init.xavier_uniform_(self.classifier.weight)

        # Logit scale parameter (temperature)
        self.logit_scale = nn.Parameter(
            torch.tensor(init_scale),
            requires_grad=learnable_scale,
        )

        logger.info(
            f"CosineClassifier: init_scale={init_scale}, "
            f"learnable_scale={learnable_scale}, "
            f"bias=False"
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized image features.

        Args:
            images: Input image batch (B, 3, H, W).

        Returns:
            L2-normalized features (B, feature_dim).
        """
        conv1_dtype = self.visual.conv1.weight.dtype
        images = images.to(dtype=conv1_dtype)

        with torch.set_grad_enabled(not self.freeze_clip):
            features = self.visual(images)

        if features.dim() > 2:
            if features.dim() == 4:
                features = features.mean(dim=[2, 3])
            else:
                features = features[:, 0]

        features = features.float()
        features = F.normalize(features, p=2, dim=-1)
        return features

    def clamp_scale(self, min_val: float = 1.0, max_val: float = 100.0) -> None:
        """Clamp logit_scale to prevent extreme values.

        Args:
            min_val: Minimum allowed scale value.
            max_val: Maximum allowed scale value.
        """
        if not self.learnable_scale:
            return
        with torch.no_grad():
            self.logit_scale.clamp_(min_val, max_val)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass with cosine similarity classification.

        logits = logit_scale * (normalized(features) @ normalized(weight).T)

        Args:
            images: Input image batch (B, 3, H, W).

        Returns:
            Logits tensor (B, num_classes).
        """
        features = self.encode_image(images)

        # Normalize classifier weights
        w_norm = F.normalize(self.classifier.weight, p=2, dim=1)

        # Cosine similarity: (B, D) @ (D, C) -> (B, C)
        logits = features @ w_norm.T

        # Scale
        scale = self.logit_scale.clamp(min=1.0, max=100.0)
        logits = logits * scale

        return logits

    def get_trainable_parameters(self):
        """Return trainable parameters.

        If learnable_scale is False, logit_scale is excluded from the
        parameter groups (frozen).
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield param


def build_model(
    config: dict,
    device: torch.device,
) -> Tuple[CosineClassifier, callable]:
    """Build the CosineClassifier model.

    Config expects:
        model.clip_model_name: "ViT-B/32"
        model.num_classes: int
        model.feature_dim: int (default 512)
        model.freeze_clip: bool (default True)
        model.init_scale: float (default 10.0)
        model.learnable_scale: bool (default True)

    Args:
        config: Configuration dictionary.
        device: torch device.

    Returns:
        Tuple of (model, preprocess_fn).
    """
    try:
        import clip
    except ImportError:
        raise ImportError(
            "The 'clip' package is required. Install it with:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    clip_model_name = config["model"]["clip_model_name"]
    num_classes = config["model"]["num_classes"]
    feature_dim = config["model"].get("feature_dim", 512)
    freeze_clip = config["model"].get("freeze_clip", True)
    init_scale = config["model"].get("init_scale", 10.0)
    learnable_scale = config["model"].get("learnable_scale", True)

    logger.info(f"Loading CLIP model: {clip_model_name}")
    clip_model, preprocess = clip.load(clip_model_name, device=device)
    clip_model.visual = clip_model.visual.float()

    model = CosineClassifier(
        clip_model=clip_model,
        num_classes=num_classes,
        feature_dim=feature_dim,
        freeze_clip=freeze_clip,
        init_scale=init_scale,
        learnable_scale=learnable_scale,
    )

    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Cosine model built: {total:,} total params, {trainable:,} trainable params"
    )

    return model, preprocess
