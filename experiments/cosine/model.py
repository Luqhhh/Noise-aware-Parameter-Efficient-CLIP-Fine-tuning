"""
Cosine classifier -- replaces the linear head with a cosine-similarity-based
classification layer. No bias term. Input features and class prototypes are
both L2-normalized before computing logits.

Supports:
  - Fixed scale: logit_scale is a buffer, not optimized
  - Learnable scale: logit_scale is a nn.Parameter, optimized separately
  - Clamping: scale clamped to [1.0, 100.0] after each optimizer step (learnable only)
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.clip_utils import load_openai_clip

logger = logging.getLogger(__name__)


class CosineClassifier(nn.Module):
    """CLIP ViT-B/32 encoder + cosine classification head.

    Args:
        clip_model: CLIP model from load_openai_clip.
        num_classes: Number of output classes.
        feature_dim: CLIP feature dimensionality (512 for ViT-B/32).
        freeze_clip: Whether to freeze the CLIP backbone.
        init_scale: Initial value for the logit scale (temperature-like).
        learnable_scale: If True, logit_scale is a trainable parameter.
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

        if init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {init_scale}")
        if init_scale > 100:
            raise ValueError(f"init_scale must be <= 100, got {init_scale}")

        self.visual = clip_model.visual
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.freeze_clip = freeze_clip
        self.learnable_scale = learnable_scale
        self.head_type = "cosine"

        # Freeze CLIP backbone
        if freeze_clip:
            for param in self.visual.parameters():
                param.requires_grad = False
            logger.info("CLIP image encoder frozen.")

        # Cosine classifier weight (class prototypes) -- no bias
        weight = torch.randn(num_classes, feature_dim)
        weight = F.normalize(weight, dim=1) * init_scale
        self.weight = nn.Parameter(weight)

        # Logit scale (temperature)
        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer("logit_scale", torch.tensor(float(init_scale)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Extract and L2-normalize image features."""
        conv1_dtype = self.visual.conv1.weight.dtype
        images = images.to(dtype=conv1_dtype)

        with torch.set_grad_enabled(not self.freeze_clip):
            features = self.visual(images)

        if features.dim() > 2:
            features = (
                features.mean(dim=[2, 3]) if features.dim() == 4 else features[:, 0]
            )

        features = features.float()
        features = F.normalize(features, p=2, dim=-1)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode, normalize, compute cosine similarity, scale.

        Returns logits of shape (B, num_classes).
        """
        features = self.encode_image(images)
        # Normalize weight on each forward pass
        weight_norm = F.normalize(self.weight, dim=1)
        logits = features @ weight_norm.T * self.clamp_scale()
        return logits

    def clamp_scale(self):
        """Clamp logit_scale to [1.0, 100.0]. No-op when learnable_scale=False."""
        if self.learnable_scale:
            with torch.no_grad():
                self.logit_scale.clamp_(min=1.0, max=100.0)
        return self.logit_scale

    def get_param_groups(self, lr, weight_decay):
        """Return optimizer param groups.

        When learnable_scale=True: logit_scale gets lr*0.1, no weight decay.
        When learnable_scale=False: only weight is optimized.
        """
        groups = [
            {
                "params": [self.weight],
                "lr": lr,
                "weight_decay": weight_decay,
            },
        ]
        if self.learnable_scale:
            groups.append({
                "params": [self.logit_scale],
                "lr": lr * 0.1,
                "weight_decay": 0.0,
            })
        return groups

    def train(self, mode: bool = True):
        """Override train() to keep CLIP backbone in eval mode when frozen."""
        super().train(mode)
        if self.freeze_clip:
            self.visual.eval()
        return self


def build_cosine_model(config: dict, device: torch.device) -> Tuple[CosineClassifier, callable]:
    """Build CosineClassifier and return CLIP preprocessing function.

    Args:
        config: Must contain model.cos_init_scale, model.cos_learnable_scale,
                model.num_classes, model.feature_dim, model.freeze_clip.
        device: torch device.

    Returns:
        Tuple of (model, preprocess_fn).
    """
    clip_model, preprocess = load_openai_clip(device)

    model_cfg = config["model"]
    model = CosineClassifier(
        clip_model=clip_model,
        num_classes=model_cfg.get("num_classes", 500),
        feature_dim=model_cfg.get("feature_dim", 512),
        freeze_clip=model_cfg.get("freeze_clip", True),
        init_scale=model_cfg.get("cos_init_scale", 10.0),
        learnable_scale=model_cfg.get("cos_learnable_scale", True),
    )
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"CosineClassifier: {total:,} params, {trainable:,} trainable")

    return model, preprocess
