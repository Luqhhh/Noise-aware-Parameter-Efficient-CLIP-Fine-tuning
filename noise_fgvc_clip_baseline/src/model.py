"""
Model definition: CLIP ViT-B/32 with a linear classification head.

Architecture:
    CLIP ViT-B/32 (frozen) → L2 Normalize → Linear(512, num_classes)

The CLIP image encoder is frozen; only the linear classifier is trained.
This design allows for a simple, fast baseline that can later be extended
with LoRA, adapters, or other parameter-efficient fine-tuning methods.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CLIPLinearClassifier(nn.Module):
    """CLIP ViT-B/32 image encoder + linear classification head.

    The CLIP backbone is frozen by default. Image features are L2-normalized
    before being passed to the linear classifier, following the common practice
    of using cosine-similarity-based classification with CLIP features.

    Args:
        clip_model: The CLIP model (from `clip.load`).
        num_classes: Number of output classes (500 for this competition).
        feature_dim: Dimensionality of CLIP image features (512 for ViT-B/32).
        freeze_clip: Whether to freeze the CLIP backbone parameters.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        num_classes: int = 500,
        feature_dim: int = 512,
        freeze_clip: bool = True,
    ):
        super().__init__()

        self.visual = clip_model.visual
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.freeze_clip = freeze_clip

        # Freeze CLIP backbone
        if freeze_clip:
            for param in self.visual.parameters():
                param.requires_grad = False
            logger.info("CLIP image encoder frozen.")

        # Linear classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Initialize the linear layer
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features using the CLIP visual encoder.

        Args:
            images: Input image batch of shape (B, 3, H, W).

        Returns:
            L2-normalized feature tensor of shape (B, feature_dim).
        """
        # CLIP ViT-B/32 loads with conv1 in fp16 on CUDA; match input dtype
        # to conv1 weight dtype for compatibility.
        conv1_dtype = self.visual.conv1.weight.dtype
        images = images.to(dtype=conv1_dtype)

        with torch.set_grad_enabled(not self.freeze_clip):
            features = self.visual(images)

        # Handle CLIP returning different shapes depending on version
        if features.dim() > 2:
            # Some CLIP versions return spatial features; pool them
            features = features.mean(dim=[2, 3]) if features.dim() == 4 else features[:, 0]

        features = features.float()
        features = F.normalize(features, p=2, dim=-1)

        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode images and classify.

        Args:
            images: Input image batch of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.encode_image(images)
        logits = self.classifier(features)
        return logits

    def get_trainable_parameters(self):
        """Return only the trainable parameters (classifier head)."""
        return filter(lambda p: p.requires_grad, self.parameters())


def build_model(config: dict, device: torch.device) -> Tuple[CLIPLinearClassifier, callable]:
    """Build the CLIPLinearClassifier model and return the CLIP preprocessing function.

    Args:
        config: Configuration dictionary. Must contain:
            - model.clip_model_name (e.g., "ViT-B/32")
            - model.num_classes (e.g., 500)
            - model.feature_dim (e.g., 512)
            - model.freeze_clip (bool)
        device: torch device to load the model on.

    Returns:
        Tuple of (model, preprocess_fn).
        - model: CLIPLinearClassifier instance.
        - preprocess_fn: CLIP preprocessing function (torchvision transform).
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

    logger.info(f"Loading CLIP model: {clip_model_name}")
    clip_model, preprocess = clip.load(clip_model_name, device=device)

    # Convert visual encoder to float32 for consistent dtype handling.
    # CLIP loads some layers in fp16 on CUDA which causes dtype mismatches.
    clip_model.visual = clip_model.visual.float()

    model = CLIPLinearClassifier(
        clip_model=clip_model,
        num_classes=num_classes,
        feature_dim=feature_dim,
        freeze_clip=freeze_clip,
    )

    model = model.to(device)

    total, trainable = _count_params(model)
    logger.info(f"Model built: {total:,} total params, {trainable:,} trainable params")

    return model, preprocess


def _count_params(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
