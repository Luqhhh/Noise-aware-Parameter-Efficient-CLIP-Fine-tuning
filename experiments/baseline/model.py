"""
Model definition: CLIP ViT-B/32 with a linear classification head.

Architecture:
    CLIP ViT-B/32 (frozen) -> L2 Normalize -> Linear(512, num_classes)

The CLIP image encoder is frozen; only the linear classifier is trained.
This design allows for a simple, fast baseline that can later be extended
with LoRA, adapters, or other parameter-efficient fine-tuning methods.

The model is built from common/clip_utils.py which centralizes CLIP loading.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.clip_utils import load_openai_clip

logger = logging.getLogger(__name__)


class CLIPLinearClassifier(nn.Module):
    """CLIP ViT-B/32 image encoder + linear classification head.

    The CLIP backbone can be fully frozen or partially unfrozen (last N
    transformer blocks, ln_post, visual.proj). All visual parameters are
    ALWAYS frozen first, then selectively unfrozen per config.

    Args:
        clip_model: The CLIP model (from `clip.load`).
        num_classes: Number of output classes (500 for this competition).
        feature_dim: Dimensionality of CLIP image features (512 for ViT-B/32).
        freeze_clip: Whether to freeze the CLIP backbone parameters.
        unfreeze_last_n_blocks: Number of transformer blocks to unfreeze
            (from the end). Only used when freeze_clip=False.
        train_ln_post: Whether to unfreeze visual.ln_post.
            Only used when freeze_clip=False.
        train_visual_proj: Whether to unfreeze visual.proj.
            Only used when freeze_clip=False.
        backbone_lr: Learning rate for unfrozen visual parameters.
        backbone_weight_decay: Weight decay for unfrozen visual parameters.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        num_classes: int = 500,
        feature_dim: int = 512,
        freeze_clip: bool = True,
        unfreeze_last_n_blocks: int = 0,
        train_ln_post: bool = False,
        train_visual_proj: bool = False,
        backbone_lr: float = 1e-5,
        backbone_weight_decay: float = 0.01,
    ):
        super().__init__()

        self.visual = clip_model.visual
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.freeze_clip = freeze_clip
        self.head_type = "linear"  # For checkpoint metadata

        # Discriminative LR config (used by train.py optimizer)
        self.backbone_lr = backbone_lr
        self.backbone_weight_decay = backbone_weight_decay
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.train_ln_post = train_ln_post
        self.train_visual_proj = train_visual_proj

        # ALWAYS freeze all visual parameters first.
        # Selective unfreezing happens below when freeze_clip=False.
        for param in self.visual.parameters():
            param.requires_grad = False

        if not freeze_clip:
            self.configure_visual_trainability(
                unfreeze_last_n_blocks=unfreeze_last_n_blocks,
                train_ln_post=train_ln_post,
                train_visual_proj=train_visual_proj,
            )
        else:
            logger.info("CLIP image encoder fully frozen.")

        # Linear classification head (always trainable)
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Initialize the linear layer
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def configure_visual_trainability(
        self,
        unfreeze_last_n_blocks: int,
        train_ln_post: bool,
        train_visual_proj: bool,
    ) -> None:
        """Selectively unfreeze CLIP visual encoder components.

        Must be called AFTER all visual parameters are frozen.
        Only the specified components get requires_grad=True.

        Args:
            unfreeze_last_n_blocks: Number of transformer blocks to unfreeze
                counting from the end (0 = none).
            train_ln_post: Unfreeze visual.ln_post (LayerNorm after blocks).
            train_visual_proj: Unfreeze visual.proj (output projection).

        Raises:
            ValueError: If unfreeze_last_n_blocks is out of range.
        """
        blocks = self.visual.transformer.resblocks
        num_blocks = len(blocks)

        if not 0 <= unfreeze_last_n_blocks <= num_blocks:
            raise ValueError(
                f"unfreeze_last_n_blocks must be in [0, {num_blocks}], "
                f"got {unfreeze_last_n_blocks}"
            )

        if unfreeze_last_n_blocks > 0:
            for block in blocks[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            logger.info(
                f"Unfrozen last {unfreeze_last_n_blocks}/{num_blocks} "
                f"transformer blocks."
            )

        if train_ln_post:
            for param in self.visual.ln_post.parameters():
                param.requires_grad = True
            logger.info("Unfrozen visual.ln_post.")

        if train_visual_proj:
            self.visual.proj.requires_grad = True
            logger.info("Unfrozen visual.proj.")

    def train(self, mode: bool = True):
        """Override train() — partially unfrozen visual stays in eval for
        frozen parts. Frozen blocks/layers stay in eval regardless of mode.

        OpenAI CLIP ViT uses LayerNorm (not BatchNorm), so running stats
        are not a concern.
        """
        super().train(mode)

        if self.freeze_clip:
            self.visual.eval()
            return self

        # Partially unfrozen: start from eval, selectively set train
        self.visual.eval()

        n = self.unfreeze_last_n_blocks
        if n > 0:
            for block in self.visual.transformer.resblocks[-n:]:
                block.train(mode)

        if self.train_ln_post:
            self.visual.ln_post.train(mode)

        return self

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
            features = (
                features.mean(dim=[2, 3]) if features.dim() == 4 else features[:, 0]
            )

        # Features are already float32 from the float()-converted visual encoder
        features = F.normalize(features, p=2, dim=-1)

        return features

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Classify pre-computed CLIP features.

        Args:
            features: Tensor of shape [B, feature_dim].

        Returns:
            Logits of shape [B, num_classes].
        """
        if features.ndim != 2:
            raise ValueError(
                f"Expected cached features with shape [B, D], "
                f"got {tuple(features.shape)}"
            )
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, "
                f"got {features.shape[-1]}"
            )

        features = F.normalize(features.float(), p=2, dim=-1)
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode images and classify.

        Args:
            images: Input image batch of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.encode_image(images)
        return self.forward_features(features)

    def get_trainable_parameters(self):
        """Return only the trainable parameters (classifier head).

        Kept for backward compatibility (cached training path).
        """
        return filter(lambda p: p.requires_grad, self.parameters())

    def get_param_groups(
        self,
        head_lr: float,
        head_weight_decay: float,
    ) -> list:
        """Return parameter groups with separate LRs for head and backbone.

        When freeze_clip=True, only the head group is returned.
        When freeze_clip=False, backbone parameters get backbone_lr and
        backbone_weight_decay (set at __init__ from config).

        Args:
            head_lr: Learning rate for the classifier head.
            head_weight_decay: Weight decay for the classifier head.

        Returns:
            List of dicts, each with keys:
                name, params, lr, weight_decay.
        """
        head_params = [
            p for p in self.classifier.parameters()
            if p.requires_grad
        ]
        backbone_params = [
            p for p in self.visual.parameters()
            if p.requires_grad
        ]

        groups = [
            {
                "name": "head",
                "params": head_params,
                "lr": head_lr,
                "weight_decay": head_weight_decay,
            }
        ]

        if backbone_params:
            groups.append({
                "name": "backbone",
                "params": backbone_params,
                "lr": self.backbone_lr,
                "weight_decay": self.backbone_weight_decay,
            })

        return groups


def build_model(
    config: dict, device: torch.device
) -> Tuple[CLIPLinearClassifier, callable]:
    """Build the CLIPLinearClassifier model and return the CLIP preprocessing function.

    Args:
        config: Configuration dictionary. Must contain:
            - model.clip_model_name (e.g., "ViT-B/32")
            - model.num_classes (e.g., 500)
            - model.feature_dim (e.g., 512)
            - model.freeze_clip (bool)
            - model.unfreeze_last_n_blocks (int, default 0)
            - model.train_ln_post (bool, default False)
            - model.train_visual_proj (bool, default False)
            - train.backbone_lr (float, default 1e-5)
            - train.backbone_weight_decay (float, default 0.01)
        device: torch device to load the model on.

    Returns:
        Tuple of (model, preprocess_fn).
        - model: CLIPLinearClassifier instance.
        - preprocess_fn: CLIP preprocessing function (torchvision transform).
    """
    logger.info(f"Loading CLIP model: {config['model']['clip_model_name']}")
    clip_model, preprocess = load_openai_clip(
        device, model_name=config["model"]["clip_model_name"]
    )
    # load_openai_clip already converts the full model to float32

    model_cfg = config["model"]
    train_cfg = config.get("train", {})

    num_classes = model_cfg["num_classes"]
    feature_dim = model_cfg.get("feature_dim", 512)
    freeze_clip = model_cfg.get("freeze_clip", True)

    model = CLIPLinearClassifier(
        clip_model=clip_model,
        num_classes=num_classes,
        feature_dim=feature_dim,
        freeze_clip=freeze_clip,
        unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 0),
        train_ln_post=model_cfg.get("train_ln_post", False),
        train_visual_proj=model_cfg.get("train_visual_proj", False),
        backbone_lr=train_cfg.get("backbone_lr", 1e-5),
        backbone_weight_decay=train_cfg.get("backbone_weight_decay", 0.01),
    )

    model = model.to(device)

    total, trainable = _count_params(model)
    logger.info(
        f"Model built: {total:,} total params, {trainable:,} trainable params"
    )

    return model, preprocess


def _count_params(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
