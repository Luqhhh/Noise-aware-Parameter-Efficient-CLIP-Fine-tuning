"""Single-source model construction for training, evaluation, and inference."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


class AdditiveLowRankParametrization(nn.Module):
    """Zero-initialised LoRA update for an ordinary 2-D weight matrix."""

    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        update = (self.lora_B @ self.lora_A).to(dtype=weight.dtype)
        return weight + self.scaling * update


class QVLowRankParametrization(nn.Module):
    """LoRA updates for Q and V slices of MultiheadAttention.in_proj_weight.

    PyTorch's ``MultiheadAttention`` consumes ``in_proj_weight`` directly, so
    replacing ``out_proj`` with a wrapper does not reliably execute a LoRA
    module. Parametrising the actual weight keeps the native, well-tested MHA
    forward path while making the effective Q/V weights trainable.
    """

    def __init__(self, embed_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        self.embed_dim = int(embed_dim)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.q_A = nn.Parameter(torch.empty(rank, embed_dim))
        self.q_B = nn.Parameter(torch.zeros(embed_dim, rank))
        self.v_A = nn.Parameter(torch.empty(rank, embed_dim))
        self.v_B = nn.Parameter(torch.zeros(embed_dim, rank))
        nn.init.kaiming_uniform_(self.q_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_A, a=math.sqrt(5))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        expected = (3 * self.embed_dim, self.embed_dim)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"Expected combined QKV weight {expected}, got {tuple(weight.shape)}"
            )
        q_update = self.q_B @ self.q_A
        v_update = self.v_B @ self.v_A
        update = torch.cat(
            [q_update, torch.zeros_like(q_update), v_update], dim=0
        ).to(dtype=weight.dtype)
        return weight + self.scaling * update


def install_visual_attention_lora(
    visual: nn.Module,
    *,
    last_n_blocks: int,
    rank: int,
    alpha: float,
    adapt_qv: bool,
    adapt_out: bool,
) -> list[int]:
    """Install identity-at-initialisation LoRA on the last CLIP ViT blocks."""
    if not adapt_qv and not adapt_out:
        raise ValueError("Visual LoRA must adapt Q/V, output projection, or both")
    try:
        blocks = visual.transformer.resblocks
    except AttributeError as exc:
        raise ValueError("Visual LoRA requires CLIP transformer residual blocks") from exc
    block_count = len(blocks)
    if not 1 <= last_n_blocks <= block_count:
        raise ValueError(
            f"lora_last_n_blocks must be in [1, {block_count}], got {last_n_blocks}"
        )
    selected = list(range(block_count - last_n_blocks, block_count))
    for index in selected:
        attention = blocks[index].attn
        embed_dim = int(attention.embed_dim)
        if adapt_qv:
            if attention.in_proj_weight is None:
                raise ValueError("Visual LoRA requires a combined MHA in_proj_weight")
            qv_lora = QVLowRankParametrization(embed_dim, rank, alpha).to(
                device=attention.in_proj_weight.device,
                dtype=attention.in_proj_weight.dtype,
            )
            parametrize.register_parametrization(
                attention, "in_proj_weight", qv_lora
            )
        if adapt_out:
            output_weight = attention.out_proj.weight
            output_lora = AdditiveLowRankParametrization(
                embed_dim, embed_dim, rank, alpha
            ).to(device=output_weight.device, dtype=output_weight.dtype)
            parametrize.register_parametrization(
                attention.out_proj,
                "weight",
                output_lora,
            )
    return selected


class ResidualFeatureAdapter(nn.Module):
    """Near-identity bottleneck adapter for frozen CLIP image features."""

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError("adapter bottleneck_dim must be positive")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("adapter residual_scale must be in (0, 1]")
        self.residual_scale = float(residual_scale)
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.activation(self.down(self.norm(features))))
        return F.normalize(features + self.residual_scale * residual, dim=-1)


class AnchoredResidualClassifier(nn.Module):
    """Frozen robust base classifier plus a small trainable task residual.

    The base parameters intentionally retain the conventional ``weight`` and
    ``bias`` state-dict names.  A checkpoint from the ordinary linear head can
    therefore initialise the anchor exactly, while the residual starts at
    zero and is the only trainable part of the classifier.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("classifier residual_scale must be in (0, 1]")
        self.residual_scale = float(residual_scale)
        self.weight = nn.Parameter(
            torch.empty(num_classes, feature_dim), requires_grad=False
        )
        self.bias = nn.Parameter(torch.empty(num_classes), requires_grad=False)
        self.residual_weight = nn.Parameter(torch.zeros(num_classes, feature_dim))
        self.residual_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weight = self.weight + self.residual_scale * self.residual_weight
        bias = self.bias + self.residual_scale * self.residual_bias
        return F.linear(features, weight, bias)


class AegisCLIP(nn.Module):
    """OpenAI CLIP ViT-B/32 with an auditable PEFT policy and linear head."""

    def __init__(
        self,
        visual: nn.Module,
        num_classes: int,
        feature_dim: int = 512,
        peft_mode: str = "frozen",
        adapter_dim: int = 128,
        adapter_scale: float = 1.0,
        lora_last_n_blocks: int = 4,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        lora_adapt_qv: bool = True,
        lora_adapt_out: bool = True,
        classifier_mode: str = "linear",
        classifier_residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if peft_mode not in {
            "frozen",
            "feature_adapter",
            "visual_ln",
            "ln_post_proj",
            "visual_lora",
        }:
            raise ValueError(f"Unsupported PEFT mode: {peft_mode}")
        self.visual = visual
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.peft_mode = peft_mode
        self.adapter_dim = int(adapter_dim)
        self.adapter_scale = float(adapter_scale)
        self.lora_last_n_blocks = int(lora_last_n_blocks)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_adapt_qv = bool(lora_adapt_qv)
        self.lora_adapt_out = bool(lora_adapt_out)
        self.lora_block_indices: list[int] = []
        if classifier_mode not in {"linear", "anchored_residual"}:
            raise ValueError(f"Unsupported classifier mode: {classifier_mode}")
        self.classifier_mode = str(classifier_mode)
        self.classifier_residual_scale = float(classifier_residual_scale)
        self.feature_adapter = (
            ResidualFeatureAdapter(
                self.feature_dim,
                self.adapter_dim,
                residual_scale=self.adapter_scale,
            )
            if self.peft_mode == "feature_adapter"
            else nn.Identity()
        )
        if self.classifier_mode == "anchored_residual":
            self.classifier = AnchoredResidualClassifier(
                feature_dim,
                num_classes,
                residual_scale=self.classifier_residual_scale,
            )
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        if self.peft_mode == "visual_lora":
            self.lora_block_indices = install_visual_attention_lora(
                self.visual,
                last_n_blocks=self.lora_last_n_blocks,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                adapt_qv=self.lora_adapt_qv,
                adapt_out=self.lora_adapt_out,
            )
        self._configure_trainability()

    def _configure_trainability(self) -> None:
        for parameter in self.visual.parameters():
            parameter.requires_grad_(False)
        for parameter in self.feature_adapter.parameters():
            parameter.requires_grad_(self.peft_mode == "feature_adapter")
        for name, parameter in self.classifier.named_parameters():
            trainable = not (
                self.classifier_mode == "anchored_residual"
                and name in {"weight", "bias"}
            )
            parameter.requires_grad_(trainable)

        if self.peft_mode == "visual_ln":
            for module in self.visual.modules():
                if isinstance(module, nn.LayerNorm):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        elif self.peft_mode == "ln_post_proj":
            for parameter in self.visual.ln_post.parameters():
                parameter.requires_grad_(True)
            self.visual.proj.requires_grad_(True)
        elif self.peft_mode == "visual_lora":
            for module in self.visual.modules():
                if isinstance(
                    module,
                    (AdditiveLowRankParametrization, QVLowRankParametrization),
                ):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)

    @property
    def visual_requires_grad(self) -> bool:
        return any(parameter.requires_grad for parameter in self.visual.parameters())

    def train(self, mode: bool = True) -> "AegisCLIP":
        super().train(mode)
        self.visual.eval()
        if mode and self.visual_requires_grad:
            for module in self.visual.modules():
                if isinstance(module, nn.LayerNorm) and any(
                    parameter.requires_grad for parameter in module.parameters()
                ):
                    module.train(True)
        self.feature_adapter.train(mode and self.peft_mode == "feature_adapter")
        self.classifier.train(mode)
        return self

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        dtype = self.visual.conv1.weight.dtype
        images = images.to(dtype=dtype)
        with torch.set_grad_enabled(self.visual_requires_grad and torch.is_grad_enabled()):
            features = self.visual(images)
        if features.ndim > 2:
            features = (
                features.mean(dim=(2, 3))
                if features.ndim == 4
                else features[:, 0]
            )
        return self.adapt_features(features)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.adapt_features(features))

    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(features.float(), dim=-1)
        return self.feature_adapter(normalized)

    def forward(
        self,
        *,
        images: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (images is None) == (features is None):
            raise ValueError("Provide exactly one of images or features")
        encoded = (
            self.encode_image(images)
            if images is not None
            else self.adapt_features(features)
        )
        logits = self.classifier(encoded)
        return (logits, encoded) if return_features else logits

    def parameter_groups(
        self,
        head_lr: float,
        head_weight_decay: float,
        backbone_lr: float,
        backbone_weight_decay: float,
    ) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = [
            {
                "name": "head",
                "params": [
                    parameter
                    for parameter in (
                        list(self.feature_adapter.parameters())
                        + list(self.classifier.parameters())
                    )
                    if parameter.requires_grad
                ],
                "lr": float(head_lr),
                "weight_decay": float(head_weight_decay),
            }
        ]
        visual = [
            parameter for parameter in self.visual.parameters() if parameter.requires_grad
        ]
        if visual:
            groups.append(
                {
                    "name": "visual",
                    "params": visual,
                    "lr": float(backbone_lr),
                    "weight_decay": float(backbone_weight_decay),
                }
            )
        return groups

    def effective_spec(self) -> dict[str, Any]:
        trainable_names = [
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        ]
        return {
            "backbone": "ViT-B/32",
            "pretrained": "openai",
            "peft_mode": self.peft_mode,
            "adapter_dim": (
                self.adapter_dim if self.peft_mode == "feature_adapter" else None
            ),
            "adapter_scale": (
                self.adapter_scale if self.peft_mode == "feature_adapter" else None
            ),
            "lora_last_n_blocks": (
                self.lora_last_n_blocks
                if self.peft_mode == "visual_lora"
                else None
            ),
            "lora_block_indices": (
                self.lora_block_indices if self.peft_mode == "visual_lora" else None
            ),
            "lora_rank": (
                self.lora_rank if self.peft_mode == "visual_lora" else None
            ),
            "lora_alpha": (
                self.lora_alpha if self.peft_mode == "visual_lora" else None
            ),
            "lora_adapt_qv": (
                self.lora_adapt_qv if self.peft_mode == "visual_lora" else None
            ),
            "lora_adapt_out": (
                self.lora_adapt_out if self.peft_mode == "visual_lora" else None
            ),
            "classifier_mode": self.classifier_mode,
            "classifier_residual_scale": (
                self.classifier_residual_scale
                if self.classifier_mode == "anchored_residual"
                else None
            ),
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "visual_requires_grad": self.visual_requires_grad,
            "trainable_names": trainable_names,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "total_parameters": sum(parameter.numel() for parameter in self.parameters()),
        }


def build_model(
    config: dict[str, Any], device: torch.device
) -> tuple[AegisCLIP, Any]:
    """Build the exact architecture described by a validated configuration."""
    model_config = config["model"]
    if model_config["backbone"] != "ViT-B/32":
        raise ValueError("Only ViT-B/32 is supported")
    if model_config["pretrained"] != "openai":
        raise ValueError("Only OpenAI pretrained weights are supported")
    try:
        import clip
    except ImportError as exc:
        raise ImportError("Install the pinned OpenAI CLIP dependency") from exc

    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.visual = clip_model.visual.float()
    model = AegisCLIP(
        visual=clip_model.visual,
        num_classes=int(model_config["num_classes"]),
        feature_dim=int(model_config.get("feature_dim", 512)),
        peft_mode=str(model_config.get("peft_mode", "frozen")),
        adapter_dim=int(model_config.get("adapter_dim", 128)),
        adapter_scale=float(model_config.get("adapter_scale", 1.0)),
        lora_last_n_blocks=int(model_config.get("lora_last_n_blocks", 4)),
        lora_rank=int(model_config.get("lora_rank", 8)),
        lora_alpha=float(model_config.get("lora_alpha", 8.0)),
        lora_adapt_qv=bool(model_config.get("lora_adapt_qv", True)),
        lora_adapt_out=bool(model_config.get("lora_adapt_out", True)),
        classifier_mode=str(model_config.get("classifier_mode", "linear")),
        classifier_residual_scale=float(
            model_config.get("classifier_residual_scale", 0.25)
        ),
    ).to(device)
    return model, preprocess
