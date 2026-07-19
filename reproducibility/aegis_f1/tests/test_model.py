from copy import deepcopy

import torch
from torch import nn

from aegis_clip.checkpoint import load_initial_weights
from aegis_clip.model import AegisCLIP


class TinyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1)
        self.ln_pre = nn.LayerNorm(4)
        self.ln_post = nn.LayerNorm(4)
        self.proj = nn.Parameter(torch.eye(4))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv1(images).mean(dim=(2, 3))
        return self.ln_post(self.ln_pre(features)) @ self.proj


class TinyAttentionBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, num_heads=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.attn(values, values, values, need_weights=False)[0]


class TinyTransformer(nn.Module):
    def __init__(self, width: int, blocks: int = 2) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(
            [TinyAttentionBlock(width) for _ in range(blocks)]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            values = block(values)
        return values


class TinyAttentionVisual(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=1)
        self.transformer = TinyTransformer(width)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.eye(width))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = self.conv1(images).mean(dim=(2, 3)).unsqueeze(0)
        values = self.transformer(values).squeeze(0)
        return self.ln_post(values) @ self.proj


def _backward(mode: str) -> AegisCLIP:
    model = AegisCLIP(TinyVisual(), num_classes=3, feature_dim=4, peft_mode=mode)
    loss = model(images=torch.randn(5, 3, 4, 4)).sum()
    loss.backward()
    return model


def test_frozen_mode_blocks_visual_gradients() -> None:
    model = _backward("frozen")
    assert not model.visual_requires_grad
    assert all(parameter.grad is None for parameter in model.visual.parameters())
    assert model.classifier.weight.grad is not None


def test_feature_adapter_starts_as_identity_and_keeps_visual_frozen() -> None:
    model = AegisCLIP(
        TinyVisual(),
        num_classes=3,
        feature_dim=4,
        peft_mode="feature_adapter",
        adapter_dim=2,
        adapter_scale=0.25,
    )
    features = torch.randn(5, 4)
    with torch.no_grad():
        adapted = model.adapt_features(features)
    assert torch.allclose(adapted, torch.nn.functional.normalize(features, dim=1))

    loss = model(features=features).sum()
    loss.backward()
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert any(name.startswith("feature_adapter.") for name in trainable)
    assert all(parameter.grad is None for parameter in model.visual.parameters())
    assert model.feature_adapter.up.weight.grad is not None
    assert model.feature_adapter.residual_scale == 0.25


def test_anchored_residual_classifier_preserves_base_and_updates_residual() -> None:
    linear = AegisCLIP(TinyVisual(), num_classes=3, feature_dim=4)
    anchored = AegisCLIP(
        TinyVisual(),
        num_classes=3,
        feature_dim=4,
        classifier_mode="anchored_residual",
        classifier_residual_scale=0.25,
    )
    incompatible = anchored.load_state_dict(linear.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "classifier.residual_weight",
        "classifier.residual_bias",
    }
    assert incompatible.unexpected_keys == []

    features = torch.randn(5, 4)
    with torch.no_grad():
        expected = linear(features=features)
        actual = anchored(features=features)
    assert torch.allclose(actual, expected)

    loss = anchored(features=features).sum()
    loss.backward()
    assert anchored.classifier.weight.grad is None
    assert anchored.classifier.bias.grad is None
    assert anchored.classifier.residual_weight.grad is not None
    assert anchored.classifier.residual_bias.grad is not None


def test_visual_ln_trains_only_layer_norms() -> None:
    model = _backward("visual_ln")
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert "visual.ln_pre.weight" in trainable
    assert "visual.ln_post.bias" in trainable
    assert "visual.conv1.weight" not in trainable
    assert "visual.proj" not in trainable
    assert model.visual.ln_post.weight.grad is not None


def test_ln_post_proj_policy_is_exact() -> None:
    model = AegisCLIP(
        TinyVisual(), num_classes=3, feature_dim=4, peft_mode="ln_post_proj"
    )
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable == {
        "visual.proj",
        "visual.ln_post.weight",
        "visual.ln_post.bias",
        "classifier.weight",
        "classifier.bias",
    }


def test_visual_lora_is_identity_then_updates_native_attention() -> None:
    visual = TinyAttentionVisual()
    reference = deepcopy(visual)
    model = AegisCLIP(
        visual,
        num_classes=3,
        feature_dim=4,
        peft_mode="visual_lora",
        lora_last_n_blocks=1,
        lora_rank=2,
        lora_alpha=2.0,
    )
    images = torch.randn(5, 3, 4, 4)
    with torch.no_grad():
        expected = torch.nn.functional.normalize(reference(images), dim=1)
        actual = model.encode_image(images)
    assert torch.allclose(actual, expected, atol=1.0e-6)

    model(images=images).sum().backward()
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert any("parametrizations.in_proj_weight.0.q_B" in name for name in trainable)
    assert any("parametrizations.weight.0.lora_B" in name for name in trainable)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name.endswith("parametrizations.in_proj_weight.original")
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for name, parameter in model.named_parameters()
        if name.endswith(("q_B", "v_B", "lora_B"))
    )


def test_visual_lora_initialises_from_frozen_checkpoint(tmp_path) -> None:
    visual = TinyAttentionVisual()
    frozen = AegisCLIP(deepcopy(visual), num_classes=3, feature_dim=4)
    lora = AegisCLIP(
        deepcopy(visual),
        num_classes=3,
        feature_dim=4,
        peft_mode="visual_lora",
        lora_last_n_blocks=1,
        lora_rank=2,
        lora_alpha=2.0,
    )
    path = tmp_path / "frozen.pt"
    torch.save({"model_state_dict": frozen.state_dict(), "epoch": 7}, path)
    state = load_initial_weights(lora, path, torch.device("cpu"))
    assert state["epoch"] == 7
    images = torch.randn(5, 3, 4, 4)
    with torch.no_grad():
        assert torch.allclose(
            lora(images=images), frozen(images=images), atol=1.0e-6
        )
