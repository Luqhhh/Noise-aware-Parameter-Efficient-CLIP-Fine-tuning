from copy import deepcopy

import torch
from torch import nn

from aegis_clip.checkpoint import load_initial_weights
from aegis_clip.model import AegisCLIP, interpolate_visual_positional_embedding


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


class TinyCLIPResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, num_heads=1)
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.ReLU(),
            nn.Linear(width * 2, width),
        )
        self.attn_mask = None

    def attention(self, values: torch.Tensor) -> torch.Tensor:
        return self.attn(values, values, values, need_weights=False)[0]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.attention(self.ln_1(values))
        return values + self.mlp(self.ln_2(values))


class TinyCLIPTransformer(nn.Module):
    def __init__(self, width: int, blocks: int = 2) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(
            [TinyCLIPResidualBlock(width) for _ in range(blocks)]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            values = block(values)
        return values


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


class TinyGridVisual(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=1, stride=1, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(width))
        self.positional_embedding = nn.Parameter(torch.randn(5, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = TinyTransformer(width, blocks=1)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.eye(width))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = self.conv1(images)
        values = values.reshape(values.shape[0], values.shape[1], -1)
        values = values.permute(0, 2, 1)
        class_token = self.class_embedding.reshape(1, 1, -1).expand(
            values.shape[0], 1, values.shape[-1]
        )
        values = torch.cat([class_token, values], dim=1)
        values = values + self.positional_embedding.unsqueeze(0)
        values = self.ln_pre(values).permute(1, 0, 2)
        values = self.transformer(values).permute(1, 0, 2)
        values = self.ln_post(values[:, 0, :])
        return values @ self.proj


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


class TinyAdapterVisual(TinyAttentionVisual):
    def __init__(self, width: int = 4) -> None:
        super().__init__(width)
        self.transformer = TinyCLIPTransformer(width)


def _backward(mode: str) -> AegisCLIP:
    model = AegisCLIP(TinyVisual(), num_classes=3, feature_dim=4, peft_mode=mode)
    loss = model(images=torch.randn(5, 3, 4, 4)).sum()
    loss.backward()
    return model


def test_visual_position_interpolation_preserves_cls_and_native_grid() -> None:
    positions = torch.randn(50, 8)
    native = interpolate_visual_positional_embedding(positions, (7, 7))
    resized = interpolate_visual_positional_embedding(positions, (9, 9))

    assert native is positions
    assert resized.shape == (82, 8)
    assert torch.equal(resized[0], positions[0])
    assert torch.isfinite(resized).all()


def test_grid_visual_supports_high_and_rectangular_resolutions() -> None:
    visual = TinyGridVisual()
    model = AegisCLIP(
        visual,
        num_classes=3,
        feature_dim=4,
        input_resolution=4,
    )

    native_images = torch.randn(2, 3, 2, 2)
    with torch.no_grad():
        expected = torch.nn.functional.normalize(visual(native_images), dim=1)
        actual = model.encode_image(native_images)
        high_res_logits = model(images=torch.randn(2, 3, 4, 4))
        rectangular_logits = model(images=torch.randn(2, 3, 1, 4))

    assert torch.allclose(actual, expected, atol=1.0e-6)
    assert high_res_logits.shape == (2, 3)
    assert rectangular_logits.shape == (2, 3)
    assert torch.isfinite(high_res_logits).all()
    assert torch.isfinite(rectangular_logits).all()


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


def test_visual_mlp_adapter_is_exact_identity_and_receives_gradients() -> None:
    visual = TinyAdapterVisual()
    reference = deepcopy(visual)
    model = AegisCLIP(
        visual,
        num_classes=3,
        feature_dim=4,
        peft_mode="visual_mlp_adapter",
        visual_adapter_last_n_blocks=1,
        visual_adapter_bottleneck=2,
        visual_adapter_scale=0.1,
        visual_adapter_dropout=0.1,
    )
    adapter = model.visual.transformer.resblocks[-1].adaptmlp
    with torch.no_grad():
        adapter.down.weight.zero_()
        adapter.down.weight[0, 0] = 1.0
        adapter.down.weight[1, 0] = -1.0
    images = torch.randn(5, 3, 4, 4)
    model.eval()
    with torch.no_grad():
        expected = torch.nn.functional.normalize(reference(images), dim=1)
        actual = model.encode_image(images)
    assert torch.equal(actual, expected)

    model.train()
    torch.nn.functional.cross_entropy(
        model(images=images), torch.tensor([0, 1, 2, 0, 1])
    ).backward()
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert any(".adaptmlp.up.weight" in name for name in trainable)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name.startswith("visual.") and ".adaptmlp." not in name
    )
    assert adapter.up.weight.grad is not None
    assert torch.count_nonzero(adapter.up.weight.grad) > 0
    assert adapter.training


def test_visual_mlp_adapter_initialises_from_frozen_checkpoint(tmp_path) -> None:
    visual = TinyAdapterVisual()
    frozen = AegisCLIP(deepcopy(visual), num_classes=3, feature_dim=4)
    adapted = AegisCLIP(
        deepcopy(visual),
        num_classes=3,
        feature_dim=4,
        peft_mode="visual_mlp_adapter",
        visual_adapter_last_n_blocks=1,
        visual_adapter_bottleneck=2,
    )
    path = tmp_path / "frozen_adapter_parent.pt"
    torch.save({"model_state_dict": frozen.state_dict(), "epoch": 4}, path)
    load_initial_weights(adapted, path, torch.device("cpu"))
    images = torch.randn(3, 3, 4, 4)
    adapted.eval()
    frozen.eval()
    with torch.no_grad():
        assert torch.equal(adapted(images=images), frozen(images=images))


def test_deep_visual_prompt_trains_only_prompt_and_classifier() -> None:
    model = AegisCLIP(
        TinyGridVisual(),
        num_classes=3,
        feature_dim=4,
        peft_mode="visual_prompt",
        visual_prompt_last_n_blocks=1,
        visual_prompt_num_tokens=2,
        visual_prompt_dropout=0.0,
    )
    images = torch.randn(5, 3, 2, 2)

    model(images=images).sum().backward()

    prompt = model.visual.visual_prompt.embeddings
    assert prompt.shape == (1, 2, 4)
    assert prompt.grad is not None
    assert torch.count_nonzero(prompt.grad) > 0
    assert model.classifier.weight.grad is not None
    assert model.visual.conv1.weight.grad is None
    assert model.visual.class_embedding.grad is None


def test_visual_prompt_initialises_from_frozen_checkpoint(tmp_path) -> None:
    frozen = AegisCLIP(TinyGridVisual(), num_classes=3, feature_dim=4)
    prompted = AegisCLIP(
        deepcopy(frozen.visual),
        num_classes=3,
        feature_dim=4,
        peft_mode="visual_prompt",
        visual_prompt_last_n_blocks=1,
        visual_prompt_num_tokens=2,
    )
    path = tmp_path / "frozen_prompt_parent.pt"
    torch.save({"model_state_dict": frozen.state_dict(), "epoch": 7}, path)

    state = load_initial_weights(prompted, path, torch.device("cpu"))

    assert state["epoch"] == 7
    assert prompted.visual.visual_prompt.embeddings.requires_grad
