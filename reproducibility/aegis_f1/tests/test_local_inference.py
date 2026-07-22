import torch
from torch import nn

from aegis_clip.local_inference import (
    attention_guided_crop,
    complementary_flip_local_fusion,
    native_visual_forward_with_patch_features,
)
from aegis_clip.model import AegisCLIP


class _TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(width, num_heads=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.attention(values, values, values, need_weights=False)[0]
        return values + output


class _TinyTransformer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([_TinyBlock(width)])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            values = block(values)
        return values


class _TinyGridVisual(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=1, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(width))
        self.positional_embedding = nn.Parameter(torch.randn(5, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = _TinyTransformer(width)
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
        return self.ln_post(values[:, 0]) @ self.proj


def test_attention_crop_zooms_toward_selected_patch() -> None:
    values = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    image = values.repeat(1, 3, 1, 1)
    attention = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]])
    crop = attention_guided_crop(
        image, attention, crop_size=4, top_patches=1
    )
    assert crop.shape == image.shape
    assert crop.mean() > image.mean()


def test_attention_crop_supports_weighted_top_patch_centroid() -> None:
    image = torch.randn(2, 3, 224, 224)
    attention = torch.rand(2, 7, 7)
    crop = attention_guided_crop(
        image, attention, crop_size=160, top_patches=5
    )
    assert crop.shape == image.shape
    assert torch.isfinite(crop).all()


def test_attention_crop_rejects_non_square_inputs() -> None:
    try:
        attention_guided_crop(
            torch.randn(1, 3, 224, 192),
            torch.rand(1, 7, 6),
            crop_size=160,
            top_patches=5,
        )
    except ValueError as exc:
        assert "square" in str(exc)
    else:
        raise AssertionError("Non-square attention crop must fail closed")


def test_complementary_fusion_preserves_identical_views() -> None:
    logits = torch.tensor([[2.0, 1.0, -1.0]])

    result = complementary_flip_local_fusion(logits, logits, logits)

    expected = torch.log_softmax(logits, dim=1)
    assert torch.allclose(result["logits"], expected, atol=1.0e-6)
    assert torch.allclose(result["m1_logits"], expected, atol=1.0e-6)
    assert torch.equal(result["flip_fused_logits"], logits)


def test_complementary_fusion_rejects_shape_mismatch() -> None:
    try:
        complementary_flip_local_fusion(
            torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 3)
        )
    except ValueError as exc:
        assert "equal" in str(exc)
    else:
        raise AssertionError("Mismatched TTA logits must fail closed")


def test_native_patch_capture_preserves_scored_logits() -> None:
    model = AegisCLIP(
        _TinyGridVisual(),
        num_classes=3,
        feature_dim=4,
        peft_mode="frozen",
    )
    model.eval()
    images = torch.randn(3, 3, 2, 2)

    with torch.no_grad():
        expected_logits, expected_features = model(
            images=images,
            return_features=True,
        )
        logits, features, patch_features = (
            native_visual_forward_with_patch_features(model, images)
        )

    assert torch.equal(logits, expected_logits)
    assert torch.equal(features, expected_features)
    assert patch_features.shape == (3, 4, 4)
    assert torch.allclose(
        patch_features.norm(dim=2),
        torch.ones(3, 4),
        atol=1.0e-6,
    )
