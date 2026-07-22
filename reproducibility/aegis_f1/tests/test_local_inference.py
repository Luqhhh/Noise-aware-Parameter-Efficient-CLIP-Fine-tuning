import torch

from aegis_clip.local_inference import (
    attention_guided_crop,
    complementary_flip_local_fusion,
)


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
