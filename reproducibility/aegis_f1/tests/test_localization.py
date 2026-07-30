import pytest
import torch

from aegis_clip.localization import (
    attention_weighted_centers,
    extract_attention_crops,
    fuse_global_local_probabilities,
    fuse_global_multilocal_probabilities,
    parse_int_sequence,
)


def test_attention_center_uses_top_k_weighted_patch_coordinates() -> None:
    attention = torch.zeros(1, 2, 49)
    attention[:, :, 0] = 1.0
    attention[:, :, 48] = 3.0
    center = attention_weighted_centers(
        attention,
        image_height=224,
        image_width=224,
        top_k=2,
    )
    # Patch centers are (16, 16) and (208, 208), weighted 1:3.
    assert torch.allclose(center, torch.tensor([[160.0, 160.0]]))


def test_attention_crop_clamps_box_and_restores_input_shape() -> None:
    images = torch.arange(3 * 224 * 224, dtype=torch.float32).reshape(
        1, 3, 224, 224
    )
    attention = torch.zeros(1, 1, 49)
    attention[0, 0, 48] = 1.0
    crops, boxes = extract_attention_crops(
        images,
        attention,
        crop_size=160,
        top_k=1,
    )
    assert crops.shape == images.shape
    assert boxes == [(64, 64, 224, 224)]


def test_probability_fusion_matches_equal_probability_average() -> None:
    global_logits = torch.tensor([[4.0, 0.0]])
    local_logits = torch.tensor([[0.0, 2.0]])
    fused = fuse_global_local_probabilities(
        global_logits,
        local_logits,
        local_weight=0.5,
    )
    expected = (
        global_logits.softmax(dim=1) + local_logits.softmax(dim=1)
    ) / 2.0
    assert torch.allclose(fused.exp(), expected)


def test_multilocal_fusion_averages_local_probabilities_before_global_fusion() -> None:
    global_logits = torch.tensor([[3.0, 0.0]])
    local_logits = [
        torch.tensor([[0.0, 2.0]]),
        torch.tensor([[1.0, 0.0]]),
    ]
    fused = fuse_global_multilocal_probabilities(
        global_logits,
        local_logits,
        local_weight=0.4,
    )
    expected = 0.6 * global_logits.softmax(dim=1) + 0.4 * torch.stack(
        [value.softmax(dim=1) for value in local_logits]
    ).mean(dim=0)
    assert torch.allclose(fused.exp(), expected)


def test_multilocal_fusion_requires_a_local_view() -> None:
    with pytest.raises(ValueError, match="At least one"):
        fuse_global_multilocal_probabilities(torch.randn(2, 3), [])


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_probability_fusion_rejects_invalid_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="local_weight"):
        fuse_global_local_probabilities(
            torch.randn(2, 3),
            torch.randn(2, 3),
            local_weight=weight,
        )


def test_parse_int_sequence_rejects_duplicates() -> None:
    assert parse_int_sequence("128, 160,192") == (128, 160, 192)
    with pytest.raises(ValueError, match="unique"):
        parse_int_sequence("160,160")
