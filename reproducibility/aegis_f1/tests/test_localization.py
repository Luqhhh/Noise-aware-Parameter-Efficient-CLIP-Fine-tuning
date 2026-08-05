import pytest
import torch

from aegis_clip.localization import (
    attention_weighted_centers,
    extract_attention_crops,
    fuse_global_local_flip_probabilities,
    fuse_global_local_probabilities,
    fuse_global_multilocal_flip_probabilities,
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


def test_global_local_flip_fusion_matches_two_stage_probability_blend() -> None:
    global_logits = torch.tensor([[4.0, 0.0]])
    local_logits = torch.tensor([[0.0, 2.0]])
    flipped_global_logits = torch.tensor([[1.0, 0.0]])
    flipped_local_logits = torch.tensor([[0.0, 3.0]])
    fused = fuse_global_local_flip_probabilities(
        global_logits,
        local_logits,
        flipped_global_logits,
        flipped_local_logits,
        local_weight=0.35,
        flip_weight=0.25,
    )
    global_probability = (
        0.75 * global_logits.softmax(dim=1)
        + 0.25 * flipped_global_logits.softmax(dim=1)
    )
    local_probability = (
        0.75 * local_logits.softmax(dim=1)
        + 0.25 * flipped_local_logits.softmax(dim=1)
    )
    expected = 0.65 * global_probability + 0.35 * local_probability
    assert torch.allclose(fused.exp(), expected)


def test_multilocal_flip_fusion_averages_paired_local_probabilities() -> None:
    global_logits = torch.tensor([[4.0, 0.0]])
    flipped_global_logits = torch.tensor([[2.0, 0.0]])
    local_logits = [
        torch.tensor([[0.0, 3.0]]),
        torch.tensor([[1.0, 0.0]]),
    ]
    flipped_local_logits = [
        torch.tensor([[0.0, 2.0]]),
        torch.tensor([[0.0, 1.0]]),
    ]
    fused = fuse_global_multilocal_flip_probabilities(
        global_logits,
        local_logits,
        flipped_global_logits,
        flipped_local_logits,
        local_weight=0.4,
        flip_weight=0.25,
    )
    global_probability = (
        0.75 * global_logits.softmax(dim=1)
        + 0.25 * flipped_global_logits.softmax(dim=1)
    )
    mean_local_probability = torch.stack(
        [
            0.75 * original.softmax(dim=1)
            + 0.25 * flipped.softmax(dim=1)
            for original, flipped in zip(local_logits, flipped_local_logits)
        ]
    ).mean(dim=0)
    expected = 0.6 * global_probability + 0.4 * mean_local_probability
    assert torch.allclose(fused.exp(), expected)


def test_multilocal_flip_fusion_rejects_mismatched_view_counts() -> None:
    with pytest.raises(ValueError, match="counts must match"):
        fuse_global_multilocal_flip_probabilities(
            torch.randn(2, 3),
            [torch.randn(2, 3)],
            torch.randn(2, 3),
            [],
        )


def test_global_local_flip_fusion_reduces_to_m1_when_flip_weight_is_zero() -> None:
    views = [torch.randn(3, 5) for _ in range(4)]
    fused = fuse_global_local_flip_probabilities(
        *views,
        local_weight=0.4,
        flip_weight=0.0,
    )
    expected = fuse_global_local_probabilities(
        views[0],
        views[1],
        local_weight=0.4,
    )
    assert torch.allclose(fused, expected)


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


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_global_local_flip_fusion_rejects_invalid_flip_weight(
    weight: float,
) -> None:
    views = [torch.randn(2, 3) for _ in range(4)]
    with pytest.raises(ValueError, match="flip_weight"):
        fuse_global_local_flip_probabilities(*views, flip_weight=weight)


def test_parse_int_sequence_rejects_duplicates() -> None:
    assert parse_int_sequence("128, 160,192") == (128, 160, 192)
    with pytest.raises(ValueError, match="unique"):
        parse_int_sequence("160,160")


def test_flip_fusion_per_branch_temperature_changes_fusion() -> None:
    torch.manual_seed(0)
    views = [torch.randn(2, 3) for _ in range(4)]
    single = fuse_global_local_flip_probabilities(
        *views, local_weight=0.4, flip_weight=0.5, temperature=1.5
    )
    per_branch = fuse_global_local_flip_probabilities(
        *views,
        local_weight=0.4,
        flip_weight=0.5,
        temperature=1.5,
        global_temperature=1.0,
        local_temperature=1.5,
    )
    # Different branch temperatures must change the fused logits.
    assert not torch.allclose(single, per_branch, atol=1.0e-6)


def test_flip_fusion_per_branch_temperature_backward_compatible() -> None:
    torch.manual_seed(1)
    views = [torch.randn(2, 3) for _ in range(4)]
    baseline = fuse_global_local_flip_probabilities(
        *views, local_weight=0.4, flip_weight=0.5, temperature=1.5
    )
    explicit = fuse_global_local_flip_probabilities(
        *views,
        local_weight=0.4,
        flip_weight=0.5,
        temperature=1.5,
        global_temperature=1.5,
        local_temperature=1.5,
    )
    # Explicit equal temperatures equal the single-temperature default.
    assert torch.allclose(baseline, explicit, atol=1.0e-6)
