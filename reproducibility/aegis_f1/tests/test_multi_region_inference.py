import pytest
import torch

from aegis_clip.multi_region_inference import (
    crop_at_center,
    discriminative_region_fusion,
)


def test_fixed_crop_zooms_toward_requested_center() -> None:
    values = torch.arange(224 * 224, dtype=torch.float32).reshape(1, 1, 224, 224)
    image = values.repeat(1, 3, 1, 1)

    upper = crop_at_center(
        image, center_y=80.0, center_x=80.0, crop_size=160
    )
    lower = crop_at_center(
        image, center_y=144.0, center_x=144.0, crop_size=160
    )

    assert upper.shape == image.shape
    assert lower.mean() > upper.mean()


def test_fixed_crop_rejects_out_of_bounds_center() -> None:
    with pytest.raises(ValueError, match="outside"):
        crop_at_center(
            torch.randn(1, 3, 224, 224),
            center_y=40.0,
            center_x=112.0,
            crop_size=160,
        )


def test_discriminative_fusion_selects_target_separating_regions() -> None:
    global_logits = torch.tensor([[4.0, 1.0, 0.0]])
    candidate_logits = torch.tensor(
        [
            [
                [5.0, 0.0, 0.0],
                [3.0, 2.9, 0.0],
                [0.0, 4.0, 0.0],
            ]
        ]
    )

    result = discriminative_region_fusion(
        global_logits, candidate_logits, top_regions=2
    )

    assert result["selected_region_indices"].shape == (1, 2)
    assert result["selected_region_indices"][0, 0].item() == 0
    assert result["logits"].argmax(dim=1).item() == 0
    assert torch.allclose(
        result["selected_region_weights"].sum(dim=1), torch.ones(1)
    )
