"""Tests for NEW01: attention ROI recovered from the ORIGINAL RGB image.

All tests are tiny, synthetic, CPU-only and model-free.
"""

import pytest
import torch

from aegis_clip.local_inference import attention_guided_crop
from aegis_clip.original_roi import (
    attention_crop_box,
    geometry_vector,
    global_box_to_original,
    inverse_round_trip_check,
    original_roi_batch,
    unflip_box_x,
)


def _gaussian_attention(grid: int, center_y: float, center_x: float, sigma: float = 2.0):
    rows = torch.arange(grid, dtype=torch.float32)
    attention = torch.exp(
        -(
            (rows[:, None] - center_y) ** 2 + (rows[None, :] - center_x) ** 2
        )
        / (2.0 * sigma * sigma)
    )
    return attention.unsqueeze(0)


def _smooth_image(size: int) -> torch.Tensor:
    coordinate = torch.linspace(0.0, 1.0, size)
    plane = 0.5 + 0.4 * torch.sin(6.0 * coordinate)[None, :] * torch.cos(
        5.0 * coordinate
    )[:, None]
    return plane.expand(3, size, size).clone().unsqueeze(0)


# ---------------------------------------------------------------------------
# 1. Round-trip: an RRC-like geometry maps the global box onto the expected
#    original pixels (within 1 px) for flip=False and flip=True.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flip", [False, True])
def test_global_box_maps_to_expected_original_pixels(flip: bool):
    height0, width0 = 400, 600
    top, left, crop_h, crop_w = 40, 60, 320, 480
    size = 384
    half = 142.5
    center_x, center_y = 200.0, 180.0
    geometry = geometry_vector(
        (height0, width0), (top, left, crop_h, crop_w), flip, size
    ).unsqueeze(0)
    box = {
        "center_x": torch.tensor([center_x]),
        "center_y": torch.tensor([center_y]),
        "half": torch.tensor([half]),
    }

    mapped = global_box_to_original(box, geometry)
    axis = size - center_x if flip else center_x
    expected_x0 = left + (axis - half) * crop_w / size
    expected_x1 = left + (axis + half) * crop_w / size
    expected_y0 = top + (center_y - half) * crop_h / size
    expected_y1 = top + (center_y + half) * crop_h / size

    assert abs(float(mapped["x0"]) - expected_x0) <= 1.0
    assert abs(float(mapped["x1"]) - expected_x1) <= 1.0
    assert abs(float(mapped["y0"]) - expected_y0) <= 1.0
    assert abs(float(mapped["y1"]) - expected_y1) <= 1.0
    assert float(mapped["x1"]) > float(mapped["x0"])
    assert float(mapped["y1"]) > float(mapped["y0"])

    # The mapped box must actually land on the marker drawn at the expected
    # original pixels.
    image = torch.zeros(3, height0, width0)
    x0 = int(round(expected_x0)) - 4
    x1 = int(round(expected_x1)) + 4
    y0 = int(round(expected_y0)) - 4
    y1 = int(round(expected_y1)) + 4
    image[:, max(y0, 0) : y1, max(x0, 0) : x1] = 1.0

    roi = original_roi_batch(
        [image], torch.tensor([[height0, width0]]), geometry, box, out_size=32
    )
    assert roi.shape == (1, 3, 32, 32)
    assert float(roi.mean()) > 0.99


def test_inverse_round_trip_is_exact_for_interior_box():
    geometry = geometry_vector((400, 600), (40, 60, 320, 480), True, 384).unsqueeze(0)
    box = {
        "center_x": torch.tensor([200.0]),
        "center_y": torch.tensor([180.0]),
        "half": torch.tensor([142.5]),
    }
    report = inverse_round_trip_check(box, geometry)
    assert report["ok"] is True
    assert report["max_abs_error"] <= 1.0e-3


# ---------------------------------------------------------------------------
# 2. `attention_crop_box` fed to the existing `attention_guided_crop`.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("attention_kind", ["constant", "gaussian"])
def test_box_matches_existing_attention_guided_crop(attention_kind: str):
    size = 64
    crop_size = 48
    top_patches = 5
    grid = 8
    if attention_kind == "constant":
        attention = torch.ones(1, grid, grid)
    else:
        attention = _gaussian_attention(grid, center_y=2.0, center_x=5.0)

    global_input = _smooth_image(size)  # [1, 3, S, S]
    box = attention_crop_box(
        attention,
        crop_size=crop_size,
        top_patches=top_patches,
        height=size,
        width=size,
    )
    reference = attention_guided_crop(
        global_input, attention, crop_size=crop_size, top_patches=top_patches
    )
    # Treat the already-SxS input as an identity-geometry "original": the mapped
    # box is then exactly the box attention_guided_crop crops internally.
    geometry = geometry_vector((size, size), (0, 0, size, size), False, size).unsqueeze(0)
    roi = original_roi_batch(
        global_input,
        torch.tensor([[size, size]]),
        geometry,
        box,
        out_size=size,
        normalize=None,
    )
    assert roi.shape == reference.shape
    assert torch.allclose(roi, reference, atol=2.0e-2), (
        (roi - reference).abs().max().item()
    )


# ---------------------------------------------------------------------------
# 3. Flip handling.
# ---------------------------------------------------------------------------
def test_flipped_roi_is_flip_of_unflipped_path():
    height0, width0 = 256, 256
    size = 64
    generator = torch.Generator().manual_seed(11)
    image = torch.rand(3, height0, width0, generator=generator)
    crop = (20, 30, 200, 220)
    geometry_flip = geometry_vector((height0, width0), crop, True, size).unsqueeze(0)
    geometry_plain = geometry_vector((height0, width0), crop, False, size).unsqueeze(0)
    box = {
        "center_x": torch.tensor([31.0]),
        "center_y": torch.tensor([25.0]),
        "half": torch.tensor([12.0]),
    }
    # The un-flipped geometry must look at the SAME original pixels, i.e. use
    # the un-flipped box.
    box_unflipped = unflip_box_x(box, torch.tensor([True]), size)

    roi_flipped = original_roi_batch(
        [image], torch.tensor([[height0, width0]]), geometry_flip, box, out_size=32
    )
    roi_plain = original_roi_batch(
        [image],
        torch.tensor([[height0, width0]]),
        geometry_plain,
        box_unflipped,
        out_size=32,
    )
    assert torch.allclose(
        roi_flipped, torch.flip(roi_plain, dims=[-1]), atol=1.0e-3
    )


# ---------------------------------------------------------------------------
# 4. Clamping and degenerate boxes.
# ---------------------------------------------------------------------------
def test_border_box_is_clamped_and_shape_is_exact():
    height0, width0 = 200, 200
    size = 64
    geometry = geometry_vector((height0, width0), (0, 0, height0, width0), False, size).unsqueeze(0)
    box = {
        "center_x": torch.tensor([2.0]),
        "center_y": torch.tensor([2.0]),
        "half": torch.tensor([10.0]),
    }
    mapped = global_box_to_original(box, geometry)
    assert float(mapped["x0"]) >= 0.0
    assert float(mapped["y0"]) >= 0.0
    assert float(mapped["x1"]) <= width0
    assert float(mapped["y1"]) <= height0
    assert float(mapped["x1"]) > float(mapped["x0"])
    assert float(mapped["y1"]) > float(mapped["y0"])

    image = torch.rand(3, height0, width0, generator=torch.Generator().manual_seed(5))
    roi = original_roi_batch(
        [image], torch.tensor([[height0, width0]]), geometry, box, out_size=48
    )
    assert roi.shape == (1, 3, 48, 48)


def test_degenerate_boxes_fail_closed():
    geometry = geometry_vector((200, 200), (0, 0, 200, 200), False, 64).unsqueeze(0)
    zero_half = {
        "center_x": torch.tensor([32.0]),
        "center_y": torch.tensor([32.0]),
        "half": torch.tensor([0.0]),
    }
    with pytest.raises(ValueError):
        global_box_to_original(zero_half, geometry)

    outside = {
        "center_x": torch.tensor([-500.0]),
        "center_y": torch.tensor([32.0]),
        "half": torch.tensor([10.0]),
    }
    with pytest.raises(ValueError):
        global_box_to_original(outside, geometry)

    image = torch.zeros(3, 200, 200)
    with pytest.raises(ValueError):
        original_roi_batch(
            [image], torch.tensor([[200, 200]]), geometry, zero_half, out_size=32
        )


# ---------------------------------------------------------------------------
# 5. The original-pixel path is not a no-op relative to the SxS input.
# ---------------------------------------------------------------------------
def test_original_roi_differs_from_cropping_the_downscaled_input():
    height0 = width0 = 512
    size = 64
    generator = torch.Generator().manual_seed(7)
    original = torch.rand(3, height0, width0, generator=generator)
    # The training global view is a heavy downscale of the original.
    global_input = torch.nn.functional.interpolate(
        original.unsqueeze(0), size=(size, size), mode="bicubic", align_corners=False
    )
    geometry = geometry_vector(
        (height0, width0), (0, 0, height0, width0), False, size
    ).unsqueeze(0)
    box = {
        "center_x": torch.tensor([size / 2.0]),
        "center_y": torch.tensor([size / 2.0]),
        "half": torch.tensor([size / 4.0]),
    }
    roi_from_original = original_roi_batch(
        [original], torch.tensor([[height0, width0]]), geometry, box, out_size=size
    )
    # Same box, but cropped out of the already-downscaled SxS input.
    identity = geometry_vector((size, size), (0, 0, size, size), False, size).unsqueeze(0)
    roi_from_global = original_roi_batch(
        global_input, torch.tensor([[size, size]]), identity, box, out_size=size
    )
    assert roi_from_original.shape == roi_from_global.shape
    assert (roi_from_original - roi_from_global).abs().max().item() > 0.1


# ---------------------------------------------------------------------------
# Interface guards.
# ---------------------------------------------------------------------------
def test_attention_crop_box_validates_inputs():
    with pytest.raises(ValueError):
        attention_crop_box(torch.ones(4, 4), crop_size=2, top_patches=1, height=8, width=8)
    with pytest.raises(ValueError):
        attention_crop_box(torch.ones(1, 8, 8), crop_size=64, top_patches=1, height=64, width=64)
    with pytest.raises(ValueError):
        attention_crop_box(torch.ones(1, 8, 8), crop_size=48, top_patches=100, height=64, width=64)
    with pytest.raises(ValueError):
        attention_crop_box(torch.ones(1, 8, 8), crop_size=48, top_patches=1, height=64, width=32)


def test_original_sizes_must_agree_with_geometry():
    image = torch.zeros(3, 100, 100)
    geometry = geometry_vector((100, 100), (0, 0, 100, 100), False, 32).unsqueeze(0)
    box = {
        "center_x": torch.tensor([16.0]),
        "center_y": torch.tensor([16.0]),
        "half": torch.tensor([8.0]),
    }
    with pytest.raises(ValueError):
        original_roi_batch(
            [image], torch.tensor([[200, 200]]), geometry, box, out_size=16
        )
