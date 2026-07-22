from PIL import Image
import pytest

from aegis_clip.image_preprocess import (
    CLIP_PIXEL_MEAN,
    ResizeLongestSideAndPad,
    select_inference_preprocess,
)


class FakeCompose:
    def __init__(self, transforms: list[object]) -> None:
        self.transforms = transforms


def test_letterbox_preserves_complete_wide_image_and_centers_padding() -> None:
    image = Image.new("RGB", (8, 4), color=(255, 0, 0))
    transform = ResizeLongestSideAndPad(8)

    result = transform(image)

    assert result.size == (8, 8)
    assert result.getpixel((0, 0)) == tuple(
        round(255.0 * value) for value in CLIP_PIXEL_MEAN
    )
    assert result.getpixel((0, 2)) == (255, 0, 0)
    assert result.getpixel((7, 5)) == (255, 0, 0)
    assert result.getpixel((7, 7)) == tuple(
        round(255.0 * value) for value in CLIP_PIXEL_MEAN
    )


def test_letterbox_leaves_square_geometry_unpadded() -> None:
    image = Image.new("RGB", (5, 5), color=(1, 2, 3))

    result = ResizeLongestSideAndPad(8)(image)

    assert result.size == (8, 8)
    assert result.getpixel((0, 0)) == (1, 2, 3)
    assert result.getpixel((7, 7)) == (1, 2, 3)


def test_native_preprocess_is_returned_by_identity() -> None:
    preprocess = object()

    assert (
        select_inference_preprocess(
            preprocess,
            mode="clip_center_crop",
            input_resolution=224,
        )
        is preprocess
    )


def test_letterbox_rejects_unusable_preprocess() -> None:
    with pytest.raises(ValueError, match="derive CLIP"):
        select_inference_preprocess(
            FakeCompose([object(), object()]),
            mode="clip_letterbox",
            input_resolution=224,
        )
