"""Auditable inference-only image geometry variants for CLIP."""

from __future__ import annotations

from typing import Any

from PIL import Image


CLIP_PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)


class ResizeLongestSideAndPad:
    """Preserve the complete image while placing it on a square CLIP canvas."""

    def __init__(
        self,
        size: int,
        *,
        fill: tuple[int, int, int] | None = None,
    ) -> None:
        if int(size) <= 0:
            raise ValueError("letterbox size must be positive")
        self.size = int(size)
        self.fill = fill or tuple(round(255.0 * value) for value in CLIP_PIXEL_MEAN)
        if len(self.fill) != 3 or any(not 0 <= int(value) <= 255 for value in self.fill):
            raise ValueError("letterbox fill must be a valid RGB tuple")

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("letterbox preprocessing requires a PIL image")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("letterbox preprocessing requires a non-empty image")
        scale = float(self.size) / float(max(width, height))
        resized_width = max(1, min(self.size, round(width * scale)))
        resized_height = max(1, min(self.size, round(height * scale)))
        resized = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.size, self.size), color=self.fill)
        left = (self.size - resized_width) // 2
        top = (self.size - resized_height) // 2
        canvas.paste(resized.convert("RGB"), (left, top))
        return canvas


def select_inference_preprocess(
    preprocess: Any,
    *,
    mode: str,
    input_resolution: int,
) -> Any:
    """Return the native CLIP transform or a full-image letterbox variant."""
    if mode == "clip_center_crop":
        return preprocess
    if mode != "clip_letterbox":
        raise ValueError(
            "input resize mode must be clip_center_crop or clip_letterbox"
        )
    transforms = list(getattr(preprocess, "transforms", []))
    if len(transforms) < 3:
        raise ValueError("Cannot derive CLIP tensor conversion and normalization")
    try:
        from torchvision.transforms import Compose
    except ImportError as exc:
        raise ImportError("clip_letterbox requires torchvision") from exc
    return Compose(
        [
            ResizeLongestSideAndPad(int(input_resolution)),
            *transforms[-3:],
        ]
    )
