"""Deterministic source-image recrops for PRELIM75 v9.

The helpers in this module are deliberately model-free.  They preserve the
native CLIP Resize(224)+CenterCrop(224) geometry, map already-selected native
attention boxes back to the decoded RGB image, and apply only the native
RGB/ToTensor/Normalize tail after a Pillow bilinear ROI resize.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode


PLAN_ID = "PRELIM75_V9_20260920"
INPUT_SIZE = 224
SOURCE_GAIN_THRESHOLD = 1.5
D0_SAMPLE_SIZE = 2048
D0_MINIMUM_SUPPORTED = 205


@dataclass(frozen=True)
class ResizeCenterTrace:
    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    crop_left: int
    crop_top: int
    crop_size: int = INPUT_SIZE

    @property
    def source_gain(self) -> float:
        return min(
            self.source_width / self.resized_width,
            self.source_height / self.resized_height,
        )

    def to_dict(self) -> dict[str, int | float]:
        return {**asdict(self), "source_gain": self.source_gain}


@dataclass(frozen=True)
class PreparedRGB:
    rgb: Image.Image
    canvas: Image.Image
    global_tensor: torch.Tensor
    trace: ResizeCenterTrace


def trace_from_dimensions(
    width: int,
    height: int,
    *,
    crop_size: int = INPUT_SIZE,
) -> ResizeCenterTrace:
    """Mirror torchvision's integer Resize(short-side)+CenterCrop geometry."""
    width, height, crop_size = int(width), int(height), int(crop_size)
    if width <= 0 or height <= 0 or crop_size <= 0:
        raise ValueError("Image and crop dimensions must be positive")
    if width <= height:
        resized_width = crop_size
        resized_height = int(crop_size * height / width)
    else:
        resized_height = crop_size
        resized_width = int(crop_size * width / height)
    crop_left = int(round((resized_width - crop_size) / 2.0))
    crop_top = int(round((resized_height - crop_size) / 2.0))
    if crop_left < 0 or crop_top < 0:
        raise ValueError("Native resize cannot support the requested center crop")
    return ResizeCenterTrace(
        source_width=width,
        source_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        crop_left=crop_left,
        crop_top=crop_top,
        crop_size=crop_size,
    )


def validate_native_preprocess(preprocess: object) -> dict[str, object]:
    """Fail closed unless the deployed transform is native OpenAI CLIP 224."""
    transforms = list(getattr(preprocess, "transforms", []))
    if len(transforms) != 5:
        raise ValueError("Expected the five-stage OpenAI CLIP preprocess")
    resize, center_crop, rgb, to_tensor, normalize = transforms
    if not isinstance(resize, Resize) or resize.size != INPUT_SIZE:
        raise ValueError("Native preprocess must start with Resize(224)")
    if resize.interpolation != InterpolationMode.BICUBIC:
        raise ValueError("Native Resize interpolation must be bicubic")
    center_size = tuple(center_crop.size) if isinstance(center_crop, CenterCrop) else ()
    if center_size != (INPUT_SIZE, INPUT_SIZE):
        raise ValueError("Native preprocess must use CenterCrop(224)")
    if not callable(rgb) or not isinstance(to_tensor, ToTensor):
        raise ValueError("Native preprocess RGB/ToTensor tail changed")
    if not isinstance(normalize, Normalize):
        raise ValueError("Native preprocess normalization changed")
    mean = tuple(float(value) for value in normalize.mean)
    std = tuple(float(value) for value in normalize.std)
    expected_mean = (0.48145466, 0.4578275, 0.40821073)
    expected_std = (0.26862954, 0.26130258, 0.27577711)
    if mean != expected_mean or std != expected_std:
        raise ValueError("Native CLIP normalization changed")
    return {
        "resize": {
            "size": INPUT_SIZE,
            "interpolation": "bicubic",
            "antialias": bool(getattr(resize, "antialias", True)),
            "max_size": getattr(resize, "max_size", None),
        },
        "center_crop": {
            "size": [INPUT_SIZE, INPUT_SIZE],
            "offset_rule": "int(round((resized_length-224)/2.0))",
        },
        "decode": "Pillow Image.open then convert RGB",
        "tail": "RGB conversion, ToTensor, Normalize",
        "normalize_mean": list(mean),
        "normalize_std": list(std),
    }


def native_tail(preprocess: object) -> Compose:
    validate_native_preprocess(preprocess)
    return Compose(list(preprocess.transforms)[2:])


def prepare_rgb(rgb: Image.Image, preprocess: object) -> PreparedRGB:
    """Retain decoded RGB, native PIL canvas, exact global tensor, and trace."""
    if not isinstance(rgb, Image.Image):
        raise TypeError("prepare_rgb requires a Pillow image")
    if rgb.mode != "RGB":
        raise ValueError("prepare_rgb requires an already decoded RGB image")
    validate_native_preprocess(preprocess)
    transforms = list(preprocess.transforms)
    trace = trace_from_dimensions(*rgb.size)
    resized = transforms[0](rgb)
    if resized.size != (trace.resized_width, trace.resized_height):
        raise RuntimeError(
            "Installed torchvision Resize geometry differs from the registered trace"
        )
    canvas = transforms[1](resized)
    if canvas.size != (INPUT_SIZE, INPUT_SIZE):
        raise RuntimeError("Native CenterCrop did not produce a 224 canvas")
    global_tensor = preprocess(rgb)
    reconstructed = native_tail(preprocess)(canvas)
    if not torch.equal(global_tensor, reconstructed):
        raise RuntimeError("Wrapped global tensor differs from native preprocess")
    return PreparedRGB(
        rgb=rgb,
        canvas=canvas,
        global_tensor=global_tensor,
        trace=trace,
    )


def _checked_box(box: Sequence[float], *, size: int = INPUT_SIZE) -> tuple[float, ...]:
    if len(box) != 4:
        raise ValueError("A crop box must have four coordinates")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Crop coordinates must be finite")
    x0, y0, x1, y1 = values
    if x0 < 0 or y0 < 0 or x1 > size or y1 > size or x1 <= x0 or y1 <= y0:
        raise ValueError("Crop box is outside the native canvas or has no area")
    return values


def unflip_box(
    box: Sequence[float], *, canvas_size: int = INPUT_SIZE
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = _checked_box(box, size=canvas_size)
    return (float(canvas_size) - x1, y0, float(canvas_size) - x0, y1)


def map_box_to_source(
    box: Sequence[float],
    trace: ResizeCenterTrace,
    *,
    flipped: bool = False,
) -> tuple[float, float, float, float]:
    """Map a native-view box to the unflipped decoded source using float bounds."""
    native = unflip_box(box, canvas_size=trace.crop_size) if flipped else _checked_box(
        box, size=trace.crop_size
    )
    x0, y0, x1, y1 = native
    mapped = (
        (trace.crop_left + x0) * trace.source_width / trace.resized_width,
        (trace.crop_top + y0) * trace.source_height / trace.resized_height,
        (trace.crop_left + x1) * trace.source_width / trace.resized_width,
        (trace.crop_top + y1) * trace.source_height / trace.resized_height,
    )
    u0, v0, u1, v1 = mapped
    tolerance = 1.0e-9
    if (
        u0 < -tolerance
        or v0 < -tolerance
        or u1 > trace.source_width + tolerance
        or v1 > trace.source_height + tolerance
        or u1 <= u0
        or v1 <= v0
    ):
        raise ValueError("Mapped source ROI is illegal")
    return mapped


def recrop_local(
    prepared: PreparedRGB,
    box: Sequence[float],
    *,
    mode: str,
    flipped: bool,
    preprocess_tail: Compose,
    native_local: torch.Tensor,
    source_gain_threshold: float = SOURCE_GAIN_THRESHOLD,
) -> torch.Tensor:
    """Return R0/R1 local tensor or the exact native tensor below the gain gate."""
    if mode not in {"R0", "R1"}:
        raise ValueError("recrop mode must be R0 or R1")
    if source_gain_threshold <= 0.0 or not math.isfinite(source_gain_threshold):
        raise ValueError("source gain threshold must be finite and positive")
    if native_local.shape != (3, INPUT_SIZE, INPUT_SIZE):
        raise ValueError("native local tensor must have shape [3,224,224]")
    if prepared.trace.source_gain < source_gain_threshold:
        return native_local
    native_box = unflip_box(box) if flipped else _checked_box(box)
    if mode == "R0":
        source = prepared.canvas
        source_box = native_box
    else:
        source = prepared.rgb
        source_box = map_box_to_source(box, prepared.trace, flipped=flipped)
    local = source.resize(
        (INPUT_SIZE, INPUT_SIZE),
        resample=Image.Resampling.BILINEAR,
        box=source_box,
        reducing_gap=None,
    )
    if flipped:
        local = local.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    tensor = preprocess_tail(local)
    if tensor.dtype != torch.float32 or tensor.shape != native_local.shape:
        raise RuntimeError("Recropped local tensor must be float32 [3,224,224]")
    if not torch.isfinite(tensor).all():
        raise RuntimeError("Recropped local tensor is non-finite")
    return tensor


def select_group_representatives(
    paths: Iterable[str],
    groups: Mapping[str, str],
    *,
    sample_size: int = D0_SAMPLE_SIZE,
    seed: int = 42,
    plan_id: str = PLAN_ID,
) -> list[dict[str, str | int]]:
    """Select canonical content-group representatives without labels or scores."""
    if sample_size <= 0:
        raise ValueError("D0 sample size must be positive")
    representatives: dict[str, str] = {}
    for raw_path in paths:
        canonical = str(raw_path).replace("\\", "/").strip().lstrip("./")
        for prefix in ("train_dedup/", "train/"):
            if canonical.startswith(prefix):
                canonical = canonical[len(prefix):]
                break
        if canonical not in groups:
            raise ValueError(f"Training path is missing from content groups: {canonical}")
        group = str(groups[canonical])
        previous = representatives.get(group)
        if previous is None or canonical < previous:
            representatives[group] = canonical
    if len(representatives) < sample_size:
        raise ValueError("Not enough unique content groups for D0")
    ranked = []
    for group, path in representatives.items():
        selection_hash = hashlib.sha256(
            f"{plan_id}/{int(seed)}/{path}".encode("utf-8")
        ).hexdigest()
        ranked.append((selection_hash, path, group))
    ranked.sort()
    return [
        {
            "rank": rank,
            "selection_sha256": selection_hash,
            "canonical_path": path,
            "content_group": group,
        }
        for rank, (selection_hash, path, group) in enumerate(
            ranked[:sample_size], start=1
        )
    ]


def d0_gate(
    source_gains: Sequence[float],
    *,
    threshold: float = SOURCE_GAIN_THRESHOLD,
    minimum_supported: int = D0_MINIMUM_SUPPORTED,
    expected_samples: int = D0_SAMPLE_SIZE,
) -> dict[str, int | float | bool | str]:
    if len(source_gains) != expected_samples:
        raise ValueError(
            f"D0 requires exactly {expected_samples} gains, got {len(source_gains)}"
        )
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in source_gains):
        raise ValueError("D0 source gains must be finite and positive")
    supported = sum(float(value) >= threshold for value in source_gains)
    passed = supported >= minimum_supported
    return {
        "passed": passed,
        "status": "passed" if passed else "closed_insufficient_source_support",
        "sample_count": len(source_gains),
        "source_gain_threshold": float(threshold),
        "supported_count": int(supported),
        "supported_fraction": float(supported / len(source_gains)),
        "minimum_supported": int(minimum_supported),
    }


def stable_path_list_sha256(paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(Path(path).as_posix()).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
