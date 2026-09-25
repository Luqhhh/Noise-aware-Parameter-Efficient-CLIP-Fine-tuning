"""NEW01: recover the attention-guided local ROI from the ORIGINAL RGB image.

The existing :func:`aegis_clip.local_inference.attention_guided_crop` zooms into
the weighted centroid of the most attended CLIP patches, but it does so on the
``S x S`` *model-input* tensor that is already the output of a
``RandomResizedCrop``.  Downscaling to ``S`` before cropping throws away detail
that the original image still has.  This module reproduces the exact box that
``attention_guided_crop`` uses, exposes it in global-input coordinates, maps it
back onto the ORIGINAL image through the recorded RRC geometry, and extracts the
ROI from the original pixels at full resolution.

Pixel convention (chosen once and used consistently)
----------------------------------------------------
All "global input" coordinates ``(u, v)`` are continuous **pixel-centre**
coordinates on the ``S x S`` model input: the centre of pixel column ``p`` is
``u = p + 0.5``, so ``u`` lives in ``(0, S)``.  ``attention_guided_crop`` uses
``theta[:, 0, 2] = 2 * center_x / width - 1`` with ``affine_grid`` at
``align_corners=False``; solving that transform shows the crop is centred on
exactly this pixel-centre coordinate, so the box returned by
:func:`attention_crop_box` is directly comparable to ``center_x``/``center_y``
inside ``attention_guided_crop``.

The training global view is produced by torchvision
``RandomResizedCrop(S, scale, ratio, BICUBIC)`` -> ``RandomHorizontalFlip(0.5)``
-> CLIP normalize.  ``RandomResizedCrop`` samples ``(i, j, h, w)`` =
``(top, left, crop_h, crop_w)`` on the ORIGINAL image, crops, then resizes to
``S x S``; the flip is applied to the ``S x S`` result.  The geometry record is
therefore ``(H0, W0, i, j, h, w, flip, S)``.

For an **un-flipped** global input the coordinate maps to the original as::

    x = j + u * w / S           y = i + v * h / S

in continuous original-image pixel-centre coordinates (original pixel column
``q`` has centre ``q + 0.5``).  ``RandomHorizontalFlip`` maps pixel ``p`` to
``S - 1 - p``; in pixel-centre coordinates that is ``u -> S - u``, so for a
**flipped** global input we first un-flip::

    u' = S - u                  x = j + u' * w / S

The half-open crop interval ``[center - half, center + half]`` is mapped through
the same transform, then clamped to ``[0, W0] x [0, H0]``.  Extracting the ROI
crops the original at that box, resizes once to ``out_size`` with bicubic, and
finally applies the sample's horizontal flip when ``flip = 1`` so the ROI
matches the appearance of the attention crop taken from the (possibly flipped)
model input.

Assumptions
-----------
* ``attention_guided_crop`` requires a square model input; so does the box math
  here.
* ``originals`` may be ``uint8``/integer (scaled by ``1/255``) or floating point
  (assumed to already lie in ``[0, 1]``).  The sampled ROI is clamped to
  ``[0, 1]`` before ``normalize`` runs.
* ``normalize`` is called per sample on a float ``[3, out_size, out_size]``
  tensor, matching torchvision ``Normalize`` and any plain callable.
* Nothing here reads the test split, a model, or a GPU.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "attention_crop_box",
    "unflip_box_x",
    "global_box_to_original",
    "original_roi_batch",
    "geometry_vector",
    "inverse_round_trip_check",
]


def _as_vector(value: Any, *, name: str, count: int) -> torch.Tensor:
    tensor = torch.as_tensor(value).float().flatten()
    if tensor.numel() != count:
        raise ValueError(
            f"{name} must have {count} element(s), got {tensor.numel()}"
        )
    return tensor


def _box_vectors(
    box: Mapping[str, torch.Tensor], count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    missing = [key for key in ("center_x", "center_y", "half") if key not in box]
    if missing:
        raise ValueError(f"box is missing key(s): {missing}")
    center_x = _as_vector(box["center_x"], name="center_x", count=count)
    center_y = _as_vector(box["center_y"], name="center_y", count=count)
    half = _as_vector(box["half"], name="half", count=count)
    return center_x, center_y, half


def attention_crop_box(
    patch_attention: torch.Tensor,
    *,
    crop_size: int,
    top_patches: int,
    height: int,
    width: int,
) -> dict[str, torch.Tensor]:
    """Re-implement the box math of ``attention_guided_crop`` exactly.

    Same ``topk`` selection, same clamped non-negative weights, same weighted
    patch-centre centroid, and the same clamp to ``[half, size - half]``.  The
    result is returned instead of being turned into an affine grid:

    ``{"center_x", "center_y", "half"}``, each ``[N]`` float32, in GLOBAL-INPUT
    pixel-centre coordinates (see the module docstring).
    """
    if patch_attention.ndim != 3:
        raise ValueError("Patch attention must be rank-3 [N, grid_h, grid_w]")
    height = int(height)
    width = int(width)
    if height != width:
        raise ValueError("Attention crop currently requires square model inputs")
    if not 0 < int(crop_size) < height:
        raise ValueError("crop_size must be smaller than the model input")
    patch_count = patch_attention.shape[1] * patch_attention.shape[2]
    if not 1 <= int(top_patches) <= patch_count:
        raise ValueError("top_patches is out of range")

    flat = patch_attention.float().flatten(1)
    weights, indices = torch.topk(flat, k=int(top_patches), dim=1)
    weights = weights.clamp_min(0.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    grid_width = patch_attention.shape[2]
    rows = torch.div(indices, grid_width, rounding_mode="floor").float()
    columns = (indices % grid_width).float()
    patch_height = height / patch_attention.shape[1]
    patch_width = width / patch_attention.shape[2]
    center_y = ((rows + 0.5) * patch_height * weights).sum(dim=1)
    center_x = ((columns + 0.5) * patch_width * weights).sum(dim=1)
    half = float(crop_size) / 2.0
    center_y = center_y.clamp(half, height - half)
    center_x = center_x.clamp(half, width - half)
    return {
        "center_x": center_x.float(),
        "center_y": center_y.float(),
        "half": torch.full_like(center_x.float(), half),
    }


def unflip_box_x(
    box: Mapping[str, torch.Tensor], flip: torch.Tensor, size: int
) -> dict[str, torch.Tensor]:
    """Return the box in the UN-flipped global-input frame.

    ``flip`` is a per-sample flag ``[N]`` (bool or 0/1).  For flipped samples the
    horizontal centre is swapped with ``size - center_x`` (pixel-centre
    convention); ``center_y`` and ``half`` are unchanged.

    This is a standalone helper: :func:`global_box_to_original` already performs
    the same un-flip internally from ``geometry[:, 6]``, so do **not** feed the
    result of this function into it (that would un-flip twice).
    """
    flip = torch.as_tensor(flip).bool().flatten()
    center_x = _as_vector(box["center_x"], name="center_x", count=flip.numel())
    center_y = _as_vector(box["center_y"], name="center_y", count=flip.numel())
    half = _as_vector(box["half"], name="half", count=flip.numel())
    flip = flip.to(center_x.device)
    size = float(size)
    if size <= 0.0:
        raise ValueError("size must be positive")
    unflipped_x = torch.where(flip, size - center_x, center_x)
    return {
        "center_x": unflipped_x.float(),
        "center_y": center_y.clone(),
        "half": half.clone(),
    }


def _as_geometry(geometry: torch.Tensor) -> torch.Tensor:
    geometry = torch.as_tensor(geometry).float()
    if geometry.ndim != 2 or geometry.shape[1] != 8:
        raise ValueError(
            "geometry must be [N, 8] = (H0, W0, i, j, h, w, flip, S)"
        )
    return geometry


def global_box_to_original(
    box: Mapping[str, torch.Tensor], geometry: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Map a global-input box to ORIGINAL image pixel coordinates.

    ``geometry`` is ``[N, 8] = (H0, W0, i, j, h, w, flip, S)``.  The box is
    un-flipped (when ``flip`` is set) and pushed through
    ``x = j + u' * w / S`` / ``y = i + v * h / S``, then clamped to the image
    extent.  Returns ``{"x0", "y0", "x1", "y1"}`` ``[N]`` float32 with
    ``x1 > x0`` and ``y1 > y0``; a degenerate (zero-area) box raises
    ``ValueError``.
    """
    geometry = _as_geometry(geometry)
    count = int(geometry.shape[0])
    center_x, center_y, half = _box_vectors(box, count)
    device = geometry.device
    center_x = center_x.to(device)
    center_y = center_y.to(device)
    half = half.to(device)

    height0 = geometry[:, 0]
    width0 = geometry[:, 1]
    top = geometry[:, 2]
    left = geometry[:, 3]
    crop_h = geometry[:, 4]
    crop_w = geometry[:, 5]
    flip = geometry[:, 6] > 0.5
    size = geometry[:, 7]

    if bool((height0 <= 0.0).any()) or bool((width0 <= 0.0).any()):
        raise ValueError("geometry contains a non-positive original size")
    if bool((crop_h <= 0.0).any()) or bool((crop_w <= 0.0).any()):
        raise ValueError("geometry contains a non-positive crop size")
    if bool((size <= 0.0).any()):
        raise ValueError("geometry contains a non-positive model input size")
    if bool((half <= 0.0).any()):
        raise ValueError("box half-width must be positive")

    unflipped_x = torch.where(flip, size - center_x, center_x)
    x0 = left + (unflipped_x - half) * crop_w / size
    x1 = left + (unflipped_x + half) * crop_w / size
    y0 = top + (center_y - half) * crop_h / size
    y1 = top + (center_y + half) * crop_h / size

    zero = torch.zeros_like(x0)
    x0 = torch.maximum(torch.minimum(x0, width0), zero)
    x1 = torch.maximum(torch.minimum(x1, width0), zero)
    y0 = torch.maximum(torch.minimum(y0, height0), zero)
    y1 = torch.maximum(torch.minimum(y1, height0), zero)

    width_span = x1 - x0
    height_span = y1 - y0
    if bool((width_span <= 0.0).any()) or bool((height_span <= 0.0).any()):
        raise ValueError(
            "degenerate box: the mapped crop has zero area inside the image"
        )
    return {
        "x0": x0.float(),
        "y0": y0.float(),
        "x1": x1.float(),
        "y1": y1.float(),
    }


def geometry_vector(
    original_size: Sequence[float],
    crop: Sequence[float],
    flip: bool | float,
    size: float,
) -> torch.Tensor:
    """Convenience builder for one geometry row ``[8]`` float32.

    ``original_size = (H0, W0)``; ``crop = (i, j, h, w)`` = (top, left, h, w);
    ``flip`` is a bool/0-1 flag; ``size = S``.
    """
    height0, width0 = (float(original_size[0]), float(original_size[1]))
    top, left, crop_h, crop_w = (
        float(crop[0]),
        float(crop[1]),
        float(crop[2]),
        float(crop[3]),
    )
    return torch.tensor(
        [height0, width0, top, left, crop_h, crop_w, 1.0 if flip else 0.0, float(size)],
        dtype=torch.float32,
    )


def _to_unit_float(image: torch.Tensor) -> torch.Tensor:
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError("each original image must be a [3, H0, W0] tensor")
    if image.dtype.is_floating_point:
        return image.float()
    return image.float() / 255.0


def original_roi_batch(
    originals: list[torch.Tensor] | torch.Tensor,
    original_sizes: torch.Tensor,
    geometry: torch.Tensor,
    box: Mapping[str, torch.Tensor],
    *,
    out_size: int,
    normalize: Callable | None = None,
) -> torch.Tensor:
    """Extract the mapped ROI from each ORIGINAL image, in one resize.

    ``originals`` is either a list of ``[3, H0, W0]`` tensors (variable size) or
    a single padded ``[N, 3, Hmax, Wmax]`` tensor paired with
    ``original_sizes`` ``[N, 2] = (H0, W0)``.  ``geometry`` is ``[N, 8]`` and
    ``box`` is in global-input coordinates.

    For each sample the ORIGINAL is cropped at the mapped box, resized once to
    ``out_size x out_size`` with bicubic sampling, horizontally flipped when
    ``geometry[:, 6] > 0.5``, clamped to ``[0, 1]`` and finally passed through
    ``normalize`` (per sample, on a float ``[3, out_size, out_size]`` tensor).
    Returns ``[N, 3, out_size, out_size]`` float32; with ``normalize=None`` the
    values are the unnormalised ``[0, 1]`` floats.
    """
    out_size = int(out_size)
    if out_size <= 0:
        raise ValueError("out_size must be positive")
    geometry = _as_geometry(geometry)
    count = int(geometry.shape[0])
    if count == 0:
        raise ValueError("geometry batch is empty")

    if isinstance(originals, (list, tuple)):
        if len(originals) != count:
            raise ValueError(
                f"got {len(originals)} original image(s) for {count} geometry row(s)"
            )
        image_list = [_to_unit_float(image) for image in originals]
        padded = None
        device = image_list[0].device
    else:
        if originals.dim() != 4 or originals.shape[1] != 3:
            raise ValueError("padded originals must be [N, 3, Hmax, Wmax]")
        if int(originals.shape[0]) != count:
            raise ValueError(
                f"padded originals have {originals.shape[0]} rows for {count} geometry row(s)"
            )
        image_list = None
        padded = originals
        device = padded.device

    geometry = geometry.to(device)
    original_box = global_box_to_original(box, geometry)
    x0 = original_box["x0"]
    y0 = original_box["y0"]
    x1 = original_box["x1"]
    y1 = original_box["y1"]

    height0 = geometry[:, 0]
    width0 = geometry[:, 1]
    flip = geometry[:, 6] > 0.5

    sizes = None
    if original_sizes is not None:
        sizes = torch.as_tensor(original_sizes).float().to(device)
        if sizes.ndim != 2 or sizes.shape[1] != 2 or int(sizes.shape[0]) != count:
            raise ValueError("original_sizes must be [N, 2] = (H0, W0)")
        if not torch.allclose(sizes[:, 0], height0, atol=1.0) or not torch.allclose(
            sizes[:, 1], width0, atol=1.0
        ):
            raise ValueError("original_sizes disagree with geometry H0/W0")

    fraction = (
        torch.arange(out_size, dtype=torch.float32, device=device) + 0.5
    ) / float(out_size)
    xs = x0[:, None] + fraction[None, :] * (x1 - x0)[:, None]
    ys = y0[:, None] + fraction[None, :] * (y1 - y0)[:, None]
    # grid_sample (align_corners=False) uses normalised pixel-index coordinates
    # whose relation to pixel-centre coordinates x is x_norm = 2 * x / dim - 1.
    grid_x = (2.0 * xs / width0[:, None] - 1.0)[:, None, :].expand(count, out_size, out_size)
    grid_y = (2.0 * ys / height0[:, None] - 1.0)[:, :, None].expand(count, out_size, out_size)
    grid = torch.stack([grid_x, grid_y], dim=-1)

    outputs: list[torch.Tensor] = []
    for index in range(count):
        if image_list is not None:
            image = image_list[index]
            expected_h = int(round(float(height0[index])))
            expected_w = int(round(float(width0[index])))
            if image.shape[-2] != expected_h or image.shape[-1] != expected_w:
                raise ValueError(
                    f"original {index} has shape {tuple(image.shape)} but geometry "
                    f"says ({expected_h}, {expected_w})"
                )
        else:
            height = int(round(float(height0[index])))
            width = int(round(float(width0[index])))
            if height > padded.shape[-2] or width > padded.shape[-1]:
                raise ValueError(
                    f"geometry ({height}, {width}) exceeds padded originals "
                    f"{tuple(padded.shape[-2:])}"
                )
            image = _to_unit_float(padded[index][:, :height, :width])
        sample = image.unsqueeze(0)
        roi = F.grid_sample(
            sample,
            grid[index].unsqueeze(0),
            mode="bicubic",
            padding_mode="border",
            align_corners=False,
        )
        roi = roi.clamp(0.0, 1.0)
        if bool(flip[index]):
            roi = torch.flip(roi, dims=[-1])
        roi = roi.squeeze(0).contiguous()
        if normalize is not None:
            roi = normalize(roi)
        outputs.append(roi.float())
    return torch.stack(outputs, dim=0)


def inverse_round_trip_check(
    box: Mapping[str, torch.Tensor],
    geometry: torch.Tensor,
    *,
    atol: float = 1.0e-3,
) -> dict[str, Any]:
    """Map a box to the original frame and back; report the round-trip error.

    Only meaningful for boxes strictly inside the image (clamping is lossy at the
    border).  Returns ``{"ok", "max_abs_error"}``.
    """
    geometry = _as_geometry(geometry)
    count = int(geometry.shape[0])
    center_x, center_y, half = _box_vectors(box, count)
    original = global_box_to_original(box, geometry)
    top = geometry[:, 2]
    left = geometry[:, 3]
    crop_h = geometry[:, 4]
    crop_w = geometry[:, 5]
    flip = geometry[:, 6] > 0.5
    size = geometry[:, 7]

    center_x_original = (original["x0"] + original["x1"]) / 2.0
    center_y_original = (original["y0"] + original["y1"]) / 2.0
    half_x_original = (original["x1"] - original["x0"]) / 2.0
    half_y_original = (original["y1"] - original["y0"]) / 2.0

    unflipped_x = (center_x_original - left) * size / crop_w
    center_x_back = torch.where(flip, size - unflipped_x, unflipped_x)
    center_y_back = (center_y_original - top) * size / crop_h
    half_x_back = half_x_original * size / crop_w
    half_y_back = half_y_original * size / crop_h

    error = torch.stack(
        [
            (center_x_back - center_x).abs(),
            (center_y_back - center_y).abs(),
            (half_x_back - half).abs(),
            (half_y_back - half).abs(),
        ]
    ).max()
    return {"ok": bool(float(error) <= float(atol)), "max_abs_error": float(error)}
