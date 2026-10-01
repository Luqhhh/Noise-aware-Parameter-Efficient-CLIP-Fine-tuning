"""Check that the intervention loses detail without altering small originals."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from probe_v1_resolution_degradation import DetailLossTransform
from aegis_clip.v1_pipeline import image_transform


@pytest.mark.parametrize('size', [(112, 250), (224, 448)])
def test_does_not_resample_small_or_boundary_images(size):
    pixels = np.random.default_rng(42).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    assert torch.equal(DetailLossTransform()(image), image_transform(512)(image))


def test_large_image_loses_detail_and_retains_native_model_geometry():
    pixels = np.random.default_rng(42).integers(0, 256, (600, 300, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    transform = DetailLossTransform()
    assert transform.reduce(image).size == (224, 448)
    reduced, original = transform(image), image_transform(512)(image)
    assert reduced.shape == original.shape == (3, 512, 512)
    assert torch.isfinite(reduced).all()
    assert not torch.equal(reduced, original)
