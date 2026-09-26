import sys
from pathlib import Path
import pytest
import torch
from clip.model import VisionTransformer
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from l05_rectangular_support import rectangle_size, rectangular_position, rectangular_visual


@pytest.mark.parametrize('size', [(500, 375), (375, 500), (1200, 100), (1, 1)])
def test_geometry_bounded_and_transpose_equivariant(size):
    w, h = rectangle_size(*size)
    assert w % 32 == h % 32 == 0 and 32 <= min(w, h) <= max(w, h) <= 576
    assert (h, w) == rectangle_size(*size[::-1])


def test_positional_identity_and_cls_preservation():
    position = torch.randn(10, 8)
    assert rectangular_position(position, 3, 3) is position
    changed = rectangular_position(position, 2, 5)
    assert changed.shape == (11, 8)
    assert torch.equal(changed[0], position[0])


def test_square_native_forward_exact_and_rectangular_nonmutating():
    torch.manual_seed(42)
    model = VisionTransformer(64, 32, 64, 1, 4, 32).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        square = torch.randn(2, 3, 64, 64)
        assert torch.equal(model(square), rectangular_visual(model, square))
        rectangular = rectangular_visual(model, torch.randn(2, 3, 64, 96))
        assert rectangular.shape == (2, 32) and torch.isfinite(rectangular).all()
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
