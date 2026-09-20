"""CPU geometry and pixel-contract tests for PRELIM75 v9."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from PIL import Image

from aegis_clip.source_recrop import (
    D0_MINIMUM_SUPPORTED,
    D0_SAMPLE_SIZE,
    INPUT_SIZE,
    d0_gate,
    map_box_to_source,
    native_tail,
    prepare_rgb,
    recrop_local,
    select_group_representatives,
    stable_path_list_sha256,
    trace_from_dimensions,
    unflip_box,
    validate_native_preprocess,
)
from aegis_clip.prelim75_source_recrop import CUDA_NUMERICS


@pytest.fixture
def preprocess():
    from clip.clip import _transform

    return _transform(INPUT_SIZE)


def _gradient(width: int, height: int) -> Image.Image:
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    array = np.empty((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = (3 * x + y) % 256
    array[:, :, 1] = (x + 5 * y) % 256
    array[:, :, 2] = (7 * x + 11 * y) % 256
    return Image.fromarray(array, mode="RGB")


def test_trace_landscape_matches_registered_example():
    trace = trace_from_dimensions(1200, 800)
    assert (trace.resized_width, trace.resized_height) == (336, 224)
    assert (trace.crop_left, trace.crop_top) == (56, 0)


def test_trace_portrait():
    trace = trace_from_dimensions(600, 1000)
    assert (trace.resized_width, trace.resized_height) == (224, 373)
    assert (trace.crop_left, trace.crop_top) == (0, 74)


def test_trace_square():
    trace = trace_from_dimensions(224, 224)
    assert trace.to_dict()["source_gain"] == 1.0
    assert (trace.crop_left, trace.crop_top) == (0, 0)


def test_trace_uses_round_to_even_for_center_offset():
    trace = trace_from_dimensions(225, 224)
    assert trace.resized_width == 225
    assert trace.crop_left == 0


@pytest.mark.parametrize("size", [(0, 2), (2, 0), (-1, 3)])
def test_trace_rejects_invalid_dimensions(size):
    with pytest.raises(ValueError, match="positive"):
        trace_from_dimensions(*size)


def test_native_preprocess_contract(preprocess):
    descriptor = validate_native_preprocess(preprocess)
    assert descriptor["resize"]["size"] == 224
    assert descriptor["center_crop"]["size"] == [224, 224]


def test_native_preprocess_rejects_changed_resize(preprocess):
    changed = copy.deepcopy(preprocess)
    changed.transforms[0].size = 256
    with pytest.raises(ValueError, match="Resize"):
        validate_native_preprocess(changed)


def test_prepare_rgb_global_tensor_is_native(preprocess):
    rgb = _gradient(1200, 800)
    prepared = prepare_rgb(rgb, preprocess)
    assert torch.equal(prepared.global_tensor, preprocess(rgb))


def test_prepare_rgb_records_actual_canvas_and_trace(preprocess):
    prepared = prepare_rgb(_gradient(801, 1201), preprocess)
    assert prepared.canvas.size == (224, 224)
    assert (prepared.trace.resized_width, prepared.trace.resized_height) == (224, 335)


def test_unflip_box():
    assert unflip_box((10, 20, 70, 80)) == (154.0, 20.0, 214.0, 80.0)


@pytest.mark.parametrize(
    "box",
    [(-1, 0, 10, 10), (0, 0, 225, 10), (10, 10, 10, 20), (0, 0, float("nan"), 2)],
)
def test_unflip_rejects_illegal_boxes(box):
    with pytest.raises(ValueError, match="Crop"):
        unflip_box(box)


def test_map_box_matches_plan_example():
    trace = trace_from_dimensions(1200, 800)
    mapped = map_box_to_source((56, 56, 168, 168), trace)
    np.testing.assert_allclose(mapped, (400, 200, 800, 600), atol=1e-12)


def test_flip_mapping_unflips_before_source_transform():
    trace = trace_from_dimensions(1200, 800)
    flipped = map_box_to_source((56, 56, 168, 168), trace, flipped=True)
    direct = map_box_to_source(unflip_box((56, 56, 168, 168)), trace)
    assert flipped == direct


@pytest.mark.parametrize("mode", ["R0", "R1"])
def test_recrop_outputs_float32_224(mode, preprocess):
    prepared = prepare_rgb(_gradient(1200, 800), preprocess)
    native = torch.zeros(3, 224, 224)
    result = recrop_local(
        prepared,
        (56, 56, 168, 168),
        mode=mode,
        flipped=False,
        preprocess_tail=native_tail(preprocess),
        native_local=native,
    )
    assert result.dtype == torch.float32
    assert result.shape == (3, 224, 224)
    assert torch.isfinite(result).all()


def test_r0_flip_is_post_roi_horizontal_flip(preprocess):
    prepared = prepare_rgb(_gradient(1200, 800), preprocess)
    native = torch.zeros(3, 224, 224)
    box = (20, 30, 100, 150)
    flipped = recrop_local(
        prepared,
        box,
        mode="R0",
        flipped=True,
        preprocess_tail=native_tail(preprocess),
        native_local=native,
    )
    direct_unflipped = recrop_local(
        prepared,
        unflip_box(box),
        mode="R0",
        flipped=False,
        preprocess_tail=native_tail(preprocess),
        native_local=native,
    )
    assert torch.equal(flipped, direct_unflipped.flip(2))


@pytest.mark.parametrize("mode", ["R0", "R1"])
def test_low_gain_fallback_is_same_tensor_object(mode, preprocess):
    prepared = prepare_rgb(_gradient(224, 224), preprocess)
    native = torch.randn(3, 224, 224)
    result = recrop_local(
        prepared,
        (10, 20, 100, 120),
        mode=mode,
        flipped=False,
        preprocess_tail=native_tail(preprocess),
        native_local=native,
    )
    assert result is native
    assert torch.equal(result, native)


def test_recrop_rejects_unknown_mode(preprocess):
    prepared = prepare_rgb(_gradient(1200, 800), preprocess)
    with pytest.raises(ValueError, match="R0 or R1"):
        recrop_local(
            prepared,
            (10, 10, 100, 100),
            mode="R2",
            flipped=False,
            preprocess_tail=native_tail(preprocess),
            native_local=torch.zeros(3, 224, 224),
        )


def test_recrop_rejects_wrong_native_shape(preprocess):
    prepared = prepare_rgb(_gradient(1200, 800), preprocess)
    with pytest.raises(ValueError, match="shape"):
        recrop_local(
            prepared,
            (10, 10, 100, 100),
            mode="R1",
            flipped=False,
            preprocess_tail=native_tail(preprocess),
            native_local=torch.zeros(3, 112, 112),
        )


def test_r0_and_r1_are_explicitly_not_pixel_equivalent(preprocess):
    prepared = prepare_rgb(_gradient(1200, 800), preprocess)
    native = torch.zeros(3, 224, 224)
    kwargs = dict(
        prepared=prepared,
        box=(56, 56, 168, 168),
        flipped=False,
        preprocess_tail=native_tail(preprocess),
        native_local=native,
    )
    r0 = recrop_local(mode="R0", **kwargs)
    r1 = recrop_local(mode="R1", **kwargs)
    assert not torch.equal(r0, r1)


def _selection_inputs():
    paths = ["train/a.jpg", "train/b.jpg", "train/c.jpg", "train/d.jpg"]
    groups = {"a.jpg": "g1", "b.jpg": "g1", "c.jpg": "g2", "d.jpg": "g3"}
    return paths, groups


def test_group_selection_uses_lexicographic_canonical_representative():
    paths, groups = _selection_inputs()
    selected = select_group_representatives(paths, groups, sample_size=3)
    assert "b.jpg" not in {row["canonical_path"] for row in selected}


def test_group_selection_is_order_independent():
    paths, groups = _selection_inputs()
    forward = select_group_representatives(paths, groups, sample_size=3)
    reverse = select_group_representatives(reversed(paths), groups, sample_size=3)
    assert forward == reverse


def test_group_selection_rejects_missing_group():
    paths, groups = _selection_inputs()
    del groups["c.jpg"]
    with pytest.raises(ValueError, match="missing"):
        select_group_representatives(paths, groups, sample_size=2)


def test_d0_gate_passes_at_fixed_minimum():
    gains = [1.5] * D0_MINIMUM_SUPPORTED + [1.0] * (
        D0_SAMPLE_SIZE - D0_MINIMUM_SUPPORTED
    )
    report = d0_gate(gains)
    assert report["passed"] is True
    assert report["supported_count"] == 205


def test_d0_gate_fails_one_below_fixed_minimum():
    gains = [1.5] * (D0_MINIMUM_SUPPORTED - 1) + [1.0] * (
        D0_SAMPLE_SIZE - D0_MINIMUM_SUPPORTED + 1
    )
    report = d0_gate(gains)
    assert report["passed"] is False
    assert report["status"] == "closed_insufficient_source_support"


def test_d0_gate_requires_exact_sample_count():
    with pytest.raises(ValueError, match="exactly"):
        d0_gate([2.0] * (D0_SAMPLE_SIZE - 1))


def test_stable_path_hash_is_order_sensitive_and_repeatable():
    first = stable_path_list_sha256(["a", "b"])
    assert first == stable_path_list_sha256(["a", "b"])
    assert first != stable_path_list_sha256(["b", "a"])


def test_v9_cuda_numerics_match_archived_generic_l1_inference():
    assert CUDA_NUMERICS == {
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "float32_matmul_precision": "highest",
    }
