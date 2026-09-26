"""D 轴：训练侧成像退化增广。

这一层此前**完全没有测试覆盖**（`train_augmentation` 在 tests/ 下零引用），
所以本文件同时承担两件事：给新增的退化层定契约，以及给既有的两个几何预设
补上「键缺失 ⇒ 行为逐位不变」的回归门。

回归门是重点：`data.train_degradation` 缺席时必须与今天逐位一致，
否则本仓库所有既有结果都会被这一次改动污染。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

from aegis_clip.degradation import (
    DEGRADATION_OPERATORS,
    build_train_degradation,
    validate_train_degradation,
)
from aegis_clip.trainer import _training_preprocess


class FakeCompose:
    """Stand-in for a CLIP preprocess: [..., ToTensor, Normalize]."""

    def __init__(self, transforms: list[object]) -> None:
        self.transforms = transforms


def clip_like_preprocess() -> Compose:
    return Compose([ToTensor(), Normalize((0.5,) * 3, (0.25,) * 3)])


def structured_image(size: int = 32) -> Image.Image:
    """A high-frequency test image: degradation must visibly perturb it."""
    ramp = np.arange(size, dtype=np.uint8)
    checker = ((ramp[:, None] // 2 + ramp[None, :] // 2) % 2) * 255
    rgb = np.stack([checker, checker.T, checker], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


# --------------------------------------------------------------------------
# 回归门：没有 train_degradation 时，行为必须与今天逐位一致
# --------------------------------------------------------------------------


def test_absent_spec_builds_nothing() -> None:
    assert build_train_degradation(None) is None
    assert build_train_degradation({}) is None


def test_clip_center_crop_is_returned_by_identity() -> None:
    preprocess = clip_like_preprocess()

    assert _training_preprocess(preprocess, {}) is preprocess
    assert _training_preprocess(preprocess, {"train_augmentation": "clip_center_crop"}) is preprocess


def test_degradation_spec_alone_does_not_invent_geometry() -> None:
    """退化是正交层：它不得悄悄改变几何预设的语义。"""
    preprocess = clip_like_preprocess()
    spec = {"jpeg": {"quality": [50, 50], "p": 0.0}}

    composed = _training_preprocess(
        preprocess, {"train_degradation": spec}
    )

    assert composed is not preprocess  # 被包了一层
    assert len(composed.transforms) == len(preprocess.transforms) + 1


# --------------------------------------------------------------------------
# 恒等性：p=0 或零强度必须一等一还原
# --------------------------------------------------------------------------


def test_probability_zero_is_a_noop() -> None:
    image = structured_image()
    chain = build_train_degradation(
        {"jpeg": {"quality": [10, 10], "p": 0.0},
         "gaussian_noise": {"sigma": [50.0, 50.0], "p": 0.0}}
    )
    torch.manual_seed(0)

    assert np.array_equal(pixels(chain(image)), pixels(image))


def test_zero_strength_is_a_noop() -> None:
    image = structured_image()
    chain = build_train_degradation(
        {"gaussian_blur": {"sigma": [0.0, 0.0], "p": 1.0},
         "gaussian_noise": {"sigma": [0.0, 0.0], "p": 1.0}}
    )
    torch.manual_seed(0)

    assert np.array_equal(pixels(chain(image)), pixels(image))


# --------------------------------------------------------------------------
# 每个算子生效时确实改变像素
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        {"jpeg": {"quality": [5, 5], "p": 1.0}},
        {"gaussian_blur": {"sigma": [2.0, 2.0], "p": 1.0}},
        {"gaussian_noise": {"sigma": [40.0, 40.0], "p": 1.0}},
    ],
)
def test_operator_changes_pixels_when_active(spec: dict) -> None:
    image = structured_image()
    chain = build_train_degradation(spec)
    torch.manual_seed(0)

    assert not np.array_equal(pixels(chain(image)), pixels(image))


def test_degradation_preserves_size_and_mode() -> None:
    image = structured_image()
    chain = build_train_degradation(
        {"jpeg": {"quality": [20, 20], "p": 1.0},
         "gaussian_blur": {"sigma": [1.0, 1.0], "p": 1.0},
         "gaussian_noise": {"sigma": [5.0, 5.0], "p": 1.0}}
    )
    torch.manual_seed(0)

    result = chain(image)

    assert result.size == image.size
    assert result.mode == "RGB"


# --------------------------------------------------------------------------
# 确定性：同种子同输出，异种子异输出
# --------------------------------------------------------------------------


def test_same_seed_reproduces_the_augmentation_stream() -> None:
    image = structured_image()
    spec = {"gaussian_noise": {"sigma": [10.0, 20.0], "p": 1.0}}
    chain = build_train_degradation(spec)

    torch.manual_seed(7)
    first = pixels(chain(image))
    torch.manual_seed(7)
    second = pixels(chain(image))

    assert np.array_equal(first, second)


def test_different_seed_changes_the_noise_realisation() -> None:
    image = structured_image()
    chain = build_train_degradation(
        {"gaussian_noise": {"sigma": [10.0, 20.0], "p": 1.0}}
    )

    torch.manual_seed(1)
    first = pixels(chain(image))
    torch.manual_seed(2)
    second = pixels(chain(image))

    assert not np.array_equal(first, second)


def test_operator_order_is_canonical_not_yaml_order() -> None:
    """键序不得影响行为 —— 否则换一行 YAML 就换了实验。"""
    image = structured_image()
    forward = build_train_degradation(
        {"jpeg": {"quality": [20, 20], "p": 1.0},
         "gaussian_noise": {"sigma": [10.0, 10.0], "p": 1.0}}
    )
    reversed_spec = build_train_degradation(
        {"gaussian_noise": {"sigma": [10.0, 10.0], "p": 1.0},
         "jpeg": {"quality": [20, 20], "p": 1.0}}
    )

    torch.manual_seed(3)
    first = pixels(forward(image))
    torch.manual_seed(3)
    second = pixels(reversed_spec(image))

    assert np.array_equal(first, second)


# --------------------------------------------------------------------------
# fail-closed 校验
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, match",
    [
        ({"sharpen": {"sigma": [1.0, 2.0], "p": 0.5}}, "unknown"),
        ({"jpeg": {"quality": [0, 50], "p": 0.5}}, "quality"),
        ({"jpeg": {"quality": [50, 101], "p": 0.5}}, "quality"),
        ({"gaussian_blur": {"sigma": [-1.0, 2.0], "p": 0.5}}, "sigma"),
        ({"gaussian_noise": {"sigma": [1.0, 2.0], "p": 1.5}}, "probability"),
        ({"gaussian_noise": {"sigma": [1.0, 2.0], "p": -0.1}}, "probability"),
        ({"gaussian_blur": {"sigma": [2.0, 1.0], "p": 0.5}}, "range"),
        ({"jpeg": {"quality": [30], "p": 0.5}}, "range"),
        ({"jpeg": {"quality": [30, 50], "p": 0.5, "strength": 1}}, "unknown"),
        ({"jpeg": 0.5}, "mapping"),
    ],
)
def test_malformed_specs_are_rejected(spec: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_train_degradation(spec)


def test_operator_names_are_the_declared_set() -> None:
    assert set(DEGRADATION_OPERATORS) == {"jpeg", "gaussian_blur", "gaussian_noise"}


def test_empty_spec_validates() -> None:
    validate_train_degradation(None)
    validate_train_degradation({})


# --------------------------------------------------------------------------
# 与既有几何预设组合，且不破坏张量与归一化
# --------------------------------------------------------------------------


def test_composes_with_weak_rrc_flip_and_still_emits_normalised_tensor() -> None:
    preprocess = clip_like_preprocess()
    train = _training_preprocess(
        preprocess,
        {"train_augmentation": "weak_rrc_flip",
         "train_degradation": {"jpeg": {"quality": [40, 60], "p": 1.0}}},
        input_resolution=32,
    )
    torch.manual_seed(0)

    out = train(structured_image())

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32


def test_degradation_runs_before_tensor_conversion_with_center_crop() -> None:
    """退化必须作用在 PIL 上、张量化之前，否则归一化会被做两次或漏做。"""
    preprocess = clip_like_preprocess()
    train = _training_preprocess(
        preprocess,
        {"train_degradation": {"jpeg": {"quality": [40, 60], "p": 1.0}}},
        input_resolution=32,
    )
    torch.manual_seed(0)

    out = train(structured_image())

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32)
    # 归一化只做一次：clean 图的张量均值应落在 Normalize 后的合理范围
    clean = preprocess(structured_image())
    assert float(out.std()) > 0.0
    assert abs(float(out.mean()) - float(clean.mean())) < 5.0


# --------------------------------------------------------------------------
# 可审计：repr 稳定，不含进程地址
# --------------------------------------------------------------------------


def test_repr_is_canonical_and_stable() -> None:
    chain = build_train_degradation(
        {"gaussian_noise": {"sigma": [1.0, 2.0], "p": 0.5},
         "jpeg": {"quality": [30, 50], "p": 0.25}}
    )

    assert repr(chain) == repr(build_train_degradation(
        {"jpeg": {"quality": [30, 50], "p": 0.25},
         "gaussian_noise": {"sigma": [1.0, 2.0], "p": 0.5}}
    ))
    assert "0x" not in repr(chain)
