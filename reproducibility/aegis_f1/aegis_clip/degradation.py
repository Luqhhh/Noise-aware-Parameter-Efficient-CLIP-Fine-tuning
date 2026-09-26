"""训练侧成像退化增广（D 轴）。

为什么单独一层：增量几何/光度增广（`weak_rrc_flip`）与**成像退化**是两类不同的
扰动。前者保持图像内容、只挪采样位置；后者破坏**来源特有的成像统计** —— JPEG
量化表、压缩块效应、传感器噪声、镜头模糊。本项目的 11.90pp 本地/平台落差在
机制上被怀疑与来源成像统计有关，退化增广就是直接针对它的那条轴，而既有的两个
几何预设与队友 V4 的 A 组（mixup / cutmix / RandAugment，全是几何+光度）都不覆盖它。

设计约束（规则 §8.2）：增广参数**必须**进配置，不得硬编码 —— 所以强度是以区间形式
写在 YAML 里、每个算子带独立施加概率，而不是把某个质量值写死在这里。

两个刻意的选择：

1. **正交层，不是第三个几何预设。** `data.train_augmentation` 的语义一个字不动；
   `data.train_degradation` 缺席时本模块返回 ``None``，训练管线与改动前逐位一致。
   这是保护本仓库全部既有结果的回归性质。
2. **算子顺序固定为 `DEGRADATION_OPERATORS`，与 YAML 键序无关。** 否则调换两行
   YAML 就换了一次实验，而实验记录里看不出来。

刻意**未**纳入 v1 的算子：降采样再放回。它是唯一与推理侧探针已证伪的那条轴
（干净几何只值 0.17–0.31pp，见 `docs/rematch750_inference_side_probe_20260922.md`）
机制重叠的算子；先排除，免得 D 轴一开始就和一条已关闭的轴纠缠不清。

随机性走 torch 全局 RNG，因此受 `set_seed(..., deterministic=True)` 掌控，与
`weak_rrc_flip` 里 torchvision 变换的行为一致。附带好处：本层的参数写在 YAML 内，
自动被既有的 `training_config_sha256` checkpoint 绑定覆盖，provenance 无需新机制。
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter


#: 施加顺序即此元组顺序，**不随 YAML 键序变化**。
DEGRADATION_OPERATORS: tuple[str, ...] = (
    "jpeg",
    "gaussian_blur",
    "gaussian_noise",
)

#: 每个算子的强度参数名，以及其合法下界（None 表示无下界）。
_OPERATOR_PARAMETER: dict[str, tuple[str, float | None]] = {
    "jpeg": ("quality", 1.0),
    "gaussian_blur": ("sigma", 0.0),
    "gaussian_noise": ("sigma", 0.0),
}

_OPERATOR_UPPER_BOUND: dict[str, float | None] = {
    "jpeg": 100.0,
    "gaussian_blur": None,
    "gaussian_noise": None,
}


def _read_range(value: Any, name: str, parameter: str) -> tuple[float, float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   for v in value)):
        raise ValueError(
            f"train_degradation.{name}.{parameter} must be a [low, high] range"
        )
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(
            f"train_degradation.{name}.{parameter} range must satisfy low <= high"
        )
    return low, high


def validate_train_degradation(spec: Any) -> None:
    """Fail closed on anything malformed. ``None`` and ``{}`` mean "disabled"."""
    if spec is None:
        return
    if not isinstance(spec, Mapping):
        raise ValueError("data.train_degradation must be a mapping")
    unknown = set(spec) - set(DEGRADATION_OPERATORS)
    if unknown:
        raise ValueError(
            f"unknown train_degradation operators: {sorted(unknown)}; "
            f"supported: {list(DEGRADATION_OPERATORS)}"
        )
    for name, options in spec.items():
        parameter, lower = _OPERATOR_PARAMETER[name]
        if not isinstance(options, Mapping):
            raise ValueError(f"train_degradation.{name} must be a mapping")
        extra = set(options) - {parameter, "p"}
        if extra:
            raise ValueError(
                f"unknown train_degradation.{name} keys: {sorted(extra)}"
            )
        if parameter not in options:
            raise ValueError(
                f"train_degradation.{name} requires {parameter}"
            )
        low, high = _read_range(options[parameter], name, parameter)
        if lower is not None and low < lower:
            raise ValueError(
                f"train_degradation.{name}.{parameter} must be >= {lower}"
            )
        upper = _OPERATOR_UPPER_BOUND[name]
        if upper is not None and high > upper:
            raise ValueError(
                f"train_degradation.{name}.{parameter} must be <= {upper}"
            )
        probability = options.get("p", 1.0)
        if (isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not 0.0 <= float(probability) <= 1.0):
            raise ValueError(
                f"train_degradation.{name}.p is an apply probability and must "
                f"be within [0, 1]"
            )


def _uniform(low: float, high: float) -> float:
    """Draw from torch's global RNG so ``set_seed`` governs the stream."""
    return float(torch.rand(1).item()) * (high - low) + low


def _apply_jpeg(image: Image.Image, quality: float) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer, format="JPEG", quality=int(round(quality))
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        decoded.load()
        return decoded.convert("RGB")


def _apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0.0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def _apply_gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0.0:
        return image
    array = torch.as_tensor(
        np.asarray(image.convert("RGB"), dtype=np.float32)
    )
    noisy = array + torch.randn(array.shape) * float(sigma)
    return Image.fromarray(
        noisy.clamp(0.0, 255.0).to(torch.uint8).numpy(), mode="RGB"
    )


_APPLY = {
    "jpeg": _apply_jpeg,
    "gaussian_blur": _apply_gaussian_blur,
    "gaussian_noise": _apply_gaussian_noise,
}


class DegradationChain:
    """Apply the configured operators to a PIL image, in canonical order."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        validate_train_degradation(spec)
        self._spec = {
            name: dict(spec[name])
            for name in DEGRADATION_OPERATORS
            if name in spec
        }

    def __call__(self, image: Image.Image) -> Image.Image:
        for name in DEGRADATION_OPERATORS:
            options = self._spec.get(name)
            if options is None:
                continue
            if _uniform(0.0, 1.0) >= float(options.get("p", 1.0)):
                continue
            parameter, _ = _OPERATOR_PARAMETER[name]
            low, high = (float(v) for v in options[parameter])
            image = _APPLY[name](image, _uniform(low, high))
        return image

    def __repr__(self) -> str:
        """Canonical: independent of YAML key order, free of process addresses."""
        parts = []
        for name in DEGRADATION_OPERATORS:
            options = self._spec.get(name)
            if options is None:
                continue
            rendered = ", ".join(
                f"{key}={options[key]}" for key in sorted(options)
            )
            parts.append(f"{name}({rendered})")
        return f"DegradationChain({', '.join(parts)})"


def build_train_degradation(spec: Any) -> DegradationChain | None:
    """Return the chain for ``spec``, or ``None`` when degradation is disabled.

    ``None`` is the load-bearing return value: the training pipeline treats it as
    "behave exactly as before this module existed".
    """
    if spec is None:
        return None
    validate_train_degradation(spec)
    if not spec:
        return None
    return DegradationChain(spec)
