# Baseline 改进修复与可直接执行的实现方案

> 项目：Noise-aware-Parameter-Efficient-CLIP-Fine-tuning  
> 审查对象：2026-07-10 GitHub `main` 分支（页面显示 42 commits）  
> 目标：先把 baseline 做成**配置可信、训练正确、实验可复现、结果可比较**的强基线，再接入噪声鲁棒算法。

---

## 1. 当前状态与结论

当前仓库已经具备以下模块：

- CLIP ViT-B/32 + Linear Head baseline；
- Cosine Classifier；
- A0～A3 数据增强；
- 冻结 CLIP 特征缓存；
- dev / confirm / final_fit 训练模式；
- 多配置文件、测试目录和验收脚本；
- B0、E0～E5、C0～C2 的实验配置。

但目前仍有两个 P0 阻断问题：

1. **YAML 中的实验配置没有可靠传递到训练逻辑。**  
   `--mode`、`--augmentation-preset`、`--use-cached-features` 等命令行参数带有非空默认值，导致配置文件中的 `experiment.mode`、`experiment.augmentation_preset`、`model.use_cached_features` 被默认值覆盖。

2. **缓存特征训练路径错误。**  
   `CachedFeatureDataset` 返回 `[B, 512]` 特征，但训练循环仍调用 `model(inputs)`；当前模型的 `forward()` 会把输入继续送入 CLIP visual encoder，而 visual encoder 需要 `[B, 3, H, W]` 图像。

因此，当前阶段不能直接运行 E0/E1 搜索，也不能根据现有代码认定 Cosine Head、增强或缓存已经产生有效提升。

---

# 2. 完成标准

baseline 改进完成必须同时满足：

- `configs/e2_augmentation.yaml` 启动后日志明确显示 `augmentation_preset=a1`；
- `configs/e0_hyper_search.yaml` 启动后日志明确显示 `use_cached_features=True`；
- 缓存特征能够完成前向、反向和优化器更新；
- online A0 与 cached A0 的分类 logits 在合理误差内一致；
- B0 可复现原始约 61.38% 验证准确率，允许随机波动；
- E0、E1 的 9 组参数搜索可以自动运行并生成汇总表；
- E2～E5 使用冻结后的超参数，不在确认划分上重新调参；
- 每次运行保存 resolved config、split seed、train seed、head、增强、最佳 epoch 和准确率；
- 至少在开发划分和两个确认划分上报告 paired delta。

建议验收阈值：

- B0 新实现相对 61.38% 的下降不超过 **0.5 个百分点**；
- cached 与 online 的单批次 logits：
  - 使用相同已编码特征时：`atol <= 1e-5`；
  - 分别经过两次 CLIP 编码时：`atol <= 1e-3`；
- 所有新增单元测试通过；
- E0/E1 搜索结果中无失败 trial、无共用 checkpoint 目录、无配置串写。

---

# 3. 修改顺序

严格按以下顺序执行：

```text
P0-1  修复配置解析
P0-2  修复 cached feature forward
P0-3  增加行为测试
P0-4  复现 B0
P1-1  自动运行 E0/E1 超参数搜索
P1-2  运行 E2～E5 消融
P1-3  多划分确认
P2    接入第一项噪声鲁棒策略
```

在 P0 全部通过前，不建议继续增加新的噪声损失或复杂模型。

---

# 4. P0-1：修复 YAML 与 CLI 配置优先级

## 4.1 修改 `experiments/baseline/train.py` 的参数默认值

将以下参数的默认值全部改为 `None`，表示"用户没有从 CLI 显式指定，应读取 YAML"。

```python
parser.add_argument(
    "--experiment-id",
    type=str,
    default=None,
)

parser.add_argument(
    "--mode",
    type=str,
    default=None,
    choices=["dev", "confirm", "final_fit"],
)

parser.add_argument(
    "--use-cached-features",
    action=argparse.BooleanOptionalAction,
    default=None,
)

parser.add_argument(
    "--augmentation-preset",
    type=str,
    default=None,
    choices=sorted(VALID_PRESETS),
)

parser.add_argument(
    "--head-type",
    type=str,
    default=None,
    choices=["linear", "cosine"],
)
```

`BooleanOptionalAction` 会同时提供：

```bash
--use-cached-features
--no-use-cached-features
```

这样才能区分：

- 未提供参数：读取 YAML；
- 显式开启；
- 显式关闭。

项目已使用较新的 Python 类型标注，建议统一要求 Python 3.10+。

---

## 4.2 新建 `common/runtime_config.py`

```python
"""Resolve runtime options from CLI and YAML.

Priority:
    explicit CLI > YAML > hard default
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict


def _pick(cli_value: Any, yaml_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if yaml_value is not None:
        return yaml_value
    return default


def resolve_runtime_args(
    args: Namespace,
    config: Dict[str, Any],
) -> Namespace:
    exp_cfg = config.get("experiment", {})
    model_cfg = config.get("model", {})

    args.experiment_id = _pick(
        args.experiment_id,
        exp_cfg.get("id"),
        "B0",
    )
    args.mode = _pick(
        args.mode,
        exp_cfg.get("mode"),
        "dev",
    )
    args.augmentation_preset = _pick(
        args.augmentation_preset,
        exp_cfg.get("augmentation_preset"),
        "a0",
    )
    args.head_type = _pick(
        args.head_type,
        exp_cfg.get("head_type"),
        "linear",
    )
    args.use_cached_features = bool(
        _pick(
            args.use_cached_features,
            model_cfg.get("use_cached_features"),
            False,
        )
    )

    # 将最终生效值写回 config，确保 checkpoint 和 snapshot 保存的是 resolved config。
    config.setdefault("experiment", {})
    config["experiment"].update(
        {
            "id": args.experiment_id,
            "mode": args.mode,
            "augmentation_preset": args.augmentation_preset,
            "head_type": args.head_type,
        }
    )
    config.setdefault("model", {})
    config["model"]["use_cached_features"] = args.use_cached_features

    config["runtime"] = {
        "experiment_id": args.experiment_id,
        "mode": args.mode,
        "augmentation_preset": args.augmentation_preset,
        "head_type": args.head_type,
        "use_cached_features": args.use_cached_features,
    }

    return args
```

---

## 4.3 在 `train.py` 中接入 resolver

在 import 区加入：

```python
from common.runtime_config import resolve_runtime_args
```

将 `main()` 开头修改为：

```python
def main():
    args = parse_args()

    config = load_config(args.config)
    args = resolve_runtime_args(args, config)

    mode = args.mode
    experiment_id = args.experiment_id
    use_cached = args.use_cached_features
    aug_preset = args.augmentation_preset
    head_type = args.head_type
```

删除原先这段重复逻辑：

```python
mode = args.mode
experiment_id = args.experiment_id
use_cached = args.use_cached_features
aug_preset = args.augmentation_preset

head_type = args.head_type
if head_type is None:
    head_type = config.get("experiment", {}).get("head_type", "linear")
```

---

## 4.4 增加配置自检

在 guard 前增加：

```python
if aug_preset not in VALID_PRESETS:
    raise ValueError(
        f"Unknown augmentation preset: {aug_preset}. "
        f"Expected one of {sorted(VALID_PRESETS)}"
    )

if mode not in {"dev", "confirm", "final_fit"}:
    raise ValueError(f"Unsupported training mode: {mode}")

if head_type not in {"linear", "cosine"}:
    raise ValueError(f"Unsupported head type: {head_type}")
```

建议将关键 resolved 值一次性输出：

```python
train_logger.info(
    "Resolved runtime: "
    f"experiment_id={experiment_id}, "
    f"mode={mode}, "
    f"head_type={head_type}, "
    f"augmentation={aug_preset}, "
    f"cached={use_cached}"
)
```

---

# 5. P0-2：修复缓存特征训练接口

## 5.1 修改 Linear 模型

修改 `experiments/baseline/model.py`。

在 `CLIPLinearClassifier` 中新增：

```python
def forward_features(self, features: torch.Tensor) -> torch.Tensor:
    """Classify pre-computed CLIP features.

    Args:
        features: Tensor of shape [B, feature_dim].

    Returns:
        Logits of shape [B, num_classes].
    """
    if features.ndim != 2:
        raise ValueError(
            f"Expected cached features with shape [B, D], "
            f"got {tuple(features.shape)}"
        )
    if features.shape[-1] != self.feature_dim:
        raise ValueError(
            f"Expected feature_dim={self.feature_dim}, "
            f"got {features.shape[-1]}"
        )

    features = F.normalize(features.float(), p=2, dim=-1)
    return self.classifier(features)
```

将原 `forward()` 改为：

```python
def forward(self, images: torch.Tensor) -> torch.Tensor:
    features = self.encode_image(images)
    return self.forward_features(features)
```

不建议让 `forward()` 根据 tensor 维度自动猜测输入类型。显式调用 `forward_features()` 更容易发现数据管线错误。

---

## 5.2 修改 Cosine 模型

修改 `experiments/cosine/model.py`。

首先修正 weight 初始化。当前代码先乘 `init_scale`，随后 forward 又归一化 weight，乘法会被抵消。改为：

```python
weight = torch.randn(num_classes, feature_dim)
weight = F.normalize(weight, dim=1)
self.weight = nn.Parameter(weight)
```

新增：

```python
def forward_features(self, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(
            f"Expected cached features with shape [B, D], "
            f"got {tuple(features.shape)}"
        )
    if features.shape[-1] != self.feature_dim:
        raise ValueError(
            f"Expected feature_dim={self.feature_dim}, "
            f"got {features.shape[-1]}"
        )

    features = F.normalize(features.float(), p=2, dim=-1)
    weight_norm = F.normalize(self.weight.float(), p=2, dim=1)

    # forward 中只读取裁剪值；参数原地裁剪仍在 optimizer.step() 后执行。
    scale = self.logit_scale.clamp(min=1.0, max=100.0)
    return scale * features @ weight_norm.t()
```

将原 `forward()` 改为：

```python
def forward(self, images: torch.Tensor) -> torch.Tensor:
    features = self.encode_image(images)
    return self.forward_features(features)
```

将原 `clamp_scale()` 改为更明确的原地方法：

```python
@torch.no_grad()
def clamp_scale(self) -> None:
    if self.learnable_scale:
        self.logit_scale.clamp_(min=1.0, max=100.0)
```

训练循环中原有：

```python
if hasattr(model, "clamp_scale"):
    model.clamp_scale()
```

可以继续使用。

---

## 5.3 修改训练 batch 解析

在 `train.py` 中新增：

```python
def _unpack_batch(
    batch_data,
    device: torch.device,
):
    """Return inputs, labels, is_cached."""
    if len(batch_data) == 3:
        images, labels, _paths = batch_data
        inputs = images.to(device, non_blocking=True)
        is_cached = False
    elif len(batch_data) == 2:
        features, labels = batch_data
        inputs = features.to(device, non_blocking=True)
        is_cached = True
    else:
        raise ValueError(
            f"Unexpected batch tuple length: {len(batch_data)}"
        )

    labels = labels.to(device, non_blocking=True)
    return inputs, labels, is_cached


def _forward_inputs(
    model: nn.Module,
    inputs: torch.Tensor,
    is_cached: bool,
) -> torch.Tensor:
    if is_cached:
        if not hasattr(model, "forward_features"):
            raise TypeError(
                f"{type(model).__name__} does not implement "
                "forward_features() required by cached training."
            )
        return model.forward_features(inputs)

    return model(inputs)
```

将 `train_one_epoch()` 中原先的 batch 解析和：

```python
logits = model(images)
```

改为：

```python
inputs, labels, is_cached = _unpack_batch(batch_data, device)

if use_amp:
    with autocast(device_type=device.type, enabled=True):
        logits = _forward_inputs(model, inputs, is_cached)
        loss = criterion(logits, labels)
else:
    logits = _forward_inputs(model, inputs, is_cached)
    loss = criterion(logits, labels)
```

统计 batch size 改为：

```python
batch_size = inputs.size(0)
```

`validate()` 同样使用 `_unpack_batch()` 和 `_forward_inputs()`。

---

## 5.4 修复 final_fit 忽略缓存的问题

当前 `main()` 中先判断 `mode == "final_fit"`，再判断 `use_cached`，因此 E0/E1 即使配置了缓存，final_fit 仍然走在线图像路径。

修改 `CachedFeatureDataset`，允许 `split_csv=None`。

在 `common/cache.py` 中修改构造函数类型语义：

```python
def __init__(
    self,
    cache_dir,
    split_csv=None,
    class_to_idx_path=None,
    dataset_root=None,
    verification="full",
):
```

修改 `_load_split()`：

```python
def _load_split(self, split_csv):
    if split_csv is None:
        self.sample_indices = list(range(len(self.all_paths)))
        return

    import pandas as pd

    df = pd.read_csv(split_csv)
    self.sample_indices = []
    path_to_idx = {p: i for i, p in enumerate(self.all_paths)}

    for _, row in df.iterrows():
        # 保留原有路径匹配和标签一致性检查逻辑
        ...
```

训练数据分支改为：

```python
if use_cached:
    cache_dir = config["cache"]["cache_dir"]
    verification = config["cache"].get("verification", "full")
    class_to_idx_path = str(
        Path(class_mapping_path) / "class_to_idx.json"
    )

    train_split_csv = (
        None
        if mode == "final_fit"
        else str(Path(split_dir) / "train.csv")
    )

    train_dataset = CachedFeatureDataset(
        cache_dir=cache_dir,
        split_csv=train_split_csv,
        class_to_idx_path=class_to_idx_path,
        dataset_root=config["data"]["train_dir"],
        verification=verification,
    )

    # 构造 cached train_loader

    if mode in {"dev", "confirm"}:
        # 构造在线 val_loader
        ...
    else:
        val_loader = None

elif mode == "final_fit":
    # 非缓存增强实验使用完整在线训练集
    ...
else:
    # 普通在线 train / val
    ...
```

---

# 6. P0-3：新增行为测试

现有测试不能只检查文件和字段是否存在，必须检查"配置是否真的生效"和"缓存特征是否真的走分类头"。

## 6.1 新建 `tests/test_runtime_config.py`

```python
from argparse import Namespace

from common.runtime_config import resolve_runtime_args


def make_args(**kwargs):
    values = {
        "experiment_id": None,
        "mode": None,
        "augmentation_preset": None,
        "head_type": None,
        "use_cached_features": None,
    }
    values.update(kwargs)
    return Namespace(**values)


def test_yaml_values_are_used_when_cli_is_absent():
    config = {
        "experiment": {
            "id": "E2",
            "mode": "dev",
            "head_type": "linear",
            "augmentation_preset": "a1",
        },
        "model": {
            "use_cached_features": False,
        },
    }

    args = resolve_runtime_args(make_args(), config)

    assert args.experiment_id == "E2"
    assert args.mode == "dev"
    assert args.head_type == "linear"
    assert args.augmentation_preset == "a1"
    assert args.use_cached_features is False


def test_cli_explicit_values_override_yaml():
    config = {
        "experiment": {
            "id": "E2",
            "mode": "dev",
            "head_type": "linear",
            "augmentation_preset": "a1",
        },
        "model": {
            "use_cached_features": False,
        },
    }

    args = resolve_runtime_args(
        make_args(
            experiment_id="E1",
            head_type="cosine",
            augmentation_preset="a0",
            use_cached_features=True,
        ),
        config,
    )

    assert args.experiment_id == "E1"
    assert args.head_type == "cosine"
    assert args.augmentation_preset == "a0"
    assert args.use_cached_features is True
```

---

## 6.2 新建 `tests/test_cached_forward.py`

```python
from types import SimpleNamespace

import torch
import torch.nn as nn

from experiments.baseline.model import CLIPLinearClassifier
from experiments.cosine.model import CosineClassifier


class DummyVisual(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=1)
        self.proj = nn.Linear(8, feature_dim)

    def forward(self, images):
        x = self.conv1(images)
        x = x.mean(dim=(2, 3))
        return self.proj(x)


def make_dummy_clip():
    return SimpleNamespace(visual=DummyVisual())


def test_linear_cached_forward_shape_and_grad():
    model = CLIPLinearClassifier(
        clip_model=make_dummy_clip(),
        num_classes=5,
        feature_dim=512,
        freeze_clip=True,
    )

    features = torch.randn(4, 512)
    logits = model.forward_features(features)

    assert logits.shape == (4, 5)

    logits.sum().backward()
    assert model.classifier.weight.grad is not None


def test_linear_online_and_feature_forward_match():
    model = CLIPLinearClassifier(
        clip_model=make_dummy_clip(),
        num_classes=5,
        feature_dim=512,
        freeze_clip=True,
    )
    model.eval()

    images = torch.randn(4, 3, 16, 16)

    with torch.no_grad():
        features = model.encode_image(images)
        online_logits = model(images)
        cached_logits = model.forward_features(features)

    torch.testing.assert_close(
        online_logits,
        cached_logits,
        rtol=1e-5,
        atol=1e-6,
    )


def test_cosine_cached_forward_shape_and_grad():
    model = CosineClassifier(
        clip_model=make_dummy_clip(),
        num_classes=5,
        feature_dim=512,
        freeze_clip=True,
        init_scale=10.0,
        learnable_scale=True,
    )

    features = torch.randn(4, 512)
    logits = model.forward_features(features)

    assert logits.shape == (4, 5)

    logits.sum().backward()
    assert model.weight.grad is not None
    assert model.logit_scale.grad is not None


def test_cosine_online_and_feature_forward_match():
    model = CosineClassifier(
        clip_model=make_dummy_clip(),
        num_classes=5,
        feature_dim=512,
        freeze_clip=True,
        init_scale=10.0,
        learnable_scale=True,
    )
    model.eval()

    images = torch.randn(4, 3, 16, 16)

    with torch.no_grad():
        features = model.encode_image(images)
        online_logits = model(images)
        cached_logits = model.forward_features(features)

    torch.testing.assert_close(
        online_logits,
        cached_logits,
        rtol=1e-5,
        atol=1e-6,
    )
```

---

## 6.3 新建配置解析验收脚本

新建 `scripts/check_resolved_configs.py`：

```python
from argparse import Namespace
from pathlib import Path

from common.runtime_config import resolve_runtime_args
from common.utils import load_config


EXPECTED = {
    "b0_regression.yaml": ("B0", "linear", "a0", False),
    "e0_hyper_search.yaml": ("E0", "linear", "a0", True),
    "e1_hyper_search.yaml": ("E1", "cosine", "a0", True),
    "e2_augmentation.yaml": ("E2", "linear", "a1", False),
    "e3_augmentation.yaml": ("E3", "linear", "a2", False),
    "e4_augmentation.yaml": ("E4", "linear", "a3", False),
    "e5_combined.yaml": ("E5", "cosine", "a3", False),
}


def empty_args():
    return Namespace(
        experiment_id=None,
        mode=None,
        augmentation_preset=None,
        head_type=None,
        use_cached_features=None,
    )


def main():
    for filename, expected in EXPECTED.items():
        path = Path("configs") / filename
        config = load_config(str(path))
        args = resolve_runtime_args(empty_args(), config)

        actual = (
            args.experiment_id,
            args.head_type,
            args.augmentation_preset,
            args.use_cached_features,
        )

        if actual != expected:
            raise AssertionError(
                f"{filename}: expected={expected}, actual={actual}"
            )

        print(f"[PASS] {filename}: {actual}")


if __name__ == "__main__":
    main()
```

运行：

```bash
pytest -q tests/test_runtime_config.py tests/test_cached_forward.py
python scripts/check_resolved_configs.py
```

---

# 7. P0-4：复现 B0

## 7.1 数据检查与划分

先修改配置中的实际数据路径，然后运行：

```bash
python scripts/check_data.py --config configs/b0_regression.yaml

python scripts/split_data.py \
  --config configs/b0_regression.yaml
```

必须确认：

- 类别数为 500；
- split seed 为 42；
- train / val 无重复路径；
- class mapping 固定；
- `train.csv` 和 `val.csv` 数量之和等于有效训练图片数。

---

## 7.2 单元测试

```bash
pytest -q
python scripts/check_resolved_configs.py
python scripts/run_acceptance.py
```

`run_acceptance.py` 不能替代 `pytest`。它只能作为附加检查。

---

## 7.3 运行 B0

```bash
python -m experiments.baseline.train \
  --config configs/b0_regression.yaml
```

训练日志必须出现：

```text
experiment_id=B0
mode=dev
head_type=linear
augmentation=a0
cached=False
```

检查：

```bash
cat outputs/b0/checkpoints/eval_results.json
cat outputs/b0/logs/train_log.csv
```

验收：

- 最佳验证准确率接近 0.6138；
- 不低于约 0.6088；
- `best.pt`、`last.pt`、`eval_results.json`、resolved config 均存在；
- checkpoint 中含 class mapping、head type、augmentation、split seed、train seed、best epoch。

若 B0 下降超过 0.5 个百分点，先检查：

1. 训练/验证划分是否与原实验一致；
2. class mapping 是否一致；
3. CLIP 权重来源是否为 OpenAI；
4. AMP、随机种子和数据预处理是否发生变化；
5. warmup 和 scheduler 是否按 step 正确更新；
6. 是否误用了随机增强；
7. 是否加载了错误 checkpoint。

---

# 8. P1-1：自动运行 E0/E1 超参数搜索

## 8.1 新建 `scripts/run_hyper_search.py`

```python
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

from common.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse trials with an existing eval_results.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser.parse_args()


def format_float(value: float) -> str:
    return f"{value:.0e}".replace("+", "")


def run_command(command, dry_run=False):
    print("+", " ".join(map(str, command)))
    if not dry_run:
        subprocess.run(command, check=True)


def main():
    args = parse_args()
    base_config = load_config(args.config)

    search_cfg = base_config.get("hyper_search")
    if not search_cfg:
        raise ValueError(
            f"No hyper_search section found in {args.config}"
        )

    lr_values = search_cfg["lr_values"]
    wd_values = search_cfg["wd_values"]

    experiment_id = base_config["experiment"]["id"]
    base_save_dir = Path(base_config["train"]["save_dir"])
    experiment_root = base_save_dir.parent
    search_root = experiment_root / "search"
    generated_root = search_root / "generated_configs"
    generated_root.mkdir(parents=True, exist_ok=True)

    if base_config["model"].get("use_cached_features", False):
        cache_dir = Path(base_config["cache"]["cache_dir"])
        if not (cache_dir / "manifest.json").exists():
            raise FileNotFoundError(
                f"Feature cache not found: {cache_dir}\n"
                f"Run: python scripts/cache_features.py "
                f"--config {args.config}"
            )

    split_dir = Path(base_config["data"]["split_dir"])
    required_split_files = [
        split_dir / "train.csv",
        split_dir / "val.csv",
        split_dir / "class_to_idx.json",
        split_dir / "idx_to_class.json",
    ]
    if not all(path.exists() for path in required_split_files):
        run_command(
            [
                sys.executable,
                "scripts/split_data.py",
                "--config",
                args.config,
            ],
            dry_run=args.dry_run,
        )

    rows = []

    for lr in lr_values:
        for wd in wd_values:
            trial_name = (
                f"lr_{format_float(float(lr))}"
                f"__wd_{format_float(float(wd))}"
            )
            trial_root = search_root / trial_name
            result_path = (
                trial_root / "checkpoints" / "eval_results.json"
            )

            trial_config = copy.deepcopy(base_config)
            trial_config["train"]["lr"] = float(lr)
            trial_config["train"]["weight_decay"] = float(wd)
            trial_config["train"]["save_dir"] = str(
                trial_root / "checkpoints"
            )
            trial_config["output"]["log_dir"] = str(
                trial_root / "logs"
            )
            trial_config["output"]["submission_dir"] = str(
                trial_root / "submissions"
            )
            trial_config.setdefault("runtime", {})
            trial_config["runtime"]["search_trial"] = trial_name

            config_path = generated_root / f"{trial_name}.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    trial_config,
                    f,
                    sort_keys=False,
                    allow_unicode=True,
                )

            if not (args.skip_existing and result_path.exists()):
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "experiments.baseline.train",
                        "--config",
                        str(config_path),
                    ],
                    dry_run=args.dry_run,
                )

            if args.dry_run:
                continue

            if not result_path.exists():
                raise FileNotFoundError(
                    f"Trial finished without result: {result_path}"
                )

            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)

            rows.append(
                {
                    "experiment_id": experiment_id,
                    "trial": trial_name,
                    "lr": float(lr),
                    "weight_decay": float(wd),
                    "best_val_acc": result["best_val_acc"],
                    "dev_best_epoch": result["dev_best_epoch"],
                    "head_type": result["head_type"],
                    "augmentation_preset": result[
                        "augmentation_preset"
                    ],
                    "config_path": str(config_path),
                    "checkpoint_path": str(
                        trial_root / "checkpoints" / "best.pt"
                    ),
                }
            )

    if args.dry_run:
        return

    rows.sort(key=lambda row: row["best_val_acc"], reverse=True)

    csv_path = search_root / "search_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    with open(
        search_root / "best_trial.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    best_config = load_config(best["config_path"])
    with open(
        search_root / "best_config.yaml",
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            best_config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Best trial: {best}")
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
```

---

## 8.2 构建一次 feature cache

先使用 E0 配置：

```bash
python scripts/cache_features.py \
  --config configs/e0_hyper_search.yaml
```

首次构建后执行完整检查。大量重复搜索时，可将配置中的：

```yaml
cache:
  verification: full
```

改为：

```yaml
cache:
  verification: quick
```

前提是：

- 数据目录未发生变化；
- 完整 fingerprint 已至少成功验证一次；
- class mapping 未改变。

否则每个 trial 初始化时重新计算全量文件内容哈希，会造成很大时间开销。

---

## 8.3 E0：Linear + A0 搜索

```bash
python scripts/run_hyper_search.py \
  --config configs/e0_hyper_search.yaml
```

断点续跑：

```bash
python scripts/run_hyper_search.py \
  --config configs/e0_hyper_search.yaml \
  --skip-existing
```

先查看计划而不执行：

```bash
python scripts/run_hyper_search.py \
  --config configs/e0_hyper_search.yaml \
  --dry-run
```

结果：

```text
outputs/e0/search/search_results.csv
outputs/e0/search/best_trial.json
outputs/e0/search/best_config.yaml
```

---

## 8.4 E1：Cosine + A0 搜索

```bash
python scripts/run_hyper_search.py \
  --config configs/e1_hyper_search.yaml
```

结果：

```text
outputs/e1/search/search_results.csv
outputs/e1/search/best_trial.json
outputs/e1/search/best_config.yaml
```

E0 和 E1 均使用相同：

- split seed；
- train seed；
- A0；
- frozen CLIP；
- epoch 数；
- 搜索网格。

因此比较：

```text
E1_best_acc - E0_best_acc
```

才可以较公平地判断 Cosine Head 的收益。

---

# 9. 缓存与在线路径的一致性检查

缓存只应改变运行速度，不应改变实验语义。

新增一个小规模 smoke test，随机读取一批 A0 图像：

1. 在线 CLIP 编码得到 features；
2. 从 cache 读取同一批 features；
3. 比较 feature 和 logits；
4. 确认分类头输入维度、归一化和标签一致。

建议输出：

```json
{
  "sample_count": 128,
  "feature_max_abs_diff": 0.0002,
  "feature_mean_abs_diff": 0.00001,
  "linear_logit_max_abs_diff": 0.0001,
  "cosine_logit_max_abs_diff": 0.0003,
  "label_mismatch": 0,
  "path_mismatch": 0
}
```

若差异明显，检查：

- cache 构建和在线路径是否使用相同 CLIP 权重；
- resize / center crop / normalization 是否完全一致；
- cache 是否使用 AMP，而在线是否使用 FP32；
- 图片 RGB 转换是否一致；
- class mapping 是否一致；
- cache 是否来自旧数据目录。

建议把 cache 视为计算优化，不作为一项精度改进计入 ablation。

---

# 10. P1-2：运行 E2～E5 消融

## 10.1 冻结超参数

E0 搜索结束后，将 E0 最优 `lr`、`weight_decay` 写入：

```text
configs/e2_augmentation.yaml
configs/e3_augmentation.yaml
configs/e4_augmentation.yaml
```

E1 搜索结束后，将 E1 最优参数写入：

```text
configs/e5_combined.yaml
configs/c0_cosine_scale.yaml
configs/c1_cosine_scale.yaml
configs/c2_cosine_scale.yaml
```

不得在 E2～E5 的验证结果出来后继续针对同一划分反复改 lr/wd，否则会扩大验证集过拟合。

---

## 10.2 运行增强实验

```bash
python scripts/split_data.py \
  --config configs/e2_augmentation.yaml

python -m experiments.baseline.train \
  --config configs/e2_augmentation.yaml

python -m experiments.baseline.train \
  --config configs/e3_augmentation.yaml

python -m experiments.baseline.train \
  --config configs/e4_augmentation.yaml
```

日志必须分别显示：

```text
E2 -> a1
E3 -> a2
E4 -> a3
```

E2～E4 都必须显示：

```text
cached=False
```

原因是随机增强不能使用预先固定的 A0 特征缓存。

---

## 10.3 选择最佳增强

只根据开发划分 seed 42 选择：

```text
best_augmentation = argmax(E2, E3, E4)
```

然后修改 `configs/e5_combined.yaml`：

```yaml
experiment:
  augmentation_preset: a1 # 示例，以真实最佳项为准
```

运行：

```bash
python -m experiments.baseline.train \
  --config configs/e5_combined.yaml
```

E5 的目标是检验：

```text
最佳增强是否与 Cosine Head 具有可叠加收益
```

不能直接假设 A3 最强。细粒度动植物分类可能依赖颜色和局部纹理，过强 ColorJitter 或 RandomErasing 可能损伤判别信息。

---

# 11. P1-3：多划分确认

## 11.1 划分原则

开发划分：

```text
split_seed = 42
```

确认划分：

```text
split_seed = 3407
split_seed = 2026
```

所有方法使用相同 split seed 和相同 train seed。

确认阶段只运行：

- E0 最优 Linear baseline；
- E1 最优 Cosine baseline；
- E5 最优组合；
- 可选：表现最好的单独增强。

不得在确认划分重新搜索超参数或重新选择增强。

---

## 11.2 paired delta

对每个 seed 计算：

```text
delta_cosine(seed) = Acc(E1, seed) - Acc(E0, seed)
delta_combined(seed) = Acc(E5, seed) - Acc(E0, seed)
```

报告：

- 每个 seed 的绝对准确率；
- paired delta；
- 平均 delta；
- 最差 seed delta；
- 标准差；
- 是否所有 seed 都为正。

结果表建议：

```csv
experiment,split_seed,train_seed,head,augmentation,lr,weight_decay,best_epoch,val_acc,delta_vs_e0
E0,42,42,linear,a0,...,...,...,...,0
E1,42,42,cosine,a0,...,...,...,...,...
E5,42,42,cosine,a1,...,...,...,...,...
E0,3407,42,linear,a0,...,...,...,...,0
E1,3407,42,cosine,a0,...,...,...,...,...
E5,3407,42,cosine,a1,...,...,...,...,...
```

只有多划分平均收益为正，且不存在明显灾难性负收益，才能将该项写成"baseline 强化"。

---

# 12. 结果汇总字段

统一在 `eval_results.json` 中增加：

```python
eval_results = {
    "experiment_id": experiment_id,
    "mode": mode,
    "config_path": args.config,
    "split_seed": config["data"].get("split_seed"),
    "train_seed": train_seed,
    "best_val_acc": float(best_val_acc),
    "dev_best_epoch": dev_best_epoch,
    "trained_epochs": epochs,
    "head_type": model.head_type,
    "augmentation_preset": aug_preset,
    "use_cached_features": use_cached,
    "learning_rate": config["train"]["lr"],
    "weight_decay": config["train"]["weight_decay"],
    "batch_size": config["train"]["batch_size"],
    "freeze_clip": config["model"].get("freeze_clip", True),
    "clip_model_name": config["model"]["clip_model_name"],
}
```

checkpoint metadata中也保存相同字段。

建议额外保存 Git commit：

```python
import subprocess


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None
```

写入：

```python
"git_commit": get_git_commit()
```

---

# 13. 调度器与 warmup 的建议修复

当前 scheduler 的 `T_max` 使用总训练 step，但 warmup 期间不调用 scheduler；因此真正执行 cosine scheduler 的 step 数少于 `T_max`。

更严谨的写法：

```python
warmup_steps = train_cfg["warmup_epochs"] * len(train_loader)
total_steps = epochs * len(train_loader)
cosine_steps = max(total_steps - warmup_steps, 1)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cosine_steps,
    eta_min=train_cfg["lr"] * 0.01,
)
```

warmup 结束后再调用 scheduler。

另外，当前 `_warmup_lr()` 用统一 `base_lr` 覆盖所有 param group，会破坏 Cosine Head 中：

```text
weight lr = lr
logit_scale lr = lr * 0.1
```

建议为每个 param group 保存初始 lr：

```python
for group in optimizer.param_groups:
    group.setdefault("initial_lr", group["lr"])
```

warmup：

```python
def _warmup_lr(optimizer, warmup_steps, current_step):
    if current_step >= warmup_steps:
        return

    scale = (current_step + 1) / max(warmup_steps, 1)

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * scale
```

调用时不再传统一 `base_lr`。

这是 Cosine Head 搜索前应修复的隐蔽公平性问题，否则 logit scale 参数的低学习率设置在 warmup 阶段会被错误覆盖。

---

# 14. 建议的完整执行顺序

## 14.1 代码正确性

```bash
pytest -q

python scripts/check_resolved_configs.py

python scripts/run_acceptance.py
```

---

## 14.2 B0 回归

```bash
python scripts/check_data.py \
  --config configs/b0_regression.yaml

python scripts/split_data.py \
  --config configs/b0_regression.yaml

python -m experiments.baseline.train \
  --config configs/b0_regression.yaml
```

---

## 14.3 建立缓存

```bash
python scripts/cache_features.py \
  --config configs/e0_hyper_search.yaml
```

---

## 14.4 E0 / E1 搜索

```bash
python scripts/run_hyper_search.py \
  --config configs/e0_hyper_search.yaml

python scripts/run_hyper_search.py \
  --config configs/e1_hyper_search.yaml
```

---

## 14.5 增强消融

先把 E0 最优参数写入 E2～E4，再运行：

```bash
python -m experiments.baseline.train \
  --config configs/e2_augmentation.yaml

python -m experiments.baseline.train \
  --config configs/e3_augmentation.yaml

python -m experiments.baseline.train \
  --config configs/e4_augmentation.yaml
```

---

## 14.6 组合实验

把最佳增强和 E1 最优参数写入 E5：

```bash
python -m experiments.baseline.train \
  --config configs/e5_combined.yaml
```

---

## 14.7 确认划分

生成 seed 3407、2026 的配置副本，分别运行 E0、E1、E5。

建议目录：

```text
outputs/confirm/seed_3407/e0/
outputs/confirm/seed_3407/e1/
outputs/confirm/seed_3407/e5/

outputs/confirm/seed_2026/e0/
outputs/confirm/seed_2026/e1/
outputs/confirm/seed_2026/e5/
```

---

# 15. 第一阶段实验矩阵

| ID  | Head                    |   Augmentation | Cache |    调参 | 目的                     |
| --- | ----------------------- | -------------: | ----: | ------: | ------------------------ |
| B0  | Linear                  |             A0 |    No |      No | 回归原始 61.38%          |
| E0  | Linear                  |             A0 |   Yes | lr × wd | 建立强化 Linear baseline |
| E1  | Cosine                  |             A0 |   Yes | lr × wd | 隔离 Cosine Head 收益    |
| E2  | Linear                  |             A1 |    No |      No | 基础随机增强             |
| E3  | Linear                  |             A2 |    No |      No | 检查 ColorJitter         |
| E4  | Linear                  |             A3 |    No |      No | 检查 RandomErasing       |
| E5  | Cosine                  | best(A1,A2,A3) |    No |      No | 检查组合收益             |
| C0  | Cosine fixed scale      |             A0 |   Yes |      No | scale 消融               |
| C1  | Cosine learnable scale  |             A0 |   Yes |      No | scale 消融               |
| C2  | Cosine alternative init |             A0 |   Yes |      No | scale 初始化消融         |

优先级：

```text
B0 -> E0 -> E1 -> E2/E3/E4 -> E5 -> C0/C1/C2
```

C0～C2 不是当前最高优先级。若 E1 本身没有稳定收益，不需要先花大量算力细调 scale。

---

# 16. 不应混入 baseline 改进的内容

以下内容应在 baseline 完成后作为独立噪声鲁棒实验，不要与 E0～E5 混在一起：

- small-loss 筛样；
- GCE / SCE / ELR；
- 伪标签修正；
- CLIP 文本原型一致性；
- 样本动态加权；
- Co-teaching；
- DivideMix；
- LoRA / Adapter；
- prototype refinement。

原因是 baseline 阶段的目标是确定：

```text
普通监督训练条件下，哪种 head、增强和优化参数最合理
```

噪声鲁棒阶段的目标才是：

```text
如何显式减弱错误标签监督
```

二者混合会导致无法判断提升来自基础工程优化，还是噪声处理策略。

---

# 18. 最终验收清单

## 代码正确性

- [ ] CLI 未指定时正确读取 YAML；
- [ ] CLI 显式参数可覆盖 YAML；
- [ ] E0/E1 实际启用缓存；
- [ ] E2/E3/E4 实际启用 A1/A2/A3；
- [ ] cached tensor 不再进入 CLIP visual encoder；
- [ ] Linear 和 Cosine 都支持 `forward_features()`；
- [ ] Cosine scale warmup 不破坏 param-group lr；
- [ ] final_fit 中缓存配置不被静默忽略；
- [ ] 所有测试通过。

## 实验完整性

- [ ] B0 回归完成；
- [ ] E0 九组搜索完成；
- [ ] E1 九组搜索完成；
- [ ] E2～E4 消融完成；
- [ ] E5 组合完成；
- [ ] seed 3407 和 2026 确认完成；
- [ ] 保存 paired delta；
- [ ] 保存所有 resolved configs；
- [ ] 每个 trial 使用独立输出目录；
- [ ] 无确认集调参。

## 结果判定

- [ ] B0 无明显回归；
- [ ] E0 相对 B0 的收益可解释；
- [ ] E1 相对 E0 在多划分上稳定；
- [ ] 最佳增强不是仅在单划分偶然提升；
- [ ] E5 组合具有稳定正增益；
- [ ] 若某项无增益，明确保留负结果，不强行纳入最终模型。

---

# 19. 推荐的 Git 提交拆分

不要把所有修改放在一个 commit 中。建议：

```text
fix: resolve runtime options from yaml and explicit cli overrides

fix: add forward_features path for cached clip embeddings

test: add runtime config and cached forward behavior tests

fix: preserve parameter-group learning rates during warmup

feat: add reproducible hyperparameter search runner

chore: add resolved config and richer evaluation metadata

exp: reproduce B0 regression baseline

exp: run E0 and E1 hyperparameter search

exp: run augmentation and combined ablations
```

这样出现精度异常时，可以快速定位是哪类修改引入了变化。

---

# 20. 最终建议

当前最重要的不是继续扩大算法模块，而是建立可信实验闭环。

本轮工作的完成标志不是：

```text
仓库中出现了 Cosine、ColorJitter、RandomErasing 和 cache 文件
```

而是：

```text
每个配置实际控制了对应训练行为；
缓存和在线路径语义一致；
所有实验能够自动运行；
结果能够按相同划分和相同规则公平比较；
提升在多个划分上可复现。
```

完成上述修复后，仓库才具备继续开发噪声标签学习方法的可靠基础。
