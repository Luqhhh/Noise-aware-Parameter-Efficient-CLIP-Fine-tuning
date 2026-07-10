# Baseline Improvements Design

**Date**: 2026-07-10
**Status**: approved
**Scope**: 4 improvements — data augmentation, cosine classifier, feature caching, multi-seed validation

## Overview

Four independent improvements to the CLIP ViT-B/32 fine-grained classification baseline.
Each improvement is an independent experiment with its own config.

### Key Design Decisions

- **split_seed** and **train_seed** are separate concepts:
  - `split_seed` controls train/val data partitioning
  - `train_seed` controls model init, DataLoader shuffle, augmentation, CUDA randomness
- **Feature caching**: encode the FULL training set once, then index by split CSV — not per-split caches
- **Multi-seed validation is a diagnostic tool, NOT an ensemble**: final submission uses ONE single-seed model
- **Ablation order**: fix one variable at a time (head type → augmentation level → combination)

---

## Revised Execution Order

```
Step 0: Reproduce baseline with fixed split_seed=42, train_seed=42
Step 1: Stratified split + separate seed control + isolated output dirs
Step 2: Cache deterministic CLIP features on FULL training set (with manifest)
Step 3: Compare Linear vs Cosine under identical conditions (both A0)
Step 4: Fix Linear head, compare A0 / A1 / A2 / A3
Step 5: Combine: Cosine + best augmentation
Step 6: Multi-split verification of baseline and best model (report mean, std, worst split)
Step 7: Select ONE single-seed model, generate pred_results.csv, compress and submit
```

### Ablation Table

| Exp ID | Head | Augmentation | Answers |
|--------|------|-------------|---------|
| E0 | Linear | A0 | Baseline reproduction |
| E1 | Cosine | A0 | Is cosine head independently beneficial? |
| E2 | Linear | A1 | Is RandomResizedCrop+Flip independently beneficial? |
| E3 | Linear | A2 | Does ColorJitter add further gain? |
| E4 | Linear | A3 | Does RandomErasing add further gain? |
| E5 | Cosine | best of A0-A4 | Do head + augmentation gains stack? |

---

## Phase 1: Infrastructure

### 1a. Feature Caching

**Motivation**: Frozen CLIP backbone encodes the same image identically every epoch. Cache once, train the head many times.

**Architecture**: Cache the **full training set** once (before any split), then let `CachedFeatureDataset` select features by split CSV paths.

**Output** (`cache/clip_vit_b32_openai/`):
```
cache/clip_vit_b32_openai/
├── features.pt       # (N_full, 512) float32
├── labels.pt         # (N_full,) int64
├── paths.json        # ["/absolute/path/to/img1.jpg", ...]
└── manifest.json     # metadata for reproducibility
```

**`manifest.json`**:
```json
{
  "backbone": "ViT-B/32",
  "pretrained_source": "openai",
  "feature_dim": 512,
  "normalized": true,
  "dtype": "float32",
  "preprocess": "clip_deterministic",
  "dataset_size": 103218,
  "dataset_fingerprint": "<sha256 of sorted paths>",
  "created_at": "2026-07-10T12:00:00"
}
```

**New file: `tools/cache_features.py`**

Encodes every image in `data/preliminary/train/` (all classes, pre-split):
1. Loads CLIP ViT-B/32 (`pretrained_source: openai`), freezes backbone
2. Uses `model.eval()` + `torch.inference_mode()` for deterministic encoding
3. Calls `encode_image()` to get L2-normalized 512-dim features
4. Verifies: `not features.requires_grad`, `torch.isfinite(features).all()`, `features.norm(dim=-1) ≈ 1.0`
5. Saves features, labels, paths, and manifest

**New file: `common/cached_dataset.py`**

```python
class CachedFeatureDataset(Dataset):
    """Loads cached features filtered by a split CSV."""
    def __init__(self, cache_dir, split_csv):
        # 1. Load full features.pt, labels.pt, paths.json
        # 2. Build path→index lookup
        # 3. For each path in split_csv, select corresponding feature+label
        # Returns (feature, label, path_str)
```

This solves the multi-split conflict: one cache, many splits.

**Design constraint**: Feature caching is only valid for deterministic preprocessing. Experiments using random augmentation MUST use online encoding.

### 1b. Seed Separation & Output Isolation

**Config changes** — two separate seeds:
```yaml
experiment:
  split_seed: 42    # controls data partitioning
  train_seed: 42    # controls init, shuffle, augmentation, CUDA
```

**`scripts/split_data.py`** enhancement:
- Accept `--split_seed` to override config
- Stratified per-class split (each class gets its own train/val split)
- Validate: every class has ≥1 sample in both train and val
- Output to `{split_dir}/split_{split_seed}/`

**Output directory isolation**:
```
outputs/
└── {experiment}/
    └── split_42/
        ├── train_42/
        │   ├── checkpoints/
        │   └── logs/
        ├── train_3407/
        └── train_2026/
```

### 1c. Evaluation Metrics Enhancement

**`experiments/baseline/evaluate.py`** — save detailed results JSON alongside logging:

```json
{
  "overall": {
    "accuracy": 0.xxxx,
    "macro_accuracy": 0.xxxx,
    "loss": 0.xxxx,
    "total_samples": N
  },
  "per_class": {
    "0": {"accuracy": 0.xx, "total": N, "correct": N},
    "...": {}
  },
  "per_sample": [
    {
      "path": "train/0001/img.jpg",
      "label": 0,
      "pred": 12,
      "confidence": 0.85,
      "margin": 0.12,
      "loss": 1.23,
      "correct": false
    }
  ]
}
```

**Field definitions** (precise):
- `confidence` = softmax probability of the predicted class
- `margin` = max probability − second-max probability
- `loss` = per-sample cross-entropy against the noisy validation label
- `correct` = pred == noisy_validation_label (NOT necessarily semantically correct)

**Important caveat**: Multi-split validation measures robustness to partition noise, not clean-test accuracy. Use it as a diagnostic, not an unbiased estimator.

**New file: `scripts/run_multiseed_eval.py`**

Runs split → train → evaluate for specified `split_seed` values with fixed `train_seed`.

### Multi-Seed Experiment Types

**Experiment A — Split Sensitivity** (priority):
```
split_seed=42,  train_seed=42
split_seed=3407, train_seed=42
split_seed=2026, train_seed=42
```
Answers: "Is model evaluation stable across different noisy validation partitions?"

**Experiment B — Training Stability** (optional, compute permitting):
```
split_seed=42, train_seed=42
split_seed=42, train_seed=3407
split_seed=42, train_seed=2026
```
Answers: "Is training sensitive to initialization and data-order randomness?"

Report output as: "Accuracy across validation splits: mean ± std (worst: X.XXXX)"

---

## Phase 2: Cosine Classifier

**Experiment directory**: `experiments/cosine_classifier/`
**Config**: `configs/cosine_classifier.yaml`

### model.py

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    """Cosine classifier with L2-normalized weights, no bias, learnable scale."""

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 500,
        init_scale: float = 10.0,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)).log())
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * features @ weight.t()

    def clamp_scale(self) -> None:
        """Call after optimizer.step() to keep scale in valid range."""
        with torch.no_grad():
            self.logit_scale.clamp_(min=math.log(1.0), max=math.log(100.0))
```

**Key details**:
- `super().__init__()` is called before any `nn.Parameter` assignment
- `clamp_scale()` called after each `optimizer.step()` to prevent gradient dead zone at clamp boundary
- Ablation planned: fixed scale=10 vs learnable scale (noisy labels may cause scale to inflate)

### train.py

Supports two modes:
- **Online mode** (default): Full pipeline, same as baseline train.py structure
- **Cached mode** (`--use-cached-features`): Uses `CachedFeatureDataset`, trains only the head

**Cached mode note**: When CLIP is fully frozen, preprocessing is deterministic, features are cached as float32 with normalization — cached and online training use identical features. Re-running online without augmentation is unnecessary. Only switch to online mode when random augmentation or trainable backbone modules are enabled.

### Config

```yaml
# configs/cosine_classifier.yaml
model:
  architecture: ViT-B/32
  pretrained_source: openai      # REQUIRED: OpenAI official weights only
  feature_dim: 512
  freeze_clip: true
  num_classes: auto              # inferred from class_to_idx.json
  init_scale: 10.0               # fixed=10 vs learnable to be ablated
train:
  lr: 0.001
  batch_size: 128                # cached mode default: 4096
  ...
```

### CLIP Weight Source Enforcement

- Config MUST specify `model.pretrained_source: openai`
- `build_model()` logs and saves: `Backbone: CLIP ViT-B/32 | Pretrained source: OpenAI official`
- `manifest.json` records `pretrained_source` — cached features are only valid with matching backbone
- OpenCLIP / LAION weights are explicitly prohibited

---

## Phase 3: Data Augmentation

**Experiment directory**: `experiments/augmentation/`
**Configs**: `configs/augmentation_a{0,1,2,3}.yaml`

### Design

Two explicit transform paths (NOT switch-based incremental composition):

**A0 — Deterministic CLIP preprocess (control)**:
```python
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

def convert_to_rgb(image):
    return image.convert("RGB")

A0_train_transform = Compose([
    convert_to_rgb,
    Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

A0_val_transform = A0_train_transform  # same deterministic transform
```

**A1-A3 — Augmented transforms**:
```python
A1_train_transform = Compose([
    convert_to_rgb,
    RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.85, 1.15),
                      interpolation=InterpolationMode.BICUBIC, antialias=True),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

A2_train_transform = Compose([
    convert_to_rgb,
    RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.85, 1.15),
                      interpolation=InterpolationMode.BICUBIC, antialias=True),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    ToTensor(),
    Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

A3_train_transform = Compose([
    convert_to_rgb,
    RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.85, 1.15),
                      interpolation=InterpolationMode.BICUBIC, antialias=True),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    ToTensor(),
    Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.5, 2.0), value=0),
])
```

**All val_transforms are the A0 deterministic transform** (no augmentation during validation).

**Design principles**:
- `RandomErasing` is placed AFTER `Normalize` with `value=0` → fills with mean in normalized space (milder than pre-normalization black fill)
- `BICUBIC` interpolation + `antialias=True` matches CLIP pretraining input distribution
- `convert_to_rgb` explicit at the start of every pipeline — some images may be grayscale or RGBA
- Conservative scale/ratio ranges to avoid cropping out discriminative regions

### Augmentation Grid

| Experiment | RandomResizedCrop | HorizontalFlip | ColorJitter | RandomErasing |
|---|---|---|---|---|
| A0 (control) | — | — | — | — |
| A1 | scale=(0.75,1.0), ratio=(0.85,1.15) | p=0.5 | — | — |
| A2 | same | p=0.5 | brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02 | — |
| A3 | same | p=0.5 | same | p=0.1, scale=(0.02,0.15) |

### model.py

Phase 3 initially uses baseline's `CLIPLinearClassifier` to isolate augmentation effect from classifier choice. After E0-E4 are complete, E5 combines Cosine + best augmentation.

### Config

```yaml
# configs/augmentation_a1.yaml (example)
experiment:
  split_seed: 42
  train_seed: 42
model:
  architecture: ViT-B/32
  pretrained_source: openai
  feature_dim: 512
  freeze_clip: true
  num_classes: auto
augmentation:
  preset: a1            # a0 | a1 | a2 | a3
train:
  lr: 0.001
  batch_size: 128
  epochs: 20
  ...
```

---

## Multi-Seed ≠ Ensemble (Explicit Prohibition)

> **Multi-seed training is used ONLY for evaluating stability and selecting hyperparameters. The final submission MUST use a single model trained with one chosen seed. Do NOT average, vote, or otherwise fuse logits/predictions across multiple seeds.**

This constraint must be documented in code comments and the README to prevent accidental rule violations.

---

## num_classes Auto-Inference

`num_classes: auto` infers from `class_to_idx.json` at runtime and saves the mapping alongside outputs:

```json
{
  "class_to_idx": {"0000": 0, "0001": 1, "...": "..."},
  "idx_to_class": {"0": "0000", "1": "0001", "...": "..."}
}
```

Inference outputs class names directly from `idx_to_class`, NOT by formatting the integer index. This avoids off-by-one errors when class folder names differ from their sorted indices.

---

## Submission Format

Final submission file: `pred_results.csv` (NOT `pred_raw.csv`)

Format: `image_name.jpg, 0001` (comma + space, 4-digit zero-padded class name)

```python
class_id_str = idx_to_class[str(pred_idx)]  # e.g. "0001", "0123"
line = f"{image_name}, {class_id_str}"        # "img.jpg, 0001"
```

Zip contains exactly one file: `pred_results.csv` at the root level.

---

## Files Changed / Created

### New Files
- `tools/cache_features.py`
- `common/cached_dataset.py`
- `scripts/run_multiseed_eval.py`
- `experiments/cosine_classifier/__init__.py`
- `experiments/cosine_classifier/model.py`
- `experiments/cosine_classifier/train.py`
- `experiments/cosine_classifier/evaluate.py`
- `experiments/cosine_classifier/infer.py`
- `experiments/augmentation/__init__.py`
- `experiments/augmentation/model.py`
- `experiments/augmentation/train.py`
- `experiments/augmentation/evaluate.py`
- `experiments/augmentation/infer.py`
- `configs/cosine_classifier.yaml`
- `configs/augmentation_a0.yaml`
- `configs/augmentation_a1.yaml`
- `configs/augmentation_a2.yaml`
- `configs/augmentation_a3.yaml`

### Modified Files
- `scripts/split_data.py` — add `--split_seed`, stratified split validation, seed-specific output dirs
- `experiments/baseline/evaluate.py` — add per-sample + per-class metrics JSON output
- `experiments/baseline/model.py` — add `pretrained_source` logging and weight source verification
- `configs/baseline.yaml` — add `experiment.split_seed`, `experiment.train_seed`, `model.pretrained_source`, `model.num_classes: auto`

### Unchanged
- `common/dataset.py`
- `common/submission.py`
- `common/utils.py`
- `experiments/baseline/train.py`
- `experiments/baseline/infer.py`

---

## Acceptance Criteria（验收方案）

### AC-1: 特征缓存

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-1.1 | 缓存脚本正确编码全量数据 | `features.pt` shape `(N_full, 512)`, `labels.pt` shape `(N_full,)`, `paths.json` 长度一致 |
| AC-1.2 | 缓存特征与在线编码一致（含推理模式要求） | 缓存时使用 `model.eval()` + `torch.inference_mode()`；验证 `not features.requires_grad`, `torch.isfinite(features).all()`, `features.norm(dim=-1) ≈ 1.0`；与在线编码 max abs diff < 1e-5 |
| AC-1.3 | `CachedFeatureDataset` 可按 split CSV 正确索引 | 给定 `split_csv`，返回的特征、标签与在线 dataset 一致 |
| AC-1.4 | `manifest.json` 包含所有必要元数据 | `backbone`, `pretrained_source`, `feature_dim`, `normalized`, `dtype`, `preprocess`, `dataset_size`, `dataset_fingerprint`, `created_at` 全部存在 |
| AC-1.5 | 缓存模式训练加速显著 | 在相同设备上，缓存模式单 epoch 时间 ≤ 在线模式的 1/10 |

### AC-2: 种子分离与多种子验证

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-2.1 | `split_seed` 产生不同且分层的划分 | 不同 split_seed 的 train.csv/val.csv 不完全相同；每个类别在 train 和 val 中均有 ≥1 样本 |
| AC-2.2 | 输出目录隔离 split 和 train seed | 目录结构为 `outputs/{exp}/split_{split_seed}/train_{train_seed}/` |
| AC-2.3 | 评估 JSON 包含所有定义字段 | `overall`（accuracy, macro_accuracy, loss, total_samples）, `per_class`, `per_sample`（path, label, pred, confidence, margin, loss, correct）全部存在 |
| AC-2.4 | `run_multiseed_eval.py` 输出 split-sensitivity 报告 | 固定 train_seed，多个 split_seed 跑完后输出 `Accuracy across validation splits: mean ± std (worst: X.XXXX)` |
| AC-2.5 | 文档明确禁止 ensemble | 代码注释和 README 声明多种子仅用于评估稳定性，最终提交为单模型 |

### AC-3: 余弦分类器

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-3.1 | `CosineClassifier` 无 bias，参数结构正确 | `set(dict(classifier.named_parameters())) == {"weight", "logit_scale"}`；`not hasattr(classifier, "bias") or classifier.bias is None` |
| AC-3.2 | 权重缩放不变性（真正验证归一化在 forward 中生效） | 对同一输入，权重乘以 3.0 前后 logits 一致：`torch.allclose(logits_before, logits_after, atol=1e-5)` |
| AC-3.3 | logit_scale 可学习且约束有效 | `logit_scale.requires_grad == True`；一次 backward 后 `logit_scale.grad is not None` 且 `torch.isfinite(grad).all()`；`logit_scale.exp() <= 100` |
| AC-3.4 | `clamp_scale()` 在 step 后正确执行 | optimizer.step() 后调用 `clamp_scale()`，logit_scale 始终在 `[log(1), log(100)]` 范围内 |
| AC-3.5 | 在线模式可正常训练 | `python -m experiments.cosine_classifier.train --config configs/cosine_classifier.yaml` 跑 1 epoch，loss 下降 |
| AC-3.6 | 缓存模式可正常训练 | `python -m experiments.cosine_classifier.train ... --use-cached-features` 跑 1 epoch，loss 下降 |
| AC-3.7 | 推理产出合法提交文件 | 生成的 `pred_results.csv` 格式为 `img.jpg, 0001`，通过 `check_submission.py` 全部检查；类别名来自 `idx_to_class` lookup 而非 index 格式化 |

### AC-4: 数据增强

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-4.1 | A0 使用完整 CLIP deterministic preprocess | A0 transform 包含 convert_to_rgb → Resize(224,BICUBIC) → CenterCrop(224) → ToTensor → Normalize(CLIP)；与 `clip.load("ViT-B/32")` 返回的 preprocess 逐像素一致 |
| AC-4.2 | val_transform 始终为确定性 CLIP preprocess | 所有 config（A0-A3）的 val_transform 均为 A0 的确定性变换 |
| AC-4.3 | A1-A3 产生随机变化（非确定性） | 同一张图经 A1/A2/A3 transform 100 次，存在多个不同结果，像素方差 > 设定阈值 |
| AC-4.4 | A0 与 baseline 精度一致（同条件） | 相同 split_seed, train_seed, batch_size, lr, optimizer 下：首个 batch 输入 tensor 完全相同（`torch.allclose(atol=1e-6)`），最终 accuracy 基本一致 |
| AC-4.5 | RandomErasing 在 Normalize 之后 | A3 transform 中 `RandomErasing` 位于 `Normalize` 之后，`value=0` 对应归一化空间均值填充 |
| AC-4.6 | 四组实验均可正常训练 | A0-A3 各跑 1 epoch，loss 正常下降 |

### AC-5: 工程与回归

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-5.1 | Baseline 不受影响 | `python -m experiments.baseline.train --config configs/baseline.yaml` 行为不变（除 evaluate.py 新增 JSON 输出外） |
| AC-5.2 | 所有新模块可 import | 所有 `experiments/*/` 下模块 `__init__.py` 正确 |
| AC-5.3 | `num_classes: auto` 正确推断 | 从 `class_to_idx.json` 自动获取 num_classes，无需手动配置 |
| AC-5.4 | `pretrained_source` 被记录和验证 | build_model 时打印 `Backbone: CLIP ViT-B/32 | Pretrained source: OpenAI official`；manifest 中记录该字段 |
| AC-5.5 | 提交格式完整合规 | `pred_results.csv` 格式 `img.jpg, 0001`；`submission.zip` 仅包含 `pred_results.csv`；类名来自 `idx_to_class` 查找 |
