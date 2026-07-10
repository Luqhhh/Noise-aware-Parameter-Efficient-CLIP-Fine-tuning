# Baseline Improvements Design

**Date**: 2026-07-10
**Status**: approved
**Scope**: 4 improvements — data augmentation, cosine classifier, feature caching, multi-split validation

## Overview

Four independent improvements to the CLIP ViT-B/32 fine-grained classification baseline.
Each improvement is an independent experiment with its own config.

### Key Design Decisions

- **`split_seed`** and **`train_seed`** are separate concepts:
  - `split_seed` controls train/val data partitioning
  - `train_seed` controls model init, DataLoader shuffle, augmentation, CUDA randomness
- **Feature caching**: encode the FULL training set once, then index by split CSV — not per-split caches
- **Multi-split validation is a diagnostic tool, NOT an ensemble**: final submission uses ONE single-seed model
- **Ablation order**: fix one variable at a time (head type → augmentation level → combination)

---

## Revised Execution Order

```
Step 0: Reproduce baseline with fixed split_seed=42, train_seed=42
Step 1: Stratified split + separate seed control + isolated output dirs
Step 2: Cache deterministic CLIP features on FULL training set (with manifest + class mapping)
Step 3: Dev split (split_seed=42): compare Linear vs Cosine under identical conditions (both A0)
Step 4: Dev split (split_seed=42): fix Linear head, compare A0 / A1 / A2 / A3
Step 5: Dev split (split_seed=42): combine Cosine + best augmentation
Step 6: Confirm splits (split_seed=3407, 2026): verify top-2 candidates only, report paired deltas
Step 7: Retrain final model with PRE-DECLARED train_seed=42, generate pred_results.csv, submit
```

**Dev/confirm split strategy**:
- `split_seed=42`: development — used for all preliminary search and screening
- `split_seed=3407, 2026`: confirmation — only top-2 candidate methods are re-evaluated here
- This controls compute cost and avoids overfitting all hyperparameters to a single split

**Final seed selection policy**:
- Final submission model uses **pre-declared** `train_seed=42`
- Multiple seeds are used ONLY for stability measurement, NOT for cherry-picking the best seed
- Do NOT select train_seed based on validation or test-set performance

### Ablation Table

| Exp ID | Head | Augmentation | Answers |
|--------|------|-------------|---------|
| E0 | Linear | A0 | Baseline reproduction |
| E1 | Cosine | A0 | Is cosine head independently beneficial? |
| E2 | Linear | A1 | Is RandomResizedCrop+Flip independently beneficial? |
| E3 | Linear | A2 | Does ColorJitter add further gain? |
| E4 | Linear | A3 | Does RandomErasing add further gain? |
| E5 | Cosine | best of A0-A3 | Do head + augmentation gains stack? |

---

## Phase 1: Infrastructure

### 1a. Feature Caching

**Motivation**: Frozen CLIP backbone encodes the same image identically every epoch. Cache once, train the head many times.

**Architecture**: Cache the **full training set** once (before any split), then let `CachedFeatureDataset` select features by split CSV paths.

**Output** (`cache/clip_vit_b32_openai/`):
```
cache/clip_vit_b32_openai/
├── features.pt          # (N_full, 512) float32
├── labels.pt            # (N_full,) int64
├── paths.json           # ["0001/img1.jpg", ...]  ← dataset-root-relative POSIX paths
├── class_to_idx.json    # {"0001": 0, "0002": 1, ...}
├── idx_to_class.json    # {"0": "0001", "1": "0002", ...}
└── manifest.json        # metadata for reproducibility
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
  "num_classes": 500,
  "dataset_root": "data/preliminary/train",
  "class_mapping_hash": "<sha256 of sorted class_to_idx items>",
  "dataset_fingerprint": "<sha256 of [(rel_path, class_name, file_size), ...]>",
  "created_at": "2026-07-10T12:00:00"
}
```

**`dataset_fingerprint` computation**:
```python
# For each image: (relative_path, class_name, file_size)
# Sort by relative_path for determinism
records = []
for img_path in sorted(all_image_paths):
    rel_path = img_path.relative_to(dataset_root).as_posix()
    class_name = img_path.parent.name
    file_size = img_path.stat().st_size
    records.append((rel_path, class_name, file_size))

fingerprint = hashlib.sha256(
    json.dumps(records, sort_keys=True).encode()
).hexdigest()
```

**Path convention**: All paths in `paths.json` and split CSVs use **dataset-root-relative POSIX paths**:
```python
rel_path = image_path.relative_to(dataset_root).as_posix()
# Example: "0001/img1.jpg", "0123/some_image.png"
```
This makes caches portable across machines and Docker mounts.

**`class_mapping_hash`**: SHA256 of `sorted(class_to_idx.items())` serialized as JSON. On loading cached features, verify `cache_class_mapping_hash == current_class_mapping_hash` — refuse to train if they differ.

**New file: `tools/cache_features.py`**

Encodes every image in `data/preliminary/train/` (all classes, pre-split):
1. Loads CLIP ViT-B/32 (`pretrained_source: openai`), freezes backbone
2. Uses `model.eval()` + `torch.inference_mode()` for deterministic encoding
3. Calls `encode_image()` then `F.normalize(features.float(), dim=-1)`
4. Verifies: `not features.requires_grad`, `torch.isfinite(features).all()`, `features.norm(dim=-1) ≈ 1.0`
5. Saves features, labels, paths (relative), class_to_idx, idx_to_class, and manifest

**Normalization convention** (applies everywhere — cache AND online):
```python
features = model.encode_image(images)
features = F.normalize(features.float(), dim=-1)
```
This is the SINGLE normalization point. `CosineClassifier.forward()` re-normalizes features (mathematically idempotent for unit vectors, but explicit). Cache manifest records `normalized: true`.

**New file: `common/cached_dataset.py`**

```python
class CachedFeatureDataset(Dataset):
    """Loads cached features filtered by a split CSV."""
    def __init__(self, cache_dir, split_csv):
        # 1. Verify manifest: class_mapping_hash matches current
        # 2. Load full features.pt, labels.pt, paths.json
        # 3. Build relative_path → index lookup
        # 4. For each path in split_csv, select corresponding feature+label
        # Returns (feature, label, path_str)
```

**Second new file: `common/transforms.py`**

Centralized transform construction to prevent preprocess drift between experiments:
```python
def build_clip_eval_transform():
    """Deterministic CLIP preprocess — used by ALL val transforms and A0 train."""
    return clip.load("ViT-B/32")[1]  # official preprocess, or equivalent


def build_train_transform(preset: str):
    """Build training transform by preset name.
    preset ∈ {"a0", "a1", "a2", "a3"}
    a0 returns the same as build_clip_eval_transform()
    a1-a3 append augmentation before normalization.
    """
```

### 1b. Seed Separation & Output Isolation

**Config changes**:
```yaml
experiment:
  split_seed: 42
  train_seed: 42
```

**`scripts/split_data.py`** enhancement:
- Accept `--split_seed` to override config
- **Stratified** per-class split: shuffle images within each class, split by `val_ratio`
- **Small-class guard**: if a class has <2 samples, raise `ValueError` with class name and count
- **Post-split validation**:
  ```python
  assert set(train_paths).isdisjoint(val_paths)
  assert len(train_paths) + len(val_paths) == len(full_paths)
  assert len(train_paths) == len(set(train_paths))  # no duplicates in train
  assert len(val_paths) == len(set(val_paths))      # no duplicates in val
  ```
- Output to `{split_dir}/split_{split_seed}/`

**Output directory isolation**:
```
outputs/
└── {experiment}/
    └── split_{split_seed}/
        └── train_{train_seed}/
            ├── checkpoints/
            └── logs/
```

### 1c. Multi-Split Evaluation

**Experiment A — End-to-End Split Sensitivity** (priority):
```
split_seed=42,  train_seed=42
split_seed=3407, train_seed=42
split_seed=2026, train_seed=42
```
Answers: *"Is the end-to-end training and evaluation pipeline stable across different stratified noisy-label splits?"*

**Experiment B — Training Stability** (optional, compute permitting):
```
split_seed=42, train_seed=42
split_seed=42, train_seed=3407
split_seed=42, train_seed=2026
```

**Paired delta reporting** (mandatory for Experiment A):

For each split `i`, compute `Δ_i = Accuracy_best(split_i) − Accuracy_baseline(split_i)`:

```
split_seed    Baseline    Best    Δ
42            0.700       0.715   +0.015
3407          0.693       0.709   +0.016
2026          0.706       0.714   +0.008
```

Report:
```
Baseline accuracy:      0.700 ± 0.007 (worst: 0.693)
Best-method accuracy:   0.713 ± 0.003 (worst: 0.709)
Paired improvement Δ:  +0.013 ± 0.004 (worst: +0.008)
Wins: 3/3 splits
```

Use **sample std** (`ddof=1`). Always output all raw values alongside summary statistics.

**New file: `scripts/run_multisplit_eval.py`**

Runs split → train → evaluate for specified `split_seed` values with fixed `train_seed`, outputs paired comparison table.

### 1d. Evaluation Metrics Enhancement

**`experiments/baseline/evaluate.py`** — save detailed results JSON:

```json
{
  "overall": {
    "accuracy": 0.xxxx,
    "macro_accuracy": 0.xxxx,
    "loss": 0.xxxx,
    "total_samples": N
  },
  "per_class": {
    "0": {"accuracy": 0.xx, "total": N, "correct": N}
  },
  "per_sample": [
    {
      "path": "0001/img.jpg",
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
    """Cosine classifier with L2-normalized weights, no bias, optional learnable scale."""

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 500,
        init_scale: float = 10.0,
        learnable_scale: bool = True,
        min_scale: float = 1.0,
        max_scale: float = 100.0,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        self.learnable_scale = learnable_scale
        self.min_scale = min_scale
        self.max_scale = max_scale

        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)).log())
        else:
            self.register_buffer(
                "logit_scale", torch.tensor(float(init_scale)).log()
            )

        nn.init.normal_(self.weight, std=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        scale = self.logit_scale.exp().clamp(max=self.max_scale)
        return scale * features @ weight.t()

    def clamp_scale(self) -> None:
        """Call after optimizer.step() to keep scale in valid range.
        Prevents gradient dead-zone at clamp boundary in forward().
        """
        with torch.no_grad():
            self.logit_scale.clamp_(
                min=math.log(self.min_scale),
                max=math.log(self.max_scale),
            )
```

**Config**:
```yaml
model:
  architecture: ViT-B/32
  pretrained_source: openai
  feature_dim: 512
  freeze_clip: true
  num_classes: auto
  cosine:
    init_scale: 10.0
    learnable_scale: true
    min_scale: 1.0
    max_scale: 100.0
```

**AMP / DDP clamp_scale() placement**:
```python
# After scaler.step() + scaler.update() in the AMP path:
scaler.step(optimizer)
scaler.update()
model.classifier.clamp_scale()  # or model.module.classifier.clamp_scale() under DDP
```

**Verification — scale invariance** (the true test that forward() normalizes weights):
```python
features = torch.randn(8, 512)
logits_before = classifier(features)
with torch.no_grad():
    classifier.weight.mul_(3.0)
logits_after = classifier(features)
assert torch.allclose(logits_before, logits_after, atol=1e-5)
```

---

## Phase 3: Data Augmentation

**Experiment directory**: `experiments/augmentation/`
**Configs**: `configs/augmentation_a{0,1,2,3}.yaml`

### Transform Construction (centralized in `common/transforms.py`)

**A0 — Official CLIP preprocess (control)**:
```python
# A0 directly reuses the official transform returned by clip.load()
# This guarantees pixel-identical behavior with baseline.
A0_train_transform = build_clip_eval_transform()
A0_val_transform = build_clip_eval_transform()
```

**A1-A3 — Augmented transforms**:
```python
from torchvision.transforms import (
    Compose, RandomResizedCrop, RandomHorizontalFlip,
    ColorJitter, ToTensor, Normalize, RandomErasing, InterpolationMode,
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def _convert_to_rgb(image):
    return image.convert("RGB")

def build_train_transform(preset: str):
    if preset == "a0":
        return build_clip_eval_transform()  # reuse official preprocess

    # A1/A2/A3 — all use RandomResizedCrop + HorizontalFlip as base
    layers = [
        _convert_to_rgb,
        RandomResizedCrop(
            224, scale=(0.75, 1.0), ratio=(0.85, 1.15),
            interpolation=InterpolationMode.BICUBIC, antialias=True,
        ),
        RandomHorizontalFlip(p=0.5),
    ]

    if preset in ("a2", "a3"):
        layers.append(ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02,
        ))

    layers.append(ToTensor())
    layers.append(Normalize(mean=CLIP_MEAN, std=CLIP_STD))

    if preset == "a3":
        # After Normalize: value=0 fills with mean in normalized space
        layers.append(RandomErasing(
            p=0.1, scale=(0.02, 0.15), ratio=(0.5, 2.0), value=0,
        ))

    return Compose(layers)
```

**All val_transforms are `build_clip_eval_transform()`** (no augmentation during validation).

### Augmentation Grid

| Experiment | RandomResizedCrop | HorizontalFlip | ColorJitter | RandomErasing |
|---|---|---|---|---|
| A0 (control) | — | — | — | — |
| A1 | scale=(0.75,1.0), ratio=(0.85,1.15) | p=0.5 | — | — |
| A2 | same | p=0.5 | brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02 | — |
| A3 | same | p=0.5 | same | p=0.1, scale=(0.02,0.15), value=0 |

### model.py

Phase 3 initially uses baseline's `CLIPLinearClassifier` to isolate augmentation effect from classifier choice. After E0-E4 complete, E5 combines Cosine + best augmentation.

---

## Submission Format

**Output file**: `pred_results.csv` (NOT `pred_raw.csv`)

Use `csv.writer` — do NOT manually insert a space after the comma:
```python
import csv

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for image_name, pred_idx in predictions:
        class_name = idx_to_class[str(pred_idx)]
        # Validation:
        assert class_name == class_name.strip()
        assert len(class_name) == 4
        assert class_name.isdigit()
        writer.writerow([image_name, class_name])
# Produces: img.jpg,0001
```

**No header row** in the output CSV.

**Zip**: `submission.zip` contains exactly one file — `pred_results.csv` at root level.

---

## Multi-Split ≠ Ensemble (Explicit Prohibition)

Multi-split training is used ONLY for evaluating stability and selecting hyperparameters.
The final submission MUST use a single model trained with the **pre-declared** `train_seed=42` and
ONE `split_seed=42` (the full training pipeline, not validation-only).

Do NOT:
- Average, vote, or fuse logits/predictions across multiple seeds
- Cherry-pick the best-performing seed after seeing validation or test-set results
- Use test-set submission scores to select among seeds

---

## CLIP Weight Source Enforcement

- Config MUST specify `model.pretrained_source: openai`
- `build_model()` logs: `Backbone: CLIP ViT-B/32 | Pretrained source: OpenAI official`
- Cache `manifest.json` records `pretrained_source` — features from different backbones are incompatible
- OpenCLIP / LAION weights are explicitly prohibited per competition rules

---

## num_classes Auto-Inference

`num_classes: auto` infers from `class_to_idx.json` at runtime:
```python
with open(split_dir / "class_to_idx.json") as f:
    class_to_idx = json.load(f)
num_classes = len(class_to_idx)
```

Inference outputs class names directly from `idx_to_class` lookup, NOT by formatting integer indices.

---

## Files Changed / Created

### New Files
- `tools/cache_features.py`
- `common/cached_dataset.py`
- `common/transforms.py`
- `scripts/run_multisplit_eval.py`
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
- `scripts/split_data.py` — add `--split_seed`, stratified split with small-class guard, post-split validation, seed-specific output dirs
- `experiments/baseline/evaluate.py` — add per-sample + per-class metrics JSON output
- `experiments/baseline/model.py` — add `pretrained_source` logging and backbone verification
- `experiments/baseline/train.py` — read `experiment.split_seed` / `experiment.train_seed`, locate split dir by split_seed, write outputs to isolated dirs, auto-infer num_classes from class_to_idx
- `experiments/baseline/infer.py` — use `train_seed` for set_seed(); load idx_to_class from checkpoint's split; output class names via idx_to_class lookup to `pred_results.csv` using csv.writer
- `common/submission.py` — use `csv.writer` instead of manual `f"{img_name}, {pred_label}\n"`; add class name validation
- `configs/baseline.yaml` — add `experiment.split_seed`, `experiment.train_seed`, `model.pretrained_source`, `model.num_classes: auto`

### Unchanged
- `common/dataset.py`
- `common/utils.py` — `set_seed()` stays as-is; callers pass the appropriate seed

---

## Acceptance Criteria（验收方案）

### AC-1: 特征缓存

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-1.1 | 缓存脚本正确编码全量数据 | `features.pt` shape `(N_full, 512)`, `labels.pt` shape `(N_full,)`, `paths.json` 长度一致；所有路径为 dataset-root-relative POSIX 格式 |
| AC-1.2 | 缓存特征与在线编码一致 | 使用 `model.eval()` + `torch.inference_mode()`；验证 `not features.requires_grad`, `torch.isfinite(features).all()`, `features.norm(dim=-1) ≈ 1.0`；与在线编码 max abs diff < 1e-5 |
| AC-1.3 | `CachedFeatureDataset` 按 split CSV 正确索引 | 给定 `split_csv`，返回的特征、标签与在线 dataset 完全一致 |
| AC-1.4 | `manifest.json` 完整 | `backbone`, `pretrained_source`, `feature_dim`, `normalized`, `dtype`, `preprocess`, `dataset_size`, `num_classes`, `dataset_root`, `class_mapping_hash`, `dataset_fingerprint`, `created_at` 全部存在 |
| AC-1.5 | `class_mapping_hash` 不一致时拒绝训练 | 修改 `class_to_idx` 后尝试用旧缓存训练，应立即报错退出 |
| AC-1.6 | `class_to_idx.json` 和 `idx_to_class.json` 已保存 | 缓存目录包含完整的类别映射文件 |
| AC-1.7 | `dataset_fingerprint` 基于 (relative_path, class_name, file_size) | 替换一张图片后 fingerprint 发生变化 |
| AC-1.8 | 缓存模式训练加速（性能目标，非 correctness gate） | 记录在线与缓存模式的 epoch 时间及加速比；缓存模式应显著降低训练时间；目标 ≥10× |

### AC-2: 种子分离与多划分验证

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-2.1 | `split_seed` 产生不同且分层的划分 | 不同 split_seed 的 train.csv/val.csv 完全不同；每个类别在 train 和 val 中均有 ≥1 样本 |
| AC-2.2 | 极小类别（<2 samples）报错 | 构造含单样本类别的数据集，split 时应抛出 `ValueError` |
| AC-2.3 | split 后验证无泄漏 | `train ∩ val = ∅`, `train ∪ val = full`, train/val 各自无重复 |
| AC-2.4 | 输出目录按 `split_{split_seed}/train_{train_seed}/` 隔离 | 不同组合的 checkpoint 和日志互不覆盖 |
| AC-2.5 | 评估 JSON 包含所有定义字段 | `overall`（accuracy, macro_accuracy, loss, total_samples）, `per_class`, `per_sample`（path, label, pred, confidence, margin, loss, correct）全部存在 |
| AC-2.6 | `run_multisplit_eval.py` 输出配对比较报告 | 固定 train_seed，多个 split_seed 跑完后输出 Baseline mean±std、Best mean±std、Paired Δ mean±std、win count (X/3)、worst paired delta |
| AC-2.7 | 文档明确禁止 ensemble + 最终 seed 预声明 | 代码注释和 README 声明：多种子仅用于评估，最终提交使用预声明的 train_seed=42 单模型 |

### AC-3: 余弦分类器

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-3.1 | `CosineClassifier` 无 bias，参数名正确 | `learnable_scale=True` 时 `set(dict(classifier.named_parameters())) == {"weight", "logit_scale"}`；`learnable_scale=False` 时不含 `logit_scale`（为 buffer）；`not hasattr(classifier, "bias") or classifier.bias is None` |
| AC-3.2 | 权重缩放不变性（验证 forward 中确实归一化权重） | 对同一输入，`classifier.weight` 乘以 3.0 后 logits 不变：`torch.allclose(logits_before, logits_after, atol=1e-5)` |
| AC-3.3 | `learnable_scale=True` 时 logit_scale 有梯度 | `logit_scale.requires_grad == True`；一次 backward 后 `logit_scale.grad is not None` 且 `torch.isfinite(grad).all()` |
| AC-3.4 | `learnable_scale=False` 时 scale 固定 | 训练前后 `logit_scale.exp()` 值不变 |
| AC-3.5 | `clamp_scale()` 约束有效 | optimizer.step() 后调用 `clamp_scale()`，`logit_scale` 始终在 `[log(min_scale), log(max_scale)]` 范围内 |
| AC-3.6 | 在线/缓存模式均可正常训练 | 各跑 1 epoch，loss 下降 |
| AC-3.7 | 推理产出合法提交文件 | `pred_results.csv` 由 `csv.writer` 生成，格式 `img.jpg,0001`（无多余空格），无 header；通过 `check_submission.py` 全部检查；类名来自 `idx_to_class` lookup 并通过 `isdigit()` + len==4 验证 |

### AC-4: 数据增强

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-4.1 | A0 和 val 使用 `build_clip_eval_transform()` 构造 | Baseline train transform = A0 train transform = 所有 val transform，均由同一函数构造；抽样验证 `torch.allclose(output_a, output_b, atol=1e-6, rtol=0)` |
| AC-4.2 | val_transform 始终为确定性 CLIP preprocess | 所有 config（A0-A3）的 val_transform 均通过 `build_clip_eval_transform()` 构造 |
| AC-4.3 | A1-A3 产生随机变化（非确定性） | 同一张图经 A1/A2/A3 transform 100 次，SHA256 hash 集合 size > 1；所有输出 shape 为 (3, 224, 224) 且值有限 |
| AC-4.4 | A0 与 baseline 精度一致（同条件） | 相同 split_seed, train_seed, batch_size, lr, optimizer 下：首个 batch 输入 tensor 完全相同（`torch.allclose(atol=1e-6)`），最终 accuracy 基本一致 |
| AC-4.5 | RandomErasing 在 Normalize 之后，value=0 | A3 transform 中 `RandomErasing` 位于 `Normalize` 之后 |
| AC-4.6 | 四组实验均可正常训练 | A0-A3 各跑 1 epoch，loss 正常下降 |

### AC-5: 工程与回归

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-5.1 | Baseline 重构后可复现 | 使用原 checkpoint、split_seed=42、train_seed=42、相同参数：数据量、类别映射、首个 batch 输入和首轮 loss 与重构前一致 |
| AC-5.2 | 所有新模块可 import | 所有 `experiments/*/` 下模块正确导入 |
| AC-5.3 | `num_classes: auto` 正确推断 | 从 `class_to_idx.json` 自动获取 num_classes，无需手动配置 |
| AC-5.4 | `pretrained_source` 被记录和验证 | `build_model()` 打印 `Backbone: CLIP ViT-B/32 \| Pretrained source: OpenAI official`；manifest 中记录该字段 |
| AC-5.5 | 提交格式完整合规 | `pred_results.csv` 无 header，格式 `img.jpg,0001`；`submission.zip` 仅包含 `pred_results.csv`；类名来自 `idx_to_class` 查找；类名通过 strip/len==4/isdigit 验证 |
