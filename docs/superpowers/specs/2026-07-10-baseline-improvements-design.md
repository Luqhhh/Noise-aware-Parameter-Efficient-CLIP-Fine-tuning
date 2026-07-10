# Baseline Improvements Design

**Date**: 2026-07-10
**Status**: approved
**Scope**: 4 improvements — data augmentation, cosine classifier, feature caching, multi-seed validation

## Overview

Four independent improvements to the CLIP ViT-B/32 fine-grained classification baseline, executed in dependency order:

```
Phase 1: Infrastructure (no dependencies)
├── Feature caching (tools/cache_features.py + common/cached_dataset.py)
└── Multi-seed validation (split_data.py enhancement + run_multiseed_eval.py)

Phase 2: Cosine Classifier (can use cached features for fast head search)
└── experiments/cosine_classifier/

Phase 3: Data Augmentation (must train online; can use cosine classifier)
└── experiments/augmentation/
```

Each improvement is an independent experiment with its own config, following the project convention of `experiments/<name>/` + `configs/<name>.yaml`.

---

## Phase 1: Infrastructure

### 1a. Feature Caching

**Motivation**: When CLIP backbone is frozen, encoding every image every epoch is redundant compute. Pre-computing L2-normalized features enables:
- Fast classifier-head-only training (large batch sizes, many epochs)
- Rapid hyperparameter search (lr, wd, optimizer, loss function)
- Multi-seed experiments at low cost

**New file: `tools/cache_features.py`**

Script that:
1. Loads CLIP ViT-B/32 with frozen backbone
2. Iterates over train.csv / val.csv splits
3. Calls `encode_image()` to get L2-normalized 512-dim features
4. Saves features, labels, and paths to disk

**Output** (`outputs/{experiment}/features/`):
```
features/
├── train_features.pt    # (N_train, 512) float32
├── train_labels.pt      # (N_train,) int64
├── train_paths.json     # ["path/to/img1.jpg", ...]
├── val_features.pt      # (N_val, 512)
├── val_labels.pt
└── val_paths.json
```

Usage:
```bash
python3 tools/cache_features.py --config configs/baseline.yaml
```

**New file: `common/cached_dataset.py`**

`CachedFeatureDataset` — loads pre-computed features from disk, returns `(feature, label, path)` tuples. When used, training can run with very large batch sizes (4096+) since each sample is just a 512-dim vector.

**Key design constraint**: Feature caching is only valid for deterministic preprocessing. Experiments using random augmentation MUST use online encoding. The train.py in each experiment supports both modes via a `--use-cached-features` flag.

### 1b. Multi-Seed Validation

**Motivation**: The current single-seed (42) 10% validation split may misrepresent true accuracy because:
- Validation labels come from noisy training data
- A single split can over/under-sample noisy examples
- Model corrections of wrong labels may appear as validation errors

**split_data.py enhancement**:
- Add `--seed` CLI argument to override config's seed
- Output to `{split_dir}/seed_{seed}/` instead of flat `{split_dir}/`

**evaluate.py enhancement** — save detailed results JSON:

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
    ...
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
    },
    ...
  ]
}
```

**New file: `scripts/run_multiseed_eval.py`**

Runs the full pipeline (split → train → evaluate) for seeds 42, 3407, 2026 and reports mean ± std accuracy.

Usage:
```bash
python3 scripts/run_multiseed_eval.py --config configs/baseline.yaml --seeds 42,3407,2026
```

---

## Phase 2: Cosine Classifier

**Experiment directory**: `experiments/cosine_classifier/`
**Config**: `configs/cosine_classifier.yaml`

### model.py

`CosineClassifier` replaces `nn.Linear`:
- Weights are L2-normalized per class
- No bias term
- Learnable temperature scale (initialized at 10.0, clamped to max 100)

```python
class CosineClassifier(nn.Module):
    def __init__(self, feature_dim=512, num_classes=500, init_scale=10.0):
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        self.logit_scale = nn.Parameter(torch.tensor(init_scale).log())
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, features):
        features = F.normalize(features, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * features @ weight.t()
```

The `CLIPLinearClassifier` wrapper is identical to baseline except the head.

### train.py — Dual Mode

Supports two modes:
- **Online mode** (default): Full pipeline, CLIP runs each epoch. Same as baseline train.py.
- **Cached mode** (`--use-cached-features`): Uses `CachedFeatureDataset`, trains only the CosineClassifier head. Enables fast hyperparameter sweeps.

**Hyperparameter search space** (cached mode):
- lr ∈ {1e-3, 3e-3, 1e-2}
- weight_decay ∈ {1e-4, 1e-3}
- init_scale ∈ {5, 10, 15, 20}

### Config

```yaml
# configs/cosine_classifier.yaml
data: ...           # Same as baseline
model:
  clip_model_name: ViT-B/32
  feature_dim: 512
  freeze_clip: true
  num_classes: 500
  init_scale: 10.0   # Cosine classifier temperature
train:
  lr: 0.001
  batch_size: 128     # Online mode; cached mode can go higher
  ...
```

---

## Phase 3: Data Augmentation

**Experiment directory**: `experiments/augmentation/`
**Configs**: `configs/augmentation_a{0,1,2,3}.yaml`

### Design

A single train.py uses config flags to compose transforms. The validate transform is always the deterministic CLIP preprocess.

**Augmentation grid**:

| Experiment | RandomResizedCrop | HorizontalFlip | ColorJitter | RandomErasing |
|---|---|---|---|---|
| A0 (control) | — | — | — | — |
| A1 | scale=(0.75,1.0), ratio=(0.85,1.15) | p=0.5 | — | — |
| A2 | same | p=0.5 | brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02 | — |
| A3 | same | p=0.5 | same | p=0.1 |

**Design principles**:
- Conservative scale/ratio ranges to avoid cropping out discriminative regions (wing patterns, petal edges, head/tail morphology)
- Light color jitter — fine-grained classes can differ by subtle color cues
- Low RandomErasing probability — occlusion augmentation, not dominant
- Validation always uses deterministic CLIP preprocess (no augmentation)

### Transform Composition

```python
train_transforms = []
if config.augmentation.use_random_resized_crop:
    train_transforms.append(RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.85, 1.15)))
if config.augmentation.use_horizontal_flip:
    train_transforms.append(RandomHorizontalFlip(p=0.5))
# Convert to tensor
train_transforms.append(ToTensor())
if config.augmentation.use_color_jitter:
    train_transforms.append(ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02))
if config.augmentation.use_random_erasing:
    train_transforms.append(RandomErasing(p=0.1))
# CLIP normalization (extracted from CLIP's preprocess)
train_transforms.append(Normalize(mean=CLIP_MEAN, std=CLIP_STD))
```

### model.py

Phase 3 默认使用 baseline 的 `CLIPLinearClassifier`（`nn.Linear` 头），目的是将增强效果与分类器选择隔离——先确认增强是否有效，再与 cosine classifier 组合。后续可加 `--use-cosine-head` 开关切换到 `CosineClassifier`。

---

## Execution Strategy

### Dependency Graph

```
cache_features.py  ←── 独立，无依赖
split_data 增强      ←── 独立，无依赖
run_multiseed_eval  ←── 依赖 split_data 增强
cosine_classifier   ←── 可用缓存特征加速（可选依赖）
augmentation        ←── 可复用 baseline 或 cosine 模型；必须在线训练
```

### Recommended Order

1. **Feature caching + Multi-seed validation** (in parallel — both are independent infrastructure)
2. **Cosine classifier** — use cached features for hyperparameter sweep, then final online training
3. **Augmentation A0-A3** — use best classifier head from step 2, online training only
4. **Combine**: best augmentation × best classifier head

### Results Tracking

All experiments log to `results/ablation.csv` with columns:
`exp_id, method, backbone, head, freeze_clip, augmentation, lr, batch_size, epochs, val_acc_mean, val_acc_std, online_acc, ckpt_path, notes`

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
- `scripts/split_data.py` — add `--seed` flag, seed-specific output dirs
- `experiments/baseline/evaluate.py` — add per-sample + per-class metrics JSON output

### Unchanged
- `common/dataset.py` — existing `TrainImageDataset` / `TestImageDataset` unchanged
- `common/submission.py` — unchanged
- `common/utils.py` — unchanged (reuse existing utilities)
- `experiments/baseline/model.py` — unchanged
- `experiments/baseline/train.py` — unchanged
- `experiments/baseline/infer.py` — unchanged

---

## Acceptance Criteria（验收方案）

### AC-1: 特征缓存

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-1.1 | `tools/cache_features.py` 能正确缓存特征 | 对 baseline split 运行，输出 `train_features.pt` shape 为 `(N_train, 512)`，`train_labels.pt` shape 为 `(N_train,)`，`train_paths.json` 长度与特征一致 |
| AC-1.2 | 缓存特征与在线编码一致 | 用同一 batch 对比在线 `encode_image()` 输出和缓存加载的特征，max abs diff < 1e-5 |
| AC-1.3 | `CachedFeatureDataset` 可正常训练 | 用缓存特征训练一个 Linear 头 1 epoch，loss 正常下降 |
| AC-1.4 | `--use-cached-features` 模式 batch_size=4096 可行 | 无 OOM，训练速度显著快于在线模式（至少 10x per epoch） |

### AC-2: 多种子验证

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-2.1 | `split_data.py --seed X` 产生不同划分 | 对 seed=42, 3407, 2026 分别运行，验证 train.csv / val.csv 不完全相同 |
| AC-2.2 | 输出目录隔离 | 三个 seed 的 split 文件分别在 `splits/seed_42/`、`splits/seed_3407/`、`splits/seed_2026/` 下 |
| AC-2.3 | 评估 JSON 包含所有指标 | `overall accuracy`、`macro accuracy`、`per_class accuracy`、`per_sample`（含 confidence, margin, loss, correct）全部存在 |
| AC-2.4 | `run_multiseed_eval.py` 输出 mean ± std | 三个 seed 跑完后打印 `Val Acc: 0.xxxx ± 0.xxxx` |

### AC-3: 余弦分类器

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-3.1 | `CosineClassifier` 无 bias | 检查 `len(list(model.classifier.parameters())) == 2`（weight + logit_scale） |
| AC-3.2 | 权重逐类归一化 | 验证 `F.normalize(weight, dim=-1)` 后每行范数为 1.0 |
| AC-3.3 | logit_scale 可学习且在范围内 | 训练前后 scale 值变化，始终 ≤ 100 |
| AC-3.4 | 在线模式可正常训练 | `python -m experiments.cosine_classifier.train --config configs/cosine_classifier.yaml` 跑 1 epoch，loss 下降 |
| AC-3.5 | 缓存模式可正常训练 | `python -m experiments.cosine_classifier.train --config configs/cosine_classifier.yaml --use-cached-features` 跑 1 epoch，loss 下降 |
| AC-3.6 | 推理产出合法提交文件 | `python -m experiments.cosine_classifier.infer` 生成的 `pred_raw.csv` 通过 `check_submission.py` 全部 9 项检查 |

### AC-4: 数据增强

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-4.1 | 四组 config 产生不同 transform | 对同一张图分别用 A0-A3 的 train_transform 跑 100 次，A0 每次完全相同，A1-A3 每次有变化 |
| AC-4.2 | val_transform 始终是确定性 CLIP preprocess | 对同一张图用 val_transform 跑 10 次，结果完全一致 |
| AC-4.3 | 四组实验均可正常训练 | A0-A3 各跑 1 epoch，loss 正常下降 |
| AC-4.4 | A0 与 baseline 结果一致 | A0（无增强）的 val accuracy 应在 baseline 的 ±0.5% 以内 |
| AC-4.5 | A3 包含所有增强 | 验证 A3 config 下 RandomResizedCrop + Flip + ColorJitter + RandomErasing 全部启用 |

### AC-5: 回归验证

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-5.1 | Baseline 不受影响 | 原有 `python -m experiments.baseline.train --config configs/baseline.yaml` 行为不变 |
| AC-5.2 | 所有新模块可 import | `python -c "from experiments.cosine_classifier.model import CosineClassifier"` 等全部通过 |
| AC-5.3 | Smoke test 通过 | `bash tools/run_smoke_test.sh`（如果已存在）全部步骤通过 |
