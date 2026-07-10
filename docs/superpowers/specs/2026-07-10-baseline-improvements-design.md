# Baseline Improvements Design

**Date**: 2026-07-10
**Status**: approved
**Scope**: 4 improvements — data augmentation, cosine classifier, feature caching, multi-split validation

## Overview

Four independent improvements to the CLIP ViT-B/32 fine-grained classification baseline.

### Key Design Decisions

- **`split_seed`** and **`train_seed`** are separate concepts
- **Feature caching**: encode FULL training set once, index by split CSV
- **Multi-split is diagnostic, NOT ensemble**: final submission = ONE model on FULL training set
- **Ablation order**: fix one variable at a time (head → augmentation → combination)
- **Dev/confirm/final-fit**: search on split_42, verify top-2 on 3407+2026, final-fit on full data
- **Canonical class mapping**: generated once from full training directory, stored at `data/preliminary/metadata/`, used by all stages (split, cache, training, inference)

---

## Revised Execution Order

```
Step 0: Save baseline regression fixture; reproduce split_seed=42, train_seed=42
Step 1: Seed separation + canonical class mapping + isolated output dirs
Step 2: Cache deterministic CLIP features on FULL training set (content SHA256 + class mapping)
Step 3: Dev split (seed=42): E0 (Linear+A0) vs E1 (Cosine+A0) — controlled + equal-budget
Step 4: Dev split (seed=42): fix Linear head, compare E0/E2/E3/E4 (A0/A1/A2/A3)
Step 5: Dev split (seed=42): E5 (Cosine + best augmentation); select top-2 candidates
Step 6: Confirm splits (seed=3407,2026): E0 + candidate-1 + candidate-2 per split; paired deltas
Step 7: Pre-specified rule selects final method and fixed epoch count
Step 8: Final-fit on FULL training set, train_seed=42, no val, no early stopping
Step 9: Generate pred_results.csv, validate, compress, submit
```

### Canonical Class Mapping

Generated once from the full training directory (before any split), stored at `data/preliminary/metadata/`:

```python
class_names = sorted(
    p.name for p in Path(train_dir).iterdir() if p.is_dir()
)
class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {str(i): name for name, i in class_to_idx.items()}

# Save to:
#   data/preliminary/metadata/class_to_idx.json
#   data/preliminary/metadata/idx_to_class.json
```

Config references:
```yaml
data:
  class_mapping_path: data/preliminary/metadata/class_to_idx.json
```

**All stages use the same canonical mapping**: split generation, feature caching, dev/confirm training, final-fit, inference. The mapping is also embedded in every checkpoint.

### Dev / Confirm / Final-Fit Strategy

| Stage | split_seed | Data | Purpose |
|-------|-----------|------|---------|
| Dev | 42 | train.csv (90%) | Search, screen, tune |
| Confirm | 3407, 2026 | train.csv (90%) | Verify top-2 only |
| Final-fit | N/A | FULL training set | Submission model |

`split_seed` applies only to dev and confirm stages. Final-fit uses the complete official training set with no split.

### Candidate & Epoch Selection Rules (pre-specified, no post-hoc tuning)

**Method selection**:
1. Compare top-2 candidates on confirm splits by mean paired delta vs baseline
2. If |Δ_candidate1 − Δ_candidate2| < 0.1pp → select the structurally simpler method
3. If any candidate degrades >0.2pp on any confirm split vs baseline → eliminated
4. Test-set submission scores are NOT used for selection

**Epoch selection** (fixed before confirm):
1. On dev split (seed=42), record `best_epoch = argmax(val_acc)`
2. Use this `best_epoch` for all confirm-split training — no per-split epoch tuning
3. After confirm, epoch count is frozen — no further adjustment
4. Final-fit uses the same frozen epoch count

**Note on epoch vs steps**: Final-fit preserves epoch count, not optimizer steps. With val_ratio=0.1, final-fit performs ~1.11× more steps than dev (90%→100% data). This is acceptable and simpler than matching step counts exactly.

### Ablation Fairness Protocol

**Controlled ablation** (E0 vs E1, E0 vs E2/E3/E4):
Identical split, train_seed, optimizer, lr, wd, epochs, batch_size, scheduler. Only the variable under test changes. Answers: "Does this change help under the baseline training config?"

**Equal-budget tuning** (separate, after controlled):
Same grid for both heads being compared:
- `lr ∈ {1e-3, 3e-3, 1e-2}`
- `weight_decay ∈ {0.0, 1e-4, 1e-3}`
- Cosine additionally: `learnable_scale ∈ {false, true}`, `init_scale ∈ {10, 20}`

Answers: "With equal tuning budget, which configuration wins?"

**Augmentation experiments** (E0/E2/E3/E4): all use the SAME training config — no per-augmentation lr/wd re-tuning. Otherwise gains cannot be purely attributed to augmentation.

---

## Phase 1: Infrastructure

### 1a. Feature Caching

**New file: `common/clip_utils.py`** — CLIP loading and feature encoding (separate from transforms):

```python
import contextlib
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


def load_openai_clip(device: torch.device, model_name: str = "ViT-B/32"):
    """Load OpenAI CLIP once per process. Returns (model, preprocess)."""
    model, preprocess = clip.load(model_name, device=device, jit=False)
    return model, preprocess


@torch.no_grad()
def encode_frozen_clip_features(
    clip_model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    """Encode images through FROZEN CLIP backbone. Returns L2-normalized float32 features.

    This function is ONLY for freeze_clip=True experiments. It uses @torch.no_grad() —
    gradient-based fine-tuning methods (LoRA, adapters, partial unfreeze) must use a
    different encoding path.
    """
    with torch.autocast(device_type=device.type, enabled=use_amp):
        features = clip_model.encode_image(images)
    return F.normalize(features.float(), dim=-1)
```

Note: `CosineClassifier.forward()` performs a defensive re-normalization of features (mathematically idempotent for unit vectors, but explicit). The single canonical normalization point is in `encode_frozen_clip_features()`.

**Output** (`cache/clip_vit_b32_openai/`):
```
cache/clip_vit_b32_openai/
├── features.pt
├── labels.pt
├── paths.json           # dataset-root-relative POSIX paths
├── class_to_idx.json    # from canonical mapping
├── idx_to_class.json
└── manifest.json
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
  "dataset_fingerprint": "<sha256 of [(rel_path, class_name, file_size, content_sha256), ...]>",
  "created_at": "..."
}
```

`dataset_fingerprint`: content SHA256 per image — detects ANY file change (content, name, size, addition, deletion).

`class_mapping_hash`: SHA256 of `sorted(class_to_idx.items())` serialized as JSON.

**New file: `common/cached_dataset.py`**

```python
class CachedFeatureDataset(Dataset):
    def __init__(self, cache_dir, split_csv, class_to_idx_path):
        # 1. Load manifest; verify class_mapping_hash matches current
        # 2. Load cached class_to_idx, idx_to_class
        #    assert current_class_to_idx == cached_class_to_idx
        #    assert current_idx_to_class == cached_idx_to_class
        # 3. Load features.pt, labels.pt, paths.json
        # 4. Verify: len(paths) == len(set(paths))
        # 5. Build relative_path → index lookup
        # 6. For each row in split_csv:
        #    - find feature via path → index
        #    - assert cached_labels[index] == split_label
```

### 1b. Transform Construction

**New file: `common/transforms.py`** (thin module — CLIP loading is in `clip_utils.py`):

```python
from torchvision.transforms import (
    Compose, RandomResizedCrop, RandomHorizontalFlip,
    ColorJitter, ToTensor, Normalize, RandomErasing, InterpolationMode,
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _convert_to_rgb(image):
    return image.convert("RGB")


def build_train_transform(preset: str, clip_eval_transform):
    """preset ∈ {"a0", "a1", "a2", "a3"}.
    a0 returns clip_eval_transform directly (official CLIP preprocess from load_openai_clip).
    """
    if preset == "a0":
        return clip_eval_transform

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
        layers.append(RandomErasing(
            p=0.1, scale=(0.02, 0.15), ratio=(0.5, 2.0), value=0,
        ))

    return Compose(layers)
```

`clip_eval_transform` is the preprocess returned by `load_openai_clip()`, passed in from outside.

### 1c. CLIP Backbone eval-mode Enforcement

```python
class CLIPClassifier(nn.Module):
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_clip:
            self.clip_model.eval()
        return self
```

### 1d. Seed Separation & DataLoader Reproducibility

```yaml
experiment:
  mode: dev                 # dev | confirm | final_fit
  split_seed: 42
  train_seed: 42
```

**DataLoader with explicit seed control**:
```python
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

generator = torch.Generator()
generator.manual_seed(train_seed)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=generator,
    pin_memory=True,
    drop_last=True,
)
```

**Verification**: Same split_seed + train_seed → two consecutive runs → first batch paths, labels, input checksum are identical.

**`scripts/split_data.py`** enhancement:
```python
n_val = max(1, round(n_samples * val_ratio))
n_val = min(n_val, n_samples - 1)
```
Post-split validation: no overlap, no duplicates, full coverage, small-class guard.

**Output directory isolation**:
```
outputs/{experiment}/
├── split_{split_seed}/train_{train_seed}/
└── final_fit/train_{train_seed}/
```

### 1e. Multi-Split Evaluation

**Experiment A — End-to-End Split Sensitivity** (priority):
```
split_seed=42,  train_seed=42
split_seed=3407, train_seed=42
split_seed=2026, train_seed=42
```

**Paired delta reporting**:

```
split_seed    Baseline    Candidate    Δ
42            0.700       0.715        +0.015
3407          0.693       0.709        +0.016
2026          0.706       0.714        +0.008

Confirmation (seed=3407,2026): wins 2/2, mean Δ +0.012, worst Δ +0.008
Pooled (seed=42,3407,2026):   wins 3/3, mean Δ +0.013, worst Δ +0.008
```

**Dev split is NOT counted as independent confirmation** — it was used for candidate selection. Confirmation wins use denominator 2. Pooled is descriptive only.

**New file: `scripts/run_multisplit_eval.py`**

### 1f. Evaluation Metrics JSON

Same as before: overall (accuracy, macro_accuracy, loss, total_samples), per_class, per_sample (path, label, pred, confidence, margin, loss, correct).

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

        if min_scale <= 0:
            raise ValueError(f"min_scale must be positive, got {min_scale}")
        if max_scale < min_scale:
            raise ValueError(f"max_scale ({max_scale}) must be >= min_scale ({min_scale})")
        if not min_scale <= init_scale <= max_scale:
            raise ValueError(
                f"init_scale ({init_scale}) must be in [{min_scale}, {max_scale}]"
            )

        self.learnable_scale = learnable_scale
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))

        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)).log())
        else:
            self.register_buffer("logit_scale", torch.tensor(float(init_scale)).log())

        nn.init.normal_(self.weight, std=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        scale = self.logit_scale.exp().clamp(min=self.min_scale, max=self.max_scale)
        return scale * features @ weight.t()

    def clamp_scale(self) -> None:
        if not self.learnable_scale:
            return
        with torch.no_grad():
            self.logit_scale.clamp_(
                min=math.log(self.min_scale),
                max=math.log(self.max_scale),
            )
```

**Optimizer — conditional construction**:
```python
param_groups = [
    {"params": [classifier.weight], "weight_decay": weight_decay},
]
if classifier.learnable_scale:
    param_groups.append(
        {"params": [classifier.logit_scale], "weight_decay": 0.0},
    )
optimizer = AdamW(param_groups, lr=lr)
```

### Checkpoint Metadata

Every checkpoint saves complete inference metadata:
```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "architecture": "ViT-B/32",
    "pretrained_source": "openai",
    "head_type": "cosine",           # or "linear"
    "augmentation_preset": "a2",     # or "a0"
    "class_to_idx": class_to_idx,
    "idx_to_class": idx_to_class,
    "feature_dim": 512,
    "num_classes": num_classes,
    "train_seed": train_seed,
    "split_seed": split_seed,        # None for final_fit
    "config": resolved_config,
}
```

Inference reads `class_to_idx` / `idx_to_class` from the checkpoint — no dependency on external split directories.

---

## Phase 3: Data Augmentation

As before — `experiments/augmentation/` with configs `augmentation_a{0,1,2,3}.yaml`. Phase 3 uses baseline's `CLIPLinearClassifier`. Transforms built by `build_train_transform(preset, clip_eval_transform)`.

---

## Final-Fit Mode

**Config**:
```yaml
experiment:
  mode: final_fit
  train_seed: 42

data:
  use_full_training_set: true
  train_dir: data/preliminary/train

train:
  epochs: 20    # Frozen — determined before confirm, NOT adjusted after
```

**Behavior**:
- `TrainImageDataset(split_csv=None)` → scans all class directories via `_load_from_directory()`
- No validation: no val loader, no `best.pt`, no early stopping
- Saves only `last.pt` at final epoch
- Output: `outputs/{experiment}/final_fit/train_{train_seed}/`

**Epoch count**: The `best_epoch` selected on dev split (seed=42) is frozen before confirm and reused unchanged through final-fit. Final-fit performs ~1.11× more optimizer steps than dev (100%/90% data) — this is acceptable and simpler than step-count matching.

---

## Submission

`csv.writer`, no header, `img.jpg,0001` format. `submission.zip` contains only `pred_results.csv`.

---

## Files Changed / Created

### New Files
- `common/clip_utils.py` — `load_openai_clip()`, `encode_frozen_clip_features()`
- `common/transforms.py` — `build_train_transform()`
- `common/cached_dataset.py` — `CachedFeatureDataset`
- `tools/cache_features.py`
- `data/preliminary/metadata/class_to_idx.json` — canonical mapping (generated)
- `data/preliminary/metadata/idx_to_class.json` — canonical mapping (generated)
- `scripts/run_multisplit_eval.py`
- `experiments/cosine_classifier/__init__.py`, `model.py`, `train.py`, `evaluate.py`, `infer.py`
- `experiments/augmentation/__init__.py`, `model.py`, `train.py`, `evaluate.py`, `infer.py`
- `configs/cosine_classifier.yaml`
- `configs/augmentation_a0.yaml`, `augmentation_a1.yaml`, `augmentation_a2.yaml`, `augmentation_a3.yaml`
- `tests/fixtures/baseline_reference.json`

### Modified Files
- `scripts/split_data.py`
- `experiments/baseline/evaluate.py`
- `experiments/baseline/model.py` — `CLIPClassifier.train()` eval enforcement, `pretrained_source` logging
- `experiments/baseline/train.py` — seed separation, seed_worker+Generator, canonical mapping, isolated dirs, final_fit mode, num_classes auto, checkpoint metadata
- `experiments/baseline/infer.py` — read mapping from checkpoint, csv.writer output
- `common/submission.py` — csv.writer, class name validation
- `configs/baseline.yaml`

### Unchanged
- `common/dataset.py` — `TrainImageDataset(split_csv=None)` already scans all class directories via `_load_from_directory()`; no modification needed for full-data support
- `common/utils.py`

---

## Acceptance Criteria

### AC-1: 特征缓存

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-1.1 | 缓存正确编码全量数据 | `features.pt` `(N_full,512)`, `labels.pt` `(N_full,)`, `paths.json` 长度一致；路径为 dataset-root-relative POSIX |
| AC-1.2 | 缓存特征与在线编码一致 | `encode_frozen_clip_features()` 统一编码路径；`model.eval()` + `torch.inference_mode()`；`torch.allclose(atol=1e-5, rtol=1e-5)` |
| AC-1.3 | `CachedFeatureDataset` 三层校验 | (1) class_mapping_hash 一致 (2) cached mapping == current mapping (3) per-sample label 一致 |
| AC-1.4 | `manifest.json` 完整 | 所有字段存在 |
| AC-1.5 | class_mapping_hash 不一致时拒绝训练 | 修改映射后用旧缓存训练 → 报错退出 |
| AC-1.6 | 缓存目录包含规范类别映射文件 | `class_to_idx.json` + `idx_to_class.json`；与当前规范映射完全相等 |
| AC-1.7 | `dataset_fingerprint` 基于 content SHA256 | 替换、增加、删除、修改任何图片 → fingerprint 变化 |
| AC-1.8 | 缓存路径无重复 | `len(paths) == len(set(paths))` |
| AC-1.9 | 缓存模式训练加速（性能目标） | 记录加速比；目标 ≥10× |

### AC-2: 种子分离与多划分验证

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-2.1 | split_seed 产生不完全相同的分层划分 | `train_paths` 不相等；`val_paths` 不相等；≥1 类别 val 集合变化；每类别 train/val 均 ≥1 样本 |
| AC-2.2 | <2 样本类别报错 | `ValueError` |
| AC-2.3 | 无泄漏无重复 | train∩val=∅, train∪val=full, 各自无重复 |
| AC-2.4 | 输出目录隔离 | `split_{split_seed}/train_{train_seed}/` |
| AC-2.5 | 确定性复现 | 相同 split_seed + train_seed 连续两次运行：首个 batch paths/labels/input_checksum 一致 |
| AC-2.6 | 评估 JSON 完整 | overall, per_class, per_sample 所有字段存在 |
| AC-2.7 | 配对报告区分 confirmation 和 pooled | Confirmation wins X/2；Pooled wins X/3；dev split 不算入独立确认 |
| AC-2.8 | final_fit 使用全量数据，无 val | `use_full_training_set: true` → 加载所有官方样本，无 split CSV，无 val loader，无 early stopping |
| AC-2.9 | 规范类别映射独立于 split | `data/preliminary/metadata/class_to_idx.json` 由全量训练目录生成；所有阶段引用同一份映射 |

### AC-3: 余弦分类器

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-3.1 | 无 bias，参数名正确 | learnable_scale=True: `named_parameters()` = {weight, logit_scale}；False: logit_scale 为 buffer |
| AC-3.2 | 权重缩放不变性 | weight × 3.0 后 logits `torch.allclose(atol=1e-5)` |
| AC-3.3 | learnable_scale=True 时 logit_scale 有梯度 | `requires_grad`, `grad is not None`, `torch.isfinite(grad).all()` |
| AC-3.4 | learnable_scale=False 时 scale 固定 | 训练前后 `logit_scale.exp()` 不变 |
| AC-3.5 | clamp_scale() 正确 | 可学习时 clamp 到 `[log(min), log(max)]`；固定时直接返回 |
| AC-3.6 | 参数范围验证 | min_scale≤0 / max_scale<min_scale / init_scale 越界 → ValueError |
| AC-3.7 | optimizer 参数组条件构造 | learnable_scale=True: logit_scale 在 optimizer 中且 wd=0；False: logit_scale 不出现在任何 param_group |
| AC-3.8 | 在线/缓存模式均可训练 | 各 1 epoch，loss 下降 |
| AC-3.9 | 推理合法 | csv.writer 生成，无 header，`img.jpg,0001`；通过 check_submission.py |

### AC-4: 数据增强

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-4.1 | A0/val 使用 clip_eval_transform（同一对象） | baseline = A0 = 所有 val transform，均为 `load_openai_clip()` 返回的 preprocess |
| AC-4.2 | build_train_transform 不内部加载 CLIP | preprocess 从外部传入 |
| AC-4.3 | A1-A3 随机变化 | 100 次 transform，SHA256 hash 集合 size > 1；shape (3,224,224)，isfinite |
| AC-4.4 | A0 与 baseline 精度一致 | 相同条件：首个 batch `torch.allclose(atol=1e-6)`；`abs(acc_a0 - acc_baseline) <= 0.001` |
| AC-4.5 | RandomErasing 在 Normalize 之后 | A3 transform 顺序验证 |
| AC-4.6 | 四组实验正常训练 | 各 1 epoch，loss 下降 |

### AC-5: 工程与回归

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-5.1 | Baseline 重构后可复现 | `tests/fixtures/baseline_reference.json`；`abs(current_loss - reference_loss) < 1e-4` |
| AC-5.2 | CLIP backbone 冻结时 eval | `model.clip_model.training == False`；`all(not p.requires_grad)`；`model.train()` 后 backbone 仍在 eval |
| AC-5.3 | 所有新模块可 import | import 无错误 |
| AC-5.4 | `num_classes: auto` 从规范映射推断 | 读取 canonical `class_to_idx.json`，无需手动配置 |
| AC-5.5 | `pretrained_source` 记录和验证 | build_model 打印；manifest 记录 |
| AC-5.6 | 提交合规 | `pred_results.csv` 无 header，`img.jpg,0001`；zip 仅含该文件；类名 strip/len==4/isdigit |
| AC-5.7 | checkpoint 包含完整推理元数据 | architecture, pretrained_source, head_type, augmentation_preset, class_to_idx, idx_to_class, feature_dim, num_classes, train_seed, config 全部保存 |
| AC-5.8 | 推理不依赖外部 split 目录 | infer.py 从 checkpoint 读取类别映射 |
| AC-5.9 | 消融公平性 | Controlled ablation: 同配置仅改单变量；Equal-budget: 同网格搜索；Augmentation: 不按增强调 lr/wd |
