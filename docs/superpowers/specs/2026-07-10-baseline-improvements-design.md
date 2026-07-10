# Baseline Improvements Design

**Date**: 2026-07-10
**Status**: approved
**Scope**: 4 improvements — data augmentation, cosine classifier, feature caching, multi-split validation

---

## Key Design Decisions

- **`split_seed`** / **`train_seed`** are separate concepts
- **Feature caching**: encode FULL training set once, index by split CSV
- **Multi-split is diagnostic, NOT ensemble**: final submission = ONE model on FULL training set
- **Ablation order**: fix one variable at a time (head → augmentation → combination)
- **Dev/confirm/final-fit**: search on split_42, verify top-2 on 3407+2026, final-fit on full data
- **Canonical class mapping**: generated once from full training directory at `data/preliminary/metadata/`, used by ALL stages
- **`drop_last=False` everywhere**: no samples silently discarded; especially critical for final_fit

---

## Revised Execution Order

```
Step 0: Save baseline regression fixture; reproduce split_seed=42, train_seed=42
Step 1: Seed separation + canonical class mapping + isolated output dirs
Step 2: Cache deterministic CLIP features on FULL training set (content SHA256 + class mapping)
Step 3: Dev split (seed=42): E0 (Linear+A0) vs E1 (Cosine+A0) — equal-budget comparison
Step 4: Dev split (seed=42): fix Linear head, compare E0/E2/E3/E4 (A0/A1/A2/A3)
Step 5: Dev split (seed=42): E5 (Cosine + best augmentation); select top-2 candidates
Step 6: Confirm splits (seed=3407,2026): E0 + candidate-1 + candidate-2 per split; paired deltas
Step 7: Pre-specified rule selects final method; epoch counts frozen per-method from Step 3-5
Step 8: Final-fit on FULL training set, train_seed=42, no val, no early stopping, drop_last=False
Step 9: Generate pred_results.csv, full coverage check, compress, submit
```

### Canonical Class Mapping

Generated once from the full training directory, stored at `data/preliminary/metadata/`:

```python
train_dir = Path("data/preliminary/train")
class_names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())

# Validate class directory names
for name in class_names:
    if len(name) != 4 or not name.isdigit():
        raise ValueError(f"Invalid class directory name: {name!r}")

expected_num_classes = 500
if len(class_names) != expected_num_classes:
    raise ValueError(
        f"Expected {expected_num_classes} classes, found {len(class_names)}"
    )

class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {str(i): name for name, i in class_to_idx.items()}
```

**Mapping file lifecycle**:
- Not exist → generate
- Exists and matches current directory listing → reuse
- Exists but inconsistent → error (requires explicit `--regenerate-class-mapping` to overwrite)

Config:
```yaml
data:
  class_mapping_path: data/preliminary/metadata/class_to_idx.json
```

All stages (split generation, caching, training, inference) reference the same canonical mapping.

### Dev / Confirm / Final-Fit Strategy

| Stage | split_seed | Data | drop_last |
|-------|-----------|------|-----------|
| Dev | 42 | train.csv (90%) | False |
| Confirm | 3407, 2026 | train.csv (90%) | False |
| Final-fit | N/A | FULL training set | False |

### Method-specific Epoch Freezing

Each method independently records its best epoch on dev split (seed=42). This epoch is frozen before confirmation and reused unchanged:

```
E0 (Linear+A0):       best_epoch =  8 on split_42 → frozen, used for confirm splits
E1 (Cosine+A0):       best_epoch = 13 on split_42 → frozen, used for confirm splits
E2 (Linear+A1):       best_epoch = 10 on split_42 → frozen
...
candidate-1: uses its own frozen epoch on 3407, 2026
candidate-2: uses its own frozen epoch on 3407, 2026
final model: uses its own frozen epoch in final_fit
```

No epoch tuning on confirm splits. Frozen epoch is recorded in checkpoint:
```json
{"selected_epoch": 13, "epoch_selection_split": 42}
```

### Candidate & Epoch Selection Rules

**Method selection**:
1. Compare top-2 candidates on confirm splits by mean paired delta vs baseline
2. If |Δ_c1 − Δ_c2| < 0.1pp → select the structurally simpler method
3. If any candidate degrades >0.2pp on any confirm split vs baseline → eliminated
4. Test-set submission scores are NOT used for selection

**Epoch selection** (per-method, frozen before confirm):
1. On dev split (seed=42): each method records `best_epoch = argmax(val_acc)`
2. Confirm splits use each method's own frozen epoch — no per-split epoch tuning
3. After confirm: epoch counts are frozen permanently
4. Final-fit uses the final method's frozen epoch

### Ablation Fairness Protocol

**True equal-budget**: Linear and Cosine each run the SAME number of trials in the main comparison.

```
Linear main search:   lr × wd = 3×3 = 9 trials
Cosine main search:   lr × wd = 3×3 = 9 trials  (init_scale=10, learnable_scale=True fixed)
```

Cosine scale experiments are a **separate internal ablation** reported independently:
```
C0: fixed scale=10
C1: learnable scale, init=10
C2: learnable scale, init=20
```

These do NOT participate in the equal-budget Linear-vs-Cosine comparison.

**Augmentation**: all E0/E2/E3/E4 use identical training config — no per-augmentation lr/wd tuning.

---

## Phase 1: Infrastructure

### 1a. CLIP Loading & Feature Encoding

**New file: `common/clip_utils.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

ALLOWED_MODEL_NAME = "ViT-B/32"
ALLOWED_PRETRAINED_SOURCE = "openai"


def load_openai_clip(
    device: torch.device,
    model_name: str = ALLOWED_MODEL_NAME,
    pretrained_source: str = ALLOWED_PRETRAINED_SOURCE,
):
    """Load OpenAI CLIP. HARD-ENFORCES ViT-B/32 + OpenAI weights."""
    if model_name != ALLOWED_MODEL_NAME:
        raise ValueError(
            f"Competition requires {ALLOWED_MODEL_NAME}, got {model_name}"
        )
    if pretrained_source != ALLOWED_PRETRAINED_SOURCE:
        raise ValueError(
            f"Only OpenAI official CLIP weights are allowed, got {pretrained_source}"
        )

    model, preprocess = clip.load(ALLOWED_MODEL_NAME, device=device, jit=False)
    return model, preprocess


@torch.no_grad()
def encode_frozen_clip_features(
    clip_model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    """Encode through FROZEN CLIP backbone. Returns L2-normalized float32 features.
    ONLY for freeze_clip=True. LoRA/adapters/unfreeze must use a different path.
    """
    with torch.autocast(device_type=device.type, enabled=use_amp):
        features = clip_model.encode_image(images)
    return F.normalize(features.float(), dim=-1)
```

Note: `CosineClassifier.forward()` performs defensive re-normalization (idempotent for unit vectors). The canonical normalization is in `encode_frozen_clip_features()`.

### 1b. Feature Caching

**Output** (`cache/clip_vit_b32_openai/`):
```
cache/clip_vit_b32_openai/
├── features.pt
├── labels.pt
├── paths.json           # dataset-root-relative POSIX paths
├── class_to_idx.json
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
  "torch_version": "2.x.y",
  "torchvision_version": "0.x.y",
  "clip_package": "openai-clip",
  "image_resolution": 224,
  "interpolation": "bicubic",
  "clip_mean": [0.48145466, 0.4578275, 0.40821073],
  "clip_std": [0.26862954, 0.26130258, 0.27577711],
  "created_at": "..."
}
```

**`dataset_fingerprint`**: content SHA256 per image — detects ANY file change.

**`class_mapping_hash`**: SHA256 of `sorted(class_to_idx.items())` as JSON.

**New file: `common/cached_dataset.py`**

```python
class CachedFeatureDataset(Dataset):
    def __init__(self, cache_dir, split_csv, class_to_idx_path):
        # 1. Load manifest
        # 2. FULL manifest validation (not just class mapping):
        expected = {
            "backbone": "ViT-B/32",
            "pretrained_source": "openai",
            "feature_dim": 512,
            "normalized": True,
            "dtype": "float32",
            "preprocess": "clip_deterministic",
        }
        for key, expected_value in expected.items():
            if manifest[key] != expected_value:
                raise ValueError(
                    f"Cache mismatch: {key}={manifest[key]!r}, "
                    f"expected {expected_value!r}"
                )
        # 3. Verify class_mapping_hash matches current
        # 4. Load cached class_to_idx, idx_to_class
        #    assert current_class_to_idx == cached_class_to_idx
        #    assert current_idx_to_class == cached_idx_to_class
        # 5. Load features.pt, labels.pt, paths.json
        # 6. Tensor validation:
        #    assert features.ndim == 2
        #    assert features.shape[1] == manifest["feature_dim"]
        #    assert labels.ndim == 1
        #    assert len(features) == len(labels) == len(paths)
        #    assert features.dtype == torch.float32
        #    assert labels.dtype == torch.int64
        #    assert torch.isfinite(features).all()
        #    assert len(paths) == len(set(paths))
        # 7. Build path → index lookup
        # 8. For each row in split_csv: verify cached_labels[index] == split_label
```

**Cache verification modes**:
```yaml
cache:
  verification: full   # full: re-compute content SHA256, compare fingerprint strictly
                       # quick: compare path count, class names, file sizes
```
Use `full` for official experiments; `quick` for development iteration.

### 1c. Transform Construction

**New file: `common/transforms.py`**

```python
from torchvision.transforms import (
    Compose, RandomResizedCrop, RandomHorizontalFlip,
    ColorJitter, ToTensor, Normalize, RandomErasing, InterpolationMode,
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
VALID_PRESETS = {"a0", "a1", "a2", "a3"}


def _convert_to_rgb(image):
    return image.convert("RGB")


def build_train_transform(preset: str, clip_eval_transform):
    if preset not in VALID_PRESETS:
        raise ValueError(
            f"Unknown augmentation preset: {preset!r}. "
            f"Expected one of {sorted(VALID_PRESETS)}"
        )
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

### 1d. CLIP Backbone eval-mode Enforcement

```python
class CLIPClassifier(nn.Module):
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_clip:
            self.clip_model.eval()
        return self
```

### 1e. Seed Separation & DataLoader Determinism

```yaml
experiment:
  mode: dev
  split_seed: 42
  train_seed: 42
```

**DataLoader — `drop_last=False` everywhere**:
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
    drop_last=False,       # ← NEVER discard samples
)
```

**split_data.py**: `n_val = max(1, round(n_samples * val_ratio))`, clamped to `min(n_val, n_samples - 1)`. Post-split validation: no overlap, no duplicates, full coverage, small-class guard.

**Output isolation**: `outputs/{experiment}/split_{split_seed}/train_{train_seed}/` and `outputs/{experiment}/final_fit/train_{train_seed}/`.

### 1f. Multi-Split Evaluation

**Experiment A — End-to-End Split Sensitivity**:
```
split_seed=42/3407/2026, train_seed=42 (fixed)
```

**Paired delta reporting** (confirm splits require re-running E0 baseline):

```
split_seed    Baseline    Candidate    Δ
3407          0.693       0.709        +0.016
2026          0.706       0.714        +0.008

Confirmation (seed=3407,2026): wins 2/2, mean Δ +0.012, worst Δ +0.008
Pooled (seed=42,3407,2026):   wins 3/3, mean Δ +0.013, worst Δ +0.008
```

Dev split is NOT independent confirmation. Confirmation denominator = 2. Pooled is descriptive only.

### 1g. Evaluation Metrics JSON

Same structure as before: overall, per_class, per_sample (path, label, pred, confidence, margin, loss, correct).

---

## Phase 2: Cosine Classifier

### model.py

```python
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
        # Validate scale parameters
        if min_scale <= 0:
            raise ValueError(...)
        if max_scale < min_scale:
            raise ValueError(...)
        if not min_scale <= init_scale <= max_scale:
            raise ValueError(...)

        self.learnable_scale = learnable_scale
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)).log())
        else:
            self.register_buffer("logit_scale", torch.tensor(float(init_scale)).log())

        nn.init.normal_(self.weight, std=0.01)

    def forward(self, features):
        features = F.normalize(features, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        scale = self.logit_scale.exp().clamp(min=self.min_scale, max=self.max_scale)
        return scale * features @ weight.t()

    def clamp_scale(self):
        if not self.learnable_scale:
            return
        with torch.no_grad():
            self.logit_scale.clamp_(
                min=math.log(self.min_scale), max=math.log(self.max_scale),
            )
```

### Optimizer — conditional param groups

```python
param_groups = [{"params": [classifier.weight], "weight_decay": weight_decay}]
if classifier.learnable_scale:
    param_groups.append({"params": [classifier.logit_scale], "weight_decay": 0.0})
optimizer = AdamW(param_groups, lr=lr)
```

### Checkpoint Metadata

```python
checkpoint = {
    "model_state_dict": ...,
    "architecture": "ViT-B/32",
    "pretrained_source": "openai",
    "head_type": "cosine",
    "augmentation_preset": "a2",
    "class_to_idx": class_to_idx,
    "idx_to_class": idx_to_class,
    "feature_dim": 512,
    "num_classes": num_classes,
    "train_seed": train_seed,
    "split_seed": split_seed if mode != "final_fit" else None,
    "training_mode": mode,
    "selected_epoch": best_epoch,
    "epoch_selection_split": 42,
    "config": resolved_config,
}
```

Inference reads mapping from checkpoint — no external split directory dependency.

---

## Phase 3: Data Augmentation

`experiments/augmentation/` with configs `augmentation_a{0,1,2,3}.yaml`. Uses baseline `CLIPLinearClassifier`. Transforms via `build_train_transform(preset, clip_eval_transform)`.

---

## Final-Fit Mode

```yaml
experiment:
  mode: final_fit
  train_seed: 42

data:
  use_full_training_set: true

train:
  epochs: 13    # Frozen per-method epoch from dev split — NOT adjusted after confirm
```

- `TrainImageDataset(split_csv=None)` scans all class directories via `_load_from_directory()`
- `drop_last=False` — every sample is used in every epoch
- No validation, no `best.pt`, no early stopping
- Saves only `last.pt`
- Output: `outputs/{experiment}/final_fit/train_{train_seed}/`

---

## Submission

```python
import csv

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for image_name, pred_idx in predictions:
        class_name = idx_to_class[str(pred_idx)]
        assert class_name == class_name.strip()
        assert len(class_name) == 4
        assert class_name.isdigit()
        writer.writerow([image_name, class_name])
```

No header. Format: `img.jpg,0001`. Zip contains only `pred_results.csv`.

**Test coverage validation**:
```python
expected_names = {p.name for p in test_image_paths}
predicted_names = [row[0] for row in submission_rows]
assert len(predicted_names) == len(expected_names)
assert len(predicted_names) == len(set(predicted_names))
assert set(predicted_names) == expected_names
```

---

## Files Changed / Created

### New Files
- `common/clip_utils.py`
- `common/transforms.py`
- `common/cached_dataset.py`
- `tools/cache_features.py`
- `data/preliminary/metadata/class_to_idx.json` (generated)
- `data/preliminary/metadata/idx_to_class.json` (generated)
- `scripts/run_multisplit_eval.py`
- `experiments/cosine_classifier/__init__.py`, `model.py`, `train.py`, `evaluate.py`, `infer.py`
- `experiments/augmentation/__init__.py`, `model.py`, `train.py`, `evaluate.py`, `infer.py`
- `configs/cosine_classifier.yaml`
- `configs/augmentation_a0.yaml`, `augmentation_a1.yaml`, `augmentation_a2.yaml`, `augmentation_a3.yaml`
- `tests/fixtures/baseline_reference.json`

### Modified Files
- `scripts/split_data.py`
- `experiments/baseline/evaluate.py`
- `experiments/baseline/model.py`
- `experiments/baseline/train.py`
- `experiments/baseline/infer.py`
- `common/submission.py`
- `configs/baseline.yaml`

### Unchanged
- `common/dataset.py` — `TrainImageDataset(split_csv=None)` already scans all directories
- `common/utils.py`

---

## Acceptance Criteria

### AC-1: 特征缓存

| # | 验收标准 |
|---|---|
| AC-1.1 | 缓存正确编码全量数据：shape 正确，路径为 POSIX relative |
| AC-1.2 | 缓存特征与在线编码一致：`encode_frozen_clip_features()` 统一路径；`torch.allclose(atol=1e-5, rtol=1e-5)` |
| AC-1.3 | `CachedFeatureDataset` 三层校验：class_mapping_hash + mapping equality + per-sample label |
| AC-1.4 | `manifest.json` 完整：所有字段存在含 torch/torchvision/clip 版本 |
| AC-1.5 | class_mapping_hash 不一致 → 拒绝训练 |
| AC-1.6 | 缓存目录含规范类别映射，与当前映射完全相等 |
| AC-1.7 | dataset_fingerprint 基于 content SHA256 → 任何图片变化都能检测 |
| AC-1.8 | 缓存路径无重复 |
| AC-1.9 | 缓存模式加速（性能目标） |
| AC-1.10 | **缓存配置兼容性**：更改 backbone、pretrained_source、feature_dim、normalized、preprocess 任一配置后加载旧缓存，必须立即拒绝训练并报错 |

### AC-2: 种子与多划分

| # | 验收标准 |
|---|---|
| AC-2.1 | split_seed 产生不完全相同的分层划分；每类别 train/val ≥1 样本 |
| AC-2.2 | <2 样本类别 → ValueError |
| AC-2.3 | 无泄漏无重复 |
| AC-2.4 | 输出目录按 `split_{split_seed}/train_{train_seed}/` 隔离 |
| AC-2.5 | 相同 seed 两次运行首个 batch 完全一致（seed_worker + Generator） |
| AC-2.6 | 评估 JSON 完整 |
| AC-2.7 | 配对报告：confirmation X/2，pooled X/3，dev split 不算独立确认 |
| AC-2.8 | **final_fit 全量训练**：加载样本数 = 训练目录图片总数；`drop_last=False`；每 epoch 处理所有样本，无丢弃 |
| AC-2.9 | 规范类别映射独立于 split；生命周期正确（生成→复用→冲突报错）；目录名校验 4-digit |

### AC-3: 余弦分类器

| # | 验收标准 |
|---|---|
| AC-3.1 | 无 bias，参数名因 learnable_scale 而异 |
| AC-3.2 | 权重缩放不变性：weight × 3.0 后 logits 不变 |
| AC-3.3 | learnable_scale=True: grad 正常；False: scale 固定 |
| AC-3.4 | clamp_scale() 正确：可学习时 clamp，固定时 no-op |
| AC-3.5 | 参数范围校验：非法值 → ValueError |
| AC-3.6 | **optimizer 条件构造**：learnable_scale=False 时 logit_scale 不在任何 param_group |
| AC-3.7 | **等预算比较**：Linear 与 Cosine 主比较使用相同 trial 数（9 vs 9）；Cosine scale 消融作为独立报告 |
| AC-3.8 | 在线/缓存模式正常训练 |
| AC-3.9 | 推理合法：csv.writer，无 header，`img.jpg,0001`，通过 check_submission.py |

### AC-4: 数据增强

| # | 验收标准 |
|---|---|
| AC-4.1 | A0/val 使用同一 `clip_eval_transform` 对象 |
| AC-4.2 | `build_train_transform()` 不内部加载 CLIP；非法 preset → ValueError |
| AC-4.3 | A1-A3 随机变化：100 次 hash 集合 size > 1 |
| AC-4.4 | A0 vs baseline：`torch.allclose(atol=1e-6)`；`abs(acc diff) <= 0.001` |
| AC-4.5 | RandomErasing 在 Normalize 之后 |
| AC-4.6 | 四组正常训练 |

### AC-5: 工程与回归

| # | 验收标准 |
|---|---|
| AC-5.1 | Baseline 回归 fixture 可复现（paths/labels/input_checksum/model_checksum/logits/loss） |
| AC-5.2 | CLIP backbone 冻结时 eval 模式，即使调用 `model.train()` |
| AC-5.3 | 所有新模块可 import |
| AC-5.4 | `num_classes: auto` 从规范映射推断 |
| AC-5.5 | `load_openai_clip()` 硬校验：非 ViT-B/32 或非 openai → ValueError |
| AC-5.6 | 提交合规：csv.writer，无 header，`img.jpg,0001`；zip 仅含 pred_results.csv |
| AC-5.7 | checkpoint 含完整推理元数据；infer 从 checkpoint 读映射 |
| AC-5.8 | **每个候选方法 epoch 独立冻结**：各自在 dev split 确定，写入 checkpoint，confirm/final-fit 不改变 |
| AC-5.9 | 消融公平性：equal-budget main comparison + Cosine scale as separate ablation |
| AC-5.10 | **测试集覆盖**：提交行数 = 唯一文件名数 = 官方测试图片数；文件名集合与测试集完全相等；无重复、无缺失、无额外项 |
