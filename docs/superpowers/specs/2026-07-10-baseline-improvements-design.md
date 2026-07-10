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
- **Multi-split validation is a diagnostic tool, NOT an ensemble**: final submission uses ONE single-seed model trained on the FULL official training set
- **Ablation order**: fix one variable at a time (head type → augmentation level → combination)
- **Dev/confirm split strategy**: search on split_42, verify top-2 on split_3407+2026, final-fit on full data

---

## Revised Execution Order

```
Step 0: Save baseline regression fixture; reproduce with split_seed=42, train_seed=42
Step 1: Stratified split + split_seed/train_seed separation + isolated output dirs
Step 2: Cache deterministic CLIP features on FULL training set (content SHA256 fingerprint + class mapping)
Step 3: Dev split (split_seed=42): E0 (Linear+A0) vs E1 (Cosine+A0)
Step 4: Dev split (split_seed=42): fix Linear head, compare E0/E2/E3/E4 (A0/A1/A2/A3)
Step 5: Dev split (split_seed=42): E5 (Cosine + best augmentation); select top-2 candidates
Step 6: Confirm splits (split_seed=3407, 2026): run E0 baseline + candidate-1 + candidate-2 on each split; report paired deltas
Step 7: Decide final method, hyperparameters, fixed epoch count
Step 8: Final-fit on FULL official training set with pre-declared train_seed=42 (no val split, no early stopping)
Step 9: Generate pred_results.csv, validate locally, compress and submit
```

**Dev/confirm/final-fit strategy**:
- `split_seed=42`: development — all preliminary search and screening
- `split_seed=3407, 2026`: confirmation — only E0 baseline + top-2 candidates, compute paired deltas
- `final_fit` mode: train on ALL official training data, fixed epochs, no validation split, no early stopping

**Final seed selection policy**:
- Final submission model uses **pre-declared** `train_seed=42`
- Multiple seeds are used ONLY for stability measurement, NOT for cherry-picking
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

**Unified feature encoding function** (used by BOTH cache and online modes):
```python
@torch.no_grad()
def encode_normalized_features(
    clip_model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    """Encode images through CLIP backbone, return L2-normalized float32 features."""
    with torch.autocast(device_type=device.type, enabled=use_amp):
        features = clip_model.encode_image(images)
    return F.normalize(features.float(), dim=-1)
```

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
  "dataset_fingerprint": "<sha256 of [(rel_path, class_name, file_size, content_sha256), ...]>",
  "created_at": "2026-07-10T12:00:00"
}
```

**`dataset_fingerprint` computation** (content SHA256 — detects ANY file change):
```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

records = []
for img_path in sorted(all_image_paths):
    rel_path = img_path.relative_to(dataset_root).as_posix()
    class_name = img_path.parent.name
    file_size = img_path.stat().st_size
    content_sha256 = sha256_file(img_path)
    records.append({
        "path": rel_path,
        "class_name": class_name,
        "file_size": file_size,
        "content_sha256": content_sha256,
    })

fingerprint = hashlib.sha256(
    json.dumps(records, sort_keys=True).encode()
).hexdigest()
```

**Path convention**: All paths in `paths.json` and split CSVs use **dataset-root-relative POSIX paths**:
```python
rel_path = image_path.relative_to(dataset_root).as_posix()
# Example: "0001/img1.jpg", "0123/some_image.png"
```

**New file: `tools/cache_features.py`**

Encodes every image in `data/preliminary/train/`:
1. Loads CLIP ViT-B/32 (`pretrained_source: openai`), freezes backbone
2. Uses `model.eval()` + `torch.inference_mode()` + `encode_normalized_features()`
3. Verifies: `not features.requires_grad`, `torch.isfinite(features).all()`, `features.norm(dim=-1) ≈ 1.0`
4. Computes `class_mapping_hash` and `dataset_fingerprint`
5. Saves features, labels, paths (relative), class_to_idx, idx_to_class, manifest

**New file: `common/cached_dataset.py`**

```python
class CachedFeatureDataset(Dataset):
    """Loads cached features filtered by a split CSV."""
    def __init__(self, cache_dir, split_csv, class_to_idx_path):
        # 1. Load manifest; verify class_mapping_hash matches current
        # 2. Load cached class_to_idx, idx_to_class
        #    assert current_class_to_idx == cached_class_to_idx
        #    assert current_idx_to_class == cached_idx_to_class
        # 3. Load features.pt, labels.pt, paths.json
        # 4. Verify: len(paths) == len(set(paths))  (no duplicate paths)
        # 5. Build relative_path → index lookup
        # 6. For each row in split_csv:
        #    - find feature via path → index
        #    - assert cached_labels[index] == split_label  (label consistency check)
        # Returns (feature, label, path_str)
```

**Second new file: `common/transforms.py`**

Centralized transform construction. The CLIP model+preprocess is loaded ONCE and the preprocess function is passed in:
```python
def build_clip_eval_transform():
    """DEPRECATED: do NOT call clip.load() here. Use the preprocess returned by
    load_openai_clip() and pass it to build_train_transform() instead."""
    ...

def build_train_transform(preset: str, clip_eval_transform):
    """Build training transform.
    preset ∈ {"a0", "a1", "a2", "a3"}
    a0 returns clip_eval_transform directly (official CLIP preprocess).
    a1-a3 compose augmentation before normalization.
    """
```

**Model loading** (called once, returns both model and preprocess):
```python
def load_openai_clip(device: torch.device, model_name: str = "ViT-B/32"):
    """Load OpenAI CLIP model and preprocess. Call ONCE per process."""
    model, preprocess = clip.load(model_name, device=device, jit=False)
    return model, preprocess
```

### 1b. CLIP Backbone eval-mode Enforcement

```python
class CLIPClassifier(nn.Module):
    def __init__(self, ...):
        ...
        self.freeze_clip = freeze_clip

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_clip:
            self.clip_model.eval()
        return self
```

Training loop sanity check:
```python
assert not model.clip_model.training
assert all(not p.requires_grad for p in model.clip_model.parameters())
```

### 1c. Seed Separation & Output Isolation

**Config changes**:
```yaml
experiment:
  mode: dev                 # dev | confirm | final_fit
  split_seed: 42
  train_seed: 42

data:
  train_dir: data/preliminary/train
  use_full_training_set: false   # true only for final_fit
```

**`scripts/split_data.py`** enhancement:
- Accept `--split_seed` to override config
- **Stratified** per-class split: `n_val = max(1, round(n_samples * val_ratio))`, clamped to `min(n_val, n_samples - 1)`
- **Small-class guard**: if a class has <2 samples, raise `ValueError("Class {name} has only {n} sample(s); cannot split")`
- **Post-split validation**:
  ```python
  assert set(train_paths).isdisjoint(val_paths)
  assert len(train_paths) + len(val_paths) == len(full_paths)
  assert len(train_paths) == len(set(train_paths))
  assert len(val_paths) == len(set(val_paths))
  ```
- Output to `{split_dir}/split_{split_seed}/`

**Output directory isolation**:
```
outputs/
└── {experiment}/
    ├── split_{split_seed}/
    │   └── train_{train_seed}/
    │       ├── checkpoints/
    │       └── logs/
    └── final_fit/
        └── train_{train_seed}/
            ├── checkpoints/
            └── logs/
```

### 1d. Multi-Split Evaluation

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

For each split `i`, compute `Δ_i = Accuracy_candidate(split_i) − Accuracy_baseline(split_i)`.

**Confirm splits must re-run baseline**: to compute paired deltas, the E0 baseline must be trained on each confirm split alongside the candidates.

```
split_seed    Baseline    Candidate    Δ
42            0.700       0.715        +0.015
3407          0.693       0.709        +0.016
2026          0.706       0.714        +0.008
```

Report:
```
Baseline accuracy:       0.700 ± 0.007 (worst: 0.693)
Candidate accuracy:      0.713 ± 0.003 (worst: 0.709)
Paired improvement Δ:   +0.013 ± 0.004 (worst: +0.008)
Wins: 3/3 splits
```

Use **sample std** (`ddof=1`). Always output all raw values alongside summary statistics.

**New file: `scripts/run_multisplit_eval.py`**

### 1e. Evaluation Metrics Enhancement

**`experiments/baseline/evaluate.py`** — save detailed results JSON:

```json
{
  "overall": {
    "accuracy": 0.xxxx,
    "macro_accuracy": 0.xxxx,
    "loss": 0.xxxx,
    "total_samples": N
  },
  "per_class": {...},
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

**Field definitions**:
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

        # Validate scale parameters
        if min_scale <= 0:
            raise ValueError(f"min_scale must be positive, got {min_scale}")
        if max_scale < min_scale:
            raise ValueError(f"max_scale ({max_scale}) must be >= min_scale ({min_scale})")
        if not min_scale <= init_scale <= max_scale:
            raise ValueError(f"init_scale ({init_scale}) must be in [{min_scale}, {max_scale}]")

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
        # Clamp both bounds for defense-in-depth
        scale = self.logit_scale.exp().clamp(min=self.min_scale, max=self.max_scale)
        return scale * features @ weight.t()

    def clamp_scale(self) -> None:
        """Call after optimizer.step() to keep scale in valid range."""
        if not self.learnable_scale:
            return
        with torch.no_grad():
            self.logit_scale.clamp_(
                min=math.log(self.min_scale),
                max=math.log(self.max_scale),
            )
```

**Optimizer — no weight decay on logit_scale**:
```python
optimizer = AdamW([
    {"params": [classifier.weight], "weight_decay": weight_decay},
    {"params": [classifier.logit_scale], "weight_decay": 0.0},
], lr=lr)
```

**AMP / DDP clamp_scale() placement**:
```python
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
model.classifier.clamp_scale()  # model.module.classifier.clamp_scale() under DDP
```

### Config

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

---

## Phase 3: Data Augmentation

**Experiment directory**: `experiments/augmentation/`
**Configs**: `configs/augmentation_a{0,1,2,3}.yaml`

### Transform Construction (centralized in `common/transforms.py`)

The CLIP model+preprocess is loaded ONCE via `load_openai_clip()`. The preprocess is passed into transform builders — `build_clip_eval_transform()` must NOT call `clip.load()` internally.

```python
def load_openai_clip(device: torch.device, model_name: str = "ViT-B/32"):
    """Load OpenAI CLIP once. Returns (model, preprocess)."""
    model, preprocess = clip.load(model_name, device=device, jit=False)
    return model, preprocess

def build_train_transform(preset: str, clip_eval_transform):
    """preset ∈ {"a0", "a1", "a2", "a3"}.
    a0 returns clip_eval_transform directly.
    """
    if preset == "a0":
        return clip_eval_transform

    # A1/A2/A3 base
    from torchvision.transforms import (
        Compose, RandomResizedCrop, RandomHorizontalFlip,
        ColorJitter, ToTensor, Normalize, RandomErasing, InterpolationMode,
    )

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def _convert_to_rgb(image):
        return image.convert("RGB")

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

**All val_transforms = `clip_eval_transform`** (the preprocess from `load_openai_clip()`).

### Augmentation Grid

| Experiment | RandomResizedCrop | HorizontalFlip | ColorJitter | RandomErasing |
|---|---|---|---|---|
| A0 (control) | — | — | — | — |
| A1 | scale=(0.75,1.0), ratio=(0.85,1.15) | p=0.5 | — | — |
| A2 | same | p=0.5 | brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02 | — |
| A3 | same | p=0.5 | same | p=0.1, scale=(0.02,0.15), value=0 |

### model.py

Phase 3 initially uses baseline's `CLIPLinearClassifier`. After E0-E4 complete, E5 combines Cosine + best augmentation.

---

## Final-Fit Mode

**Config**:
```yaml
experiment:
  mode: final_fit
  train_seed: 42

data:
  use_full_training_set: true    # All official training images
  train_dir: data/preliminary/train

train:
  epochs: 20                     # Fixed, determined from dev experiments
  # No val_ratio, no split_dir needed
```

**Behavior**:
- `use_full_training_set: true` → `TrainImageDataset` scans ALL class directories (no split CSV filtering)
- No validation during training (no val loader, no best.pt selection, no early stopping)
- Saves only `last.pt` at the final epoch
- Output to `outputs/{experiment}/final_fit/train_{train_seed}/`

**Epoch selection**: determined from dev experiments (Step 3-5). Use the epoch where best val accuracy was achieved, or a conservative fixed count. Do NOT tune epoch count on the full training set.

---

## Submission Format

**Output file**: `pred_results.csv`

Use `csv.writer` — do NOT manually insert a space after the comma:
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

**No header row**. Produces: `img.jpg,0001`

**Zip**: `submission.zip` contains exactly one file — `pred_results.csv` at root level.

---

## Multi-Split ≠ Ensemble (Explicit Prohibition)

Multi-split training is used ONLY for evaluating stability and selecting hyperparameters.

The final submission uses **one model trained on the full official training set** with
the pre-declared `train_seed=42` and fixed epoch count. `split_seed` does NOT apply to final-fit mode.

Do NOT:
- Average, vote, or fuse logits/predictions across multiple seeds
- Cherry-pick the best-performing seed after seeing validation or test-set results
- Use test-set submission scores to select among seeds

---

## CLIP Weight Source Enforcement

- Config MUST specify `model.pretrained_source: openai`
- `build_model()` logs: `Backbone: CLIP ViT-B/32 | Pretrained source: OpenAI official`
- Cache `manifest.json` records `pretrained_source`
- OpenCLIP / LAION weights are explicitly prohibited per competition rules

---

## num_classes Auto-Inference

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
- `tests/fixtures/baseline_reference.json`

### Modified Files
- `scripts/split_data.py` — add `--split_seed`, stratified split with `n_val = max(1, round(n * ratio))`, small-class guard, post-split validation, seed-specific output dirs
- `experiments/baseline/evaluate.py` — add per-sample + per-class metrics JSON output
- `experiments/baseline/model.py` — add `pretrained_source` logging, `CLIPClassifier.train()` eval-mode enforcement, unified `encode_normalized_features()`
- `experiments/baseline/train.py` — read `experiment.split_seed`/`experiment.train_seed`/`experiment.mode`, locate split dir, isolated output dirs, auto-infer num_classes, support `final_fit` mode (full training set, no val)
- `experiments/baseline/infer.py` — use `train_seed` for set_seed(); load idx_to_class from checkpoint; output `pred_results.csv` via csv.writer
- `common/submission.py` — use csv.writer instead of manual string formatting; add class name validation
- `configs/baseline.yaml` — add `experiment.mode`, `experiment.split_seed`, `experiment.train_seed`, `data.use_full_training_set`, `model.pretrained_source`, `model.num_classes: auto`

### Unchanged
- `common/dataset.py`
- `common/utils.py` — `set_seed()` stays as-is; callers pass the appropriate seed

---

## Acceptance Criteria（验收方案）

### AC-1: 特征缓存

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-1.1 | 缓存脚本正确编码全量数据 | `features.pt` shape `(N_full, 512)`, `labels.pt` shape `(N_full,)`, `paths.json` 长度一致；所有路径为 dataset-root-relative POSIX 格式 |
| AC-1.2 | 缓存特征与在线编码一致（统一使用 `encode_normalized_features()`） | 使用 `model.eval()` + `torch.inference_mode()`；验证 `not features.requires_grad`, `torch.isfinite(features).all()`, `features.norm(dim=-1) ≈ 1.0`；与在线编码 `torch.allclose(atol=1e-5, rtol=1e-5)`（若两边强制相同设备/精度/编码路径，否则可用 1e-4） |
| AC-1.3 | `CachedFeatureDataset` 三层校验 | (1) `class_mapping_hash` 一致；(2) `cached_class_to_idx == current_class_to_idx` 且 `cached_idx_to_class == current_idx_to_class`；(3) `cached_labels[index] == split_label` per sample |
| AC-1.4 | `manifest.json` 完整 | 所有字段存在：`backbone`, `pretrained_source`, `feature_dim`, `normalized`, `dtype`, `preprocess`, `dataset_size`, `num_classes`, `dataset_root`, `class_mapping_hash`, `dataset_fingerprint`, `created_at` |
| AC-1.5 | `class_mapping_hash` 不一致时拒绝训练 | 修改 `class_to_idx` 后用旧缓存训练，立即报错退出 |
| AC-1.6 | `class_to_idx.json` 和 `idx_to_class.json` 已保存且通过校验 | 缓存目录包含完整类别映射，与当前映射完全相等 |
| AC-1.7 | `dataset_fingerprint` 基于 `(rel_path, class_name, file_size, content_sha256)` | 替换、增加、删除或修改任何一张图片后 fingerprint 发生变化 |
| AC-1.8 | 缓存路径无重复 | `len(paths) == len(set(paths))` |
| AC-1.9 | 缓存模式训练加速（性能目标，非 correctness gate） | 记录在线与缓存模式的 epoch 时间及加速比；缓存模式应显著降低训练时间；目标 ≥10× |

### AC-2: 种子分离与多划分验证

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-2.1 | `split_seed` 产生不同但不完全相异的划分 | `train_paths_seed42 != train_paths_seed3407`；`val_paths_seed42 != val_paths_seed3407`；至少一个类别的验证样本集合发生变化；每类别在 train/val 中均 ≥1 样本 |
| AC-2.2 | 极小类别（<2 samples）报错 | 构造含单样本类别的数据集，split 时抛出 `ValueError` |
| AC-2.3 | split 后验证无泄漏、无重复 | `train ∩ val = ∅`, `train ∪ val = full`, train/val 各自无重复 |
| AC-2.4 | 输出目录按 `split_{split_seed}/train_{train_seed}/` 隔离 | 不同组合的 checkpoint 和日志互不覆盖 |
| AC-2.5 | 评估 JSON 包含所有定义字段 | `overall`, `per_class`, `per_sample`（path, label, pred, confidence, margin, loss, correct）全部存在 |
| AC-2.6 | `run_multisplit_eval.py` 输出配对比较报告（含 baseline） | 每个 confirm split 上先训练 E0 baseline，再训练 candidate；输出 Baseline mean±std、Candidate mean±std、Paired Δ mean±std、win count (X/3)、worst paired delta |
| AC-2.7 | 文档明确禁止 ensemble + 最终 seed 预声明 | 代码注释和 README 声明：多种子仅用于评估，最终提交使用预声明的 `train_seed=42` + 全量训练集 |
| AC-2.8 | `final_fit` 模式使用全量数据 | `use_full_training_set: true` 时加载所有官方训练样本，不使用 split CSV；无 val loader，无 early stopping |

### AC-3: 余弦分类器

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-3.1 | `CosineClassifier` 无 bias，参数名正确 | `learnable_scale=True` 时 `set(dict(classifier.named_parameters())) == {"weight", "logit_scale"}`；`learnable_scale=False` 时 `logit_scale` 为 buffer 不在 parameters 中；`not hasattr(classifier, "bias") or classifier.bias is None` |
| AC-3.2 | 权重缩放不变性 | 对同一输入，`classifier.weight` 乘以 3.0 后 logits 不变：`torch.allclose(atol=1e-5)` |
| AC-3.3 | `learnable_scale=True` 时 logit_scale 有梯度 | `logit_scale.requires_grad == True`；一次 backward 后 `logit_scale.grad is not None` 且 `torch.isfinite(grad).all()` |
| AC-3.4 | `learnable_scale=False` 时 scale 固定 | 训练前后 `logit_scale.exp()` 值不变 |
| AC-3.5 | `clamp_scale()` 约束有效 | optimizer.step() 后调用，`logit_scale` 始终在 `[log(min_scale), log(max_scale)]` 内；固定 scale 时 `clamp_scale()` 直接返回 |
| AC-3.6 | 参数范围验证 | 传入 `min_scale <= 0`、`max_scale < min_scale`、`init_scale` 越界时均抛出 `ValueError` |
| AC-3.7 | logit_scale 无 weight decay | 检查 optimizer param_groups 中 logit_scale 的 `weight_decay == 0.0` |
| AC-3.8 | 在线/缓存模式均可正常训练 | 各跑 1 epoch，loss 下降 |
| AC-3.9 | 推理产出合法提交文件 | `pred_results.csv` 由 csv.writer 生成，无 header，格式 `img.jpg,0001`；通过 `check_submission.py` 全部检查 |

### AC-4: 数据增强

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-4.1 | A0 和 val 使用 `clip_eval_transform`（同一对象） | Baseline train transform = A0 train transform = 所有 val transform，均为 `load_openai_clip()` 返回的 preprocess；抽样 `torch.allclose(atol=1e-6, rtol=0)` |
| AC-4.2 | `build_clip_eval_transform()` 不调用 `clip.load()` | transform builder 接受外部传入的 preprocess 函数，不内部加载模型 |
| AC-4.3 | A1-A3 产生随机变化 | 同一张图 100 次 transform，SHA256 hash 集合 size > 1；所有输出 shape (3, 224, 224) 且 `torch.isfinite` |
| AC-4.4 | A0 与 baseline 精度一致 | 相同 split_seed, train_seed, 参数下：首个 batch 输入 `torch.allclose(atol=1e-6)`；在确定性训练条件下 prediction 完全一致；若 CUDA 算子无法完全确定，则 `abs(acc_a0 - acc_baseline) <= 0.001` |
| AC-4.5 | RandomErasing 在 Normalize 之后，value=0 | A3 transform 中 `RandomErasing` 位于 `Normalize` 之后 |
| AC-4.6 | 四组实验均可正常训练 | A0-A3 各跑 1 epoch，loss 正常下降 |

### AC-5: 工程与回归

| # | 验收标准 | 验证方式 |
|---|---|---|
| AC-5.1 | Baseline 重构后可复现 | 首先生成 `tests/fixtures/baseline_reference.json`（含 first_batch_paths, first_batch_labels, first_batch_input_checksum, initial_loss）；重构后同条件运行，`abs(current_initial_loss - reference_initial_loss) < 1e-4` |
| AC-5.2 | CLIP backbone 冻结时始终处于 eval 模式 | `model.clip_model.training == False`；`all(not p.requires_grad for p in model.clip_model.parameters())`；调用 `model.train()` 后 backbone 仍在 eval |
| AC-5.3 | 所有新模块可 import | 所有 `experiments/*/` 下模块正确导入 |
| AC-5.4 | `num_classes: auto` 正确推断 | 从 `class_to_idx.json` 自动获取，无需手动配置 |
| AC-5.5 | `pretrained_source` 被记录和验证 | `build_model()` 打印 `Backbone: CLIP ViT-B/32 | Pretrained source: OpenAI official`；manifest 记录 |
| AC-5.6 | 提交格式完整合规 | `pred_results.csv` 无 header，格式 `img.jpg,0001`；`submission.zip` 仅含 `pred_results.csv`；类名通过 strip/len==4/isdigit 验证 |
