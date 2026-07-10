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
- **`drop_last=False` everywhere**

---

## Experiment Naming Convention

| ID | Configuration | Purpose |
|----|-------------|---------|
| **B0** | Original Linear+A0, FULL original training protocol (lr=1e-3, wd=1e-4, epochs=20, AdamW, CosineAnnealingLR, warmup=1, AMP, best.pt checkpoint) | Verify code refactor didn't change baseline |
| **E0** | Linear+A0, best of 9-trial lr×wd search | Tuned linear baseline for fair head comparison |
| **E1** | Cosine+A0, best of 9-trial lr×wd search (init_scale=10, learnable_scale=True fixed) | Test cosine head vs tuned linear |
| **E2** | Linear+A1, use E0's tuned lr/wd | Test RandomResizedCrop+Flip |
| **E3** | Linear+A2, use E0's tuned lr/wd | Test +ColorJitter |
| **E4** | Linear+A3, use E0's tuned lr/wd | Test +RandomErasing |
| **E5** | Cosine + best augmentation, Cosine config from E1 | Test head×augmentation combination |
| **C0** | Cosine+A0, fixed scale=10 | Cosine scale internal ablation (NOT in candidate pool) |
| **C1** | Cosine+A0, learnable scale init=10 | Cosine scale internal ablation (NOT in candidate pool) |
| **C2** | Cosine+A0, learnable scale init=20 | Cosine scale internal ablation (NOT in candidate pool) |

**Key rule**: E2/E3/E4 use the SAME lr/wd/scheduler/batch_size as E0 — only augmentation preset changes. Each method independently freezes its own best_epoch on dev split.

**Candidate pool**: C0/C1/C2 are explanatory internal ablations only — they do NOT enter the top-2 candidate pool and are NOT used to reselect E1 or E5 configurations. E1 always uses init_scale=10, learnable_scale=True. E5 inherits E1's Cosine config. C0/C1/C2 are reported independently in results.

**Cached features guard**: Feature caching is ONLY valid for augmentation=a0 AND freeze_clip=True. The following experiments can use cached features: B0, E0, E1, C0, C1, C2. The following CANNOT: E2, E3, E4, E5 (random augmentation must be re-applied each epoch). Hard enforcement:
```python
if use_cached_features and augmentation_preset != "a0":
    raise ValueError("Cached features only valid for deterministic A0 preprocessing")
if use_cached_features and not freeze_clip:
    raise ValueError("Cached features require freeze_clip=True")
```

**Progress tracking**:
```
B0 (original baseline, full original training protocol)
  → E0 (tuned baseline)              ← engineering + hyperparameter gain
    → E1 (cosine head)                ← head improvement
    → E2/E3/E4 (augmentation)         ← augmentation improvement
      → E5 (head × augmentation)      ← combined gain
```

**B0 regression protocol**: B0 MUST reproduce the complete original training protocol, not just lr/wd. This includes: optimizer=AdamW, scheduler=CosineAnnealingLR, epochs=20, warmup_epochs=1, batch_size=128, AMP enabled, max_grad_norm=1.0, checkpoint_policy=best_val, split_seed=42, train_seed=42. B0 does NOT adopt any new tuning rules (no 9-trial search, no per-method epoch freezing). Its sole purpose is answering: "Did infrastructure refactoring change the original baseline?"

B0 regression fixture should save the resolved config:
```json
{
  "optimizer": "AdamW",
  "lr": 0.001,
  "weight_decay": 0.0001,
  "batch_size": 128,
  "epochs": 20,
  "scheduler": "CosineAnnealingLR",
  "warmup_epochs": 1,
  "amp": true,
  "max_grad_norm": 1.0,
  "checkpoint_policy": "best_val",
  "split_seed": 42,
  "train_seed": 42
}
```

**Confirm-stage paired delta**: `Δ_i = Acc_candidate,i − Acc_E0,i` (relative to tuned linear baseline).

B0 is reported separately as a one-time regression check.

---

## Revised Execution Order

```
Step 0: Save B0 regression fixture; reproduce with original hyperparameters
Step 1: Seed separation + canonical class mapping + isolated output dirs
Step 2: Cache deterministic CLIP features on FULL training set (dual fingerprint + class mapping)
Step 3: Dev split (seed=42): B0 → E0 (9-trial lr×wd) → E1 (9-trial lr×wd) — equal-budget
Step 4: Dev split (seed=42): E2/E3/E4 using E0's tuned lr/wd; each freezes its own best_epoch
Step 5: Dev split (seed=42): E5 + C0/C1/C2; select top-2 candidates
Step 6: Confirm splits (seed=3407,2026): E0 + candidate-1 + candidate-2 per split; paired deltas vs E0
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

expected = config["data"]["expected_num_classes"]  # 500 preliminary, 1500 second_round, 1000 semifinal
if len(class_names) != expected:
    raise ValueError(f"Expected {expected} classes, found {len(class_names)}")

class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {str(i): name for name, i in class_to_idx.items()}
```

Config:
```yaml
data:
  stage: preliminary
  expected_num_classes: 500
  class_mapping_path: data/preliminary/metadata/class_to_idx.json
```

**Mapping file lifecycle**: not exist → generate; exists and matches → reuse; exists but inconsistent → error (needs `--regenerate-class-mapping`).

### Dev / Confirm / Final-Fit

| Stage | split_seed | Data | Baseline for Δ |
|-------|-----------|------|----------------|
| Dev | 42 | train.csv (90%) | B0 (one-time regression) |
| Confirm | 3407, 2026 | train.csv (90%) | E0 (tuned linear, rerun per split) |
| Final-fit | N/A | FULL training set | N/A |

### Method-specific Epoch Freezing

Each method independently records its best_epoch on dev split:
```
B0: best_epoch =  8 on split_42 → frozen (for regression verification only)
E0: best_epoch =  8 on split_42 → frozen, used on confirm splits
E1: best_epoch = 13 on split_42 → frozen, used on confirm splits
E2: best_epoch = 10 on split_42 → frozen
...
```

Confirm splits use each method's own frozen epoch. No epoch tuning on confirm.
Final-fit uses the selected method's frozen epoch.

### Candidate & Epoch Selection Rules

**Method selection**:
1. Compare top-2 candidates on confirm splits by mean paired delta vs E0 (tuned linear)
2. If |Δ_c1 − Δ_c2| < 0.1pp → select structurally simpler method
3. If any candidate degrades >0.2pp on any confirm split vs E0 → eliminated
4. Test-set submission scores are NOT used

**Epoch selection** (per-method, frozen before confirm):
1. Each method records `best_epoch = argmax(val_acc)` on dev split (seed=42)
2. Confirm splits use each method's own frozen epoch — no tuning
3. After confirm: all epoch counts permanently frozen
4. Final-fit uses the selected method's frozen epoch

### Ablation Fairness Protocol

**Equal-budget**: E0 and E1 each run 9 trials (3 lr × 3 wd). Cosine main comparison fixes init_scale=10, learnable_scale=True.

**Cosine scale ablation (C0/C1/C2)**: reported independently, NOT part of equal-budget comparison.

**Augmentation (E2/E3/E4)**: use E0's tuned lr/wd/scheduler/batch_size — no per-augmentation hyperparameter tuning.

---

## Infrastructure

### CLIP Loading & Feature Encoding

**`common/clip_utils.py`**:

```python
ALLOWED_MODEL_NAME = "ViT-B/32"
ALLOWED_PRETRAINED_SOURCE = "openai"


def load_openai_clip(device, model_name=ALLOWED_MODEL_NAME,
                     pretrained_source=ALLOWED_PRETRAINED_SOURCE):
    if model_name != ALLOWED_MODEL_NAME:
        raise ValueError(f"Requires {ALLOWED_MODEL_NAME}, got {model_name}")
    if pretrained_source != ALLOWED_PRETRAINED_SOURCE:
        raise ValueError(f"Only OpenAI weights allowed, got {pretrained_source}")
    model, preprocess = clip.load(ALLOWED_MODEL_NAME, device=device, jit=False)
    return model, preprocess


@torch.no_grad()
def encode_frozen_clip_features(clip_model, images, device, use_amp=False):
    with torch.autocast(device_type=device.type, enabled=use_amp):
        features = clip_model.encode_image(images)
    return F.normalize(features.float(), dim=-1)
```

### Feature Caching

**Output**: `cache/{stage}/clip_vit_b32_openai/`

Stage-separated cache directories:
```
cache/preliminary/clip_vit_b32_openai/
cache/second_round/clip_vit_b32_openai/
cache/semifinal/clip_vit_b32_openai/
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
  "class_mapping_hash": "<sha256>",
  "dataset_quick_fingerprint": "<sha256(rel_path, class_name, file_size)>",
  "dataset_full_fingerprint": "<sha256(rel_path, class_name, file_size, content_sha256)>",
  "torch_version": "2.x.y",
  "torchvision_version": "0.x.y",
  "clip_package": "openai-clip",
  "clip_version": null,
  "clip_commit": "<git commit if available, or null>",
  "clip_source_path": "/path/to/clip/installation",
  "pillow_version": "x.y.z",
  "python_version": "3.x.y",
  "image_resolution": 224,
  "interpolation": "bicubic",
  "clip_mean": [0.48145466, 0.4578275, 0.40821073],
  "clip_std": [0.26862954, 0.26130258, 0.27577711],
  "created_at": "..."
}
```

**Dual fingerprint**: `quick` traverses directories and reads file metadata (path, class_name, file_size via stat()) without reading image content; `full` additionally reads all file bytes and computes content SHA256. Both are pre-computed at cache creation time and stored in manifest.

**Cache verification modes**:
```yaml
cache:
  verification: full   # full: recompute content SHA256, compare dataset_full_fingerprint
                       # quick: recompute from file metadata only, compare dataset_quick_fingerprint
```

**`CachedFeatureDataset`**:
```python
class CachedFeatureDataset(Dataset):
    def __init__(self, cache_dir, split_csv, class_to_idx_path,
                 dataset_root, verification="full"):
        # 1. Load manifest
        # 2. HARD-FAIL on incompatible fields:
        #    backbone, pretrained_source, feature_dim, normalized, dtype, preprocess
        # 3. WARN on version differences:
        #    torch_version, torchvision_version, clip_version, pillow_version
        # 4. Verify fingerprint (quick or full) against dataset_root
        # 5. class_mapping_hash + cached mapping == current mapping
        # 6. Tensor validation: ndim, shape, dtype, finite, path uniqueness
        # 7. Per-sample label consistency check
```

**Version fields**: incompatible fields (backbone, pretrained_source, etc.) mismatch → `ValueError`. Environment version fields (torch, torchvision, clip, pillow) mismatch → warning only (patch version differences shouldn't invalidate valid caches).

### Transform Construction

**`common/transforms.py`**: `build_train_transform(preset, clip_eval_transform)` with `VALID_PRESETS = {"a0", "a1", "a2", "a3"}`. Unknown preset → `ValueError`.

### CLIP Backbone eval-mode Enforcement

`CLIPClassifier.train()` overrides to keep backbone in eval when frozen.

### DataLoader — `drop_last=False`, Deterministic

`seed_worker` + `torch.Generator` for reproducibility. `drop_last=False` everywhere.

### Multi-Split Evaluation

Paired deltas vs E0 (tuned linear, rerun on each confirm split). Output: confirmation wins X/2, pooled wins X/3.

---

## Cosine Classifier

Same as previous revision. Key points:
- Conditional optimizer param_groups (logit_scale excluded when learnable_scale=False)
- `clamp_scale()` no-op when fixed scale
- Forward clamps both min and max bounds
- Checkpoint embeds `class_to_idx`, `idx_to_class`, `head_type`, `augmentation_preset`, `trained_epochs`, `epoch_selection_split`, `epoch_selection_policy`, `training_mode`, `split_seed` (None for final_fit)

**Checkpoint epoch fields**:
```python
checkpoint = {
    "trained_epochs": selected_epoch,            # how many epochs this model was trained
    "epoch_selection_split": 42,                  # which split was used to choose epoch
    "epoch_selection_policy": "dev_best_epoch_frozen_before_confirm",
}
# dev mode additionally saves:
if mode == "dev":
    checkpoint["dev_best_epoch"] = best_epoch     # the argmax(val_acc) epoch
```

In final_fit, `trained_epochs` = the frozen epoch from dev, and `dev_best_epoch` = same value. No "best_epoch" is computed during final_fit (no validation).

---

## Data Augmentation

`experiments/augmentation/`. Config `augmentation_a{0,1,2,3}.yaml`. Uses baseline `CLIPLinearClassifier`. E2/E3/E4 use E0's tuned lr/wd.

---

## Final-Fit Mode

```yaml
experiment:
  mode: final_fit
  train_seed: 42

data:
  use_full_training_set: true

train:
  epochs: 13    # Frozen per-method epoch from dev split
```

- `TrainImageDataset(split_csv=None)` scans all class directories
- `drop_last=False`
- No validation, no best.pt, no early stopping
- `trained_epochs` in checkpoint = frozen dev epoch

---

## Submission

`csv.writer`, no header, `img.jpg,0001`. Zip contains only `pred_results.csv`.

**Pre-submission checks**:
```python
# 1. Test file name uniqueness
test_names = [p.name for p in test_image_paths]
assert len(test_names) == len(set(test_names)), \
    "Test set contains duplicate basenames"

# 2. Coverage
expected_names = set(test_names)
predicted_names = [row[0] for row in submission_rows]
assert len(predicted_names) == len(expected_names)
assert len(predicted_names) == len(set(predicted_names))
assert set(predicted_names) == expected_names

# 3. Class name validation per row
for name, class_name in submission_rows:
    assert class_name == class_name.strip()
    assert len(class_name) == 4
    assert class_name.isdigit()
```

---

## Acceptance Criteria

### AC-1: 特征缓存

| # | 验收标准 |
|---|---|
| AC-1.1 | 缓存正确编码全量数据：shape 正确，路径 POSIX relative |
| AC-1.2 | `encode_frozen_clip_features()` 统一编码路径；`torch.allclose(atol=1e-5)` |
| AC-1.3 | CachedFeatureDataset 全量校验：manifest 硬字段→error，版本字段→warning；fingerprint 验证（quick/full）；三层映射校验；tensor 校验；逐样本 label 校验 |
| AC-1.4 | manifest 完整：含 quick+full 双 fingerprint，torch/torchvision/clip/pillow/python 版本 |
| AC-1.5 | class_mapping_hash 不一致 → 拒绝训练 |
| AC-1.6 | 缓存目录含规范类别映射 |
| AC-1.7 | full_fingerprint 基于 content SHA256 → 任何图片变化可检测 |
| AC-1.8 | 缓存路径无重复 |
| AC-1.9 | 缓存模式加速（性能目标） |
| AC-1.10 | 更改 backbone/pretrained_source/feature_dim/normalized/preprocess → 拒绝训练 |
| AC-1.11 | **缓存模式 guard**：preset != a0 或 freeze_clip=False 时启用 `--use-cached-features` → ValueError，程序拒绝运行 |

### AC-2: 种子与多划分

| # | 验收标准 |
|---|---|
| AC-2.1 | split_seed 产生不完全相同的分层划分；每类别 train/val ≥1 |
| AC-2.2 | <2 样本类别 → ValueError |
| AC-2.3 | 无泄漏无重复 |
| AC-2.4 | 输出目录隔离 |
| AC-2.5 | 相同 seed 两次运行首个 batch 完全一致 |
| AC-2.6 | 评估 JSON 完整 |
| AC-2.7 | 配对报告 vs E0：confirmation X/2，pooled X/3 |
| AC-2.8 | final_fit：样本数=目录图片总数；drop_last=False；每 epoch 无丢弃 |
| AC-2.9 | 规范类别映射生命周期正确；`expected_num_classes` 从 config 读取；stage 隔离缓存目录 |

### AC-3: 余弦分类器

| # | 验收标准 |
|---|---|
| AC-3.1 | 无 bias；参数名因 learnable_scale 而异 |
| AC-3.2 | weight 缩放不变性 |
| AC-3.3 | learnable_scale=True: grad 正常；False: scale 固定 |
| AC-3.4 | clamp_scale() 正确 |
| AC-3.5 | 参数范围校验 → ValueError |
| AC-3.6 | optimizer 条件构造：learnable_scale=False 时 logit_scale 不在 param_group |
| AC-3.7 | **E0 vs E1 等预算**：各 9 trials；C0/C1/C2 独立报告 |
| AC-3.8 | 在线/缓存模式正常训练 |
| AC-3.9 | 推理合法：csv.writer，无 header，通过 check_submission.py |

### AC-4: 数据增强

| # | 验收标准 |
|---|---|
| AC-4.1 | A0/baseline/val 来自同一官方 preprocess 构造路径；抽样输出逐元素一致（不要求对象身份 `is`） |
| AC-4.2 | build_train_transform 不内部加载 CLIP；非法 preset → ValueError |
| AC-4.3 | A1-A3 随机变化：100 次 hash 集合 size > 1 |
| AC-4.4 | **E2/E3/E4 使用 E0 的 lr/wd/scheduler/batch_size**；只改 augmentation preset；各自独立冻结 epoch |
| AC-4.5 | RandomErasing 在 Normalize 之后 |
| AC-4.6 | 四组正常训练 |

### AC-5: 工程与回归

| # | 验收标准 |
|---|---|
| AC-5.1 | **B0** 回归 fixture 可复现（mapping→paths→labels→checksum→logits→loss 分层检查） |
| AC-5.2 | CLIP backbone 冻结时 eval，即使 `model.train()` |
| AC-5.3 | 所有新模块可 import |
| AC-5.4 | `num_classes: auto` 从规范映射推断 |
| AC-5.5 | `load_openai_clip()` 硬校验：非 ViT-B/32 或非 openai → ValueError |
| AC-5.6 | 提交合规：csv.writer，无 header，`img.jpg,0001`；zip 仅含 pred_results.csv |
| AC-5.7 | checkpoint 含完整推理元数据；infer 从 checkpoint 读映射；epoch 字段语义清晰（trained_epochs/dev_best_epoch/epoch_selection_split/epoch_selection_policy） |
| AC-5.8 | 每个方法 epoch 独立冻结；confirm/final-fit 不改变 |
| AC-5.9 | 消融公平性：E0/E1 等预算；C0-C2 独立；E2-E4 固定 E0 超参 |
| AC-5.10 | 测试集覆盖：先检查 basename 唯一性；行数=唯一文件名数=测试图片数；集合完全相等；无重复无缺失无额外 |
| AC-5.11 | **B0 完整训练协议匹配**：B0 的 resolved config（optimizer, scheduler, epochs, batch_size, warmup, AMP, max_grad_norm, checkpoint_policy）与重构前 baseline reference 一致；B0 不采用新增调优规则（无 9-trial search，无 per-method epoch freezing） |
