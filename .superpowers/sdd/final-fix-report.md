# Final Fix Report — Cosine & Augmentation Trainer Refactoring

## Summary

Refactored `experiments/baseline/train.py` to be model-agnostic (supports `--head-type linear|cosine`), and reduced `experiments/cosine/train.py` and `experiments/augmentation/train.py` to thin import-only wrappers. All training logic now lives in a single canonical location.

## Changes Made

### 1. `experiments/baseline/train.py` — Head type support
- Added `--head-type` argument (choices: linear/cosine, default: linear)
- Added `--cos-init-scale` and `--cos-learnable-scale` arguments (override config)
- When `head_type == "cosine"`: builds `CosineClassifier` from `experiments.cosine.model`
- When `head_type == "linear"`: builds `CLIPLinearClassifier` from `experiments.baseline.model`
- `_build_optimizer_and_scheduler()`: uses `model.get_param_groups(lr, wd)` for cosine head (handles logit_scale separately), falls back to `model.get_trainable_parameters()` for linear
- `train_one_epoch()`: calls `model.clamp_scale()` after each optimizer step for cosine head
- Removed unused top-level `from .model import build_model` (now imported conditionally)

### 2. `experiments/cosine/train.py` — Thin wrapper
- Replaced entire 548-line standalone copy with an import-only wrapper that passes `--head-type cosine` to the baseline trainer

### 3. `experiments/cosine/evaluate.py` & `infer.py` — Thin wrappers
- Both replaced with thin wrappers importing from `experiments.baseline.evaluate` / `experiments.baseline.infer`

### 4. `experiments/augmentation/train.py` — Thin wrapper
- Replaced entire 648-line standalone copy with an import-only wrapper (no extra args — augmentation preset is read from config by baseline trainer)

### 5. `experiments/baseline/evaluate.py` & `infer.py` — Head type support
- Added `--head-type`, `--cos-init-scale`, `--cos-learnable-scale` arguments
- Conditional model building based on head_type (same pattern as train.py)
- Removed unused top-level `from .model import build_model`

### 6. `configs/e5_combined.yaml` — Fixed augmentation preset
- Changed `augmentation_preset: best_aug` to `augmentation_preset: a3` with a comment noting it's a placeholder

### 7. Redundant `.float()` calls removed
- `experiments/baseline/model.py`: removed `features = features.float()` in `encode_image()` (redundant — visual encoder is already float32 from `load_openai_clip`)
- `experiments/baseline/model.py`: removed `clip_model.visual = clip_model.visual.float()` in `build_model()` (redundant — `load_openai_clip` already does `model = model.float()`)
- `experiments/cosine/model.py`: removed `features = features.float()` in `encode_image()` (same reason)

### 8. Tests
- All 57 tests pass
- All cross-module imports verified

## Key Design Decisions

- **Config-driven head_type**: The `experiment.head_type` field in YAML configs is the primary source. CLI `--head-type` overrides it. This matches the existing configs which already have `head_type: cosine` or `head_type: linear`.
- **Duck typing for optimizer**: The code checks `hasattr(model, "get_param_groups")` rather than passing `head_type` through the optimizer builder — cleaner separation of concerns.
- **Duck typing for clamp_scale**: Same pattern — `hasattr(model, "clamp_scale")` after `optimizer.step()`.
- **E5 `augmentation_preset: a3`**: Placeholder value. Must be updated after ablation studies identify the best augmentation preset.
