# P0 Fixes Report

Date: 2026-07-10

## Changes Made

### P0-1: Fix YAML vs CLI config priority

**Files modified:**
- `experiments/baseline/train.py` — Changed `--mode`, `--use-cached-features`, `--augmentation-preset` defaults from hardcoded values to `None`. Changed `--use-cached-features` to use `argparse.BooleanOptionalAction` so explicit CLI flags are distinguishable from "not provided". Replaced inline `mode`/`head_type`/`use_cached`/`aug_preset` resolution with `resolve_runtime_args()` call. Added validation checks for resolved values. Added combined resolved-runtime logging line.
- `common/runtime_config.py` — New file. Contains `_pick()` helper and `resolve_runtime_args()` function implementing priority: explicit CLI > YAML > hard default. Writes resolved values back into config dict for checkpoint/snapshot preservation.

### P0-2: Fix cached feature training interface

**Files modified:**
- `experiments/baseline/model.py` — Added `forward_features()` method to `CLIPLinearClassifier`. Refactored `forward()` to call `self.forward_features(features)`.
- `experiments/cosine/model.py` — Removed `* init_scale` from weight initialization (gets normalized away in forward). Added `forward_features()` method. Refactored `forward()` to call `self.forward_features(features)`. Refactored `clamp_scale()` to use `@torch.no_grad()` decorator and return `None`.
- `experiments/baseline/train.py` — Added `_unpack_batch()` and `_forward_inputs()` helper functions. Updated `train_one_epoch()` and `validate()` to use them, so cached feature batches are dispatched to `model.forward_features()` instead of `model()` (which would attempt CLIP visual encoding on feature tensors).
- `common/cache.py` — Modified `CachedFeatureDataset._load_split()` to accept `split_csv=None` (full dataset mode).
- `experiments/baseline/train.py` — Restructured data branching: `if use_cached:` checked first (handles final_fit+cached via `split_csv=None`), then `elif mode == "final_fit":` (non-cached final_fit), then `else:` (standard online). This ensures E0/E1 final_fit with cached features correctly uses the cached dataset.

### P0-3: Behavioral tests

**Files created:**
- `tests/test_runtime_config.py` — Tests that YAML values are used when CLI args are absent, and CLI values override YAML.
- `tests/test_cached_forward.py` — Tests `forward_features()` shape + gradient flow for both Linear and Cosine models, and tests that online and cached forward paths produce identical logits.
- `scripts/check_resolved_configs.py` — Validates all 7 experiment config files resolve to expected runtime values.

**Files modified (pre-existing tests):**
- `tests/test_cosine.py` — Updated `test_cosine_fixed_scale` and `test_cosine_clamp` to reflect that `clamp_scale()` now returns `None` (previously returned the clamped tensor).

## Test Results

```
tests/test_runtime_config.py ...                          [PASS] (2/2)
tests/test_cached_forward.py ...                          [PASS] (4/4)
tests/test_cosine.py (updated) ...                        [PASS] (2/2)
All other existing tests ...                              [PASS] (57/57)
```

Only pre-existing failure: `test_integration.py::test_full_pipeline_smoke` (unrelated — missing `generate_class_mapping` in `common/class_mapping.py`).

Config check: 7/7 pass.

## Git Commits

Three separate commits (one per P0 fix) as required.
