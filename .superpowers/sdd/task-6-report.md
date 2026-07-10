# Task 6: CachedFeatureDataset with Full Validation

## Status: Complete

## Changes

### Modified: `common/cache.py`
- Appended `CachedFeatureDataset` class (223 lines) to the existing `FeatureCacheBuilder`
- Class validates 6 aspects on init:
  1. Manifest hard fields (`backbone`, `pretrained_source`, `feature_dim`, `normalized`, `dtype`, `preprocess`, `feature_encode_amp`, `autocast_dtype`) -- raises `ValueError` on mismatch
  2. Environment version fields (torch, torchvision, python) -- logs warnings on differences
  3. Dataset fingerprint (quick or full) -- raises `ValueError` on mismatch
  4. Tensor validation: ndim==2, shape[1]==512, dtype==float32, finite values, no duplicate paths
  5. `class_mapping_hash` verification against current `class_to_idx.json`
  6. Per-sample label consistency between CSV split and cached labels
- Returns `(feature_tensor, label)` per sample (no image path)
- Dependencies use existing `compute_quick_fingerprint`, `compute_full_fingerprint` from same module

### Created: `tests/test_cache.py`
- `make_dummy_cache_dir()` helper to create a minimal valid cache directory for testing
- `test_cached_dataset_rejects_missing_manifest()` -- verifies `FileNotFoundError` is raised when no `manifest.json` exists

## Commit

```
723f75e feat: add CachedFeatureDataset with full validation
```

## Test Results

```
tests/test_cache.py::test_cached_dataset_rejects_missing_manifest PASSED
```

All 26 existing tests continue to pass (no regressions).

---

## Task 6 Review Fixes

**Date:** 2026-07-10

### Fixes Applied

#### `common/cache.py`

1. **Issue 1 (Important): `autocast_dtype: None` bypassed hard-field check**
   - Added `_MISSING = object()` sentinel at module level
   - Changed `_validate_hard_fields` to use `EXPECTED_HARD_VALUES.get(field, _MISSING)` and `expected is not _MISSING` comparison
   - `EXPECTED_HARD_VALUES["autocast_dtype"] = None` is now actually enforced

2. **Issue 2 (Minor): Dead `img_path.name` fallback in `_load_split`**
   - Added defense-in-depth comment explaining the bare filename lookup is a final fallback

3. **Issue 3 (Minor): Local re-imports in `_check_version_fields` and `_build_manifest`**
   - Moved `import sys` and `import torchvision` to module-level imports
   - Removed all local `import sys`, `import torch as _torch`, `import torchvision as _tv` from inside methods
   - Updated references from `_torch.__version__` to `torch.__version__` and `_tv.__version__` to `torchvision.__version__`

4. **Issue 5 (Minor): No existence check for `class_to_idx_path`**
   - Added `if not Path(class_to_idx_path).exists(): raise FileNotFoundError(...)` before opening

#### `tests/test_cache.py`

5. **Issue 4 (Minor): Weak test coverage**
   - Added `manifest.json` creation to `make_dummy_cache_dir` helper with valid hard fields and computed `class_mapping_hash`
   - Added `test_cached_dataset_hard_field_mismatch` -- corrupts `backbone` field in manifest, verifies `ValueError`
   - Added `test_cached_dataset_successful_load` -- creates real dataset directory structure, computes quick fingerprint, builds split CSV, and verifies full load pipeline succeeds

### Test Results (28/28 pass)

```
tests/test_cache.py::test_cached_dataset_rejects_missing_manifest PASSED
tests/test_cache.py::test_cached_dataset_hard_field_mismatch PASSED
tests/test_cache.py::test_cached_dataset_successful_load PASSED
```
