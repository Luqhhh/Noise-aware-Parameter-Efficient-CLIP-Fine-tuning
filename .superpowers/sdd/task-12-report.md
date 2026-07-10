# Task 12 Report: B0 Regression Fixture

## Summary

Created `experiments/baseline/b0_regression.py` with the `B0_FIXTURE` dictionary
and `save_b0_fixture()` function. Created `configs/b0_regression.yaml` with
the B0-specific configuration.

## Files Created

| File | Lines | Description |
|------|-------|-------------|
| `experiments/baseline/b0_regression.py` | ~50 | B0_FIXTURE dict + save_b0_fixture() helper |
| `configs/b0_regression.yaml` | ~55 | B0 configuration YAML |

## Key Design Decisions

1. **B0_FIXTURE dict**: Contains the resolved hyperparameters (lr=1e-3,
   wd=1e-4, epochs=20, batch_size=128, AdamW, CosineAnnealingLR, warmup=1,
   AMP, max_grad_norm=1.0, split_seed=42, train_seed=42). This serves as a
   regression anchor — any refactoring that changes B0 results must be detected.

2. **save_b0_fixture()**: Saves the fixture as JSON for easy comparison. The
   output path is logged on save.

3. **B0 config uses online encoding**: `model.use_cached_features: false` is
   explicit. B0 is the only experiment that MUST NOT use cached features.

4. **B0 does NOT adopt new tuning rules**: The config specifies the original
   hyperparameters directly (no 9-trial search, no per-method epoch freezing).

## Usage

```python
from experiments.baseline.b0_regression import B0_FIXTURE, save_b0_fixture

save_b0_fixture("outputs/baseline")
```

```bash
python -m experiments.baseline.train --config configs/b0_regression.yaml
```

## Dependencies

- `json`, `pathlib.Path` (stdlib)
- `experiments/baseline/train.py` (when training with this config)
