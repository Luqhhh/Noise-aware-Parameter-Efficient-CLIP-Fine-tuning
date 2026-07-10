# Task 10 Report: Augmentation Experiments

## Summary

Created `experiments/augmentation/` with thin wrappers reusing the baseline
CLIPLinearClassifier. The augmentation train script is self-contained (imports
build_model from baseline) while evaluate/infer are one-line wrappers around
the baseline equivalents.

## Files Created

| File | Lines | Description |
|------|-------|-------------|
| `experiments/augmentation/__init__.py` | 0 | Package marker (empty) |
| `experiments/augmentation/train.py` | ~320 | Full training script with A1/A2/A3 transforms |
| `experiments/augmentation/evaluate.py` | 15 | One-line wrapper around baseline evaluate |
| `experiments/augmentation/infer.py` | 15 | One-line wrapper around baseline infer |

## Key Design Decisions

1. **Online encoding enforced**: The train script explicitly checks
   `use_cached_features` in config and raises `ValueError` if True. Random
   augmentations must be recomputed each epoch.

2. **Augmentation presets**: Three presets (A1/A2/A3) defined with increasing
   strength:
   - A1: RandomResizedCrop (scale=0.8-1.0) + RandomHorizontalFlip
   - A2: A1 + ColorJitter (brightness/contrast/saturation/hue)
   - A3: A2 + RandomErasing (after Normalize, p=0.5)

3. **Validation uses deterministic CLIP preprocess**: No augmentation on the
   validation set, identical to baseline behavior.

4. **drop_last=False**: All DataLoaders use `drop_last=False` per the spec.

5. **Config-driven hyperparameters**: LR, weight_decay, scheduler, and
   batch_size are read from the config file. E2/E3/E4 are expected to use
   E0's tuned values from their respective configs.

6. **Checkpoint metadata**: Checkpoints include `augmentation_preset` and
   `head_type` fields for traceability.

## Usage

```bash
# Train with augmentation preset a1 (E2)
python -m experiments.augmentation.train --config configs/e2_augmentation.yaml --preset a1

# Train with a2 (E3)
python -m experiments.augmentation.train --config configs/e3_augmentation.yaml --preset a2

# Train with a3 (E4)
python -m experiments.augmentation.train --config configs/e4_augmentation.yaml --preset a3

# Evaluate
python -m experiments.augmentation.evaluate --config configs/e2_augmentation.yaml \
    --ckpt outputs/e2/checkpoints/best.pt

# Infer
python -m experiments.augmentation.infer --config configs/e2_augmentation.yaml \
    --ckpt outputs/e2/checkpoints/best.pt
```

## Dependencies

- `experiments/baseline/model.py` for `build_model` (CLIPLinearClassifier)
- `common/dataset.py` for `TrainImageDataset`
- `common/utils.py` for config loading, logging, seed management
- `torchvision.transforms` for augmentation pipelines
