"""
Augmentation experiment trainer. Delegates to baseline trainer.

Augmentation configs (e2/e3/e4) set experiment.augmentation_preset in YAML,
so the baseline trainer reads augmentation_preset from config. Online encoding
(use_cached_features=False) is enforced by the baseline's guard mechanism.

Usage:
    python -m experiments.augmentation.train --config configs/e2_augmentation.yaml
    python -m experiments.augmentation.train --config configs/e3_augmentation.yaml
    python -m experiments.augmentation.train --config configs/e4_augmentation.yaml

This is a thin wrapper. All training logic lives in experiments.baseline.train.
"""

from experiments.baseline.train import main

if __name__ == "__main__":
    main()
