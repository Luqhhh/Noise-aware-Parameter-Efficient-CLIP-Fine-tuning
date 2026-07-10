"""
Evaluation script for augmentation experiments.

Thin wrapper around experiments.baseline.evaluate — evaluation protocol
is identical (deterministic CLIP preprocess, no augmentation).

Usage:
    python -m experiments.augmentation.evaluate --config configs/augmentation.yaml \
        --ckpt outputs/augmentation/checkpoints/best.pt
"""

from experiments.baseline.evaluate import main

if __name__ == "__main__":
    main()
