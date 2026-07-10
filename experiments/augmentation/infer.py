"""
Inference script for augmentation experiments.

Thin wrapper around experiments.baseline.infer — inference protocol is
identical (deterministic CLIP preprocess, no augmentation at test time).

Usage:
    python -m experiments.augmentation.infer --config configs/augmentation.yaml \
        --ckpt outputs/augmentation/checkpoints/best.pt
"""

from experiments.baseline.infer import main

if __name__ == "__main__":
    main()
