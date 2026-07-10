"""
Cosine experiment inference. Delegates to baseline inference with head_type=cosine.

Usage:
    python -m experiments.cosine.infer --config configs/e1_hyper_search.yaml \
        --ckpt outputs/e1/checkpoints/best.pt

This is a thin wrapper. All inference logic lives in experiments.baseline.infer.
"""

import sys
from experiments.baseline.infer import main

if __name__ == "__main__":
    sys.argv.append("--head-type")
    sys.argv.append("cosine")
    main()
