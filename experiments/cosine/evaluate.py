"""
Cosine experiment evaluator. Delegates to baseline evaluator with head_type=cosine.

Usage:
    python -m experiments.cosine.evaluate --config configs/e1_hyper_search.yaml \
        --ckpt outputs/e1/checkpoints/best.pt

This is a thin wrapper. All evaluation logic lives in experiments.baseline.evaluate.
"""

import sys
from experiments.baseline.evaluate import main

if __name__ == "__main__":
    sys.argv.append("--head-type")
    sys.argv.append("cosine")
    main()
