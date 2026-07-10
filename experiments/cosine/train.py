"""
Cosine experiment trainer. Delegates to baseline trainer with head_type=cosine.

Usage:
    python -m experiments.cosine.train --config configs/e1_hyper_search.yaml
    python -m experiments.cosine.train --config configs/c0_cosine_scale.yaml

This is a thin wrapper. All training logic lives in experiments.baseline.train.
"""

import sys
from experiments.baseline.train import main

if __name__ == "__main__":
    sys.argv.append("--head-type")
    sys.argv.append("cosine")
    main()
