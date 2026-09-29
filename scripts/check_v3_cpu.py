#!/usr/bin/env python3
"""Official v3 initialization/384px forward on CPU, without training."""
from pathlib import Path
import argparse
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))
from v2.model import V2Classifier
from v2.plan import dump, require, sha
from v3.core import isolated_cpu_rng
from v3.plan import verify
from v3.runtime import state_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists(), "Refusing to overwrite CPU verification")
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), "CPU verification must not initialize CUDA")
    plan = verify(args.plan)
    hashes = []
    for _ in range(2):
        with isolated_cpu_rng(plan["recipe"]["seed"]):
            model = V2Classifier(plan["model_recipe"]).eval()
        hashes.append(state_sha256(model.state_dict()))
        with torch.no_grad():
            logits = model(torch.zeros(1, 3, plan["recipe"]["image_size"], plan["recipe"]["image_size"]))
        require(tuple(logits.shape) == (1, len(plan["classes"])) and torch.isfinite(logits).all(),
                "Invalid official 384px forward")
        count = sum(p.numel() for p in model.parameters())
        require(all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()),
                "All visual/head parameters must be trainable FP32")
        del model
    require(hashes[0] == hashes[1], "The two arm initializations differ")
    require(not torch.cuda.is_initialized(), "CPU verification initialized CUDA")
    report = dict(status="official_cpu_forward_passed", plan_sha256=sha(args.plan),
                  logits_shape=list(logits.shape), trainable_parameters=count,
                  same_initial_weights=True, initial_model_sha256=hashes[0],
                  gpu_started=False, training_started=False, cuda_initialized=False,
                  local_metrics=None, platform_metrics=None)
    dump(args.output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
