#!/usr/bin/env python3
"""Read-only real-source preflight and FP32/eval conversion checks; never use CUDA."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))

import torch
from aegis_clip.runtime import atomic_json_dump, sha256_file
from v1_continuation.plan import inspect_source, load_config, source_hashes
from v1_continuation.runtime import initial_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    records = []
    for route in ("wft448", "lr512"):
        for population in ("dev", "full"):
            path = ROOT / "configs/v1_continuation" / f"{route}_{population}.yaml"
            cfg = load_config(path)
            context, supervision = inspect_source(cfg)
            record = dict(experiment_id=cfg["experiment_id"], partition=cfg["partition"],
                source_binding=context.binding, parent_binding=context.parent_binding, source_hashes=source_hashes(cfg),
                config_sha256=sha256_file(path), train_rows=len(context.train), val_rows=len(context.val),
                active_train_rows=int((supervision["weights"] > 0).sum()), logical_batch_size=context.config["train"]["batch_size"],
                source_population_verified=True, full_prepared=False)
            if population == "dev":
                model, check = initial_model(dict(config=cfg), context)
                record.update(zero_update=check, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                              cosine_head_preserved=model.head.__class__.__name__ == "CosineHead")
                del model
                gc.collect()
            records.append(record)
            print(cfg["experiment_id"], "source verified", "zero check passed" if population == "dev" else "not prepared", flush=True)
    report = dict(status="cpu_verified", gpu_initialized=torch.cuda.is_initialized(), training_started=False,
                  records=records, local_score=None, platform_score=None)
    assert report["gpu_initialized"] is False
    atomic_json_dump(report, output)


if __name__ == "__main__":
    main()
