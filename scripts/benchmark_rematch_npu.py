"""Bounded real-trainer NPU throughput probes; never use test images or select by accuracy.

The probe exits before validation/checkpoint selection. Each case starts from the
same stage LP, uses the full training sampler/schedule, and measures complete
steps after warmup. Optional synchronized section timing is diagnostic only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import functools
import json
import hashlib
import os
from pathlib import Path
import statistics
import time
import types

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


class ProbeComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--optimizer", choices=["adamw", "foreach", "fused"], default="adamw")
    parser.add_argument("--norm", choices=["original", "vector", "foreach"], default="original")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--production", action="store_true", help="Use integrated trainer options, without optimizer/norm substitution")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--loss-phase", choices=["ce", "gce"], default="ce")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum() or min(args.steps, args.batch) <= 0:
        raise ValueError("Invalid bounded probe arguments")
    from aegis_clip.device import resolve_device
    from aegis_clip.config import load_config
    import aegis_clip.trainer as trainer
    resolve_device("npu:0")
    torch.set_num_threads(args.threads)
    config_path = ROOT / f"configs/npu_probe_{args.name}.local.yaml"
    raw = yaml.safe_load((ROOT / "configs/rematch750_ft_npu.yaml").read_text())
    raw["project"]["experiment_id"] = args.name
    raw["output"]["root"] = "../outputs/npu_tuning/benchmarks"
    raw["train"].update(batch_size=args.batch, num_workers=args.workers,
                        prefetch_factor=args.prefetch, log_every_steps=20,
                        npu_pin_memory=args.pin_memory)
    if args.loss_phase == "gce":
        raw["loss"]["ce_warmup_epochs"] = 0
    if args.production:
        raw["train"]["optimizer_impl"] = {"adamw": "default", "foreach": "foreach", "fused": "npu_fused_adamw"}[args.optimizer]
        raw["train"]["gradient_norm_impl"] = {"original": "sum_squares", "foreach": "foreach", "vector": "vector"}[args.norm]
    if config_path.exists():
        raise FileExistsError(config_path)
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = load_config(config_path)
    timings = []
    waits = []
    sections = defaultdict(list)
    state = {"index": -1, "optimizer": None}
    result = dict(arguments=vars(args), status="running", training_source=config["data"]["train_csv"],
                  test_data_used=False, config_path=str(config_path), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), ROOT / "reproducibility/aegis_f1/aegis_clip/trainer.py",
                                           ROOT / "reproducibility/aegis_f1/aegis_clip/optim.py"] if p.exists()})
    out = ROOT / f"results/npu_tuning/{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def timed(name, function):
        @functools.wraps(function)
        def wrapper(*a, **kw):
            if not args.profile or state["index"] < args.warmup:
                return function(*a, **kw)
            torch.npu.synchronize()
            start = time.perf_counter()
            value = function(*a, **kw)
            torch.npu.synchronize()
            sections[name].append(time.perf_counter() - start)
            return value
        return wrapper

    original_adam = torch.optim.AdamW

    def make_optimizer(*a, **kw):
        if args.optimizer == "fused":
            from torch_npu.optim import NpuFusedAdamW
            kw.pop("foreach", None)
            optimizer = NpuFusedAdamW(*a, **kw)
            # Fused storage requires persistent gradient buffers.
            zero_grad = optimizer.zero_grad
            optimizer.zero_grad = lambda set_to_none=False: zero_grad(set_to_none=False)
        else:
            kw["foreach"] = args.optimizer == "foreach"
            optimizer = original_adam(*a, **kw)
        state["optimizer"] = optimizer
        if args.profile:
            optimizer.step = types.MethodType(timed("optimizer", optimizer.step.__func__), optimizer)
        return optimizer

    if args.production:
        import aegis_clip.optim as optim_module
        original_builder = optim_module.build_adamw
        def production_builder(*a, **kw):
            optimizer = original_builder(*a, **kw)
            state["optimizer"] = optimizer
            if args.profile:
                optimizer.step = types.MethodType(timed("optimizer", optimizer.step.__func__), optimizer)
            return optimizer
        optim_module.build_adamw = production_builder
    else:
        torch.optim.AdamW = make_optimizer
    if args.norm != "original" and not args.production:
        def gradient_norm(parameters, implementation=None):
            grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
            if not grads:
                return 0.0
            norms = torch._foreach_norm(grads) if args.norm == "foreach" else [torch.linalg.vector_norm(g) for g in grads]
            return float(torch.linalg.vector_norm(torch.stack(norms)))
        trainer._gradient_norm = gradient_norm
    trainer._gradient_norm = timed("gradient_norm", trainer._gradient_norm)
    torch.nn.utils.clip_grad_norm_ = timed("clip_grad_norm", torch.nn.utils.clip_grad_norm_)
    torch.Tensor.backward = timed("backward", torch.Tensor.backward)
    trainer._per_sample_loss = timed("loss", trainer._per_sample_loss)
    from aegis_clip.model import AegisCLIP
    AegisCLIP.forward = timed("forward", AegisCLIP.forward)
    original_loader = trainer.DataLoader

    class MeasuredLoader:
        def __init__(self, loader):
            self.loader = loader

        def __len__(self):
            return len(self.loader)

        def __iter__(self):
            iterator = iter(self.loader)
            for index in range(args.warmup + args.steps):
                state["index"] = index
                torch.npu.synchronize()
                start = time.perf_counter()
                batch = next(iterator)
                if args.pin_memory and index == 0:
                    assert batch["images"].is_pinned(), "Requested NPU pinned memory was not active"
                wait = time.perf_counter() - start
                yield batch
                torch.npu.synchronize()
                if index >= args.warmup:
                    timings.append(time.perf_counter() - start)
                    waits.append(wait)
            raise ProbeComplete()

    def make_loader(*a, **kw):
        loader = original_loader(*a, **kw)
        return MeasuredLoader(loader) if kw.get("shuffle") or kw.get("sampler") else loader

    trainer.DataLoader = make_loader
    torch.npu.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        trainer.train(config)
        raise RuntimeError("Probe unexpectedly exhausted training")
    except ProbeComplete:
        steps = sorted({int(v["step"]) for v in state["optimizer"].state.values() if "step" in v})
        assert steps == [args.warmup + args.steps], steps
        result.update(status="passed", optimizer_step_values=steps,
                      mean_step_seconds=statistics.mean(timings), median_step_seconds=statistics.median(timings),
                      images_per_second=args.batch / statistics.mean(timings),
                      mean_loader_wait_seconds=statistics.mean(waits),
                      peak_allocated_gib=torch.npu.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.npu.max_memory_reserved()/2**30,
                      sections_seconds_per_step={k: sum(v)/args.steps for k, v in sections.items()},
                      step_seconds=timings, loader_wait_seconds=waits)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["total_wall_seconds"] = time.perf_counter() - start
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k:v for k,v in result.items() if k not in {"step_seconds", "loader_wait_seconds"}}, indent=2))


if __name__ == "__main__":
    main()
