"""Implement/inspect the team-designed Aegis-Aligned448 strategy; compute requires an explicit flag."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "targets", "train", "calibrate", "infer"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--execute", action="store_true", help="Explicitly enable the requested compute stage")
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.action != "preflight" and not args.execute:
        parser.error("Implementation-only default: compute stages require --execute")
    if args.resume and args.action != "train":
        parser.error("--resume is only valid for train")
    if args.action in ("calibrate", "infer") and not args.checkpoint:
        parser.error("Specify one explicit --checkpoint")
    from aegis_clip.aligned448_pipeline import StageContext, load_recipe, prepare_targets, train, calibrate, infer
    from aegis_clip.runtime import atomic_json_dump
    context = StageContext(load_recipe(args.config, args.source_root))
    if args.action == "preflight":
        report = context.summary()
        if args.report:
            atomic_json_dump(report, Path(args.report))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Local CUDA is unavailable; no remote or NPU fallback")
    functions = {"targets": lambda: prepare_targets(context, args.device),
                 "train": lambda: train(context, args.device, args.resume),
                 "calibrate": lambda: calibrate(context, args.checkpoint, args.device),
                 "infer": lambda: infer(context, args.checkpoint, args.device)}
    print(functions[args.action]())


if __name__ == "__main__":
    main()
