"""Implement/inspect the team-designed v1 strategy; compute requires an explicit flag."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "targets", "train", "calibrate", "infer", "export-swa"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--execute", action="store_true", help="Explicitly enable the requested compute stage")
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--report")
    parser.add_argument("--training-dir", help="Source epoch directory for export-swa; default: recipe output/training")
    parser.add_argument("--output-dir", help="New independent export-swa output directory")
    parser.add_argument("--start-epoch", type=int, help="First averaged epoch; default: recipe swa_start")
    parser.add_argument("--end-epoch", type=int, help="Last averaged epoch; default: recipe epochs")
    parser.add_argument("--weight-source", choices=["ema", "raw"], help="Averaged weights; default: ema")
    args = parser.parse_args()
    if args.action != "preflight" and not args.execute:
        parser.error("Implementation-only default: compute stages require --execute")
    if args.resume and args.action != "train":
        parser.error("--resume is only valid for train")
    if args.action in ("calibrate", "infer") and not args.checkpoint:
        parser.error("Specify one explicit --checkpoint")
    if args.action == "export-swa":
        if not args.output_dir:
            parser.error("export-swa requires a new --output-dir")
        if args.device != "cpu":
            parser.error("export-swa is a CPU operation; use --device cpu")
        if args.checkpoint:
            parser.error("export-swa reads an epoch window from --training-dir")
    elif any(value is not None for value in (args.training_dir, args.output_dir, args.start_epoch,
                                            args.end_epoch, args.weight_source)):
        parser.error("SWA export options are only valid for export-swa")
    from aegis_clip.v1_pipeline import StageContext, load_recipe, prepare_targets, train, calibrate, infer
    from aegis_clip.runtime import atomic_json_dump
    context = StageContext(load_recipe(args.config, args.source_root))
    if args.action == "preflight":
        report = context.summary()
        if args.report:
            atomic_json_dump(report, Path(args.report))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.action == "export-swa":
        from aegis_clip.v1_export import export_swa
        try:
            checkpoint = export_swa(context, args.output_dir, training_dir=args.training_dir,
                                    start_epoch=args.start_epoch, end_epoch=args.end_epoch,
                                    weight_source=args.weight_source or "ema")
        except (FileNotFoundError, FileExistsError, ValueError) as error:
            parser.error(str(error))
        print(checkpoint)
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
