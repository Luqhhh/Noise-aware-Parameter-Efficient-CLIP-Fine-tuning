"""Verify the real official 448px B/32 strategy without initializing CUDA."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from aegis_clip.b448_pipeline import StageContext, load_recipe
from aegis_clip.b448_strategy import build_classifier
from aegis_clip.runtime import atomic_json_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU check requires a process without initialized CUDA")
    torch.set_num_threads(2)
    start = time.monotonic()
    context = StageContext(load_recipe(args.config))
    model = build_classifier(context.official, len(context.classes), context.config["model"], "cpu")
    model.eval()
    size = context.config["model"]["image_size"]
    torch.manual_seed(42)
    with torch.inference_mode():
        logits = model(torch.randn(1, 3, size, size))
    if logits.shape != (1, len(context.classes)) or not torch.isfinite(logits).all():
        raise RuntimeError("Official CPU forward failed")
    if torch.cuda.is_initialized() or any(p.device.type != "cpu" for p in model.parameters()):
        raise RuntimeError("CPU check initialized or used an accelerator")
    report = dict(status="passed", device="cpu", cuda_initialized=False, image_size=size,
        output_shape=list(logits.shape), adapted_modules=len(model.adapted_modules),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        backbone="OpenAI official CLIP ViT-B/32", training_started=False,
        elapsed_seconds=round(time.monotonic() - start, 3), binding=context.binding)
    atomic_json_dump(report, Path(args.report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
