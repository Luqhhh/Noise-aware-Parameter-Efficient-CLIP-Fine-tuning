"""Measure one fixed v1 recipe on local CUDA without retaining fitted weights."""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from aegis_clip.runtime import atomic_json_dump
from aegis_clip.v1_pipeline import StageContext, image_transform, load_recipe, loader_for, seed_training
from aegis_clip.v1_strategy import (
    build_classifier, knn_signals, target_probabilities, weighted_mixup_loss,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Local CUDA cost measurement requires --execute")
    context = StageContext(load_recipe(args.config))
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA is unavailable")
    cfg = context.config["train"]
    seed_training(context.config["project"]["seed"])
    model = build_classifier(context.official, len(context.classes), context.config["model"], "cuda")
    opt = torch.optim.AdamW([
        dict(params=[p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")],
             lr=cfg["lora_lr"], weight_decay=0.0),
        dict(params=list(model.head.parameters()), lr=cfg["head_lr"], weight_decay=.01)])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    rows = context.train[:5 * cfg["batch_size"]]
    loader = loader_for(context, rows, image_transform(context.config["model"]["image_size"],
        training=True, crop_min=cfg["crop_min_scale"]))
    labels = torch.tensor([int(r["label"]) for r in rows], device="cuda")
    probabilities = target_probabilities(labels, labels, torch.zeros_like(labels, dtype=torch.float32),
        len(context.classes), cfg["label_smoothing"])
    times, losses, updates = [], [], 0
    torch.cuda.reset_peak_memory_stats()
    model.train()
    previous = time.monotonic()
    for batch, (images, indices) in enumerate(loader):
        images, indices = images.to("cuda"), indices.to("cuda")
        permutation = torch.randperm(len(images), device="cuda")
        lam = float(np.random.beta(cfg["mixup_alpha"], cfg["mixup_alpha"])) if cfg["mixup_alpha"] else 1.0
        images = lam * images + (1 - lam) * images[permutation]
        opt.zero_grad(set_to_none=True)
        autocast = torch.autocast("cuda", dtype=torch.float16) if cfg["amp"] else contextlib.nullcontext()
        with autocast:
            loss = weighted_mixup_loss(model(images), probabilities[indices],
                torch.ones(len(images), device="cuda"), permutation, lam)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite CUDA probe loss")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        old = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        updates += int(scaler.get_scale() >= old)
        torch.cuda.synchronize()
        now = time.monotonic()
        if batch >= 2:
            times.append(now - previous)
        losses.append(float(loss.detach()))
        previous = now
    if updates < 1:
        raise ValueError("CUDA probe had no successful optimizer update")
    training_peak = torch.cuda.max_memory_allocated()
    model.eval()
    eval_times = []
    with torch.no_grad():
        for images, _ in loader_for(context, rows[:3 * cfg["batch_size"]],
                image_transform(context.config["model"]["image_size"])):
            started = time.monotonic()
            logits = model(images.to("cuda"))
            torch.cuda.synchronize()
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite CUDA inference logits")
            eval_times.append(time.monotonic() - started)
    del model, opt, probabilities, logits, loss, images
    torch.cuda.empty_cache()
    features = context.features().to("cuda")
    all_labels = torch.tensor([int(r["label"]) for r in context.train], device="cuda")
    group_names = {name: i for i, name in enumerate(sorted({r["content_group"] for r in context.train}))}
    groups = torch.tensor([group_names[r["content_group"]] for r in context.train], device="cuda")
    query_rows = min(1024, len(features))
    denoise = context.config["denoise"]
    torch.cuda.synchronize()
    started = time.monotonic()
    knn_signals(features[:query_rows], all_labels[:query_rows], features, all_labels, len(context.classes),
        k=denoise["knn_k"], query_groups=groups[:query_rows], gallery_groups=groups,
        query_chunk=denoise["query_chunk"], gallery_chunk=denoise["gallery_chunk"])
    torch.cuda.synchronize()
    knn_seconds = time.monotonic() - started
    step = float(np.mean(times))
    inference = float(np.mean(eval_times[1:]))
    train_batches = int(np.ceil(len(context.train) / cfg["batch_size"]))
    val_batches = int(np.ceil(len(context.val) / cfg["batch_size"]))
    test_batches = int(np.ceil(context.manifest["test_samples"] / cfg["batch_size"]))
    views = len(context.config["decode"]["scales"]) * (2 if context.config["decode"]["flip"] else 1)
    estimate = step * train_batches * cfg["epochs"] + inference * (
        val_batches * (cfg["epochs"] + 1 + views) + test_batches * views)
    estimate += knn_seconds * len(features) / query_rows
    report = dict(status="passed", device=torch.cuda.get_device_name(), binding=context.binding,
        batch_size=cfg["batch_size"], image_size=context.config["model"]["image_size"],
        warmup_batches=2, measured_train_batches=3, successful_optimizer_updates=updates,
        measured_step_seconds=times, mean_step_seconds=step, mean_inference_batch_seconds=inference,
        training_peak_allocated_mib=training_peak / 2**20, query_rows=query_rows,
        knn_probe_seconds=knn_seconds, all_rows_estimated_knn_seconds=knn_seconds * len(features) / query_rows,
        conservative_full_train_rows=len(context.train), estimated_pipeline_hours=estimate / 3600,
        estimate_excludes_teacher_and_startup=True, fitted_weights_saved=False, losses=losses)
    atomic_json_dump(report, Path(args.report))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
