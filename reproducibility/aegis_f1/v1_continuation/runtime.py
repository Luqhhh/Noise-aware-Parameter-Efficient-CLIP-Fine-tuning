"""Explicit local execution of fixed four-epoch continuations and single-model inference."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from aegis_clip.runtime import atomic_json_dump, seed_worker, sha256_file
from aegis_clip.v1_pipeline import Images, image_transform, load_artifact, read_rows, save_artifact, seed_training
from aegis_clip.v1_strategy import (WeightAverage, build_classifier, load_trainable_state,
                                  target_probabilities, trainable_state, using_weights)
from v3.report import paired_metrics
from .model import convert_parent
from .plan import ROOT, verify


def require_idle_cuda():
    """No waiting daemon, preemption, remote fallback, or implicit execution."""
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    active = [line.strip() for line in result.stdout.splitlines()
              if line.strip() and line.strip() != "[Not Supported]"]
    if active:
        raise RuntimeError("Local CUDA is occupied; run this fixed task after existing work finishes")
    if "[Not Supported]" in result.stdout:
        raise RuntimeError("Cannot establish that local CUDA is idle")
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA is unavailable; no NPU/remote fallback")


def initial_model(plan, context):
    cfg = plan["config"]
    model = build_classifier(context.official, len(context.classes), context.config["model"], "cpu")
    parent = load_artifact(cfg["parent_checkpoint"], context.parent_binding)
    # Never initialize the head from teacher_state or reload optimizer/scheduler.
    load_trainable_state(model, parent["selected_state"])
    sample_rows = context.train[:2]
    images = torch.stack([Images(context.train_root, sample_rows, image_transform(448))[i][0]
                          for i in range(len(sample_rows))])
    model, check = convert_parent(model, cfg["route"], images)
    return model, check


def optimizer_for(model, recipe):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        head = name.startswith("head.")
        if recipe["optimizer"] == "v1_adamw":
            decay = .01 if head else 0.
        else:
            decay = recipe["weight_decay"] if parameter.ndim >= 2 and not name.endswith("bias") else 0.
        key = (head, decay)
        groups.setdefault(key, []).append(parameter)
    return torch.optim.AdamW([dict(params=parameters, lr=recipe["head_lr"] if head else recipe["backbone_lr"],
                                   weight_decay=decay) for (head, decay), parameters in groups.items()])


def scheduler_for(optimizer, recipe, steps_per_epoch):
    total = recipe["epochs"] * steps_per_epoch
    if recipe["scheduler"] == "v1_onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, [group["lr"] for group in optimizer.param_groups],
                                                   total_steps=total, pct_start=.1, anneal_strategy="cos")
    warmup = max(1, math.ceil(recipe["warmup_epochs"] * steps_per_epoch))
    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1., (step-warmup)/max(1, total-warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def mixed_backward(model, images, probabilities, weights, permutation, lam, micro_batch, scaler, amp=False):
    """Mix ONCE across the actual logical batch; normalize by its total reliability."""
    mixed_images = lam * images + (1-lam) * images[permutation]
    mass = weights[:, None] * probabilities
    mixed_mass = lam * mass + (1-lam) * mass[permutation]
    denominator = weights.sum().clamp_min(1e-8)
    loss_total = 0.
    for start in range(0, len(images), micro_batch):
        stop = start + micro_batch
        autocast = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
        with autocast:
            logits = model(mixed_images[start:stop])
            loss = -(mixed_mass[start:stop] * logits.float().log_softmax(1)).sum() / denominator
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite continuation loss")
        scaler.scale(loss).backward()
        loss_total += float(loss.detach())
    return loss_total


def logical_update(model, optimizer, images, probabilities, weights, permutation, lam,
                   micro_batch, scaler, amp, grad_clip):
    """Keep every logical update when FP16 loss scaling overflows.

    Recompute the same augmented/mixed batch with a lower scale and identical
    forward RNG. No optimizer, scheduler or EMA step happens on failed attempts.
    Persistent nonfinite gradients at scale 1 still fail explicitly.
    """
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(images.device) if images.is_cuda else None
    initial_scale = scaler.get_scale()
    retries = []
    for attempt in range(17):
        if attempt:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, images.device)
        optimizer.zero_grad(set_to_none=True)
        loss = mixed_backward(model, images, probabilities, weights, permutation, lam,
                              micro_batch, scaler, amp)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(norm):
            scale = scaler.get_scale()
            if not amp or scale <= 1 or attempt == 16:
                raise ValueError("Persistent nonfinite gradient norm after bounded AMP scale reduction")
            # unscale_ already recorded found_inf; use GradScaler's normal
            # backoff so its growth counter resets as well.
            scaler.update()
            if scaler.get_scale() >= scale:
                raise ValueError("AMP scale failed to decrease on a nonfinite gradient")
            retries.append(dict(old_scale=scale, new_scale=scaler.get_scale()))
            continue
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            raise ValueError("Unexpected skipped optimizer update despite finite gradients")
        return loss, dict(gradient_norm=float(norm), amp_initial_scale=initial_scale,
                          amp_final_scale=scaler.get_scale(), amp_retries=retries)
    raise AssertionError("Unreachable AMP retry state")


def eval_loader(plan, context, rows, scale, root=None):
    cfg = plan["config"]
    return DataLoader(Images(root or context.train_root, rows,
                            image_transform(cfg["recipe"]["image_size"], scale=scale)),
                      # Inference has no saved backward activations. Keep the v1
                      # batch32 throughput, independent of training accumulation.
                      batch_size=plan["logical_batch_size"], num_workers=cfg["num_workers"],
                      worker_init_fn=seed_worker, shuffle=False)


@torch.no_grad()
def evaluate(plan, context, model, rows, device, root=None, progress_path=None, phase=None):
    model.eval()
    classes = len(plan["classes"])
    result = torch.zeros(len(rows), classes)
    recipe = plan["config"]["recipe"]
    started, last_report = time.monotonic(), 0.
    for view, scale in enumerate(recipe["views"], 1):
        data = eval_loader(plan, context, rows, scale, root)
        for batch, (images, indices) in enumerate(data, 1):
            images = images.to(device)
            # Fixed FP32 eval, identical for the baseline and each raw/EMA candidate.
            logits = model(images)
            if recipe["flip"]:
                logits = (logits + model(images.flip(3))) / 2
            result[indices] += logits.float().cpu() / len(recipe["views"])
            now = time.monotonic()
            if progress_path and (batch == 1 or batch == len(data) or now-last_report >= 60):
                atomic_json_dump(dict(status="evaluating", phase=phase, view=view, views=len(recipe["views"]),
                    scale=scale, batch=batch, batches=len(data), rows=len(rows),
                    elapsed_seconds=now-started), progress_path)
                last_report = now
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite continuation logits")
    return result


def comparison(plan, labels, baseline, candidate):
    result = paired_metrics(labels, baseline, candidate, len(plan["classes"]), plan["groups"])
    for group in result["slices"].values():
        group["candidate"] = group.pop("v1_supervision")
    # Also report the predefined head classes; use the same arithmetic as other slices.
    head_groups = dict(plan["groups"], tail_classes=plan["groups"]["head_classes"])
    head = paired_metrics(labels, baseline, candidate, len(plan["classes"]), head_groups)["slices"]["tail"]
    head["candidate"] = head.pop("v1_supervision")
    result["slices"]["head"] = head
    result.update(decision="evidence_requires_review", automatic_full=False,
                  original_label_counts=True, clean_labels_known=False, platform_gain_known=False)
    return result


def fit(model, loader, targets, plan, output, *, device="cpu", evaluation=None, binding=None):
    """Shared CPU-testable loop; fixed exports, full population never evaluated."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    recipe, cfg = plan["config"]["recipe"], plan["config"]
    if len(loader) < 1:
        raise ValueError("No logical batches")
    if plan["config"]["partition"] == "full_train" and evaluation is not None:
        raise ValueError("FULL may not evaluate/select using the overlapping old validation split")
    seed_training(cfg["seed"])
    opt = optimizer_for(model, recipe)
    scheduler = scheduler_for(opt, recipe, len(loader))
    amp = str(device).startswith("cuda") and plan["inherited_train"]["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    # References avoid cloning the entire full tower every update.
    live_state = lambda: {name: p.detach() for name, p in model.named_parameters() if p.requires_grad}
    ema = WeightAverage(live_state(), recipe["ema_decay"])
    average = None
    tensor_targets = {k: torch.as_tensor(targets[k], device=device) for k in
                      ("targets", "labels", "weights", "original_alpha")}
    probabilities = target_probabilities(tensor_targets["targets"], tensor_targets["labels"],
        tensor_targets["original_alpha"], len(plan["classes"]), plan["inherited_train"]["label_smoothing"])
    trajectory = dict(binding=binding, seed=cfg["seed"], parent=plan["config"]["parent_checkpoint"])
    history, updates, start_time, last_report = [], 0, time.monotonic(), 0.
    amp_retries = []
    atomic_json_dump(amp_retries, output / "amp_retries.json")
    def payload(epoch, state, policy):
        return dict(epoch=epoch, selected_state={k: v.detach().cpu().clone() for k, v in state.items()},
                    selected_policy=policy, route=cfg["route"], classes=plan["classes"],
                    recipe=recipe, partition=cfg["partition"], trajectory=trajectory,
                    optimizer_updates=updates, optimizer=None, scheduler=None,
                    complete=epoch == recipe["epochs"], bias=None)
    try:
        for epoch in range(1, recipe["epochs"]+1):
            model.train()
            losses = []
            for batch, (images, indices) in enumerate(loader, 1):
                images, indices = images.to(device), indices.to(device)
                permutation = torch.randperm(len(images), device=device)
                alpha = plan["inherited_train"]["mixup_alpha"]
                lam = float(np.random.beta(alpha, alpha)) if alpha else 1.
                loss, numeric = logical_update(model, opt, images, probabilities[indices],
                    tensor_targets["weights"][indices], permutation, lam, cfg["micro_batch_size"],
                    scaler, amp, recipe["grad_clip"])
                if numeric["amp_retries"]:
                    event = dict(epoch=epoch, batch=batch, optimizer_update=updates+1, **numeric)
                    amp_retries.append(event)
                    atomic_json_dump(amp_retries, output / "amp_retries.json")
                    print(json.dumps(dict(status="amp_same_batch_recomputed", **event)), flush=True)
                scheduler.step(); ema.update(live_state()); updates += 1
                losses.append(loss)
                now = time.monotonic()
                if batch == 1 or batch == len(loader) or now-last_report >= 60:
                    progress = dict(status="training", epoch=epoch, batch=batch, batches=len(loader),
                                    optimizer_updates=updates, loss=float(np.mean(losses)),
                                    elapsed_seconds=now-start_time)
                    atomic_json_dump(progress, output / "progress.json")
                    print(json.dumps(progress), flush=True)
                    last_report = now
            raw = payload(epoch, live_state(), "raw")
            ema_payload = payload(epoch, ema.state, "ema")
            save_artifact(output / f"epoch_{epoch:02d}_raw.pt", raw, binding)
            save_artifact(output / f"epoch_{epoch:02d}_ema.pt", ema_payload, binding)
            if epoch in recipe["average_epochs"]:
                if average is None:
                    average = WeightAverage(ema_payload["selected_state"])
                average.update(ema_payload["selected_state"])
            entry = dict(epoch=epoch, loss=float(np.mean(losses)), optimizer_updates=updates,
                         amp_recomputed_batches=len(amp_retries),
                         ema_updates=ema.count, initial_ema_coefficient=recipe["ema_decay"]**ema.count)
            if evaluation:
                entry["raw"] = evaluation(model, epoch, "raw")
                with using_weights(model, ema.state):
                    entry["ema"] = evaluation(model, epoch, "ema")
            history.append(entry)
        atomic_json_dump(history, output / "history.json")
        if average is None or average.count != 3:
            raise ValueError("Fixed EMA2-4 window incomplete")
        final = output / "selected.pt"
        save_artifact(final, payload(4, ema.state, "last_ema"), binding)
        averaged = payload(4, average.state, "ema_swa_2_4")
        averaged["average_epochs"] = [2, 3, 4]
        averaged["sources"] = {str(output / f"epoch_{e:02d}_ema.pt"):
                               sha256_file(output / f"epoch_{e:02d}_ema.pt") for e in (2, 3, 4)}
        save_artifact(output / "ema_swa.pt", averaged, binding)
        if evaluation:
            with using_weights(model, average.state):
                averaged_metrics = evaluation(model, 4, "ema_swa_2_4")
        else:
            averaged_metrics = None
        atomic_json_dump(dict(status="training_complete", epochs=4, optimizer_updates=updates,
                              selected_checkpoint=str(final), average_checkpoint=str(output / "ema_swa.pt"),
                              average_metrics=averaged_metrics, elapsed_seconds=time.monotonic()-start_time),
                         output / "status.json")
        return history, averaged_metrics
    except Exception as error:
        atomic_json_dump(dict(status="incomplete", error=str(error), optimizer_updates=updates,
                              automatically_resumed=False), output / "status.json")
        raise


def run(plan_path, output, device="cuda"):
    plan, context, targets = verify(plan_path)
    if device != "cuda":
        raise ValueError("Real training requires local CUDA; CPU is for tests/zero-update checks")
    require_idle_cuda()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model, check = initial_model(plan, context)
    atomic_json_dump(check, output / "zero_update.json")
    model.to(device)
    labels = np.array([int(r["label"]) for r in context.val])
    artifacts = {}
    baseline = None
    if context.val:
        logits = evaluate(plan, context, model, context.val, device, progress_path=output / "evaluation_progress.json",
                          phase="untrained_parent_baseline")
        baseline = logits.argmax(1).numpy()
        baseline_path = output / "baseline.npz"
        np.savez_compressed(baseline_path, predictions=baseline, labels=labels,
                            image_paths=[r["image_path"] for r in context.val], logits=logits.numpy())
        artifacts[str(baseline_path)] = sha256_file(baseline_path)
    # Freeze this baseline before the first optimizer update (especially LR512).
    atomic_json_dump(dict(status="baseline_complete" if context.val else "full_no_validation",
                          recipe=plan["config"]["recipe"], train_started=False), output / "baseline_status.json")
    def evaluation(current, epoch, policy):
        logits = evaluate(plan, context, current, context.val, device, progress_path=output / "evaluation_progress.json",
                          phase=f"epoch_{epoch:02d}_{policy}")
        prediction = logits.argmax(1).numpy()
        path = output / f"val_epoch{epoch:02d}_{policy}.npz"
        np.savez_compressed(path, predictions=prediction, labels=labels,
                            image_paths=[r["image_path"] for r in context.val])
        artifacts[str(path)] = sha256_file(path)
        if epoch == 4 and policy in ("ema", "ema_swa_2_4"):
            paired_path = output / f"paired_{policy}.csv"
            with paired_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image_path", "label", "parent_prediction", "candidate_prediction", "correction", "regression"])
                for row, truth, a, b in zip(context.val, labels, baseline, prediction):
                    writer.writerow([row["image_path"], int(truth), int(a), int(b),
                                     int(a != truth and b == truth), int(a == truth and b != truth)])
            artifacts[str(paired_path)] = sha256_file(paired_path)
        return comparison(plan, labels, baseline, prediction)
    active = np.flatnonzero(targets["weights"] > 0)
    subset = {key: value[active] for key, value in targets.items()}
    rows = [context.train[i] for i in active]
    dataset = Images(context.train_root, rows, image_transform(plan["config"]["recipe"]["image_size"],
        training=True, crop_min=plan["inherited_train"]["crop_min_scale"]))
    loader = DataLoader(dataset, batch_size=plan["logical_batch_size"], shuffle=True, drop_last=False,
        num_workers=plan["config"]["num_workers"], worker_init_fn=seed_worker)
    binding = dict(plan_sha256=sha256_file(plan_path), source_binding=context.binding,
                   parent_sha256=sha256_file(plan["config"]["parent_checkpoint"]),
                   targets_sha256=sha256_file(plan["config"]["targets"]))
    history, average_metrics = fit(model, loader, subset, plan, output / "training", device=device,
                                   evaluation=evaluation if context.val else None, binding=binding)
    for name in ("zero_update.json", "baseline_status.json", "training/history.json", "training/status.json",
                 "training/amp_retries.json",
                 "training/selected.pt", "training/ema_swa.pt"):
        path = output / name
        artifacts[str(path)] = sha256_file(path)
    report = dict(status="development_complete" if context.val else "full_training_complete", route=plan["config"]["route"],
        partition=plan["config"]["partition"], experiment_id=plan["experiment_id"], epochs=4,
        recipe=plan["config"]["recipe"], plan=str(Path(plan_path).resolve()), plan_sha256=sha256_file(plan_path),
        validation_is_independent=bool(context.val), baseline_before_training=baseline is not None,
        paired=dict(last_ema=history[-1].get("ema"), ema_swa_2_4=average_metrics), trajectory=history,
        frozen_groups=plan["groups"], source_binding=plan["source_binding"], parent_binding=plan["parent_binding"],
        parent_checkpoint=plan["config"]["parent_checkpoint"], targets=plan["config"]["targets"],
        artifacts=artifacts, bias="disabled", automatic_full=False, platform_score=None,
        submission_status="not_generated", clean_labels_known=False)
    atomic_json_dump(report, output / "report.json")
    return report


def infer(plan_path, run_root, checkpoint, output):
    from aegis_clip.submission import create_submission
    plan, context, _ = verify(plan_path)
    root, cp = Path(run_root).resolve(), Path(checkpoint).resolve()
    report = json.loads((root / "report.json").read_text())
    if (report["plan_sha256"] != sha256_file(plan_path) or report["epochs"] != 4
            or report["status"] not in ("development_complete", "full_training_complete")):
        raise ValueError("Training report and plan disagree")
    if cp not in (root / "training/selected.pt", root / "training/ema_swa.pt"):
        raise ValueError("Use one explicit fixed last EMA or EMA2-4 export")
    if sha256_file(cp) != report["artifacts"][str(cp)]:
        raise ValueError("Selected checkpoint changed")
    binding = dict(plan_sha256=sha256_file(plan_path), source_binding=context.binding,
                   parent_sha256=sha256_file(plan["config"]["parent_checkpoint"]),
                   targets_sha256=sha256_file(plan["config"]["targets"]))
    payload = load_artifact(cp, binding)
    if payload["complete"] is not True or payload["epoch"] != 4 or payload["bias"] is not None:
        raise ValueError("Inference requires the completed frozen single checkpoint")
    require_idle_cuda()
    model, _ = initial_model(plan, context)
    load_trainable_state(model, payload["selected_state"])
    model.to("cuda").eval()
    rows = read_rows(Path(context.reference["data"]["dataset_manifest"]).parent / "test_manifest.csv")
    names = [Path(r["image_path"]).name for r in rows]
    if len(names) != context.manifest["test_samples"] or sorted(names) != sorted(p.name for p in context.test_root.iterdir() if p.is_file()):
        raise ValueError("Official test coverage changed")
    logits = evaluate(plan, context, model, rows, "cuda", context.test_root,
                      progress_path=root / "inference_progress.json", phase=payload["selected_policy"])
    predictions = [(name, context.classes[index]) for name, index in zip(names, logits.argmax(1).tolist())]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Submission output must be new")
    create_submission(predictions, names, output, cp, inference_mode="fixed_continuation_views",
        tta_risk_acknowledged=True, valid_labels=set(context.classes), space_after_comma=True,
        extra_manifest=dict(binding=binding, decoder=plan["config"]["recipe"], weights=payload["selected_policy"],
                            bias=None, test_usage="inference_only", platform_score=None))
    checked = subprocess.run([sys.executable, str(ROOT / "scripts/check_submission.py"),
        "--test_dir", str(context.test_root), "--class-mapping", context.reference["data"]["class_mapping"],
        "--csv", str(output / "pred_results.csv"), "--zip", str(output / "submission.zip")], capture_output=True, text=True)
    (output / "submission_check.log").write_text(checked.stdout+checked.stderr)
    checked.check_returncode()
    atomic_json_dump(dict(status="package_ready", checkpoint=str(cp), checkpoint_sha256=sha256_file(cp),
                          rows=len(rows), csv_sha256=sha256_file(output / "pred_results.csv"),
                          zip_sha256=sha256_file(output / "submission.zip"), platform_score=None), output / "report.json")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["zero-check", "train", "infer"])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.action != "zero-check" and not args.execute:
        parser.error("Compute requires --execute; FULL preparation also requires DEV evidence")
    if args.action == "zero-check":
        plan, context, _ = verify(args.plan)
        _, check = initial_model(plan, context)
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        atomic_json_dump(check, output)
        print(json.dumps(check))
    elif args.action == "train":
        print(run(args.plan, args.output)["status"])
    else:
        if not args.run_root or not args.checkpoint:
            parser.error("infer requires --run-root and --checkpoint")
        print(infer(args.plan, args.run_root, args.checkpoint, args.output))


if __name__ == "__main__":
    main()
