"""Explicitly authorized local v3 pair; --help/import never probes CUDA."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from aegis_clip.v1_strategy import WeightAverage, target_probabilities, using_weights
from v2 import training_utils
from v2.model import V2Classifier
from v2.plan import dump, json_read, require, sha
from v2.runtime import cpu_state, device_after_authorization, save_checkpoint
from .core import PairedImages, isolated_cpu_rng, mix_supervision, replay_seed, weighted_backward
from .plan import ARMS, authorize, read_rows


def state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def data_loader(dataset, cfg, batch_size):
    workers = cfg["data"]["num_workers"]
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                      drop_last=False, pin_memory=False, persistent_workers=False,
                      generator=torch.Generator(device="cpu").manual_seed(cfg["train"]["seed"]),
                      multiprocessing_context="spawn" if workers else None,
                      timeout=120 if workers else 0,
                      prefetch_factor=cfg["data"]["prefetch_factor"] if workers else None)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_for(device):
    return (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else nullcontext


def verify_images(root, records):
    root = Path(root).resolve()
    for row in records:
        path = (root / row["relative_path"]).resolve()
        require(path.is_relative_to(root), "Image path escapes official training root")
        require(sha(path) == row["source_sha256"], f"Training pixels changed: {row['relative_path']}")


@torch.no_grad()
def evaluate(model, data, labels, classes, device):
    model.eval()
    predictions = np.full(len(labels), -1, np.int64)
    seen = np.zeros(len(labels), np.int64)
    for images, truth, indices in data:
        index = indices.numpy()
        require(np.array_equal(truth.numpy(), labels[index]), "Validation labels changed")
        with autocast_for(device)():
            logits = model(images.to(device))
        require(torch.isfinite(logits).all(), "Nonfinite validation logits")
        predictions[index] = logits.float().argmax(1).cpu().numpy()
        seen[index] += 1
    require((seen == 1).all(), "Validation must visit every row exactly once")
    counts = np.bincount(labels, minlength=classes)
    correct = np.bincount(labels[predictions == labels], minlength=classes)
    present = counts > 0
    metrics = dict(macro=float((correct[present] / counts[present]).mean()),
                   micro=float((predictions == labels).mean()), rows=len(labels), classes_present=int(present.sum()))
    return metrics, predictions


def fit_arm(model, plan, workspace, cfg, train_rows, val_rows, draws, supervision,
            arm, destination, device):
    """Shared CPU-testable loop; the public operation authorizes before CUDA."""
    destination.mkdir(parents=True, exist_ok=False)
    seed, classes = cfg["train"]["seed"], len(plan["classes"])
    root = Path(plan["train_root"])
    initial_hash = state_sha256(model.state_dict())
    optimizer = training_utils.build_optimizer(model, cfg["train"])
    scheduler = training_utils.build_scheduler(optimizer, cfg["train"]["epochs"], cfg["train"]["warmup_epochs"],
                                              plan["steps_per_epoch"], cfg["train"]["min_lr_ratio"])
    ema = WeightAverage(model.state_dict(), cfg["train"]["ema_decay"])
    original = torch.from_numpy(supervision["labels"].copy()).long()
    if arm == "original":
        targets, weights, alpha = original, torch.ones(len(original)), torch.zeros(len(original))
    else:
        targets = torch.from_numpy(supervision["targets"].copy()).long()
        weights = torch.from_numpy(supervision["weights"].copy()).float()
        alpha = torch.from_numpy(supervision["original_alpha"].copy()).float()
    probabilities = target_probabilities(targets, original, alpha, classes, cfg["augment"]["label_smoothing"])
    val_labels = np.array([r["label"] for r in val_rows])
    val_ds = training_utils.ManifestDataset(root, val_rows,
        training_utils.build_eval_transform(cfg["data"]["image_size"], cfg["data"]["eval_resize_ratio"]))
    val_loader = data_loader(val_ds, cfg, plan["recipe"]["micro_batch_size"])
    train_transform = training_utils.build_train_transform(cfg["data"]["image_size"], cfg["augment"])
    history = []
    updates, zero_mass_batches = 0, 0
    actual_epochs = cfg["train"]["epochs"]
    dump(destination / "status.json", dict(status="running", arm=arm, plan_sha256=sha(workspace / "plan.json")))
    with (destination / "steps.jsonl").open("w") as trace:
        for epoch in range(1, actual_epochs + 1):
            dataset = PairedImages(root, train_rows, train_transform, draws[epoch - 1], seed, epoch)
            batches = iter(data_loader(dataset, cfg, cfg["data"]["batch_size"]))
            count = plan["steps_per_epoch"]
            model.train()
            for batch in range(count):
                images, indices = next(batches)
                expected = draws[epoch - 1][batch * cfg["data"]["batch_size"]:(batch + 1) * cfg["data"]["batch_size"]]
                require(np.array_equal(indices.numpy(), expected), "Frozen training draw order changed")
                w = weights[indices]
                zero_mass_batches += int(float(w.sum()) == 0)
                images, masses, denominator, mixed = mix_supervision(
                    images, probabilities[indices], w, cfg["augment"], replay_seed(seed, "mix", epoch, batch))
                images, masses, denominator = images.to(device), masses.to(device), denominator.to(device)
                optimizer.zero_grad(set_to_none=True)
                rates = [group["lr"] for group in optimizer.param_groups]
                loss = weighted_backward(model, images, masses, denominator,
                                         plan["recipe"]["micro_batch_size"], autocast_for(device))
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                require(torch.isfinite(norm), "Nonfinite gradient; stop without retries")
                optimizer.step()
                scheduler.step()
                ema.update(model.state_dict())
                updates += 1
                synchronize(device)
                # Intentionally excludes arm-specific loss/gradients. Exact trace
                # equality checks all shared draws/mixing/LRs and update counts.
                shared = dict(epoch=epoch, batch=batch, optimizer_updates=updates, ema_updates=ema.count,
                              indices_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                              augmentation_seed=replay_seed(seed, "image", epoch, batch * cfg["data"]["batch_size"], int(expected[0])),
                              lr=rates, **mixed)
                trace.write(json.dumps(shared, sort_keys=True) + "\n")
                if batch == 0 or (batch + 1) % cfg["train"]["log_every"] == 0:
                    print(json.dumps(dict(arm=arm, epoch=epoch, batch=batch + 1, updates=updates, loss=loss)), flush=True)
                del images, masses, denominator
            raw_metrics, _ = evaluate(model, val_loader, val_labels, classes, device)
            with using_weights(model, ema.state):
                ema_metrics, final_predictions = evaluate(model, val_loader, val_labels, classes, device)
            history.append(dict(epoch=epoch, updates=updates, raw=raw_metrics, ema=ema_metrics))
            payload = dict(model=cpu_state(model), ema={k: v.detach().cpu().clone() for k, v in ema.state.items()},
                           optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), history=history,
                           epoch=epoch, updates=updates, initial_model_sha256=initial_hash,
                           rng_cpu=torch.get_rng_state(),
                           rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                           binding=dict(plan_sha256=sha(workspace / "plan.json"), arm=arm, probe=False,
                                        complete=epoch == actual_epochs))
            save_checkpoint(destination / f"epoch{epoch:02d}.pt", payload)
            del payload
            dump(destination / "history.json", history)
    result = dict(status="complete", arm=arm, updates=updates,
                  initial_model_sha256=initial_hash, plan_sha256=sha(workspace / "plan.json"),
                  zero_mass_batches=zero_mass_batches, trace_sha256=sha(destination / "steps.jsonl"))
    require(updates == plan["updates_per_arm"] and ema.count == updates, "Wrong fixed update endpoint")
    selected = dict(model={k: v.detach().cpu().clone() for k, v in ema.state.items()},
                    binding=dict(plan_sha256=sha(workspace / "plan.json"), arm=arm, weights="last_ema",
                                 epoch=actual_epochs, updates=updates, complete=True), metrics=history[-1]["ema"])
    save_checkpoint(destination / "selected.pt", selected)
    np.savez(destination / "predictions.npz", labels=val_labels, predictions=final_predictions,
             image_paths=np.array([r["image_path"] for r in val_rows]))
    result.update(weights="last_ema", checkpoint_sha256=sha(destination / "selected.pt"),
                  predictions_sha256=sha(destination / "predictions.npz"))
    dump(destination / "status.json", result)
    return result


def run_pair(plan_path, authorization_path):
    operation = "train"
    _, plan = authorize(plan_path, authorization_path, operation)  # before ANY CUDA query
    started = time.monotonic()
    workspace = Path(plan_path).resolve().parent
    destination = workspace / "runs" / operation
    require(not destination.exists(), "Refusing to overwrite an existing pair operation")
    destination.mkdir(parents=True)
    cfg = json_read(workspace / "shared_config.json")
    train_rows, val_rows = read_rows(workspace / "train_manifest.csv"), read_rows(workspace / "val_manifest.csv")
    draws = np.load(workspace / "draws.npy", allow_pickle=False)
    with np.load(workspace / "supervision.npz", allow_pickle=False) as payload:
        supervision = {key: payload[key].copy() for key in payload.files}
    results = {}
    try:
        # All original train/holdout bytes are checked; no test images are read.
        verify_images(plan["train_root"], train_rows + val_rows)
        device = device_after_authorization()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for arm in ARMS:
            with isolated_cpu_rng(cfg["train"]["seed"]):
                model = V2Classifier(plan["model_recipe"])
            model = model.to(device)
            require(all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()), "v3 requires full FP32 visual FT")
            results[arm] = fit_arm(model, plan, workspace, cfg, train_rows, val_rows, draws, supervision,
                                   arm, destination / arm, device)
            del model
        require(results[ARMS[0]]["initial_model_sha256"] == results[ARMS[1]]["initial_model_sha256"], "Initial weights differ")
        require((destination / ARMS[0] / "steps.jsonl").read_bytes() ==
                (destination / ARMS[1] / "steps.jsonl").read_bytes(), "Pair execution diverged")
        terminal = dict(status="complete", plan_sha256=sha(plan_path),
                        operation=operation, elapsed_seconds=time.monotonic() - started, arms=results,
                        automatic_ladder=False, platform_metrics=None)
        dump(destination / "status.json", terminal)
        from .report import compare
        compare(plan_path, destination, workspace / "comparison")
        terminal["elapsed_seconds"] = time.monotonic() - started
        dump(destination / "status.json", terminal)
        return terminal
    except Exception as exc:
        dump(destination / "status.json", dict(status="stopped", plan_sha256=sha(plan_path), operation=operation,
              error=str(exc), elapsed_seconds=time.monotonic() - started, automatic_retry=False,
              automatic_ladder=False, platform_metrics=None))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--plan", required=True)
    p.add_argument("--authorization", required=True)
    args = parser.parse_args()
    run_pair(args.plan, args.authorization)


if __name__ == "__main__":
    main()
