"""Explicitly gated local GPU operations. Import/--help does not initialize CUDA.

Stage initialization transfers raw weights only; optimizer and scheduler reset.
Partial stages are saved for audit, never accepted as completed parent models.
No background runner, implicit resume, ensemble, prior fitting or test labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import time
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .core import build_view, check_checkpoint, logical_backward, reference, sample_weights
from .model import LocalFTClassifier
from .plan import (ROOT, STAGES, authorize, dump, json_read, require, sha,
                   verify_prepared)


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds):
        self.start = time.monotonic()
        self.limit = float(seconds)

    def check(self):
        if time.monotonic() - self.start >= self.limit:
            raise BudgetExpired("Bounded operation budget exhausted; no automatic restart")


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_checkpoint(path, payload):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    dump(path.with_suffix(".binding.json"), dict(sha256=sha(path), binding=payload["binding"]))


def read_checkpoint(path, plan, stage):
    path = Path(path)
    require(path.is_file() and path.with_suffix(".binding.json").is_file(), "Missing locally bound parent checkpoint")
    status = json_read(path.parent / "status.json")
    require(status["status"] == "complete" and status["updates"] == plan["stages"][stage]["total_updates"],
            "Parent stage did not complete within its operation budget")
    sidecar = json_read(path.with_suffix(".binding.json"))
    require(sidecar["sha256"] == sha(path), "Parent checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(sidecar["binding"] == payload.get("binding"), "Checkpoint/sidecar mismatch")
    check_checkpoint(payload, plan, stage)
    return payload


def choose_parent(payload):
    require(payload["metrics"]["chosen"] == "raw" and "ema" not in payload,
            "Only raw parent weights are allowed; EMA remains unconfirmed")
    return payload["model"]


def records_and_split(workspace):
    records = reference.read_manifest(workspace / "train_manifest.csv")
    with (workspace / "train_manifest.csv").open(newline="") as f:
        extra = list(csv.DictReader(f))
    for rec, row in zip(records, extra):
        rec.update(source_sha256=row["source_sha256"], content_group=row["content_group"])
    split = json_read(workspace / "split.json")
    return records, split


def verify_images(root, records, budget):
    root = Path(root).resolve()
    for r in records:
        path = (root / r["relative_path"]).resolve()
        require(path.is_relative_to(root), "Image path escapes official training root")
        expected = r["source_sha256"]
        require(sha(path) == expected, f"Training pixels changed: {r['relative_path']}")
        budget.check()


def device_after_authorization():
    require(torch.cuda.is_available(), "Local CUDA required; no CPU/NPU fallback for this GPU operation")
    require(torch.cuda.is_bf16_supported(), "Recipe requires local BF16 support")
    return torch.device("cuda")


def loader(dataset, cfg, batch_size, sampler=None):
    workers = cfg["data"]["num_workers"]
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      drop_last=sampler is not None, shuffle=False, num_workers=workers,
                      pin_memory=True, persistent_workers=False,
                      prefetch_factor=cfg["data"]["prefetch_factor"] if workers else None)


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root, files, transform):
        self.root, self.files, self.transform = root, files, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.transform(reference.load_image(self.root / self.files[index])), index


@torch.no_grad()
def evaluate(model, data_loader, classes, device, budget):
    model.eval()
    correct, total = np.zeros(classes, np.int64), np.zeros(classes, np.int64)
    predictions, labels_all = [], []
    for images, labels, _ in data_loader:
        budget.check()
        with amp():
            pred = model(images.to(device, non_blocking=True)).float().argmax(1).cpu().numpy()
        truth = labels.numpy()
        np.add.at(correct, truth, pred == truth); np.add.at(total, truth, 1)
        predictions.extend(pred.tolist()); labels_all.extend(truth.tolist())
    present = total > 0
    return dict(accuracy=float(correct.sum() / total.sum()),
                macro_accuracy=float((correct[present] / total[present]).mean()),
                classes_present=int(present.sum())), predictions, labels_all


def initialize_model(plan, workspace, stage, device):
    index = STAGES.index(stage)
    parent_path = None if index == 0 else workspace / "runs" / STAGES[index - 1] / "best.pt"
    payload = None if parent_path is None else read_checkpoint(parent_path, plan, STAGES[index - 1])
    # Fresh s1 official model; subsequent stages load ALL tensors strictly.
    model = LocalFTClassifier(plan["recipe"])
    if payload is not None:
        model.load_state_dict(choose_parent(payload), strict=True)
    return model.to(device), dict(path="official_openai" if parent_path is None else str(parent_path),
                                  sha256=plan["recipe"]["official_sha256"] if parent_path is None else sha(parent_path),
                                  weights="official" if payload is None else payload["metrics"]["chosen"])


def train(plan_path, authorization, stage, probe_steps=0):
    operation = "probe" if probe_steps else "train"
    auth = authorize(plan_path, authorization, operation, stage)
    budget = Budget(auth["max_seconds"])
    plan = verify_prepared(plan_path)
    workspace = Path(plan_path).resolve().parent
    cfg = json_read(workspace / plan["stages"][stage]["config"])
    require(cfg["train"].get("weight_averaging") is False and "ema_decay" not in cfg["train"],
            "Stage execution must disable weight averaging")
    records, split = records_and_split(workspace)
    final = cfg["local_replay"]["final_stage"]
    require(not final or not probe_steps, "Probe dev stages; final V13 has no independent holdout")
    train_indices = list(range(len(records))) if final else split["train"]
    selected = [records[i] for i in train_indices]
    weights = sample_weights([r["label"] for r in selected], plan["recipe"]["num_classes"])
    verify_images(cfg["data"]["train_dir"], records, budget)
    destination = workspace / ("probes" if probe_steps else "runs") / stage
    destination.mkdir(parents=True, exist_ok=False)
    device = device_after_authorization()
    reference.set_seed(cfg["train"]["seed"])
    generator = torch.Generator().manual_seed(cfg["train"]["seed"])
    sampler = WeightedRandomSampler(weights, len(selected), replacement=True, generator=generator)
    train_ds = reference.ManifestDataset(Path(cfg["data"]["train_dir"]), selected,
                reference.build_train_transform(cfg["data"]["image_size"], cfg["augment"]), cfg["data"]["decode_cap"])
    train_loader = loader(train_ds, cfg, cfg["data"]["batch_size"], sampler)
    # Dev stages keep the project's grouped holdout and a fixed center view.
    val_loader = None
    if not final:
        val_ds = reference.ManifestDataset(Path(cfg["data"]["train_dir"]), [records[i] for i in split["val"]],
                    reference.build_eval_transform(cfg["data"]["eval_size"], cfg["data"]["eval_resize_ratio"]), cfg["data"]["decode_cap"])
        val_loader = loader(val_ds, cfg, cfg["local_replay"]["micro_batch_size"])
    model, parent = initialize_model(plan, workspace, stage, device)
    optimizer = reference.build_optimizer(model, cfg["train"])
    scheduler = reference.build_scheduler(optimizer, cfg["train"]["epochs"], cfg["train"]["warmup_epochs"],
                                          len(train_loader), cfg["train"]["min_lr_ratio"])
    binding = dict(experiment_id=plan["experiment_id"], data_version=plan["data_version"], source_commit=plan["source_commit"],
                   official_sha256=plan["recipe"]["official_sha256"], manifest_sha256=sha(workspace / "train_manifest.csv"),
                   workspace=str(workspace), stage=stage, stage_config_sha256=plan["stages"][stage]["config_sha256"],
                   plan_sha256=sha(plan_path), parent=parent, complete=False, completed_epochs=0)
    history, best_score, updates, seconds = [], -math.inf, 0, []
    epoch = -1

    def payload(metrics):
        return dict(model=cpu_state(model),
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                             cuda=torch.cuda.get_rng_state_all(), sampler=generator.get_state()),
                    binding=dict(binding), config=cfg, epoch=epoch, global_step=updates, metrics=metrics, history=history,
                    num_classes=plan["recipe"]["num_classes"], image_size=cfg["data"]["image_size"])

    try:
        for epoch in range(cfg["train"]["epochs"]):
            model.train()
            loss_sum = 0.0
            batches = iter(train_loader)
            for _ in range(len(train_loader)):
                budget.check()
                if probe_steps:
                    torch.cuda.synchronize()
                start = time.monotonic()
                # Include CPU decoding/augmentation and DataLoader wait in cost.
                images, labels, _ = next(batches)
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                targets = reference.one_hot(labels, plan["recipe"]["num_classes"], cfg["augment"]["label_smoothing"])
                images, targets, _ = reference.apply_mixup_cutmix(images, targets, cfg["augment"]["mixup"],
                    cfg["augment"]["cutmix"], cfg["augment"]["mix_prob"], plan["recipe"]["num_classes"])
                optimizer.zero_grad(set_to_none=True)
                loss = logical_backward(model, images, targets, cfg["local_replay"]["micro_batch_size"], amp)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                optimizer.step(); scheduler.step()
                updates += 1; loss_sum += loss
                if probe_steps:
                    torch.cuda.synchronize(); seconds.append(time.monotonic() - start)
                if updates % cfg["train"]["log_every"] == 0:
                    print(json.dumps(dict(stage=stage, epoch=epoch, updates=updates, loss=loss, elapsed=time.monotonic()-budget.start)), flush=True)
                if probe_steps and updates >= probe_steps:
                    break
            del images, targets
            evaluation_start = time.monotonic()
            raw_metrics, score = None, 0.0
            chosen = "raw"
            if val_loader is not None:
                raw_metrics, predictions, truth = evaluate(model, val_loader, plan["recipe"]["num_classes"], device, budget)
                score = raw_metrics["macro_accuracy"] + .5 * raw_metrics["accuracy"]
                np.savez_compressed(destination / f"holdout_epoch{epoch}.npz", indices=np.asarray(split["val"]),
                                    labels=np.asarray(truth), raw=np.asarray(predictions))
            validation_seconds = time.monotonic() - evaluation_start
            if probe_steps:
                require(len(seconds) >= 3, "At least three probe updates required")
                checkpoint_start = time.monotonic()
                temporary = destination / "cost_checkpoint.pt"
                save_checkpoint(temporary, payload(dict(chosen="raw", incomplete=True)))
                checkpoint_seconds = time.monotonic() - checkpoint_start
                temporary.unlink()
                temporary.with_suffix(".binding.json").unlink()
                elapsed = time.monotonic() - budget.start
                setup_seconds = max(elapsed - sum(seconds) - validation_seconds - checkpoint_seconds, 0.0)
                # Per-epoch last and potential best, then sealing; conservative.
                checkpoint_writes = 2 * cfg["train"]["epochs"] + 2
                budget.check()
                dump(destination / "cost.json", dict(status="cost_probe_only", no_candidate=True, seconds_per_update=seconds,
                     measured_seconds_per_update=max(seconds[1:]), measured_validation_seconds=validation_seconds,
                     measured_checkpoint_seconds=checkpoint_seconds, measured_setup_seconds=setup_seconds,
                     measured_overhead_seconds=setup_seconds+checkpoint_writes*checkpoint_seconds,
                     dataloader_wait_included=True, elapsed_seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     complete_holdout_rows=len(split["val"]), parent=parent, plan_sha256=sha(plan_path)))
                return
            entry = dict(epoch=epoch, chosen=chosen, val=raw_metrics,
                         selection_score=score, train_loss=loss_sum / len(train_loader),
                         validation_is_independent=not final, validation_seconds=validation_seconds)
            history.append(entry)
            binding["completed_epochs"] = epoch + 1
            snapshot = payload(entry)
            save_checkpoint(destination / "last.pt", snapshot)
            if not final and score > best_score:
                best_score = score
                save_checkpoint(destination / "best.pt", snapshot)
            dump(destination / "history.json", history)
            budget.check()
        # Seal weights only after every planned epoch and the budget check passed.
        last = torch.load(destination / "last.pt", map_location="cpu", weights_only=False)
        last["binding"].update(complete=True, completed_epochs=cfg["train"]["epochs"])
        save_checkpoint(destination / "last.pt", last)
        if not final:
            best = torch.load(destination / "best.pt", map_location="cpu", weights_only=False)
            best["binding"].update(complete=True, completed_epochs=cfg["train"]["epochs"])
            save_checkpoint(destination / "best.pt", best)
        budget.check()
        dump(destination / "status.json", dict(status="complete", updates=updates, elapsed_seconds=time.monotonic()-budget.start,
                                               parent=parent, raw_last=str(destination / "last.pt"), platform_metrics=None))
    except Exception as error:
        binding["complete"] = False
        if not probe_steps:
            save_checkpoint(destination / "interrupted.pt", payload(dict(chosen="raw", incomplete=True)))
        dump(destination / "status.json", dict(status="incomplete", error=str(error), updates=updates,
                                               elapsed_seconds=time.monotonic()-budget.start, automatically_resumed=False))
        raise


def write_submission(output, files, predictions, classes):
    output = Path(output)
    require(len(files) == len(predictions) and len(set(files)) == len(files), "Prediction coverage mismatch")
    require(all(isinstance(p, (int, np.integer)) and not isinstance(p, (bool, np.bool_))
                and 0 <= int(p) < classes for p in predictions), "Bad label type/range")
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "pred_results.csv"
    with csv_path.open("w", newline="") as f:
        for name, label in zip(files, predictions):
            require(Path(name).name == name and "," not in name and "\n" not in name and "\r" not in name, "Unsafe test filename")
            f.write(f"{name}, {int(label):04d}\n")
    with zipfile.ZipFile(output / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname="pred_results.csv")


def infer(plan_path, authorization, output):
    auth = authorize(plan_path, authorization, "infer")
    budget = Budget(auth["max_seconds"])
    plan = verify_prepared(plan_path)
    workspace = Path(plan_path).resolve().parent
    cp = workspace / "runs/v13_final/last.pt"
    payload = read_checkpoint(cp, plan, "v13_final")
    require(payload["epoch"] == 4 and payload["metrics"]["chosen"] == "raw", "V13 inference requires final raw last epoch")
    root = Path(plan["recipe"]["test_root"])
    with (Path(plan["recipe"]["stage_artifacts"]) / "test_manifest.csv").open(newline="") as f:
        files = sorted(Path(r["image_path"]).name for r in csv.DictReader(f))
    require(len(files) == plan["recipe"]["test_rows"] and len(set(files)) == len(files), "Wrong official test list")
    actual = sorted(p.name for p in root.iterdir() if p.is_file())
    require(actual == files, "Missing or extra test images")
    device = device_after_authorization()
    model = LocalFTClassifier(plan["recipe"]).to(device)
    model.load_state_dict(payload["model"], strict=True); model.eval()
    TestDataset = InferenceDataset
    sums = np.zeros((len(files), plan["recipe"]["num_classes"]), np.float32)
    cfg = payload["config"]
    with torch.no_grad():
        for name in plan["recipe"]["views"]:
            data = loader(TestDataset(root, files, build_view(name, 576)), cfg, plan["recipe"]["micro_batch_size"])
            for images, indices in data:
                budget.check()
                with amp():
                    logits = model(images.to(device, non_blocking=True))
                sums[indices.numpy()] += logits.float().softmax(1).cpu().numpy()
    predictions = sums.argmax(1)
    write_submission(output, files, predictions, plan["recipe"]["num_classes"])
    subprocess.run(["python3", str(ROOT / "scripts/check_submission.py"), "--test_dir", str(root),
        "--num-classes", "750", "--csv", str(Path(output)/"pred_results.csv"), "--zip", str(Path(output)/"submission.zip")], check=True)
    dump(Path(output) / "report.json", dict(status="package_ready", checkpoint=str(cp), checkpoint_sha256=sha(cp),
         weights="raw", views=plan["recipe"]["views"], rows=len(files), elapsed_seconds=time.monotonic()-budget.start,
         csv_sha256=sha(Path(output)/"pred_results.csv"), zip_sha256=sha(Path(output)/"submission.zip"), platform_metrics=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "probe", "infer"):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True)
        p.add_argument("--authorization", required=True)
        if name in ("train", "probe"):
            p.add_argument("--stage", choices=STAGES, required=True)
        if name == "probe":
            p.add_argument("--steps", type=int, default=5)
        if name == "infer":
            p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command in ("train", "probe"):
        if args.command == "probe":
            require(3 <= args.steps <= 10, "Probe is bounded to 3-10 logical updates")
        train(args.plan, args.authorization, args.stage, args.steps if args.command == "probe" else 0)
    else:
        infer(args.plan, args.authorization, args.output)


if __name__ == "__main__":
    main()
