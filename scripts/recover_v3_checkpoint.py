"""Finish a durable original-arm endpoint, then run the frozen supervision arm.

Invoke with PYTHONPATH pointing to the code already bound by the original plan.
This wrapper never edits that code, the epoch checkpoint, or the execution trace.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from v2 import training_utils
from v2.model import V2Classifier
from v2.plan import dump, json_read, require, sha
from v2.runtime import device_after_authorization, save_checkpoint
from v3 import runtime
from v3.core import isolated_cpu_rng
from v3.plan import authorize, read_rows
from v3.report import compare


def checked_endpoint(plan_path, plan, expected_checkpoint_sha):
    """Check the exact final checkpoint and the complete, durable update trace."""
    plan_path = Path(plan_path).resolve()
    original = plan_path.parent / "runs/train/original"
    epoch = plan["recipe"]["epochs"]
    checkpoint = original / f"epoch{epoch:02d}.pt"
    binding = dict(plan_sha256=sha(plan_path), arm="original", probe=False, complete=True)
    require(sha(checkpoint) == expected_checkpoint_sha, "Recovery checkpoint changed")
    require(json_read(checkpoint.with_suffix(".binding.json")) ==
            dict(sha256=expected_checkpoint_sha, binding=binding), "Recovery sidecar changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(payload["binding"] == binding, "Recovery checkpoint binding changed")
    require(payload["epoch"] == epoch and payload["updates"] == plan["updates_per_arm"],
            "Recovery requires the fixed completed original endpoint")
    require(payload["history"] == json_read(original / "history.json") and
            [row["epoch"] for row in payload["history"]] == list(range(1, epoch + 1)) and
            all(row["updates"] == row["epoch"] * plan["steps_per_epoch"] for row in payload["history"]),
            "Recovery history is incomplete or changed")
    trace = [json.loads(line) for line in (original / "steps.jsonl").read_text().splitlines()]
    require(len(trace) == plan["updates_per_arm"], "Recovery trace is incomplete")
    for index, row in enumerate(trace):
        require(row["optimizer_updates"] == row["ema_updates"] == index + 1 and
                row["epoch"] == index // plan["steps_per_epoch"] + 1 and
                row["batch"] == index % plan["steps_per_epoch"], "Recovery trace sequence changed")
    require(payload["ema"] and all(torch.isfinite(v).all() for v in payload["ema"].values()),
            "Recovery EMA is empty or nonfinite")
    return original, payload


def export_original(model, plan_path, plan, original, payload, loader, val_rows, device):
    """Re-evaluate the stored EMA; preserve raw checkpoint/history/trace bytes."""
    for name in ("selected.pt", "selected.binding.json", "predictions.npz"):
        require(not (original / name).exists(), f"Refusing to overwrite recovered {name}")
    model.load_state_dict(payload["ema"], strict=True)
    labels = np.array([row["label"] for row in val_rows], dtype=np.int64)
    metrics, predictions = runtime.evaluate(model, loader, labels, len(plan["classes"]), device)
    recorded = payload["history"][-1]["ema"]
    require(metrics["rows"] == recorded["rows"] and metrics["classes_present"] == recorded["classes_present"] and
            all(abs(metrics[key] - recorded[key]) < 1e-10 for key in ("macro", "micro")),
            "Recovered EMA validation differs from the completed epoch")
    selected = dict(model=payload["ema"], metrics=recorded,
                    binding=dict(plan_sha256=sha(plan_path), arm="original", weights="last_ema",
                                 epoch=plan["recipe"]["epochs"], updates=plan["updates_per_arm"], complete=True))
    save_checkpoint(original / "selected.pt", selected)
    np.savez(original / "predictions.npz", labels=labels, predictions=predictions,
             image_paths=np.array([row["image_path"] for row in val_rows]))
    result = dict(status="complete", arm="original", updates=plan["updates_per_arm"],
                  initial_model_sha256=payload["initial_model_sha256"], plan_sha256=sha(plan_path),
                  zero_mass_batches=0, trace_sha256=sha(original / "steps.jsonl"), weights="last_ema",
                  checkpoint_sha256=sha(original / "selected.pt"),
                  predictions_sha256=sha(original / "predictions.npz"))
    dump(original / "status.json", result)
    print(json.dumps(dict(stage="original_export_complete", metrics=metrics, **result)), flush=True)
    return result


def recover(plan_path, authorization_path, expected_checkpoint_sha):
    plan_path = Path(plan_path).resolve()
    _, plan = authorize(plan_path, authorization_path, "train")  # before CUDA
    # Frozen module identity matters: latest main may have a different v1 binding.
    for module in (runtime, training_utils):
        require(str(Path(module.__file__).resolve()) in plan["inputs"], "Use the plan-bound frozen code")
    workspace, started = plan_path.parent, time.monotonic()
    run = workspace / "runs/train"
    for path in (run / "v1_supervision", workspace / "comparison", workspace / "submission"):
        require(not path.exists(), f"Recovery requires an unstarted destination: {path}")
    original, payload = checked_endpoint(plan_path, plan, expected_checkpoint_sha)
    for name in ("selected.pt", "selected.binding.json", "predictions.npz"):
        require(not (original / name).exists(), f"Refusing to overwrite recovered {name}")
    # Optimizer, raw weights and RNG are unnecessary: this arm finished training.
    payload = {key: payload[key] for key in ("ema", "history", "initial_model_sha256")}
    gc.collect()
    cfg = json_read(workspace / "shared_config.json")
    train_rows, val_rows = read_rows(workspace / "train_manifest.csv"), read_rows(workspace / "val_manifest.csv")
    print(json.dumps(dict(stage="verify_training_pixels", checkpoint_sha256=expected_checkpoint_sha)), flush=True)
    runtime.verify_images(plan["train_root"], train_rows + val_rows)
    device = device_after_authorization()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with isolated_cpu_rng(cfg["train"]["seed"]):
        model = V2Classifier(plan["model_recipe"])
    require(runtime.state_sha256(model.state_dict()) == payload["initial_model_sha256"], "Original initialization changed")
    model = model.to(device)
    val_dataset = training_utils.ManifestDataset(Path(plan["train_root"]), val_rows,
        training_utils.build_eval_transform(cfg["data"]["image_size"], cfg["data"]["eval_resize_ratio"]))
    loader = runtime.data_loader(val_dataset, cfg, plan["recipe"]["micro_batch_size"])
    print(json.dumps(dict(stage="original_ema_validation", rows=len(val_rows))), flush=True)
    results = {"original": export_original(model, plan_path, plan, original, payload, loader, val_rows, device)}
    del model, payload, loader
    gc.collect()
    torch.cuda.empty_cache()
    draws = np.load(workspace / "draws.npy", allow_pickle=False)
    with np.load(workspace / "supervision.npz", allow_pickle=False) as data:
        supervision = {key: data[key].copy() for key in data.files}
    with isolated_cpu_rng(cfg["train"]["seed"]):
        model = V2Classifier(plan["model_recipe"])
    require(runtime.state_sha256(model.state_dict()) == results["original"]["initial_model_sha256"], "Arm initial weights differ")
    model = model.to(device)
    require(all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()), "v3 requires full FP32 visual FT")
    print(json.dumps(dict(stage="v1_supervision_start", epochs=cfg["train"]["epochs"])), flush=True)
    results["v1_supervision"] = runtime.fit_arm(model, plan, workspace, cfg, train_rows, val_rows, draws,
                                               supervision, "v1_supervision", run / "v1_supervision", device)
    require((original / "steps.jsonl").read_bytes() == (run / "v1_supervision/steps.jsonl").read_bytes(),
            "Recovered paired execution diverged")
    terminal = dict(status="complete", operation="train", plan_sha256=sha(plan_path), arms=results,
                    elapsed_seconds=time.monotonic() - started, original_retrained=False,
                    recovered_checkpoint_sha256=expected_checkpoint_sha)
    dump(run / "status.json", terminal)
    terminal["comparison"] = compare(plan_path, run, workspace / "comparison")
    dump(run / "status.json", terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.plan, args.authorization, args.checkpoint_sha256)), flush=True)


if __name__ == "__main__":
    main()
