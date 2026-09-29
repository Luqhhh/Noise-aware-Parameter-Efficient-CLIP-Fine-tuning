"""Export the fixed v3 supervision arm as one checked submission package."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from aegis_clip.v1_pipeline import StageContext, load_recipe
from v2 import training_utils
from v2.model import V2Classifier
from v2.plan import dump, json_read, require, sha
from v2.runtime import Budget, InferenceDataset, write_submission
from .plan import verify
from .runtime import autocast_for, data_loader


def checked_source(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = verify(plan_path)
    workspace = plan_path.parent
    run = workspace / "runs/train"
    terminal = json_read(run / "status.json")
    require(terminal["status"] == "complete" and terminal["plan_sha256"] == sha(plan_path),
            "v3 fixed pair is incomplete or belongs to another plan")
    report = json_read(workspace / "comparison/report.json")
    require(report["status"] == "local_result" and report["plan_sha256"] == sha(plan_path) and
            report["paired_trace_verified"] is True, "v3 pair comparison is unverified")
    arm = "v1_supervision"
    status = json_read(run / arm / "status.json")
    checkpoint = run / arm / "selected.pt"
    require(status["status"] == "complete" and status["weights"] == "last_ema" and
            status["updates"] == plan["updates_per_arm"] and status["plan_sha256"] == sha(plan_path) and
            status["checkpoint_sha256"] == sha(checkpoint) == report["checkpoint_sha256"][arm],
            "v3 selected checkpoint is incomplete or has changed")
    selected = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(selected["binding"] == dict(plan_sha256=sha(plan_path), arm=arm, weights="last_ema",
                                        epoch=plan["recipe"]["epochs"], updates=plan["updates_per_arm"],
                                        complete=True), "v3 selected checkpoint binding changed")
    require(set(selected["model"]) and all(torch.isfinite(v).all() for v in selected["model"].values()),
            "v3 selected checkpoint contains nonfinite or empty parameters")
    return plan, checkpoint, selected


def test_files(plan):
    context = StageContext(load_recipe(plan["recipe"]["source_v1_config"]))
    require(context.binding == plan["v1_binding"] and context.classes == plan["classes"],
            "v3 source data or class mapping changed")
    base = Path(context.reference["data"]["dataset_manifest"]).parent
    with (base / "test_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    files = sorted(Path(row["image_path"]).name for row in rows)
    require(len(files) == context.manifest["test_samples"] and len(files) == len(set(files)),
            "v3 test manifest has duplicate or missing rows")
    require(all(Path(row["image_path"]).parts == ("test", Path(row["image_path"]).name) for row in rows),
            "v3 test manifest contains a path outside test")
    require(sorted(p.name for p in context.test_root.iterdir() if p.is_file()) == files,
            "v3 test directory differs from the current stage manifest")
    return context, files


def deliver(plan_path, output, max_seconds, device="cuda"):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    require(device in ("cuda", "cpu") and not output.exists(), "Fresh output and CPU/CUDA device required")
    require(math.isfinite(max_seconds) and max_seconds > 0, "Finite positive inference budget required")
    plan, checkpoint, selected = checked_source(plan_path)
    context, files = test_files(plan)
    budget = Budget(max_seconds)
    if device == "cuda":
        require(torch.cuda.is_available(), "Local CUDA required for v3 delivery")
    model = V2Classifier(plan["model_recipe"]).to(device)
    model.load_state_dict(selected["model"], strict=True)
    model.eval()
    cfg = json_read(plan_path.parent / "shared_config.json")
    transform = training_utils.build_eval_transform(cfg["data"]["image_size"],
                                                    cfg["data"]["eval_resize_ratio"])
    loader = data_loader(InferenceDataset(context.test_root, files, transform), cfg,
                         plan["recipe"]["micro_batch_size"])
    predictions = np.full(len(files), -1, np.int64)
    with torch.no_grad():
        for images, indices in loader:
            budget.check()
            with autocast_for(torch.device(device))():
                logits = model(images.to(device))
            require(torch.isfinite(logits).all(), "Nonfinite test logits")
            predictions[indices.numpy()] = logits.float().argmax(1).cpu().numpy()
    require(((predictions >= 0) & (predictions < len(plan["classes"]))).all(),
            "Incomplete v3 test prediction coverage")
    budget.check()
    write_submission(output, files, predictions, len(plan["classes"]))
    command = [sys.executable, str(Path(__file__).resolve().parents[3] / "scripts/check_submission.py"),
               "--test_dir", str(context.test_root), "--class-mapping",
               str(context.reference["data"]["class_mapping"]), "--csv", str(output / "pred_results.csv"),
               "--zip", str(output / "submission.zip")]
    checked = subprocess.run(command, capture_output=True, text=True, check=False)
    (output / "submission_check.log").write_text(checked.stdout + checked.stderr)
    checked.check_returncode()
    report = dict(status="package_ready", strategy="v3", arm="v1_supervision", platform_score=None,
                  plan_sha256=sha(plan_path), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                  weights="last_ema", decoder="384_center_crop", rows=len(files),
                  csv_sha256=sha(output / "pred_results.csv"), zip_sha256=sha(output / "submission.zip"),
                  elapsed_seconds=time.monotonic() - budget.start)
    dump(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    print(deliver(args.plan, args.output, args.max_seconds, args.device))


if __name__ == "__main__":
    main()
