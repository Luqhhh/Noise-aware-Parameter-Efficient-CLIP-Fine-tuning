"""CPU verification of the completed frozen v3 pair and its submission package."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

from v2.plan import dump, json_read, require, sha
from v3.delivery import checked_source, test_files
from v3.plan import ARMS, read_rows
from v3.report import paired_metrics


def verify(plan_path, controller_path, output):
    plan_path, controller_path, output = [Path(x).resolve() for x in (plan_path, controller_path, output)]
    require(not output.exists(), "Final verification requires a fresh archive")
    plan, selected_path, _ = checked_source(plan_path)
    workspace = plan_path.parent
    run = workspace / "runs/train"
    controller = json_read(controller_path)
    require(controller["status"] == "completed" and controller["stage"] == "delivered" and
            controller["plan_sha256"] == sha(plan_path), "Controller did not complete this plan")
    report = json_read(workspace / "comparison/report.json")
    rows = read_rows(workspace / "val_manifest.csv")
    labels = np.array([row["label"] for row in rows])
    predictions, initial, traces, histories = {}, {}, {}, {}
    for arm in ARMS:
        root = run / arm
        status = json_read(root / "status.json")
        require(status["status"] == "complete" and status["updates"] == plan["updates_per_arm"] and
                status["plan_sha256"] == sha(plan_path), "Arm endpoint changed")
        for name, key in (("selected.pt", "checkpoint_sha256"), ("predictions.npz", "predictions_sha256"),
                          ("steps.jsonl", "trace_sha256")):
            require(sha(root / name) == status[key], f"Changed {arm}/{name}")
        require(status["checkpoint_sha256"] == report["checkpoint_sha256"][arm], "Comparison source changed")
        with np.load(root / "predictions.npz", allow_pickle=False) as data:
            require(np.array_equal(data["labels"], labels) and
                    np.array_equal(data["image_paths"], [row["image_path"] for row in rows]), "Holdout rows changed")
            predictions[arm] = data["predictions"].copy()
        initial[arm], traces[arm] = status["initial_model_sha256"], (root / "steps.jsonl").read_bytes()
        histories[arm] = json_read(root / "history.json")
    require(initial[ARMS[0]] == initial[ARMS[1]] and traces[ARMS[0]] == traces[ARMS[1]], "Pair replay differs")
    trace_rows = [json.loads(line) for line in traces[ARMS[0]].splitlines()]
    require(len(trace_rows) == plan["updates_per_arm"] and
            all(row["optimizer_updates"] == row["ema_updates"] == index + 1 and
                row["epoch"] == index // plan["steps_per_epoch"] + 1 and
                row["batch"] == index % plan["steps_per_epoch"] for index, row in enumerate(trace_rows)),
            "Incomplete paired update trace")
    recomputed = paired_metrics(labels, predictions[ARMS[0]], predictions[ARMS[1]], len(plan["classes"]),
                                json_read(workspace / "groups.json"))
    require(all(recomputed[key] == report[key] for key in recomputed), "Pair metrics do not reproduce")
    for arm in ARMS:
        require(histories[arm][-1]["updates"] == plan["updates_per_arm"] and
                all(histories[arm][-1]["ema"][key] == report["slices"]["all"][arm][key]
                    for key in ("macro", "micro")), "History and final predictions differ")
    context, files = test_files(plan)
    package = workspace / "submission"
    delivery = json_read(package / "report.json")
    require(delivery["status"] == "package_ready" and delivery["rows"] == len(files) and
            delivery["plan_sha256"] == sha(plan_path) and
            delivery["checkpoint_sha256"] == sha(selected_path), "Submission binding changed")
    require(sha(package / "pred_results.csv") == delivery["csv_sha256"] and
            sha(package / "submission.zip") == delivery["zip_sha256"], "Submission digest changed")
    with zipfile.ZipFile(package / "submission.zip") as archive:
        require(archive.namelist() == ["pred_results.csv"] and
                archive.read("pred_results.csv") == (package / "pred_results.csv").read_bytes(), "ZIP/CSV differ")
    command = [sys.executable, str(Path(__file__).resolve().parent / "check_submission.py"),
               "--test_dir", str(context.test_root), "--class-mapping", str(context.reference["data"]["class_mapping"]),
               "--csv", str(package / "pred_results.csv"), "--zip", str(package / "submission.zip")]
    checked = subprocess.run(command, capture_output=True, text=True, check=True)
    require("All checks passed!" in checked.stdout + checked.stderr, "Formal submission checks did not pass")
    output.mkdir(parents=True)
    sources = {"controller_final.json": controller_path, "pair_status.json": run / "status.json",
               "comparison.json": workspace / "comparison/report.json", "paired.csv": workspace / "comparison/paired.csv",
               "per_class.csv": workspace / "comparison/per_class.csv", "delivery_report.json": package / "report.json",
               "original_history.json": run / "original/history.json", "supervision_history.json": run / "v1_supervision/history.json",
               "supervision_complete_status.json": run / "v1_supervision/status.json",
               "selected.binding.json": selected_path.with_suffix(".binding.json")}
    for name, source in sources.items():
        shutil.copyfile(source, output / name)
    (output / "submission_check.log").write_text(checked.stdout + checked.stderr)
    validation = dict(status="completed_delivered", verified_at=datetime.now(timezone.utc).isoformat(),
                      plan_sha256=sha(plan_path), paired_trace_verified=True, updates_per_arm=len(trace_rows),
                      metrics_reproduced=True, test_rows=len(files), submission_checks_passed=True, zip_csv_identical=True,
                      csv=str(package / "pred_results.csv"), zip=str(package / "submission.zip"),
                      csv_sha256=delivery["csv_sha256"], zip_sha256=delivery["zip_sha256"],
                      checkpoint_sha256=delivery["checkpoint_sha256"], selected_weights="last_ema",
                      original_retrained=False, platform_score=None, decision=recomputed["decision"],
                      automatic_ladder=False, checker_command=command)
    dump(output / "final_validation.json", validation)
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--controller-status", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.plan, args.controller_status, args.output), indent=2))


if __name__ == "__main__":
    main()
