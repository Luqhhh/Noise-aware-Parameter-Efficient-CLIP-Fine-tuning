"""Recheck one fixed v1 SWA candidate against the original epoch-12 EMA."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_by_path(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["image_path"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate paths: {path}")
    return indexed


def checked_metrics(rows, column, classes, recorded):
    total, correct = [0] * classes, [0] * classes
    for row in rows.values():
        label, prediction = int(row["label"]), int(row[column])
        if not 0 <= label < classes or not 0 <= prediction < classes:
            raise ValueError("Label or prediction outside class range")
        total[label] += 1
        correct[label] += prediction == label
    if not all(total):
        raise ValueError("Full validation must cover all classes")
    measured = dict(samples=len(rows), correct=sum(correct),
                    micro=sum(correct) / len(rows),
                    macro=sum(c / n for c, n in zip(correct, total)) / classes)
    for key in ("samples", "correct"):
        if measured[key] != recorded[key]:
            raise ValueError(f"Recorded {key} disagrees with predictions")
    for key in ("micro", "macro"):
        if not math.isclose(measured[key], recorded[key], abs_tol=1e-7):
            raise ValueError(f"Recorded {key} disagrees with predictions")
    if total != recorded["per_class_samples"] or correct != recorded["per_class_correct"]:
        raise ValueError("Recorded per-class counts disagree with predictions")
    return measured


def submission(path, expected_rows):
    csv_path, zip_path = path / "pred_results.csv", path / "submission.zip"
    raw = csv_path.read_bytes()
    with zipfile.ZipFile(zip_path) as handle:
        if handle.namelist() != ["pred_results.csv"] or handle.read("pred_results.csv") != raw:
            raise ValueError("ZIP does not contain the exact external CSV")
    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    predictions = {name: label.strip() for name, label in rows}
    if len(rows) != expected_rows or len(predictions) != expected_rows:
        raise ValueError("Wrong submission count or duplicate names")
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export = json.loads((args.candidate / "export_report.json").read_text())
    if export["epochs"] != list(range(4, 13)) or export["selected_policy"] != "swa_ema":
        raise ValueError("This report requires the fixed epoch 4-12 EMA SWA recipe")
    for source in export["sources"]:
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("Original epoch changed after export")
    files = [args.original / "validation_report.json", args.original / "validation_predictions.csv",
             args.candidate / "validation_report.json", args.candidate / "validation_predictions.csv",
             args.candidate / "export_report.json", args.candidate / "config.yaml",
             args.candidate / "selected.pt", args.candidate / "calibration.pt"]
    reports = [json.loads(files[i].read_text()) for i in (0, 2)]
    original, candidate = [rows_by_path(files[i]) for i in (1, 3)]
    if len(candidate) != 14880 or original.keys() != candidate.keys():
        raise ValueError("Validation paths differ or expected full validation is incomplete")
    if any(original[p]["label"] != candidate[p]["label"] for p in original):
        raise ValueError("Original-label validation changed")
    if reports[0]["views"] != reports[1]["views"]:
        raise ValueError("Original and SWA decoders differ")
    if reports[1]["bias_fitted_on_this_split"] is not True or reports[1]["independent_test_score"] is not False:
        raise ValueError("Calibration must be identified as training-side development evaluation")
    metrics, pairing = {}, {}
    paired_path = args.candidate / "paired_vs_epoch12.csv"
    with paired_path.open("w", newline="") as handle:
        fields = ["image_path", "label", "original_raw", "swa_raw",
                  "original_calibrated", "swa_calibrated"]
        writer = csv.writer(handle)
        writer.writerow(fields)
        for path, row in candidate.items():
            parent = original[path]
            writer.writerow([path, row["label"], parent["raw_prediction"], row["raw_prediction"],
                             parent["calibrated_prediction"], row["calibrated_prediction"]])
    files.append(paired_path)
    for mode in ("raw", "calibrated"):
        column = f"{mode}_prediction"
        before = checked_metrics(original, column, 750, reports[0][mode])
        after = checked_metrics(candidate, column, 750, reports[1][mode])
        fixed = regressed = changed = 0
        for path, row in candidate.items():
            old = original[path][column]
            new, label = row[column], row["label"]
            fixed += old != label and new == label
            regressed += old == label and new != label
            changed += old != new
        if fixed - regressed != after["correct"] - before["correct"]:
            raise ValueError("Paired net correction disagrees with accuracy change")
        metrics[mode] = dict(original=before, swa=after,
            macro_delta_pp=100 * (after["macro"] - before["macro"]),
            micro_delta_pp=100 * (after["micro"] - before["micro"]))
        pairing[mode] = dict(corrected=fixed, regressed=regressed,
                             net_corrected=fixed - regressed, predictions_changed=changed)
    old_test = submission(args.original / "submission", 37444)
    new_test = submission(args.candidate / "submission", 37444)
    if old_test.keys() != new_test.keys():
        raise ValueError("Test names changed")
    files.extend(args.candidate / "submission" / name for name in
                 ("pred_results.csv", "submission.zip", "submission_check.log", "manifest.json"))
    manifest = json.loads((args.candidate / "submission/manifest.json").read_text())
    if manifest["selected_policy"] != "swa_ema":
        raise ValueError("Submission is not bound to SWA weights")
    for key, path in (("checkpoint_sha256", args.candidate / "selected.pt"),
                      ("calibration_sha256", args.candidate / "calibration.pt"),
                      ("prediction_csv_sha256", args.candidate / "submission/pred_results.csv"),
                      ("submission_zip_sha256", args.candidate / "submission/submission.zip")):
        if manifest[key] != sha256(path):
            raise ValueError(f"Submission manifest hash disagrees: {key}")
    if "All checks passed" not in (args.candidate / "submission/submission_check.log").read_text():
        raise ValueError("Formal submission checker did not pass")
    summary = dict(experiment_id="V1_SWA_TRIAL_20260930", status="completed_delivered",
        training_started=False, window=export["epochs"], weight_source="ema",
        source_training_dir=export["source_training_dir"], binding=export["binding"],
        sources=export["sources"], decoder=reports[1]["views"],
        metrics=metrics, paired=pairing, bias_fitted_on_val_dev=True,
        independent_test_score=False, platform_score=None,
        l05_fixed_local_reference=dict(macro=0.758265, micro=0.766532,
            swa_minus_l05_macro_pp=100 * (metrics["calibrated"]["swa"]["macro"] - 0.758265),
            swa_minus_l05_micro_pp=100 * (metrics["calibrated"]["swa"]["micro"] - 0.766532)),
        submission_rows=37444, submission_checks_passed=9, zip_csv_bytes_identical=True,
        test_predictions_changed_vs_original=sum(old_test[p] != new_test[p] for p in old_test),
        candidate=str(args.candidate.resolve()),
        verified_files=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in files])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"metrics": metrics, "paired": pairing}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
