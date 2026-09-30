"""Independently verify a completed full-population v1 EMA-SWA delivery on CPU."""
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import torch

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_pipeline import StageContext, load_artifact, load_recipe, validate_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    context = StageContext(load_recipe(args.config))
    cfg = context.config
    assert cfg["source"]["partition"] == "full_train" and cfg["train"]["swa_enabled"]
    assert not context.val and context.train == context.full
    out = Path(cfg["output"]["root"])
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "completed" and status["stage"] == "delivered"
    targets = load_artifact(out / "targets.pt", context.binding)
    assert targets["image_paths"] == [r["image_path"] for r in context.full]
    used = int((targets["weights"] > 0).sum())
    target_hash = sha256_file(out / "targets.pt")
    final = out / "training/selected.pt"
    selected = load_artifact(final, context.binding)
    window = list(range(cfg["train"]["swa_start"], cfg["train"]["epochs"] + 1))
    assert selected["selected_policy"] == "swa_ema" and selected["swa_epochs"] == window
    assert selected["swa"]["count"] == len(window) and selected["swa"]["decay"] is None
    assert selected["targets_sha256"] == target_hash
    assert [r["epoch"] for r in selected["history"]] == list(range(1, cfg["train"]["epochs"] + 1))
    assert all(r["optimizer_steps"] > 0 and math.isfinite(r["loss"]) for r in selected["history"])
    expected, sources = {}, []
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        path = out / f"training/epoch_{epoch:02d}.pt"
        payload = load_artifact(path, context.binding)
        assert payload["epoch"] == epoch and payload["trajectory"] == selected["trajectory"]
        assert payload["targets_sha256"] == target_hash and payload["optimizer"] is not None
        if epoch in window:
            assert payload["ema"]["decay"] == cfg["train"]["ema_decay"]
            state = payload["ema"]["state"]
            assert state.keys() == selected["selected_state"].keys()
            for name, tensor in state.items():
                expected[name] = expected.get(name, 0) + tensor.double() / len(window)
        sources.append(dict(epoch=epoch, path=str(path), sha256=sha256_file(path)))
        del payload
    maximum = 0.0
    for name, average in expected.items():
        actual = selected["selected_state"][name].double()
        maximum = max(maximum, float((actual - average).abs().max()))
        torch.testing.assert_close(actual, average, rtol=1e-5, atol=1e-7)
    calibration = load_artifact(out / "calibration.pt", context.binding)
    validate_calibration(calibration, context, final)
    validation = json.loads((out / "validation_report.json").read_text())
    assert validation["score_scope"] == "training_included_diagnostic"
    assert validation["calibration_split_in_training_population"] is True
    assert validation["independent_test_score"] is False
    with (out / "validation_predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["image_path"], int(r["label"])) for r in rows] == [
        (r["image_path"], int(r["label"])) for r in context.calibration]
    labels = np.array([int(r["label"]) for r in rows])
    counts = np.bincount(labels, minlength=len(context.classes))
    assert (counts > 0).all()
    metrics = {}
    for mode in ("raw", "calibrated"):
        correct = np.array([int(r[f"{mode}_prediction"]) for r in rows]) == labels
        per_class = np.bincount(labels[correct], minlength=len(context.classes))
        metrics[mode] = dict(samples=len(rows), correct=int(correct.sum()),
                            micro=float(correct.mean()), macro=float((per_class / counts).mean()))
        for key, value in metrics[mode].items():
            assert math.isclose(value, validation[mode][key], rel_tol=1e-6, abs_tol=1e-7)
    submission = out / "submission"
    manifest = json.loads((submission / "manifest.json").read_text())
    assert manifest["binding"] == context.binding and manifest["selected_policy"] == "swa_ema"
    hashes = {"checkpoint_sha256": sha256_file(final),
              "calibration_sha256": sha256_file(out / "calibration.pt"),
              "prediction_csv_sha256": sha256_file(submission / "pred_results.csv"),
              "submission_zip_sha256": sha256_file(submission / "submission.zip")}
    assert all(manifest[key] == value for key, value in hashes.items())
    with zipfile.ZipFile(submission / "submission.zip") as archive:
        assert archive.namelist() == ["pred_results.csv"]
        assert archive.read("pred_results.csv") == (submission / "pred_results.csv").read_bytes()
    with (submission / "pred_results.csv").open() as handle:
        predictions = list(csv.reader(handle))
    assert len(predictions) == context.manifest["test_samples"] == manifest["prediction_count"]
    desktop = Path(status["desktop_zip"])
    assert sha256_file(desktop) == hashes["submission_zip_sha256"] == status["desktop_zip_sha256"]
    assert "All checks passed" in (submission / "submission_check.log").read_text()
    report = dict(experiment_id=cfg["project"]["experiment_id"], status="completed_delivered",
        started_at=status["started_at"], completed_at=status["completed_at"],
        elapsed_seconds=status["elapsed_seconds"], binding=context.binding, training_partition="full_train",
        training_population=len(context.full), student_used_samples=used, classes=len(context.classes),
        epochs=cfg["train"]["epochs"], swa_epochs=window, selected_policy="swa_ema", sources=sources,
        independent_float64_swa_check=True, swa_max_absolute_difference=maximum, metrics=metrics,
        score_scope="training_included_diagnostic", independent_validation_available=False,
        platform_score=None, submission_rows=len(predictions), submission_checks_passed=9,
        zip_csv_bytes_identical=True, hashes=hashes, checkpoint=str(final),
        csv=str(submission / "pred_results.csv"), zip=str(submission / "submission.zip"),
        desktop_zip=str(desktop), automatic_v3_resume=False, v3_status="paused_by_user")
    atomic_json_dump(report, Path(args.report))
    print(json.dumps({k: v for k, v in report.items() if k not in ("sources", "binding")}, indent=2))


if __name__ == "__main__":
    main()
