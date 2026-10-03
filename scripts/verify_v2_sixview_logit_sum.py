"""Independently replay six cached views, fixed bias, and every CSV/ZIP decision."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from aegis_clip.runtime import atomic_json_dump, sha256_file
from verify_v1_768_full_delivery import numpy_uniform_bias


def verify_sum(views, scores):
    """Replay each IEEE float32 addition in float64, including its rounding."""
    exact = np.zeros(scores.shape, dtype=np.float64)
    rounded = np.zeros_like(exact)
    magnitude = np.zeros_like(exact)
    for view in views:
        value = view.astype(np.float64)
        assert np.isfinite(value).all()
        exact += value
        magnitude += np.abs(value)
        rounded = (rounded + value).astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(scores, rounded)
    eps = np.finfo(np.float32).eps
    bound = len(views)*eps/(1-len(views)*eps)*magnitude
    assert np.all(np.abs(rounded-exact) <= bound)
    return rounded, dict(float32_rounding_entries=int((rounded != exact).sum()),
                         max_rounding_error=float(np.abs(rounded-exact).max()))


def probability_rounding_report(actual, expected, classes, views):
    # Conservative sequential-sum error bound: softmax normalization over C
    # classes plus the V-term mean. Native GPU replay is checked separately.
    eps = np.finfo(np.float32).eps
    operations = classes + views + 2
    gamma = operations*eps/(1-operations*eps)
    np.testing.assert_allclose(actual, expected, rtol=gamma, atol=np.finfo(np.float32).tiny)
    return dict(relative_bound=float(gamma), operation_count=operations,
                max_absolute_error=float(np.abs(actual-expected).max()))


def verify(cfg, config_path, output, source_snapshot=None):
    started = time.monotonic()
    output = Path(output)
    root = Path(__file__).resolve().parents[1]
    report = json.loads((output / "report.json").read_text())
    binding = json.loads((output / "binding.json").read_text())
    assert report["binding"] == binding
    assert binding["config_sha256"] == sha256_file(config_path)
    assert report["reduction"] == cfg["reduction"] == "sum_view_logits"
    assert report["views"] == cfg["views"] == [f"scale{s}{f}" for s in (448, 512, 576) for f in ("", "_flip")]
    assert report["bias_iterations"] == cfg["bias_iterations"] == 200
    assert report["bias_strength"] == cfg["bias_strength"] == 1.0
    assert sha256_file(output / "source_delivery_receipt.json") == binding["source_receipt_sha256"]
    for name, expected in binding["source_hashes"].items():
        assert sha256_file(Path(cfg["delivery_root"]) / name) == expected, name
    source_root = Path(source_snapshot) if source_snapshot is not None else root
    for name, expected in binding["source_code"].items():
        assert sha256_file(source_root / name) == expected, name
    for name, expected in report["cache_sha256"].items():
        assert sha256_file(output / name) == expected, name
    manifest = json.loads((Path(cfg["stage_root"]) / "dataset_manifest.json").read_text())
    shape = (manifest["test_samples"], manifest["num_classes"])
    assert binding["local_test_pixels_checked"] == shape[0]
    views = np.load(output / "view_logits.npy", mmap_mode="r", allow_pickle=False)
    scores = np.load(output / "summed_logits.npy", allow_pickle=False)
    mean = np.load(output / "replayed_mean_probabilities.npy", allow_pickle=False)
    assert views.shape == (6, *shape) and scores.shape == mean.shape == shape
    expected_scores, rounding = verify_sum(views, scores)
    expected_mean = np.zeros(shape, dtype=np.float64)
    for view in views:
        assert np.isfinite(view).all()
        z = view.astype(np.float64)
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        expected_mean += p / p.sum(1, keepdims=True) / 6
    probability_rounding = probability_rounding_report(mean, expected_mean, shape[1], 6)
    assert sha256_file(cfg["incumbent_probabilities"]) == binding["incumbent_probabilities_sha256"] == cfg["incumbent_probabilities_sha256"]
    previous = np.load(cfg["incumbent_probabilities"], mmap_mode="r", allow_pickle=False)
    native_error = float(np.abs(mean-previous).max())
    native_changes = int((mean.argmax(1) != previous.argmax(1)).sum())
    assert native_error == report["native_replay"]["max_probability_error"] <= cfg["native_replay_max_probability_error"]
    assert native_changes == report["native_replay"]["changed_raw_predictions"] <= shape[0]*cfg["native_replay_max_disagreement_fraction"]
    assert report["cold_replay"] == json.loads((output / "cold_replay.json").read_text())
    assert report["cold_replay"]["status"] == "passed" and report["cold_replay"]["max_absolute_error"] == 0
    with np.load(output / "predictions.npz", allow_pickle=False) as data:
        names, bias = data["names"].tolist(), data["bias"]
        raw, corrected = data["raw"], data["calibrated"]
    with (Path(cfg["stage_root"]) / "test_manifest.csv").open(newline="") as stream:
        official = sorted(Path(row["image_path"]).name for row in csv.DictReader(stream))
    assert names == official and len(set(names)) == shape[0]
    assert bias.shape == (shape[1],) and np.isfinite(bias).all()
    np.testing.assert_array_equal(raw, expected_scores.argmax(1))
    np.testing.assert_array_equal(corrected, (expected_scores+bias).argmax(1))
    independent_bias = numpy_uniform_bias(expected_scores, 200)
    np.testing.assert_allclose(bias, independent_bias, atol=2e-5, rtol=0)
    np.testing.assert_array_equal(corrected, (expected_scores+independent_bias).argmax(1))
    assert report["bias_changed_predictions"] == int((raw != corrected).sum())
    assert report["logit_sum_vs_probability_mean_raw_changes"] == int((raw != mean.argmax(1)).sum())
    packages = {}
    for name, predictions in (("submission_raw", raw), ("submission_bias", corrected)):
        package = output / name
        expected = "".join(f"{n}, {int(p):04d}\n" for n, p in zip(names, predictions)).encode()
        assert (package / "pred_results.csv").read_bytes() == expected
        with zipfile.ZipFile(package / "submission.zip") as archive:
            assert archive.namelist() == ["pred_results.csv"]
            assert archive.read("pred_results.csv") == expected
        meta = json.loads((package / "manifest.json").read_text())
        assert meta["binding"] == binding and meta["weights"] == "full_epoch5_raw"
        assert meta["reduction"] == "sum_view_logits" and meta["views"] == cfg["views"]
        assert meta["checkpoint_sha256"] == binding["source_hashes"]["runs/full_576/last.pt"]
        assert meta["test_statistical_fitting"] == (name == "submission_bias")
        assert meta["model_parameter_updates"] is False and meta["test_labels_used"] is False
        assert meta["platform_score"] is None and "All checks passed" in (package / "submission_check.log").read_text()
        packages[name] = dict(rows=len(names), csv_sha256=sha256_file(package / "pred_results.csv"),
                             zip_sha256=sha256_file(package / "submission.zip"))
    return dict(status="passed", rows=len(names), view_logits_sum_replayed=True,
        sum_rounding=rounding,
        probability_rounding=probability_rounding,
        original_bound_source_root=str(source_root.resolve()),
        verification_source_sha256=sha256_file(Path(__file__)),
        mean_probability_control_replayed=True, full_float64_decision_agreement=True,
        bias_max_absolute_error=float(np.abs(bias-independent_bias).max()),
        source_and_cache_sha_verified=True, csv_zip_bytes_replayed=True, packages=packages,
        elapsed_seconds=time.monotonic()-started, platform_score=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source-snapshot", help="Explicit SHA-bound original source snapshot for a documented verifier repair")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    report = verify(cfg, args.config, args.output, args.source_snapshot)
    atomic_json_dump(report, args.report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
