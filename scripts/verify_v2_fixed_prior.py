"""Independent CPU replay of fixed V2 prior fitting and both submission packages."""
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


def verify(cfg, config_path, output):
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    report = json.loads((output / "report.json").read_text())
    binding = json.loads((output / "binding.json").read_text())
    assert report["binding"] == binding
    assert binding["config_sha256"] == sha256_file(config_path)
    assert binding["plan_sha256"] == cfg["expected_plan_sha256"]
    assert report["views"] == cfg["views"] and report["reduction"] == cfg["reduction"]
    assert report["bias_iterations"] == cfg["bias_iterations"] == 200
    assert report["bias_strength"] == cfg["bias_strength"] == 1.0
    assert sha256_file(output / "source_delivery_receipt.json") == binding["source_receipt_sha256"]
    for name, expected in binding["source_hashes"].items():
        assert sha256_file(Path(cfg["delivery_root"]) / name) == expected, name
    for name, expected in binding["source_code"].items():
        assert sha256_file(root / name) == expected, name
    assert sha256_file(output / "predictions.npz") == report["predictions_sha256"]
    with np.load(output / "predictions.npz", allow_pickle=False) as data:
        p, scores, bias = data["probabilities"], data["scores"], data["bias"]
        names = data["names"].tolist()
        saved_raw, saved_corrected = data["raw"], data["calibrated"]
    manifest = json.loads((Path(cfg["stage_root"]) / "dataset_manifest.json").read_text())
    assert p.shape == scores.shape == (manifest["test_samples"], manifest["num_classes"])
    assert binding["local_test_pixels_checked"] == manifest["test_samples"]
    assert bias.shape == (manifest["num_classes"],)
    assert np.isfinite(p).all() and (p >= 0).all() and np.isfinite(bias).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=2e-6, rtol=0)
    with (Path(cfg["stage_root"]) / "test_manifest.csv").open(newline="") as stream:
        official_names = sorted(Path(row["image_path"]).name for row in csv.DictReader(stream))
    assert names == official_names and len(set(names)) == len(names)
    # Independent float64 logarithms and fit, plus exact cached decision replay.
    expected_scores = np.log(np.maximum(p.astype(np.float64), cfg["probability_floor"]))
    np.testing.assert_allclose(scores, expected_scores, atol=1e-14, rtol=0)
    independent_bias = numpy_uniform_bias(scores, cfg["bias_iterations"])
    np.testing.assert_allclose(bias, independent_bias, atol=2e-5, rtol=0)
    raw = p.argmax(1)
    corrected = (scores + bias).argmax(1)
    np.testing.assert_array_equal(raw, scores.argmax(1))
    np.testing.assert_array_equal(raw, saved_raw)
    np.testing.assert_array_equal(corrected, saved_corrected)
    float64_corrected = (scores.astype(np.float64) + independent_bias).argmax(1)
    # Do not silently accept a different decision in the independent fit.
    np.testing.assert_array_equal(corrected, float64_corrected)
    packages = {}
    for directory, predictions in (("submission_raw", raw), ("submission_bias", corrected)):
        package = output / directory
        expected = "".join(f"{name}, {label:04d}\n" for name, label in zip(names, predictions)).encode()
        assert (package / "pred_results.csv").read_bytes() == expected
        with zipfile.ZipFile(package / "submission.zip") as archive:
            assert archive.namelist() == ["pred_results.csv"]
            assert archive.read("pred_results.csv") == expected
        meta = json.loads((package / "manifest.json").read_text())
        assert meta["binding"] == binding and meta["model_parameter_updates"] is False
        assert meta["test_labels_used"] is False and meta["platform_score"] is None
        assert meta["weights"] == "full_epoch5_raw" and meta["views"] == cfg["views"]
        profile = cfg.get("decoder_profile", "original_four_view")
        assert meta["decoder_profile"] == profile
        assert meta["input_sizes"] == ([448, 512, 576] if profile == "multiscale_flip_six" else [576])
        assert meta["test_statistical_fitting"] == (directory == "submission_bias")
        assert "All checks passed" in (package / "submission_check.log").read_text()
        packages[directory] = dict(csv_sha256=sha256_file(package / "pred_results.csv"),
                                   zip_sha256=sha256_file(package / "submission.zip"), rows=len(names))
    with (Path(cfg["delivery_root"]) / "submission/pred_results.csv").open(newline="") as stream:
        original = list(csv.reader(stream))
    assert [row[0] for row in original] == names
    original_predictions = np.array([int(row[1]) for row in original])
    assert report["raw_remote_disagreement"] == int((raw != original_predictions).sum())
    mismatched = p[raw != original_predictions]
    ordered = np.sort(mismatched, axis=1)
    margin = float((ordered[:, -1]-ordered[:, -2]).max()) if len(ordered) else 0.0
    assert report["raw_remote_disagreement_margin_max"] == margin
    if cfg.get("decoder_profile", "original_four_view") == "original_four_view":
        assert report["raw_comparison_is_same_decoder"] is True
        assert len(mismatched) <= len(raw)*cfg["raw_replay_max_disagreement_fraction"]
        assert margin <= cfg["raw_replay_max_probability_margin"] and report["raw_replay_passed"]
    else:
        assert cfg["decoder_profile"] == "multiscale_flip_six"
        assert cfg["image_sizes"] == [448, 512, 576]
        assert cfg["views"] == [f"scale{size}{suffix}" for size in (448, 512, 576) for suffix in ("", "_flip")]
        assert report["raw_comparison_is_same_decoder"] is False and report["raw_replay_passed"] is None
    assert report["bias_changed_predictions"] == int((raw != corrected).sum())
    return dict(status="passed", rows=len(names), full_float64_decision_agreement=True,
                decoder_profile=cfg.get("decoder_profile", "original_four_view"), views=cfg["views"],
                bias_max_absolute_error=float(np.abs(bias-independent_bias).max()),
                input_and_code_sha_verified=True, csv_zip_bytes_replayed=True,
                raw_remote_disagreement=report["raw_remote_disagreement"],
                bias_changed_predictions=report["bias_changed_predictions"], packages=packages,
                elapsed_seconds=time.monotonic()-started, platform_score=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    report = verify(cfg, args.config, Path(args.output))
    atomic_json_dump(report, args.report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
