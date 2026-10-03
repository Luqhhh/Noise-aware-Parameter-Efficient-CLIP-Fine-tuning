"""One authorized V2 six-view logit SUM and fixed uniform-prior candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_v2_fixed_prior as parent
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_test_bias import fit_test_uniform_bias
from v2.runtime import InferenceDataset, amp, write_submission

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    cfg = parent.read_json(path)
    base = parent.load_config(ROOT / "configs/v2_sixview_bias_20261002.json")
    fixed = {k: v for k, v in base.items() if k not in ("experiment_id", "reduction")}
    fixed.update(experiment_id="V2_SIXVIEW_LOGIT_SUM_20261003", reduction="sum_view_logits",
                 promotion_baseline_percent=74.6688387992735,
                 native_replay_max_probability_error=0.01,
                 native_replay_max_disagreement_fraction=0.001)
    parent.require(all(cfg.get(k) == v for k, v in fixed.items()), "Fixed logit-sum protocol changed")
    parent.require(len(cfg["incumbent_probabilities_sha256"]) == 64, "Missing incumbent cache binding")
    return cfg


@torch.inference_mode()
def collect_views(model, root, files, cfg, output):
    shape = (len(cfg["views"]), len(files), model.head.out_features)
    cache = np.lib.format.open_memmap(output / "view_logits.npy", mode="w+", dtype=np.float32, shape=shape)
    scores = np.zeros(shape[1:], dtype=np.float32)
    mean = np.zeros_like(scores)
    started = time.monotonic()
    for view_index, view in enumerate(cfg["views"]):
        data = DataLoader(InferenceDataset(root, files, parent.inference_view(cfg, view)),
                          batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"],
                          pin_memory=True, persistent_workers=False, prefetch_factor=2)
        visited = np.zeros(len(files), dtype=bool)
        for batch, (images, indices) in enumerate(data):
            with amp():
                logits = model(images.cuda(non_blocking=True))
            values = logits.float()
            matrix = values.cpu().numpy()
            probabilities = values.softmax(1).cpu().numpy()
            parent.require(np.isfinite(matrix).all(), "Nonfinite view logits")
            ix = indices.numpy()
            parent.require(not visited[ix].any(), "Repeated inference indices")
            cache[view_index, ix] = matrix
            scores[ix] += matrix  # Deliberately SUM: never divide the logits by six.
            mean[ix] += probabilities
            visited[ix] = True
            if batch % 200 == 0:
                progress = dict(status="inference", view=view, completed_views=view_index,
                                view_rows=int(visited.sum()), rows=len(files), seconds=time.monotonic()-started)
                atomic_json_dump(progress, output / "progress.json")
                print(json.dumps(progress), flush=True)
        parent.require(visited.all(), "Incomplete view coverage")
        cache.flush()
    mean /= len(cfg["views"])
    np.save(output / "summed_logits.npy", scores)
    np.save(output / "replayed_mean_probabilities.npy", mean)
    return scores, mean


def check_native_replay(mean, cfg):
    path = Path(cfg["incumbent_probabilities"])
    parent.require(sha256_file(path) == cfg["incumbent_probabilities_sha256"], "Incumbent cache changed")
    previous = np.load(path, mmap_mode="r", allow_pickle=False)
    parent.require(previous.shape == mean.shape, "Incumbent cache shape mismatch")
    error = float(np.abs(mean-previous).max())
    changed = int((mean.argmax(1) != previous.argmax(1)).sum())
    report = dict(max_probability_error=error, changed_raw_predictions=changed, rows=len(mean),
                  exact_probabilities=bool(np.array_equal(mean, previous)),
                  passed=error <= cfg["native_replay_max_probability_error"] and
                  changed <= len(mean)*cfg["native_replay_max_disagreement_fraction"])
    return report


@torch.inference_mode()
def cold_replay(model, root, files, cfg, output):
    cache = np.load(output / "view_logits.npy", mmap_mode="r", allow_pickle=False)
    rows = min(cfg["probe_rows"], len(files))
    maximum = 0.0
    for view_index, view in enumerate(cfg["views"]):
        data = DataLoader(InferenceDataset(root, files[:rows], parent.inference_view(cfg, view)),
                          batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
        for images, indices in data:
            with amp():
                actual = model(images.cuda(non_blocking=True))
            actual = actual.float().cpu().numpy()
            expected = cache[view_index, indices.numpy()]
            maximum = max(maximum, float(np.abs(actual-expected).max()))
    parent.require(maximum == 0.0, "Cold-load view logits differ from complete inference")
    return dict(status="passed", rows=rows, views=len(cfg["views"]), max_absolute_error=maximum)


def run(cfg, config_path, output):
    parent.require(not output.exists(), "Output exists; no implicit restart or overwrite")
    parent.ensure_idle_cuda()
    started = time.monotonic()
    inputs = parent.verified_delivery(cfg)
    pixels = parent.verify_test_pixels(cfg, inputs["manifest"])
    parent.require(sha256_file(cfg["incumbent_probabilities"]) == cfg["incumbent_probabilities_sha256"],
                   "Incumbent cache changed before inference")
    output.mkdir(parents=True)
    codes = parent.source_code_hashes()
    for name in ("scripts/run_v2_sixview_logit_sum.py", "scripts/verify_v2_sixview_logit_sum.py"):
        codes[name] = sha256_file(ROOT / name)
    binding = dict(config_sha256=sha256_file(config_path), source_code=codes,
                   source_hashes=inputs["source_hashes"], source_receipt_sha256=inputs["receipt_sha256"],
                   plan_sha256=cfg["expected_plan_sha256"], experiment_id=cfg["experiment_id"],
                   local_test_pixels_checked=pixels,
                   incumbent_probabilities_sha256=cfg["incumbent_probabilities_sha256"])
    atomic_json_dump(binding, output / "binding.json")
    (output / "source_delivery_receipt.json").write_bytes(inputs["receipt_bytes"])
    manifest, files = inputs["manifest"], inputs["files"]
    root = Path(manifest["test_root"])
    model = parent.load_model(cfg, inputs["payload"]["model"], manifest["num_classes"])
    scores, mean = collect_views(model, root, files, cfg, output)
    replay = check_native_replay(mean, cfg)
    atomic_json_dump(replay, output / "native_replay.json")
    parent.require(replay["passed"], "Incumbent probability decoder failed fixed replay tolerances")
    del model
    torch.cuda.empty_cache()
    model = parent.load_model(cfg, inputs["payload"]["model"], manifest["num_classes"])
    cold = cold_replay(model, root, files, cfg, output)
    atomic_json_dump(cold, output / "cold_replay.json")
    del model, inputs["payload"]
    torch.cuda.empty_cache()
    bias = fit_test_uniform_bias(torch.from_numpy(scores).cuda(), iterations=cfg["bias_iterations"]).numpy()
    raw = scores.argmax(1)
    corrected = (scores.astype(np.float64) + bias.astype(np.float64)).argmax(1)
    np.savez(output / "predictions.npz", names=np.asarray(files), bias=bias, raw=raw, calibrated=corrected)
    for name, predictions in (("submission_raw", raw), ("submission_bias", corrected)):
        package = output / name
        write_submission(package, files, predictions, manifest["num_classes"])
        check = subprocess.run([sys.executable, str(ROOT / "scripts/check_submission.py"),
                  "--test_dir", str(root), "--class-mapping", str(Path(cfg["stage_root"]) / "class_to_idx.json"),
                  "--csv", str(package / "pred_results.csv"), "--zip", str(package / "submission.zip")],
                  capture_output=True, text=True)
        (package / "submission_check.log").write_text(check.stdout + check.stderr)
        check.check_returncode()
        atomic_json_dump(dict(binding=binding, checkpoint_sha256=inputs["source_hashes"][parent.FINAL],
            weights="full_epoch5_raw", views=cfg["views"], input_sizes=cfg["image_sizes"],
            reduction=cfg["reduction"], bias_iterations=200, bias_strength=1.0, test_prior="official_uniform",
            test_statistical_fitting=name == "submission_bias", model_parameter_updates=False,
            test_labels_used=False, platform_score=None, promotion_baseline_percent=cfg["promotion_baseline_percent"],
            status="checked_pending_independent_verification"), package / "manifest.json")
    caches = {name: sha256_file(output / name) for name in (
        "view_logits.npy", "summed_logits.npy", "replayed_mean_probabilities.npy", "predictions.npz",
        "cold_replay.json", "native_replay.json")}
    report = dict(status="packages_checked_pending_independent_verification", binding=binding,
        output=str(output.resolve()), rows=len(files), classes=manifest["num_classes"],
        views=cfg["views"], input_sizes=cfg["image_sizes"], reduction=cfg["reduction"],
        bias_iterations=200, bias_strength=1.0, cache_sha256=caches, native_replay=replay, cold_replay=cold,
        bias_changed_predictions=int((raw != corrected).sum()),
        logit_sum_vs_probability_mean_raw_changes=int((raw != mean.argmax(1)).sum()),
        elapsed_seconds=time.monotonic()-started, training=False, model_parameter_updates=False,
        platform_score=None, promotion_baseline_percent=cfg["promotion_baseline_percent"],
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(), batch_size=cfg["batch_size"]))
    atomic_json_dump(report, output / "report.json")
    from verify_v2_sixview_logit_sum import verify
    verified = verify(cfg, config_path, output)
    atomic_json_dump(verified, output / "independent_verification.json")
    report.update(status="completed_verified_delivery", total_seconds=time.monotonic()-started)
    atomic_json_dump(report, output / "report.json")
    atomic_json_dump(dict(status=report["status"], seconds=report["total_seconds"]), output / "progress.json")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(2)
    run(cfg, args.config, args.output)


if __name__ == "__main__":
    main()
