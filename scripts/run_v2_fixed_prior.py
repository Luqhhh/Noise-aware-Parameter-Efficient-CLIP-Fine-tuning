"""One local, read-only V2 probability-prior transfer after verified delivery.

No remote connection, training, model selection, or upload. The original V2
four-view probability reduction is preserved; its log is the calibration score.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_test_bias import fit_test_uniform_bias
from v2.core import check_checkpoint
from v2.model import V2Classifier
from v2.runtime import InferenceDataset, amp, write_submission
from v2.training_utils import build_view

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ["resize576", "center", "flip", "resize806.4"]
FINAL = "runs/full_576/last.pt"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_config(path):
    cfg = read_json(path)
    fixed = dict(data_version="20260921", views=VIEWS, image_size=576,
                 bias_iterations=200, bias_strength=1.0,
                 reduction="log_mean_view_softmax", probability_floor=1e-30,
                 raw_replay_max_disagreement_fraction=0.001, raw_replay_max_probability_margin=0.01,
                 test_prior="official_uniform", training=False,
                 parameter_search=False, platform_upload=False)
    require(all(cfg.get(k) == v for k, v in fixed.items()), "Fixed prior protocol changed")
    require(cfg["batch_size"] == 16 and cfg["num_workers"] == 2 and cfg["probe_rows"] == 64,
            "Fixed inference resources changed")
    return cfg


def local_data(cfg):
    stage = Path(cfg["stage_root"])
    manifest = read_json(stage / "dataset_manifest.json")
    require(manifest["data_version"] == cfg["data_version"] and not manifest["external_data"],
            "Wrong local official stage")
    for name in ("class_to_idx.json", "test_manifest.csv", "val_dev.csv", "train_dev.csv", "full_train.csv"):
        require(sha256_file(stage / name) == manifest["files"][name], "Local stage asset changed: " + name)
    mapping = read_json(stage / "class_to_idx.json")
    require(mapping == {f"{i:04d}": i for i in range(manifest["num_classes"])},
            "V2 integer labels do not match official mapping")
    with (stage / "test_manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    files = sorted(Path(row["image_path"]).name for row in rows)
    require(len(files) == len(set(files)) == manifest["test_samples"], "Invalid official test filenames")
    root = Path(manifest["test_root"])
    require(sorted(p.name for p in root.iterdir() if p.is_file()) == files, "Local test coverage mismatch")
    return manifest, files


def metadata_member(archive, name):
    """Read exactly one regular member; never extract archive paths to disk."""
    matches = [m for m in archive.getmembers() if m.name == name]
    require(len(matches) == 1 and matches[0].isfile(), "Missing or ambiguous metadata: " + name)
    require(matches[0].size <= 128 * 1024 * 1024, "Unexpectedly large metadata member")
    return archive.extractfile(matches[0]).read()


def verified_delivery(cfg):
    root = Path(cfg["delivery_root"])
    receipt_path = root / "delivery_receipt.json"
    require(receipt_path.is_file(), "Waiting for verified local V2 delivery receipt")
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    require(receipt.get("submission_check_passed") is True and
            receipt.get("plan_sha256") == cfg["expected_plan_sha256"],
            "Unverified or wrong local V2 delivery")
    items = {item["relative"]: item for item in receipt["files"]}
    require(len(items) == len(receipt["files"]), "Duplicate delivery members")
    required = [FINAL, "delivery_metadata.tar.gz", "submission/report.json",
                "submission/pred_results.csv", "submission/submission.zip"]
    hashes = {}
    for name in required:
        require(name in items, "Missing required delivered file: " + name)
        path = root / name
        require(path.is_file() and path.stat().st_size == items[name]["bytes"], "Incomplete delivery: " + name)
        hashes[name] = sha256_file(path)
        require(hashes[name] == items[name]["sha256"], "Delivered SHA mismatch: " + name)
    with tarfile.open(root / "delivery_metadata.tar.gz", "r:gz") as archive:
        plan_bytes = metadata_member(archive, "plan.json")
        require(digest_bytes(plan_bytes) == cfg["expected_plan_sha256"], "Frozen V2 plan changed")
        plan = json.loads(plan_bytes)
        controller = json.loads(metadata_member(archive, "controller.json"))
        sidecar = json.loads(metadata_member(archive, "runs/full_576/last.binding.json"))
        require(controller["status"] == "completed_delivered" and controller["training_completed"] is True
                and controller["plan_sha256"] == cfg["expected_plan_sha256"], "Original V2 controller incomplete")
        for stage in ("s2_448", "s3_576", "full_576"):
            status = json.loads(metadata_member(archive, f"runs/{stage}/status.json"))
            require(status["status"] == "complete" and
                    status["updates"] == plan["stages"][stage]["total_updates"], "Incomplete V2 stage: " + stage)
        config_bytes = metadata_member(archive, plan["stages"]["full_576"]["config"])
        require(digest_bytes(config_bytes) == plan["stages"]["full_576"]["config_sha256"],
                "Original final-stage config changed")
        final_config = json.loads(config_bytes)
    manifest, files = local_data(cfg)
    recipe = plan["recipe"]
    require(plan["data_version"] == cfg["data_version"] and recipe["views"] == cfg["views"]
            and recipe["official_sha256"] == cfg["official_sha256"]
            and recipe["num_classes"] == manifest["num_classes"]
            and recipe["test_rows"] == manifest["test_samples"]
            and recipe["official_train_rows"] == manifest["train_samples"], "V2/local data or decoder mismatch")
    for name in ("class_to_idx.json", "test_manifest.csv", "full_train.csv", "train_dev.csv", "val_dev.csv"):
        original = [v for k, v in plan["inputs"].items() if Path(k).name == name]
        require(original == [manifest["files"][name]], "Source/local stage asset differs: " + name)
    for name in ("reproducibility/aegis_f1/v2/model.py", "reproducibility/aegis_f1/v2/training_utils.py",
                 "reproducibility/aegis_f1/aegis_clip/model.py"):
        original = [v for k, v in plan["inputs"].items() if k.endswith("/" + name)]
        require(original == [sha256_file(ROOT / name)], "Frozen source inference code differs: " + name)
    report = read_json(root / "submission/report.json")
    require(report["status"] == "package_ready" and report["weights"] == "raw"
            and report["views"] == VIEWS and report["rows"] == len(files)
            and report["checkpoint_sha256"] == hashes[FINAL]
            and report["csv_sha256"] == hashes["submission/pred_results.csv"]
            and report["zip_sha256"] == hashes["submission/submission.zip"], "Original package binding mismatch")
    require(sidecar["sha256"] == hashes[FINAL], "Checkpoint sidecar SHA mismatch")
    payload = torch.load(root / FINAL, map_location="cpu", weights_only=False)
    check_checkpoint(payload, plan, "full_576")
    require(payload["binding"] == sidecar["binding"] and payload["epoch"] == 4
            and payload["binding"]["plan_sha256"] == cfg["expected_plan_sha256"]
            and payload["global_step"] == 9290 and payload["metrics"]["chosen"] == "raw"
            and payload["config"] == final_config and payload["image_size"] == cfg["image_size"],
            "Not the original complete V2 final raw checkpoint")
    require(all(torch.isfinite(value).all() for value in payload["model"].values()), "Nonfinite raw model")
    with zipfile.ZipFile(root / "submission/submission.zip") as archive:
        require(archive.namelist() == ["pred_results.csv"] and
                archive.read("pred_results.csv") == (root / "submission/pred_results.csv").read_bytes(),
                "Original ZIP/CSV byte mismatch")
    original_names, original_predictions = read_predictions(root / "submission/pred_results.csv")
    require(original_names == files and np.all((original_predictions >= 0) &
            (original_predictions < manifest["num_classes"])), "Original CSV/order/mapping mismatch")
    return dict(plan=plan, payload=payload, manifest=manifest, files=files,
                original_predictions=original_predictions, source_hashes=hashes,
                receipt_sha256=digest_bytes(receipt_bytes), receipt_bytes=receipt_bytes)


def read_predictions(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.reader(stream))
    require(all(len(row) == 2 and len(row[1].strip()) == 4 and row[1].strip().isdigit()
                for row in rows), "Invalid prediction rows")
    return [row[0] for row in rows], np.array([int(row[1]) for row in rows], dtype=np.int64)


def ensure_idle_cuda():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                             "--format=csv,noheader"], capture_output=True, text=True, check=True)
    require(not result.stdout.strip(), "Other CUDA process exists; do not preempt: " + result.stdout.strip())
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Local BF16 CUDA required")


def load_model(cfg, state, classes):
    recipe = dict(official_checkpoint=cfg["official_checkpoint"], official_sha256=cfg["official_sha256"],
                  num_classes=classes, gradient_checkpointing=False)
    model = V2Classifier(recipe)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.cuda().eval()


@torch.inference_mode()
def collect_probabilities(model, root, files, cfg, output):
    sums = np.zeros((len(files), model.head.out_features), dtype=np.float32)
    started = time.monotonic()
    for view_index, view in enumerate(cfg["views"]):
        data = DataLoader(InferenceDataset(root, files, build_view(view, cfg["image_size"])),
                          batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"],
                          pin_memory=True, persistent_workers=False, prefetch_factor=2)
        visited = np.zeros(len(files), dtype=bool)
        for batch, (images, indices) in enumerate(data):
            with amp():
                logits = model(images.cuda(non_blocking=True))
            probs = logits.float().softmax(1).cpu().numpy()
            require(np.isfinite(probs).all(), "Nonfinite view probabilities")
            ix = indices.numpy()
            require(not visited[ix].any(), "Repeated inference indices")
            sums[ix] += probs
            visited[ix] = True
            if batch % 200 == 0:
                progress = dict(status="inference", view=view, completed_views=view_index,
                                view_rows=int(visited.sum()), rows=len(files), seconds=time.monotonic()-started)
                atomic_json_dump(progress, output / "progress.json")
                print(json.dumps(progress), flush=True)
        require(visited.all(), "Incomplete view coverage")
    means = sums / len(cfg["views"])
    require(np.allclose(means.sum(1), 1, atol=2e-6, rtol=0), "View probability normalization failed")
    return means


def probability_scores(probabilities, floor=1e-30):
    values = np.asarray(probabilities)
    require(values.ndim == 2 and min(values.shape) > 0 and values.shape[1] > 1
            and np.isfinite(values).all() and np.all(values >= 0)
            and np.allclose(values.sum(1), 1, rtol=0, atol=2e-6), "Invalid mean probabilities")
    # Float64 log keeps distinct float32 probabilities distinct at near ties.
    scores = np.log(np.maximum(values.astype(np.float64), floor))
    require(np.array_equal(values.argmax(1), scores.argmax(1)), "Log conversion changed raw argmax")
    return scores


def raw_replay(probabilities, original, cfg):
    raw = probabilities.argmax(1)
    changed = raw != original
    ordered = np.sort(probabilities[changed], axis=1) if changed.any() else None
    margin = float((ordered[:, -1]-ordered[:, -2]).max()) if changed.any() else 0.0
    count = int(changed.sum())
    return dict(raw_remote_disagreement=count, raw_remote_disagreement_margin_max=margin,
                raw_replay_passed=count <= len(raw)*cfg["raw_replay_max_disagreement_fraction"]
                and margin <= cfg["raw_replay_max_probability_margin"])


def source_code_hashes():
    names = ["scripts/run_v2_fixed_prior.py", "scripts/verify_v2_fixed_prior.py",
             "scripts/verify_v1_768_full_delivery.py", "reproducibility/aegis_f1/v2/runtime.py",
             "reproducibility/aegis_f1/v2/model.py", "reproducibility/aegis_f1/v2/training_utils.py",
             "reproducibility/aegis_f1/aegis_clip/model.py",
             "reproducibility/aegis_f1/aegis_clip/v1_test_bias.py",
             "reproducibility/aegis_f1/aegis_clip/prior_alignment.py"]
    return {name: sha256_file(ROOT / name) for name in names}


def verify_test_pixels(cfg, manifest):
    with (Path(cfg["stage_root"]) / "test_manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = Path(manifest["test_root"]) / Path(row["image_path"]).name
        require(sha256_file(path) == row["file_sha256"], "Official test pixels changed: " + path.name)
    return len(rows)


def run(cfg, config_path, output):
    require(not output.exists(), "Output exists; no implicit restart or overwrite")
    inputs = verified_delivery(cfg)
    checked_pixels = verify_test_pixels(cfg, inputs["manifest"])
    ensure_idle_cuda()
    output.mkdir(parents=True)
    started = time.monotonic()
    binding = dict(config_sha256=sha256_file(config_path), source_hashes=inputs["source_hashes"],
                   source_receipt_sha256=inputs["receipt_sha256"], source_code=source_code_hashes(),
                   local_test_pixels_checked=checked_pixels,
                   plan_sha256=cfg["expected_plan_sha256"], experiment_id=cfg["experiment_id"])
    atomic_json_dump(binding, output / "binding.json")
    (output / "source_delivery_receipt.json").write_bytes(inputs["receipt_bytes"])
    manifest = inputs["manifest"]
    model = load_model(cfg, inputs["payload"]["model"], manifest["num_classes"])
    del inputs["payload"]
    probabilities = collect_probabilities(model, Path(manifest["test_root"]), inputs["files"], cfg, output)
    del model
    torch.cuda.empty_cache()
    np.save(output / "mean_probabilities.npy", probabilities)
    replay = raw_replay(probabilities, inputs["original_predictions"], cfg)
    atomic_json_dump(replay, output / "raw_replay.json")
    require(replay["raw_replay_passed"], "Original raw replay outside frozen numerical tolerance; diagnose cache")
    scores = probability_scores(probabilities, cfg["probability_floor"])
    bias = fit_test_uniform_bias(torch.from_numpy(scores).cuda(), iterations=cfg["bias_iterations"]).numpy()
    raw = probabilities.argmax(1)
    calibrated = (scores + bias).argmax(1)
    np.savez(output / "predictions.npz", probabilities=probabilities, scores=scores, bias=bias,
             names=np.asarray(inputs["files"]), raw=raw, calibrated=calibrated)
    for name, predictions in (("submission_raw", raw), ("submission_bias", calibrated)):
        destination = output / name
        write_submission(destination, inputs["files"], predictions, manifest["num_classes"])
        checked = subprocess.run([sys.executable, str(ROOT / "scripts/check_submission.py"),
                    "--test_dir", manifest["test_root"], "--class-mapping",
                    str(Path(cfg["stage_root"]) / "class_to_idx.json"), "--csv",
                    str(destination / "pred_results.csv"), "--zip", str(destination / "submission.zip")],
                    capture_output=True, text=True)
        (destination / "submission_check.log").write_text(checked.stdout + checked.stderr)
        checked.check_returncode()
        atomic_json_dump(dict(binding=binding, checkpoint_sha256=inputs["source_hashes"][FINAL],
            views=cfg["views"], reduction=cfg["reduction"], iterations=cfg["bias_iterations"],
            strength=cfg["bias_strength"], test_statistical_fitting=name == "submission_bias",
            model_parameter_updates=False, test_labels_used=False, weights="full_epoch5_raw",
            official_permission_source=cfg["official_permission_source"], platform_score=None,
            status="unpromoted_candidate_pending_independent_verification"), destination / "manifest.json")
    report = dict(status="packages_checked_pending_independent_verification", binding=binding,
        output=str(output.resolve()), rows=len(raw), classes=manifest["num_classes"],
        predictions_sha256=sha256_file(output / "predictions.npz"),
        **replay,
        bias_changed_predictions=int((raw != calibrated).sum()), zero_probabilities=int((probabilities == 0).sum()),
        bias_min=float(bias.min()), bias_max=float(bias.max()),
        elapsed_seconds=time.monotonic()-started, platform_score=None, training=False,
        test_statistical_fitting=True, model_parameter_updates=False,
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(), batch_size=cfg["batch_size"]))
    atomic_json_dump(report, output / "report.json")
    from verify_v2_fixed_prior import verify
    verification = verify(cfg, config_path, output)
    atomic_json_dump(verification, output / "independent_verification.json")
    report.update(status="completed_verified_delivery")
    atomic_json_dump(report, output / "report.json")
    print(json.dumps(report), flush=True)


def probe(cfg, source, expected_sha, output):
    """Architecture cost probe on 64 train-side images; no candidate or score."""
    require(not output.exists(), "Probe output exists")
    require(sha256_file(source) == expected_sha, "Known S1 probe checkpoint changed")
    state = torch.load(source, map_location="cpu", weights_only=False)
    require(state["binding"]["data_version"] == cfg["data_version"]
            and state["binding"]["stage"] == "s1_384" and state["binding"]["complete"]
            and state["binding"]["official_sha256"] == cfg["official_sha256"]
            and state["global_step"] == 13930 and state["metrics"]["chosen"] == "raw",
            "Probe must use the verified current-stage S1 raw parent")
    manifest, _ = local_data(cfg)
    with (Path(cfg["stage_root"]) / "val_dev.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))[:cfg["probe_rows"]]
    files = [row["image_path"].removeprefix("train/") for row in rows]
    for row, name in zip(rows, files):
        require(sha256_file(Path(manifest["train_root"]) / name) == row["file_sha256"],
                "Probe pixels differ from the current-stage manifest")
    ensure_idle_cuda()
    output.mkdir(parents=True)
    model = load_model(cfg, state["model"], manifest["num_classes"])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    probabilities = collect_probabilities(model, Path(manifest["train_root"]), files, cfg, output)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    probability_scores(probabilities, cfg["probability_floor"])
    report = dict(status="architecture_cost_probe_passed", no_candidate=True, training=False,
                  test_images_used=False, images=len(files), views=cfg["views"], seconds=seconds,
                  rough_full_inference_seconds=seconds / len(files) * manifest["test_samples"],
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  checkpoint_sha256=expected_sha, source=str(source),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  note="S1 weights measure the same V2 architecture only; no full-model validation or score")
    atomic_json_dump(report, output / "cost.json")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--action", choices=("check", "run", "probe"), default="check")
    parser.add_argument("--probe-parent")
    parser.add_argument("--probe-parent-sha256")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.action == "probe":
        require(args.probe_parent and args.probe_parent_sha256, "Explicit verified probe parent required")
        probe(cfg, Path(args.probe_parent), args.probe_parent_sha256, Path(args.output))
    elif args.action == "run":
        run(cfg, args.config, Path(args.output))
    else:
        receipt = Path(cfg["delivery_root"]) / "delivery_receipt.json"
        if not receipt.is_file():
            print(json.dumps(dict(status="waiting_for_local_verified_delivery", path=str(receipt))))
        else:
            inputs = verified_delivery(cfg)
            print(json.dumps(dict(status="verified_ready_for_local_inference", hashes=inputs["source_hashes"])))


if __name__ == "__main__":
    main()
