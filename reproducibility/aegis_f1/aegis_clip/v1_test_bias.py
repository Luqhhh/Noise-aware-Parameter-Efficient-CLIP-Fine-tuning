"""Explicit research path for user-requested transductive test-marginal bias.

The ordinary val_dev calibration validator remains unchanged. This module
records test statistical fitting and retains raw predictions. The user relayed
official confirmation of balanced-test-prior correction on 2026-10-01.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from aegis_clip.runtime import atomic_json_dump, sha256_file


@torch.no_grad()
def fit_test_uniform_bias(logits, *, iterations=200):
    """Match mean softmax probabilities to 1/C, without accessing labels.

    This balances soft probability mass; argmax counts need not be equal.
    Centering the bias removes its unidentifiable additive constant.
    """
    if (logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2
            or iterations < 1 or not torch.isfinite(logits).all()):
        raise ValueError("Invalid test logit matrix or iteration count")
    from aegis_clip.prior_alignment import fit_prior_bias
    bias, _ = fit_prior_bias(logits, max_iterations=iterations, damping=1.0,
                             fixed_iterations=True)
    if not torch.isfinite(bias).all():
        raise ValueError("Nonfinite test-uniform bias")
    return bias.cpu()


@torch.no_grad()
def distribution_report(logits):
    logits = logits.float()
    counts = torch.bincount(logits.argmax(1), minlength=logits.shape[1])
    mass = logits.softmax(1).mean(0)
    relative = mass * logits.shape[1] - 1
    return dict(argmax_counts=counts.tolist(), argmax_min=int(counts.min()),
        argmax_max=int(counts.max()), argmax_std=float(counts.float().std(unbiased=False)),
        soft_mass=mass.tolist(), soft_uniform_max_relative_error=float(relative.abs().max()),
        soft_uniform_rms_relative_error=float(relative.square().mean().sqrt()))


def infer_test_uniform(context, checkpoint, device):
    from aegis_clip.submission import create_submission
    from aegis_clip.v1_pipeline import (
        ROOT, collect_logits, image_transform, load_artifact, loader_for,
        load_selected, read_rows, save_artifact,
    )
    decoder = context.config["decode"]
    if (decoder["bias_source"] != "test_uniform_experimental"
            or decoder.get("experimental_test_bias") is not True
            or decoder.get("logit_reduction") != "sum"):
        raise ValueError("Test-marginal inference requires the explicit research recipe")
    output = Path(context.config["output"]["root"])
    base = Path(context.reference["data"]["dataset_manifest"]).parent
    rows = read_rows(base / "test_manifest.csv")
    names = [Path(r["image_path"]).name for r in rows]
    if len(names) != context.manifest["test_samples"] or len(names) != len(set(names)):
        raise ValueError("Test manifest count or unique filenames changed")
    checkpoint_hash = sha256_file(checkpoint)
    cached_path = output / "test_logits.pt"
    started = time.monotonic()
    if cached_path.exists():
        cache = load_artifact(cached_path, context.binding)
        if (cache["checkpoint_sha256"] != checkpoint_hash or cache["decoder"] != decoder
                or cache["names"] != names or cache["class_names"] != context.classes):
            raise ValueError("Test logits are not bound to the selected model and test order")
        logits, selected_policy = cache["logits"], cache["selected_policy"]
    else:
        model, payload = load_selected(context, checkpoint, device)
        logits = torch.zeros((len(rows), len(context.classes)))
        for scale in decoder["scales"]:
            loader = loader_for(context, rows,
                image_transform(context.config["model"]["image_size"], scale=scale),
                root=context.test_root)
            # collect_logits averages the flip pair; undo that averaging to
            # reproduce aic_new's six-view SUM before its bias fitting.
            result = collect_logits(model, loader, device, len(context.classes), flip=decoder["flip"])
            logits += result * (2 if decoder["flip"] else 1)
            progress = dict(status="test_inference", completed_scale=scale,
                scales=decoder["scales"], samples=len(rows), seconds=time.monotonic()-started)
            atomic_json_dump(progress, output / "inference_progress.json")
            print(json.dumps(progress), flush=True)
        selected_policy = payload["selected_policy"]
        save_artifact(cached_path, dict(logits=logits, names=names, class_names=context.classes,
            decoder=decoder, checkpoint_sha256=checkpoint_hash, selected_policy=selected_policy), context.binding)
        del model
    if tuple(logits.shape) != (len(rows), len(context.classes)) or not torch.isfinite(logits).all():
        raise ValueError("Invalid bound test logit matrix")
    calibration_path = output / "test_calibration.pt"
    if calibration_path.exists():
        fitted = load_artifact(calibration_path, context.binding)
        if (fitted["checkpoint_sha256"] != checkpoint_hash or fitted["decoder"] != decoder
                or fitted["test_logits_sha256"] != sha256_file(cached_path)
                or fitted["source"] != "unlabelled_current_stage_test_logits"
                or fitted["test_data_used"] is not True
                or tuple(fitted["bias"].shape) != (len(context.classes),)
                or not torch.isfinite(fitted["bias"]).all()):
            raise ValueError("Test calibration is not bound to these logits")
        bias = fitted["bias"]
    else:
        bias = fit_test_uniform_bias(logits.to(device), iterations=decoder["bias_iterations"])
        save_artifact(calibration_path, dict(bias=bias, checkpoint_sha256=checkpoint_hash,
            decoder=decoder, test_logits_sha256=sha256_file(cached_path),
            source="unlabelled_current_stage_test_logits", test_data_used=True,
            samples=len(rows), target="uniform_soft_probability_mass",
            backbone_or_head_updates=False, official_permission_confirmed=True,
            permission_source="user-relayed official confirmation 2026-10-01"), context.binding)
    calibrated = logits + decoder["bias_strength"] * bias
    raw_predictions, fitted_predictions = logits.argmax(1), calibrated.argmax(1)
    report = dict(experiment_id=context.config["project"]["experiment_id"],
        samples=len(rows), classes=len(context.classes), logit_reduction="sum",
        views=len(decoder["scales"])*(2 if decoder["flip"] else 1),
        iterations=decoder["bias_iterations"], strength=decoder["bias_strength"],
        raw=distribution_report(logits), calibrated=distribution_report(calibrated),
        predictions_changed=int((raw_predictions != fitted_predictions).sum()),
        bias_min=float(bias.min()), bias_max=float(bias.max()),
        test_statistical_fitting=True, test_labels_used=False,
        model_parameter_updates=False, independent_test_score=False, platform_score=None,
        official_permission_confirmed=True, permission_source="user-relayed official confirmation 2026-10-01",
        test_logits_sha256=sha256_file(cached_path),
        calibration_sha256=sha256_file(calibration_path), checkpoint_sha256=checkpoint_hash,
        elapsed_seconds=time.monotonic()-started)
    atomic_json_dump(report, output / "test_bias_report.json")
    for directory, predictions, usage in (
        ("submission_raw", raw_predictions, "inference_only"),
        ("submission", fitted_predictions, "unlabelled_test_marginal_bias_fitting"),
    ):
        destination = output / directory
        create_submission([(name, context.classes[index]) for name, index in zip(names, predictions.tolist())],
            names, destination, checkpoint, inference_mode="v1_fixed_multiscale_flip_"+directory,
            tta_risk_acknowledged=True, valid_labels=set(context.classes), space_after_comma=True,
            extra_manifest=dict(binding=context.binding, decoder=decoder, selected_policy=selected_policy,
                status="unpromoted_research_candidate", platform_score=None, test_usage=usage,
                test_statistical_fitting=directory=="submission", official_permission_confirmed=True,
                permission_source="user-relayed official confirmation 2026-10-01",
                raw_test_logits_sha256=sha256_file(cached_path),
                calibration_sha256=sha256_file(calibration_path) if directory=="submission" else None))
        checked = subprocess.run([sys.executable, str(ROOT / "scripts/check_submission.py"),
            "--test_dir", str(context.test_root), "--class-mapping", context.reference["data"]["class_mapping"],
            "--csv", str(destination / "pred_results.csv"), "--zip", str(destination / "submission.zip")],
            capture_output=True, text=True)
        (destination / "submission_check.log").write_text(checked.stdout + checked.stderr)
        checked.check_returncode()
    return output / "submission"
