"""PRELIM75 v8: frozen training-source class-bias calibration for G0 and L1."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
import yaml

from aegis_clip.calibration_binding import protocol_sha256, validate_frozen_prior
from aegis_clip.checkpoint import _atomic_torch_save
from aegis_clip.config import validate_config
from aegis_clip.features import canonical_sample_path
from aegis_clip.localization import fuse_global_multilocal_flip_probabilities
from aegis_clip.prelim75 import (
    OfficialImages,
    gpu_setup,
    load_composite,
    loader,
    repository_root,
    source_manifest,
    supervision,
)
from aegis_clip.prelim75_recovery import _teacher_views, fuse_teacher_probabilities
from aegis_clip.prelim75_trusted_ce import original_supervision_sha256
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.source_bias import (
    SCORE_SEMANTICS,
    build_inference_protocol_descriptor,
    fit_bounded_source_bias,
)


PLAN_ID = "PRELIM75_V8_20260919"
FIXED_REVIEW_COMMIT = "3f83d69f9a4496976e7d50beecd7e0baa0400d4d"
CANDIDATES = ("B0", "B1")
REQUIRED_TRAIN_ROWS = 103218
REQUIRED_TEST_ROWS = 24967
REQUIRED_CLASSES = 500
DATASET_ID = "preliminary_official_train_103218_v8"
INFERENCE_BATCH_SIZE = 64
INFERENCE_PROTOCOL = {
    "input_resize_mode": "clip_center_crop",
    "local_view": "attention_multiscale",
    "local_crop_sizes": [112, 128, 144, 160],
    "local_scale_weights": [0.2, 0.3, 0.4, 0.1],
    "local_top_k": 5,
    "local_weight": 0.4,
    "local_temperature": 1.5,
    "adapt_local_features": True,
    "adapt_part_token_features": True,
    "tta": "horizontal_flip",
    "tta_fusion": "mean_probabilities",
    "tta_temperature": 1.5,
    "tta_view_weight": 0.5,
    "batch_size": INFERENCE_BATCH_SIZE,
    "score_semantics": SCORE_SEMANTICS,
}
FITTING_PROTOCOL = {
    "regularization": 0.01,
    "bound": math.log(4.0),
    "strength": 0.90,
    "chunk_size": 4096,
    "solver": "L-BFGS-B",
    "dtype": "float64",
    "maxiter": 100,
    "maxfun": 300,
    "maxls": 30,
    "maxcor": 10,
    "ftol": 1e-12,
    "gtol": 1e-8,
}
EXPECTED_REFERENCES = {
    "G0_no_prior": 69.2274,
    "L1_no_prior": 69.2794,
    "G0_legacy_test_batch_prior0.9": 72.4677,
    "target": 75.0,
}

_PATH_KEYS = (
    "train_csv",
    "trust",
    "class_mapping",
    "groups",
    "train_root",
    "test_root",
    "legacy_prior_submission",
    "output",
)
_HASHED_PATH_KEYS = (
    "train_csv",
    "trust",
    "class_mapping",
    "groups",
    "legacy_prior_submission",
)


def _candidate_output(plan: dict, name: str) -> Path:
    return Path(plan["output"]) / name


def _official_source_identities(plan: dict, paths: list[str]) -> list[str]:
    train_root = Path(plan["train_root"]).resolve()
    identities = []
    for raw_path in paths:
        relative = Path(raw_path)
        parts = (
            relative.parts[1:]
            if relative.parts and relative.parts[0] == train_root.name
            else relative.parts
        )
        resolved = train_root.joinpath(*parts).resolve()
        if not resolved.is_relative_to(train_root) or not resolved.is_file():
            raise ValueError("Calibration source is not an official training-root sample")
        identities.append(resolved.relative_to(train_root).as_posix())
    if len(identities) != len(set(identities)):
        raise ValueError("Repeated official training-source identity")
    return identities


def load_v8_plan(path: str | Path) -> dict:
    source = Path(path).resolve()
    plan = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("PRELIM75 v8 config root must be a mapping")
    allowed = {
        "plan_id",
        "seed",
        "fixed_review_commit",
        "dataset_id",
        "num_workers",
        "original_supervision_sha256",
        "candidates",
        "inference_protocol",
        "fitting",
        "budget",
        "reference_platform_percent",
        *_PATH_KEYS,
        *(key + "_sha256" for key in _HASHED_PATH_KEYS),
    }
    if set(plan) != allowed:
        raise ValueError(
            "Unknown or incomplete fixed PRELIM75 v8 plan: "
            f"missing={sorted(allowed - set(plan))}, extra={sorted(set(plan) - allowed)}"
        )
    if (
        plan.get("plan_id") != PLAN_ID
        or plan.get("seed") != 42
        or plan.get("fixed_review_commit") != FIXED_REVIEW_COMMIT
        or plan.get("dataset_id") != DATASET_ID
        or plan.get("num_workers") != 4
    ):
        raise ValueError("PRELIM75 v8 fixed plan identity changed")
    if plan.get("inference_protocol") != INFERENCE_PROTOCOL:
        raise ValueError("PRELIM75 v8 inference protocol changed")
    if plan.get("fitting") != FITTING_PROTOCOL:
        raise ValueError("PRELIM75 v8 fitting protocol changed")
    if plan.get("reference_platform_percent") != EXPECTED_REFERENCES:
        raise ValueError("PRELIM75 v8 reference score table changed")
    if plan.get("budget") != {
        "backbone_optimizer_updates": 0,
        "maximum_train_forwards": 2,
        "maximum_cpu_fits": 2,
        "maximum_test_candidates": 2,
        "gpu_action_seconds": 7200,
        "cpu_fit_seconds": 3600,
    }:
        raise ValueError("PRELIM75 v8 budget changed")
    candidates = plan.get("candidates")
    if not isinstance(candidates, dict) or tuple(candidates) != CANDIDATES:
        raise ValueError("PRELIM75 v8 candidates must be ordered B0 then B1")
    expected_candidate_fields = {
        "model_id",
        "checkpoint",
        "checkpoint_sha256",
        "baseline_submission",
        "baseline_submission_sha256",
        "baseline_platform_percent",
    }
    for name in CANDIDATES:
        candidate = candidates[name]
        if set(candidate) != expected_candidate_fields:
            raise ValueError(f"Incomplete PRELIM75 v8 candidate {name}")
        for key in ("checkpoint", "baseline_submission"):
            candidate[key] = str((source.parent / candidate[key]).resolve())
        for key in ("checkpoint", "baseline_submission"):
            if sha256_file(candidate[key]) != candidate[key + "_sha256"]:
                raise ValueError(f"Frozen PRELIM75 v8 {name} {key} hash mismatch")
    for key in _PATH_KEYS:
        plan[key] = str((source.parent / plan[key]).resolve())
    for key in _HASHED_PATH_KEYS:
        if sha256_file(plan[key]) != plan[key + "_sha256"]:
            raise ValueError(f"Frozen PRELIM75 v8 asset hash mismatch: {key}")
    train_root = Path(plan["train_root"]).resolve()
    test_root = Path(plan["test_root"]).resolve()
    if train_root == test_root or train_root.is_relative_to(test_root) or test_root.is_relative_to(train_root):
        raise ValueError("Official train and test roots must be disjoint")
    plan["_config_path"] = str(source)
    plan["_config_sha256"] = sha256_file(source)
    return plan


def _check_checkpoint(plan: dict, name: str) -> dict:
    candidate = plan["candidates"][name]
    checkpoint = torch.load(candidate["checkpoint"], map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    metadata = checkpoint.get("prelim75_training", {})
    if config.get("project", {}).get("experiment_id") != candidate["model_id"]:
        raise ValueError(f"{name} checkpoint experiment id mismatch")
    if metadata.get("original_supervision_sha256") != plan["original_supervision_sha256"]:
        raise ValueError(f"{name} checkpoint original supervision binding mismatch")
    if metadata.get("prior_in_inference") is not False:
        raise ValueError(f"{name} checkpoint unexpectedly contains an inference prior")
    if "local_feature_adapter" not in checkpoint or "part_token_adapter" not in checkpoint:
        raise ValueError(f"{name} checkpoint is missing O3/PTA adapters")
    if name == "B0":
        if metadata.get("plan_id") != "PRELIM75_V6_20260918" or metadata.get("name") != "G0":
            raise ValueError("B0 is not the registered v6 G0 checkpoint")
    else:
        if metadata.get("plan_id") != "PRELIM75_V7_20260918" or metadata.get("name") != "L1":
            raise ValueError("B1 is not the registered v7 L1 checkpoint")
    return checkpoint


def _zip_csv_sha256(path: str | Path) -> str:
    with ZipFile(path) as archive:
        if archive.namelist() != ["pred_results.csv"]:
            raise ValueError(f"Submission structure changed: {path}")
        return hashlib.sha256(archive.read("pred_results.csv")).hexdigest()


def preflight_v8(plan: dict) -> dict:
    root = repository_root()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", plan["fixed_review_commit"], "HEAD"],
        cwd=root,
    ).returncode:
        raise ValueError("Fixed reviewed v8 base is not an ancestor of HEAD")
    data = supervision(plan)
    if len(data["paths"]) != REQUIRED_TRAIN_ROWS or data["targets"].shape != (
        REQUIRED_TRAIN_ROWS,
        REQUIRED_CLASSES,
    ):
        raise ValueError("Official v8 training identity changed")
    if original_supervision_sha256(data) != plan["original_supervision_sha256"]:
        raise ValueError("Original v8 supervision differs from the registered tensors")
    train_root = Path(plan["train_root"]).resolve()
    test_root = Path(plan["test_root"]).resolve()
    for raw_path in data["paths"]:
        relative = Path(raw_path)
        parts = relative.parts[1:] if relative.parts and relative.parts[0] == train_root.name else relative.parts
        resolved = train_root.joinpath(*parts).resolve()
        if not resolved.is_relative_to(train_root) or resolved.is_relative_to(test_root) or not resolved.is_file():
            raise ValueError("Non-training source entered PRELIM75 v8")
    counts = data["n_eff"]
    if counts.shape != (REQUIRED_CLASSES,) or not torch.isfinite(counts).all() or bool((counts <= 0).any()):
        raise ValueError("Every PRELIM75 v8 class must have positive effective supervision")

    checkpoint_metadata = {}
    baseline_csv_sha256 = {}
    for name in CANDIDATES:
        checkpoint = _check_checkpoint(plan, name)
        baseline_csv_sha256[name] = _zip_csv_sha256(
            plan["candidates"][name]["baseline_submission"]
        )
        checkpoint_metadata[name] = {
            "model_id": checkpoint["config"]["project"]["experiment_id"],
            "checkpoint_sha256": plan["candidates"][name]["checkpoint_sha256"],
            "baseline_submission_sha256": plan["candidates"][name]["baseline_submission_sha256"],
            "baseline_zip_csv_sha256": baseline_csv_sha256[name],
            "upstream_provenance_complete": False,
        }
    _zip_csv_sha256(plan["legacy_prior_submission"])
    return {
        "status": "passed",
        "plan_id": PLAN_ID,
        "config_sha256": plan["_config_sha256"],
        "fixed_review_commit": plan["fixed_review_commit"],
        "dataset_id": plan["dataset_id"],
        "official_train_rows": len(data["paths"]),
        "classes": REQUIRED_CLASSES,
        "positive_weight_rows": int((data["weights"] > 0).sum()),
        "effective_class_support_min": float(counts.min()),
        "effective_class_support_max": float(counts.max()),
        "original_supervision_sha256": plan["original_supervision_sha256"],
        "train_csv_sha256": plan["train_csv_sha256"],
        "trust_sha256": plan["trust_sha256"],
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "candidates": checkpoint_metadata,
        "inference_protocol": copy.deepcopy(INFERENCE_PROTOCOL),
        "fitting": copy.deepcopy(FITTING_PROTOCOL),
        "test_data_used_for_fit": False,
        "test_batch_prior_read": False,
        "backbone_optimizer_created": False,
        "backbone_optimizer_updates": 0,
        "upstream_provenance_complete": False,
        "platform_upload_authorized": False,
    }


def _runtime_config(plan: dict, name: str, checkpoint: dict) -> dict:
    config = copy.deepcopy(checkpoint["config"])
    config["project"]["dataset_id"] = plan["dataset_id"]
    for key in ("train_csv", "class_mapping", "train_root", "test_root"):
        config["data"][key] = plan[key]
    config["output"]["root"] = str(_candidate_output(plan, name))
    config["data"]["expected_official_train_samples"] = REQUIRED_TRAIN_ROWS
    config["data"]["expected_test_samples"] = REQUIRED_TEST_ROWS
    config["data"]["external_data"] = False
    config["data"]["test_usage"] = "inference_only"
    config["data"]["train_augmentation"] = "clip_center_crop"
    config["evaluation"]["inference_batch_size"] = int(
        checkpoint["config"]["evaluation"].get("inference_batch_size", 128)
    )
    validate_config(config)
    return config


def prepare_v8_source(plan: dict) -> dict:
    output = Path(plan["output"])
    source_dir = output / "source"
    source_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data = supervision(plan)
    train_root = Path(plan["train_root"]).resolve()
    identities = _official_source_identities(plan, data["paths"])
    manifest_path = source_dir / "fit_sample_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_path", "label"))
        writer.writerows(zip(identities, data["labels"].tolist()))

    content_hashes = set()
    for index, identity in enumerate(identities):
        sample = train_root.joinpath(identity).resolve()
        if not sample.is_relative_to(train_root) or not sample.is_file():
            raise ValueError("Source manifest contains a non-official training image")
        content_hashes.add(sha256_file(sample))
        if index % 4096 == 0:
            atomic_json_dump(
                {
                    "status": "hashing_official_training_source",
                    "completed_samples": index,
                    "total_samples": len(data["paths"]),
                    "elapsed_seconds": time.monotonic() - started,
                },
                source_dir / "status.json",
            )
    group_set_path = source_dir / "fit_group_set.json"
    atomic_json_dump(sorted(content_hashes), group_set_path)
    for name in CANDIDATES:
        candidate_dir = _candidate_output(plan, name)
        candidate_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = _check_checkpoint(plan, name)
        runtime_config = _runtime_config(plan, name, checkpoint)
        runtime_path = candidate_dir / "runtime_config.yaml"
        runtime_path.write_text(
            yaml.safe_dump(runtime_config, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
    result = {
        "status": "checks_passed",
        "fit_scope": "training_overlap_calibration",
        "source_authenticity_verified": True,
        "authorizes_calibration": True,
        "upstream_provenance_complete": False,
        "rows": len(data["paths"]),
        "unique_paths": len(set(data["paths"])),
        "unique_content_groups": len(content_hashes),
        "fit_sample_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "fit_group_set": {
            "path": str(group_set_path),
            "sha256": sha256_file(group_set_path),
        },
        "original_train_csv": {
            "path": plan["train_csv"],
            "sha256": plan["train_csv_sha256"],
        },
        "trust_bundle": {
            "path": plan["trust"],
            "sha256": plan["trust_sha256"],
        },
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "original_supervision_sha256": plan["original_supervision_sha256"],
        "test_data_used": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json_dump(result, source_dir / "source_identity_audit.json")
    atomic_json_dump(result, source_dir / "status.json")
    return result


def _inference_args(plan: dict, name: str, prior_config: str | None) -> argparse.Namespace:
    candidate_dir = _candidate_output(plan, name)
    return argparse.Namespace(
        checkpoint=plan["candidates"][name]["checkpoint"],
        config=str(candidate_dir / "runtime_config.yaml"),
        output_dir=str(candidate_dir / "submission"),
        tta="horizontal_flip",
        tta_fusion="mean_probabilities",
        tta_temperature=1.5,
        tta_view_weight=0.5,
        acknowledge_tta_risk=True,
        local_view="attention_multiscale",
        local_crop_size=160,
        local_crop_sizes="112,128,144,160",
        local_scale_weights="0.20,0.30,0.40,0.10",
        local_top_k=5,
        local_weight=0.4,
        local_temperature=1.5,
        adapt_local_features=True,
        adapt_part_token_features=True,
        acknowledge_local_view_risk=True,
        prior_alignment_strength=0.0,
        prior_alignment_iterations=50,
        prior_config=prior_config,
        acknowledge_balanced_test_prior=False,
        overwrite=False,
        device="cuda",
        input_resize_mode="clip_center_crop",
        batch_size=INFERENCE_BATCH_SIZE,
        dump_logits=None,
        dump_branch_logits=None,
    )


@torch.no_grad()
def cache_v8_scores(plan: dict, name: str) -> dict:
    if name not in CANDIDATES:
        raise ValueError("Unknown PRELIM75 v8 candidate")
    candidate_dir = _candidate_output(plan, name)
    cache_dir = candidate_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = gpu_setup()
    data = supervision(plan)
    identities = _official_source_identities(plan, data["paths"])
    checkpoint_path = plan["candidates"][name]["checkpoint"]
    model, preprocess, checkpoint, o3, pta = load_composite(checkpoint_path, device)
    model.float().eval().requires_grad_(False)
    o3.float().eval().requires_grad_(False)
    pta.float().eval().requires_grad_(False)
    runtime_config = yaml.safe_load(
        (candidate_dir / "runtime_config.yaml").read_text(encoding="utf-8")
    )
    descriptor = build_inference_protocol_descriptor(
        _inference_args(plan, name, None), checkpoint, runtime_config, preprocess, device
    )
    descriptor_sha256 = protocol_sha256(descriptor)
    dataset = OfficialImages(data["paths"], plan["train_root"], preprocess)
    stream = loader(dataset, INFERENCE_BATCH_SIZE, plan)
    scores = torch.empty((len(dataset), REQUIRED_CLASSES), dtype=torch.float32)
    completed = 0
    logsumexp_max_abs = 0.0
    first_batch_alignment = None
    for batch_index, batch in enumerate(stream):
        images = batch["images"].to(device, non_blocking=True)
        indices = batch["index"].long()
        expected_paths = [data["paths"][index] for index in indices.tolist()]
        if list(batch["path"]) != expected_paths:
            raise RuntimeError("Training score cache row order changed")
        global_logits, local_logits, _ = _teacher_views(model, o3, pta, images)
        fused = fuse_global_multilocal_flip_probabilities(
            global_logits[:, 0],
            list(local_logits[:, 0].unbind(dim=1)),
            global_logits[:, 1],
            list(local_logits[:, 1].unbind(dim=1)),
            local_weight=0.4,
            flip_weight=0.5,
            temperature=1.5,
            global_temperature=1.5,
            local_temperature=1.5,
            local_scale_weights=(0.2, 0.3, 0.4, 0.1),
        )
        if fused.dtype != torch.float32 or not torch.isfinite(fused).all():
            raise RuntimeError("Training fused score cache is not finite FP32")
        current_logsumexp = torch.logsumexp(fused, dim=1)
        logsumexp_max_abs = max(logsumexp_max_abs, float(current_logsumexp.abs().max()))
        if float(current_logsumexp.abs().max()) > 2e-5:
            raise RuntimeError("Training fused scores are not normalized log probabilities")
        if batch_index == 0:
            established = fuse_teacher_probabilities(global_logits, local_logits)[
                "probabilities"
            ].clamp_min(torch.finfo(torch.float32).tiny).log()
            torch.testing.assert_close(fused, established, atol=1e-5, rtol=1e-5)
            if not torch.equal(fused.argmax(1), established.argmax(1)):
                raise RuntimeError("Existing no-prior wrapper argmax alignment failed")
            first_batch_alignment = {
                "rows": len(fused),
                "max_abs_score_difference": float((fused - established).abs().max()),
                "argmax_disagreements": int((fused.argmax(1) != established.argmax(1)).sum()),
                "atol": 1e-5,
                "rtol": 1e-5,
            }
        scores[indices] = fused.cpu()
        completed += len(indices)
        if batch_index % 20 == 0 or completed == len(dataset):
            atomic_json_dump(
                {
                    "status": "running",
                    "candidate": name,
                    "completed_samples": completed,
                    "total_samples": len(dataset),
                    "elapsed_seconds": time.monotonic() - started,
                },
                cache_dir / "status.json",
            )
    if completed != len(dataset) or not torch.isfinite(scores).all():
        raise RuntimeError("Incomplete or non-finite PRELIM75 v8 training score cache")
    payload = {
        "logits": scores,
        "paths": identities,
        "fit_scope": "training_overlap_calibration",
        "score_semantics": SCORE_SEMANTICS,
        "stage": "preliminary",
        "dataset_id": plan["dataset_id"],
        "fit_checkpoint_sha256": plan["candidates"][name]["checkpoint_sha256"],
        "inference_protocol_sha256": descriptor_sha256,
        "inference_protocol": descriptor,
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "original_supervision_sha256": plan["original_supervision_sha256"],
        "test_data_used": False,
    }
    score_path = cache_dir / "validation_logits.pt"
    _atomic_torch_save(payload, score_path)
    prediction_counts = scores.argmax(1).bincount(minlength=REQUIRED_CLASSES)
    result = {
        "status": "complete_checks_passed",
        "candidate": name,
        "rows": scores.shape[0],
        "classes": scores.shape[1],
        "dtype": str(scores.dtype),
        "tensor_element_bytes": scores.numel() * scores.element_size(),
        "expected_tensor_element_bytes": 206436000,
        "score_semantics": SCORE_SEMANTICS,
        "logsumexp_max_abs": logsumexp_max_abs,
        "first_batch_no_prior_wrapper_alignment": first_batch_alignment,
        "prediction_count_min": int(prediction_counts.min()),
        "prediction_count_max": int(prediction_counts.max()),
        "prediction_empty_classes": int((prediction_counts == 0).sum()),
        "checkpoint_sha256": plan["candidates"][name]["checkpoint_sha256"],
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "original_supervision_sha256": plan["original_supervision_sha256"],
        "inference_protocol_sha256": descriptor_sha256,
        "validation_logits": str(score_path),
        "validation_logits_sha256": sha256_file(score_path),
        "test_data_used": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
    }
    atomic_json_dump(result, cache_dir / "manifest.json")
    atomic_json_dump(result, cache_dir / "status.json")
    del model, o3, pta, scores
    torch.cuda.empty_cache()
    return result


def fit_v8_bias(plan: dict, name: str) -> dict:
    if name not in CANDIDATES:
        raise ValueError("Unknown PRELIM75 v8 candidate")
    candidate_dir = _candidate_output(plan, name)
    fit_dir = candidate_dir / "fit"
    fit_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data = supervision(plan)
    identities = _official_source_identities(plan, data["paths"])
    cache_manifest = json.loads(
        (candidate_dir / "cache/manifest.json").read_text(encoding="utf-8")
    )
    score_path = candidate_dir / "cache/validation_logits.pt"
    if (
        cache_manifest.get("status") != "complete_checks_passed"
        or sha256_file(score_path) != cache_manifest.get("validation_logits_sha256")
    ):
        raise ValueError("Incomplete or changed v8 frozen score cache")
    payload = torch.load(score_path, map_location="cpu", weights_only=True)
    if (
        payload.get("paths") != identities
        or payload.get("fit_scope") != "training_overlap_calibration"
        or payload.get("score_semantics") != SCORE_SEMANTICS
        or payload.get("fit_checkpoint_sha256")
        != plan["candidates"][name]["checkpoint_sha256"]
        or payload.get("original_supervision_sha256")
        != plan["original_supervision_sha256"]
    ):
        raise ValueError("Frozen score producer binding mismatch")
    bias, fit_report = fit_bounded_source_bias(
        payload["logits"],
        data["weights"],
        data["targets"],
        regularization=FITTING_PROTOCOL["regularization"],
        bound=FITTING_PROTOCOL["bound"],
        chunk_size=FITTING_PROTOCOL["chunk_size"],
        maxiter=FITTING_PROTOCOL["maxiter"],
        maxfun=FITTING_PROTOCOL["maxfun"],
        maxls=FITTING_PROTOCOL["maxls"],
        maxcor=FITTING_PROTOCOL["maxcor"],
        ftol=FITTING_PROTOCOL["ftol"],
        gtol=FITTING_PROTOCOL["gtol"],
    )
    source_identity = json.loads(
        (Path(plan["output"]) / "source/source_identity_audit.json").read_text(
            encoding="utf-8"
        )
    )
    source_audit = {
        "status": "checks_passed",
        "plan_id": PLAN_ID,
        "candidate": name,
        "stage": "preliminary",
        "dataset_id": plan["dataset_id"],
        "fit_scope": "training_overlap_calibration",
        "fit_checkpoint_sha256": plan["candidates"][name]["checkpoint_sha256"],
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "inference_protocol_sha256": payload["inference_protocol_sha256"],
        "original_supervision_sha256": plan["original_supervision_sha256"],
        "score_semantics": SCORE_SEMANTICS,
        "source_authenticity_verified": source_identity["source_authenticity_verified"],
        "authorizes_calibration": source_identity["authorizes_calibration"],
        "upstream_provenance_complete": False,
        "test_data_used": False,
        "fit_sample_manifest": source_identity["fit_sample_manifest"],
        "fit_group_set": source_identity["fit_group_set"],
        "validation_logits": {
            "path": str(score_path),
            "sha256": cache_manifest["validation_logits_sha256"],
        },
        "original_train_csv": source_identity["original_train_csv"],
        "trust_bundle": source_identity["trust_bundle"],
        "producer_checks": {
            "checkpoint_bound": True,
            "protocol_bound": True,
            "path_order_exact": True,
            "finite_fp32_scores": True,
            "log_probability_rows_checked": True,
            "test_data_used": False,
        },
    }
    source_audit_path = candidate_dir / "source_audit.json"
    atomic_json_dump(source_audit, source_audit_path)
    record = {
        "schema_version": 2,
        "plan_id": PLAN_ID,
        "candidate": name,
        "stage": "preliminary",
        "dataset_id": plan["dataset_id"],
        "fit_scope": "training_overlap_calibration",
        "target_checkpoint_sha256": plan["candidates"][name]["checkpoint_sha256"],
        "fit_checkpoint_sha256": plan["candidates"][name]["checkpoint_sha256"],
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "num_classes": REQUIRED_CLASSES,
        "inference_protocol_sha256": payload["inference_protocol_sha256"],
        "original_supervision_sha256": plan["original_supervision_sha256"],
        "score_semantics": SCORE_SEMANTICS,
        "bias": bias.tolist(),
        "strength": FITTING_PROTOCOL["strength"],
        "strength_source": "fixed PRELIM75 v8 design inherited from recorded 0.90 application strength; vector not inherited",
        "calibration_design_record": {
            "plan_id": PLAN_ID,
            "config_sha256": plan["_config_sha256"],
            "method": "bounded_class_balanced_training_source_bias",
        },
        "test_data_used": False,
        "fit_sample_manifest_sha256": source_audit["fit_sample_manifest"]["sha256"],
        "fit_group_set_sha256": source_audit["fit_group_set"]["sha256"],
        "validation_logits_sha256": source_audit["validation_logits"]["sha256"],
        "original_train_csv_sha256": source_audit["original_train_csv"]["sha256"],
        "trust_bundle_sha256": source_audit["trust_bundle"]["sha256"],
        "source_audit": {
            "path": str(source_audit_path),
            "sha256": sha256_file(source_audit_path),
        },
        "upstream_provenance_complete": False,
        "fit_report": fit_report,
    }
    record_path = fit_dir / "frozen_bias.json"
    atomic_json_dump(record, record_path)
    context = {
        "stage": "preliminary",
        "dataset_id": plan["dataset_id"],
        "train_root": plan["train_root"],
        "target_checkpoint_sha256": plan["candidates"][name]["checkpoint_sha256"],
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "num_classes": REQUIRED_CLASSES,
        "inference_protocol_sha256": payload["inference_protocol_sha256"],
        "original_supervision_sha256": plan["original_supervision_sha256"],
    }
    validated_bias, validated_strength = validate_frozen_prior(
        record, context, base_dir=record_path.parent
    )
    if not torch.equal(validated_bias, bias) or validated_strength != FITTING_PROTOCOL["strength"]:
        raise RuntimeError("Frozen source-bias binding validation changed the fitted values")
    result = {
        "status": "complete_checks_passed",
        "candidate": name,
        "fit_report": fit_report,
        "bias_sha256": hashlib.sha256(bias.numpy().tobytes()).hexdigest(),
        "bias_min": float(bias.min()),
        "bias_max": float(bias.max()),
        "strength": FITTING_PROTOCOL["strength"],
        "record_path": str(record_path),
        "record_sha256": sha256_file(record_path),
        "source_audit_sha256": sha256_file(source_audit_path),
        "binding_validation_passed": True,
        "test_data_used": False,
        "upstream_provenance_complete": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json_dump(result, fit_dir / "result.json")
    atomic_json_dump(result, fit_dir / "status.json")
    return result


def _read_submission(path: Path) -> list[tuple[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 2:
                raise ValueError(f"Malformed submission row in {path}")
            rows.append((row[0].strip(), row[1].strip()))
    return rows


def deliver_v8(plan: dict, name: str) -> dict:
    if name not in CANDIDATES:
        raise ValueError("Unknown PRELIM75 v8 candidate")
    candidate_dir = _candidate_output(plan, name)
    fit_result = json.loads(
        (candidate_dir / "fit/result.json").read_text(encoding="utf-8")
    )
    if fit_result.get("status") != "complete_checks_passed":
        raise ValueError("PRELIM75 v8 candidate did not pass fixed fitting checks")
    prior = candidate_dir / "fit/frozen_bias.json"
    output = candidate_dir / "submission"
    if output.exists():
        raise FileExistsError("Do not overwrite an existing PRELIM75 v8 submission")
    args = _inference_args(plan, name, str(prior))
    command = [
        sys.executable,
        "-u",
        "-m",
        "aegis_clip.cli.infer",
        "--checkpoint",
        args.checkpoint,
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
        "--local-view",
        args.local_view,
        "--local-crop-sizes",
        args.local_crop_sizes,
        "--local-scale-weights",
        args.local_scale_weights,
        "--local-top-k",
        str(args.local_top_k),
        "--local-weight",
        str(args.local_weight),
        "--local-temperature",
        str(args.local_temperature),
        "--adapt-local-features",
        "--adapt-part-token-features",
        "--tta",
        args.tta,
        "--tta-fusion",
        args.tta_fusion,
        "--tta-temperature",
        str(args.tta_temperature),
        "--tta-view-weight",
        str(args.tta_view_weight),
        "--acknowledge-local-view-risk",
        "--acknowledge-tta-risk",
        "--batch-size",
        str(args.batch_size),
        "--prior-config",
        str(prior),
    ]
    with (candidate_dir / "inference.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    rows = _read_submission(output / "pred_results.csv")
    (output / "pred_results.csv").write_text(
        "".join(f"{image_name}, {label}\n" for image_name, label in rows),
        encoding="utf-8",
    )
    with ZipFile(output / "submission.zip", "w", ZIP_DEFLATED) as archive:
        archive.write(output / "pred_results.csv", arcname="pred_results.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "prediction_csv_sha256": sha256_file(output / "pred_results.csv"),
            "submission_zip_sha256": sha256_file(output / "submission.zip"),
            "csv_delimiter": "comma_space",
            "prelim75_v8_candidate": name,
        }
    )
    atomic_json_dump(manifest, output / "manifest.json")
    check_command = [
        sys.executable,
        str(repository_root() / "scripts/check_submission.py"),
        "--test_dir",
        plan["test_root"],
        "--num-classes",
        str(REQUIRED_CLASSES),
        "--csv",
        str(output / "pred_results.csv"),
        "--zip",
        str(output / "submission.zip"),
    ]
    with (candidate_dir / "submission_validation.log").open("w") as log:
        subprocess.run(check_command, check=True, stdout=log, stderr=subprocess.STDOUT)
    raw = (output / "pred_results.csv").read_bytes()
    with ZipFile(output / "submission.zip") as archive:
        if archive.namelist() != ["pred_results.csv"] or archive.read("pred_results.csv") != raw:
            raise ValueError("PRELIM75 v8 ZIP/CSV byte identity failed")
    baseline_path = Path(plan["candidates"][name]["baseline_submission"])
    with ZipFile(baseline_path) as archive:
        baseline_temp = candidate_dir / "baseline_pred_results.csv"
        baseline_temp.write_bytes(archive.read("pred_results.csv"))
    baseline_rows = _read_submission(baseline_temp)
    baseline_temp.unlink()
    if [row[0] for row in rows] != [row[0] for row in baseline_rows]:
        raise ValueError("Calibrated and baseline submission row orders differ")
    changed = sum(left[1] != right[1] for left, right in zip(rows, baseline_rows))
    if len(rows) != REQUIRED_TEST_ROWS or len({row[0] for row in rows}) != REQUIRED_TEST_ROWS:
        raise ValueError("Official PRELIM75 v8 test coverage mismatch")
    report = {
        "status": "submission_ready_pending_platform",
        "candidate": name,
        "checkpoint_path": plan["candidates"][name]["checkpoint"],
        "checkpoint_sha256": plan["candidates"][name]["checkpoint_sha256"],
        "baseline_platform_percent": plan["candidates"][name]["baseline_platform_percent"],
        "baseline_submission_sha256": plan["candidates"][name]["baseline_submission_sha256"],
        "calibration_record_sha256": sha256_file(prior),
        "csv_sha256": sha256_file(output / "pred_results.csv"),
        "zip_sha256": sha256_file(output / "submission.zip"),
        "inference_manifest_sha256": sha256_file(output / "manifest.json"),
        "inference_command": command,
        "validation_command": check_command,
        "validation_exit_code": 0,
        "rows": REQUIRED_TEST_ROWS,
        "unique_prediction_classes": len({row[1] for row in rows}),
        "changed_predictions_vs_no_prior": changed,
        "prediction_coverage_forced_to_500": False,
        "zip_internal_csv_byte_equal": True,
        "single_checkpoint": True,
        "source_bound_frozen_bias": True,
        "strength": FITTING_PROTOCOL["strength"],
        "test_data_used_for_fit": False,
        "test_batch_statistics_used": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "online_accuracy": None,
        "online_exact_correct": None,
        "actual_platform_upload_time": None,
    }
    atomic_json_dump(report, candidate_dir / "candidate_report.json")
    atomic_json_dump(report, candidate_dir / "status.json")
    return report
