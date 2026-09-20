"""Fixed PRELIM75 v9 source-recrop experiment and fail-closed audits."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import torch
import yaml
import torchvision
from PIL import Image, ImageFile, __version__ as PILLOW_VERSION
from torch.utils.data import DataLoader, Dataset

from aegis_clip.data import load_class_mapping, resolve_image_path
from aegis_clip.features import canonical_sample_path
from aegis_clip.local_inference import adapted_dual_local_view_logits
from aegis_clip.localization import (
    extract_attention_crops,
    forward_features_with_last_block_attention,
    forward_with_last_block_attention,
    fuse_global_multilocal_flip_probabilities,
)
from aegis_clip.prelim75 import gpu_setup, load_composite, repository_root
from aegis_clip.runtime import atomic_json_dump, seed_worker, sha256_file
from aegis_clip.source_recrop import (
    D0_MINIMUM_SUPPORTED,
    D0_SAMPLE_SIZE,
    INPUT_SIZE,
    PLAN_ID,
    SOURCE_GAIN_THRESHOLD,
    PreparedRGB,
    ResizeCenterTrace,
    d0_gate,
    native_tail,
    prepare_rgb,
    recrop_local,
    select_group_representatives,
    stable_path_list_sha256,
    trace_from_dimensions,
    validate_native_preprocess,
)


FIXED_REVIEW_COMMIT = "cbea37b03b75b91436b536cacd646f0e53a792e2"
CANDIDATES = ("R0", "R1")
REQUIRED_TRAIN_ROWS = 103218
REQUIRED_DIAGNOSTIC_ROWS = 10316
REQUIRED_TEST_ROWS = 24967
REQUIRED_CLASSES = 500
CHECKPOINT_SHA256 = "cd485c7beed2781ac13d08fbd33d63b8f1c46971be008966f7280c67cec1fcee"
FALLBACK_ZIP_SHA256 = "55d25f1fea9c57ba3431164d53d21f34d20baa2a49c805ad87cca188b5634c5a"
FALLBACK_CSV_SHA256 = "5277b2c33f63b951ea2236ae2e50b8c574ef9f8d2f989e8239fcce7b065c0171"
CLASS_MAPPING_SHA256 = "3edfd9e48e2f171b5d452577c52e0d189ae99dbfed3759d3e4f164fc090a6647"
TRAIN_CSV_SHA256 = "7643e120589e69d6bdc0c54abc605a7271d78c41aed1a638534ea43e4c0c4a90"
VAL_CSV_SHA256 = "54a790b35f836cfba4c19cbb5fe38c4b1b37aab62cc9d477f9285496b2d5568e"
GROUPS_SHA256 = "41e2668e0fa5e10291051c14a9c75fc96096a69ae7d36d84d0e774270a99bb87"
LOCAL_SCALES = (112, 128, 144, 160)
LOCAL_SCALE_WEIGHTS = (0.2, 0.3, 0.4, 0.1)
INFERENCE_BATCH_SIZE = 64
INFERENCE_PROTOCOL = {
    "input_size": 224,
    "local_scales": list(LOCAL_SCALES),
    "local_scale_weights": list(LOCAL_SCALE_WEIGHTS),
    "attention_top_k": 5,
    "global_weight": 0.6,
    "local_weight": 0.4,
    "flip_weight": 0.5,
    "global_temperature": 1.5,
    "local_temperature": 1.5,
    "source_gain_threshold": 1.5,
    "local_kernel": "Pillow_BILINEAR_float_box_reducing_gap_None",
    "batch_size": 64,
    "amp": False,
    "prior": False,
}
EXPECTED_REFERENCES = {
    "L1_no_prior": 69.2794,
    "G0_legacy_test_batch_prior0.9": 72.4677,
    "target": 75.0,
}
ImageFile.LOAD_TRUNCATED_IMAGES = True

_PATH_KEYS = (
    "checkpoint",
    "fallback_submission",
    "fallback_csv",
    "fallback_manifest",
    "train_csv",
    "val_csv",
    "groups",
    "class_mapping",
    "train_root",
    "test_root",
    "output",
)
_HASHED_KEYS = (
    "checkpoint",
    "fallback_submission",
    "fallback_csv",
    "train_csv",
    "val_csv",
    "groups",
    "class_mapping",
)


def load_v9_plan(path: str | Path) -> dict:
    source = Path(path).resolve()
    plan = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("PRELIM75 v9 config root must be a mapping")
    allowed = {
        "plan_id",
        "seed",
        "fixed_review_commit",
        "num_workers",
        "d0",
        "inference_protocol",
        "diagnostic",
        "budget",
        "reference_platform_percent",
        *_PATH_KEYS,
        *(key + "_sha256" for key in _HASHED_KEYS),
    }
    if set(plan) != allowed:
        raise ValueError(
            "Unknown or incomplete PRELIM75 v9 plan: "
            f"missing={sorted(allowed - set(plan))}, extra={sorted(set(plan) - allowed)}"
        )
    if (
        plan["plan_id"] != PLAN_ID
        or int(plan["seed"]) != 42
        or plan["fixed_review_commit"] != FIXED_REVIEW_COMMIT
        or int(plan["num_workers"]) != 4
    ):
        raise ValueError("PRELIM75 v9 fixed plan identity changed")
    if plan["d0"] != {
        "sample_size": D0_SAMPLE_SIZE,
        "minimum_supported": D0_MINIMUM_SUPPORTED,
        "source_gain_threshold": SOURCE_GAIN_THRESHOLD,
        "selection_salt": f"{PLAN_ID}/42/",
    }:
        raise ValueError("PRELIM75 v9 D0 protocol changed")
    if plan["inference_protocol"] != INFERENCE_PROTOCOL:
        raise ValueError("PRELIM75 v9 inference protocol changed")
    if plan["diagnostic"] != {
        "rows": REQUIRED_DIAGNOSTIC_ROWS,
        "maximum_drop_pp": 2.0,
        "scope": "training_overlap_engineering_only",
    }:
        raise ValueError("PRELIM75 v9 diagnostic protocol changed")
    if plan["budget"] != {
        "optimizer_updates": 0,
        "maximum_platform_candidates": 2,
        "gpu_action_seconds": 7200,
    }:
        raise ValueError("PRELIM75 v9 budget changed")
    if plan["reference_platform_percent"] != EXPECTED_REFERENCES:
        raise ValueError("PRELIM75 v9 reference score table changed")
    for key in _PATH_KEYS:
        plan[key] = str((source.parent / plan[key]).resolve())
    for key in _HASHED_KEYS:
        expected = plan[key + "_sha256"]
        if sha256_file(plan[key]) != expected:
            raise ValueError(f"Frozen PRELIM75 v9 asset hash mismatch: {key}")
    expected_hashes = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "fallback_submission_sha256": FALLBACK_ZIP_SHA256,
        "fallback_csv_sha256": FALLBACK_CSV_SHA256,
        "train_csv_sha256": TRAIN_CSV_SHA256,
        "val_csv_sha256": VAL_CSV_SHA256,
        "groups_sha256": GROUPS_SHA256,
        "class_mapping_sha256": CLASS_MAPPING_SHA256,
    }
    for key, expected in expected_hashes.items():
        if plan[key] != expected:
            raise ValueError(f"Registered PRELIM75 v9 digest changed: {key}")
    train_root = Path(plan["train_root"]).resolve()
    test_root = Path(plan["test_root"]).resolve()
    if (
        train_root == test_root
        or train_root.is_relative_to(test_root)
        or test_root.is_relative_to(train_root)
    ):
        raise ValueError("Official train and test roots must be disjoint")
    plan["_config_path"] = str(source)
    plan["_config_sha256"] = sha256_file(source)
    return plan


def _read_submission(path: str | Path) -> list[tuple[str, str]]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, skipinitialspace=True):
            if len(row) != 2:
                raise ValueError(f"Malformed submission row in {path}")
            rows.append((row[0].strip(), row[1].strip()))
    return rows


def _checkpoint_audit(plan: dict) -> dict:
    checkpoint = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    metadata = checkpoint.get("prelim75_training", {})
    state = checkpoint.get("model_state_dict", {})
    if config.get("project", {}).get("experiment_id") != "PRELIM75_V7_L1":
        raise ValueError("v9 checkpoint is not PRELIM75_V7_L1")
    if metadata.get("plan_id") != "PRELIM75_V7_20260918" or metadata.get("name") != "L1":
        raise ValueError("v9 checkpoint training metadata is not registered L1")
    if metadata.get("prior_in_inference") is not False:
        raise ValueError("v9 checkpoint unexpectedly requires an inference prior")
    if config.get("model", {}).get("peft_mode") != "full_finetune":
        raise ValueError("v9 requires the complete L1 full-finetune visual model")
    required_state = {
        "visual.conv1.weight",
        "visual.positional_embedding",
        "classifier.weight",
        "classifier.bias",
    }
    if not required_state.issubset(state):
        raise ValueError("v9 checkpoint is missing visual or shared-head state")
    if "local_feature_adapter" not in checkpoint or "part_token_adapter" not in checkpoint:
        raise ValueError("v9 checkpoint is missing O3 or PTA")
    return {
        "experiment_id": "PRELIM75_V7_L1",
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "prelim75_plan_id": metadata["plan_id"],
        "prelim75_name": metadata["name"],
        "peft_mode": "full_finetune",
        "model_state_tensors": len(state),
        "local_feature_adapter_present": True,
        "part_token_adapter_present": True,
        "prior_in_inference": False,
        "upstream_provenance_complete": bool(
            metadata.get("upstream_provenance_complete", False)
        ),
    }


def preflight_v9(plan: dict) -> dict:
    root = repository_root()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", plan["fixed_review_commit"], "HEAD"],
        cwd=root,
    ).returncode:
        raise ValueError("Fixed reviewed v9 base is not an ancestor of HEAD")
    checkpoint = _checkpoint_audit(plan)
    mapping, _ = load_class_mapping(plan["class_mapping"])
    if len(mapping) != REQUIRED_CLASSES:
        raise ValueError("v9 requires the registered 500-class mapping")
    frame = pd.read_csv(plan["train_csv"])
    paths = [canonical_sample_path(value) for value in frame.image_path.astype(str)]
    if len(paths) != REQUIRED_TRAIN_ROWS or len(set(paths)) != REQUIRED_TRAIN_ROWS:
        raise ValueError("Official v9 training list identity changed")
    labels = frame.label.astype(int).to_numpy()
    if labels.min() < 0 or labels.max() >= REQUIRED_CLASSES:
        raise ValueError("Training labels are outside the official class mapping")
    train_root = Path(plan["train_root"]).resolve()
    for path in paths:
        resolved = resolve_image_path(train_root, path).resolve()
        if not resolved.is_relative_to(train_root) or not resolved.is_file():
            raise ValueError("Training list contains a missing or non-official path")
    groups = json.loads(Path(plan["groups"]).read_text(encoding="utf-8"))
    if set(paths) != set(groups):
        raise ValueError("Content-group map does not exactly cover the training list")
    unique_groups = len(set(str(value) for value in groups.values()))
    if unique_groups != 101980:
        raise ValueError("Registered unique content-group count changed")
    val = pd.read_csv(plan["val_csv"])
    val_paths = [canonical_sample_path(value) for value in val.image_path.astype(str)]
    if len(val_paths) != REQUIRED_DIAGNOSTIC_ROWS or len(set(val_paths)) != len(val_paths):
        raise ValueError("Registered overlap diagnostic list changed")
    training_labels = dict(zip(paths, labels.tolist()))
    if any(training_labels.get(path) != int(label) for path, label in zip(val_paths, val.label)):
        raise ValueError("Diagnostic list is not an exact labeled subset of training")
    test_paths = sorted(
        path
        for path in Path(plan["test_root"]).rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if len(test_paths) != REQUIRED_TEST_ROWS or len({path.name for path in test_paths}) != REQUIRED_TEST_ROWS:
        raise ValueError("Official test basename identity changed")
    fallback_rows = _read_submission(plan["fallback_csv"])
    if len(fallback_rows) != REQUIRED_TEST_ROWS or set(name for name, _ in fallback_rows) != {
        path.name for path in test_paths
    }:
        raise ValueError("Archived L1 CSV does not cover the official test set")
    valid_labels = set(mapping)
    if any(label not in valid_labels for _, label in fallback_rows):
        raise ValueError("Archived L1 CSV contains an invalid class label")
    with ZipFile(plan["fallback_submission"]) as archive:
        if archive.namelist() != ["pred_results.csv"]:
            raise ValueError("Archived L1 ZIP structure changed")
        if archive.read("pred_results.csv") != Path(plan["fallback_csv"]).read_bytes():
            raise ValueError("Archived L1 ZIP and CSV are not byte-identical")
    manifest = json.loads(Path(plan["fallback_manifest"]).read_text(encoding="utf-8"))
    if (
        manifest.get("checkpoint_sha256") != plan["checkpoint_sha256"]
        or manifest.get("prediction_csv_sha256") != plan["fallback_csv_sha256"]
        or manifest.get("submission_zip_sha256") != plan["fallback_submission_sha256"]
    ):
        raise ValueError("Archived L1 submission manifest binding changed")
    return {
        "status": "passed",
        "plan_id": PLAN_ID,
        "fixed_review_commit": FIXED_REVIEW_COMMIT,
        "config_sha256": plan["_config_sha256"],
        "checkpoint": checkpoint,
        "train_rows": len(paths),
        "diagnostic_rows": len(val_paths),
        "test_rows": len(test_paths),
        "classes": len(mapping),
        "unique_content_groups": unique_groups,
        "train_paths_sha256": stable_path_list_sha256(paths),
        "diagnostic_paths_sha256": stable_path_list_sha256(val_paths),
        "test_basenames_sha256": stable_path_list_sha256([path.name for path in test_paths]),
        "asset_sha256": {
            key: plan[key + "_sha256"] for key in _HASHED_KEYS
        },
        "fallback_manifest_sha256": sha256_file(plan["fallback_manifest"]),
        "test_dimensions_read_for_d0": False,
        "test_predictions_read_for_parameter_selection": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
    }


def _native_preprocess_without_model():
    from clip.clip import _transform

    preprocess = _transform(INPUT_SIZE)
    validate_native_preprocess(preprocess)
    return preprocess


def run_d0(plan: dict) -> dict:
    output = Path(plan["output"])
    destination = output / "d0"
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frame = pd.read_csv(plan["train_csv"])
    paths = [canonical_sample_path(value) for value in frame.image_path.astype(str)]
    groups = json.loads(Path(plan["groups"]).read_text(encoding="utf-8"))
    selected = select_group_representatives(paths, groups)
    preprocess = _native_preprocess_without_model()
    train_root = Path(plan["train_root"]).resolve()
    rows = []
    for row in selected:
        path = resolve_image_path(train_root, row["canonical_path"]).resolve()
        if not path.is_relative_to(train_root) or not path.is_file():
            raise ValueError("D0 selected a missing or non-official training image")
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                trace = trace_from_dimensions(*rgb.size)
                resized = preprocess.transforms[0](rgb)
                if resized.size != (trace.resized_width, trace.resized_height):
                    raise RuntimeError("D0 trace differs from installed torchvision")
        except Exception as exc:
            raise RuntimeError(f"D0 failed to decode selected image: {path}") from exc
        rows.append({**row, **trace.to_dict()})
    selection_path = destination / "selected_groups.csv"
    with selection_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gains = [float(row["source_gain"]) for row in rows]
    gate = d0_gate(gains)

    def quantiles(values: Iterable[float]) -> dict[str, float]:
        array = np.asarray(list(values), dtype=np.float64)
        return {
            name: float(np.quantile(array, value))
            for name, value in (("min", 0), ("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9), ("max", 1))
        }

    result = {
        **gate,
        "plan_id": PLAN_ID,
        "selection_rule": f"sha256({PLAN_ID}/42/canonical_path)",
        "canonical_representative_rule": "lexicographic_minimum_path_per_content_group",
        "labels_read_for_selection": False,
        "trust_read": False,
        "model_forward_calls": 0,
        "test_images_read": False,
        "selected_groups_csv": str(selection_path),
        "selected_groups_csv_sha256": sha256_file(selection_path),
        "selected_paths_sha256": stable_path_list_sha256(
            [str(row["canonical_path"]) for row in rows]
        ),
        "train_csv_sha256": plan["train_csv_sha256"],
        "groups_sha256": plan["groups_sha256"],
        "source_width": quantiles(row["source_width"] for row in rows),
        "source_height": quantiles(row["source_height"] for row in rows),
        "source_short_side": quantiles(
            min(row["source_width"], row["source_height"]) for row in rows
        ),
        "source_gain": quantiles(gains),
        "preprocess_contract": validate_native_preprocess(preprocess),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json_dump(result, destination / "result.json")
    atomic_json_dump(result, destination / "status.json")
    return result


class SourceImageDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[str | Path],
        root: str | Path,
        preprocess: object,
        *,
        names_are_basenames: bool,
    ) -> None:
        self.paths = list(paths)
        self.root = Path(root).resolve()
        self.preprocess = preprocess
        self.names_are_basenames = names_are_basenames

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        raw = self.paths[index]
        path = Path(raw).resolve() if Path(raw).is_absolute() else resolve_image_path(
            self.root, str(raw)
        ).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Source-image inference path escaped the registered root")
        name = path.name if self.names_are_basenames else canonical_sample_path(raw)
        try:
            with Image.open(path) as image:
                prepared = prepare_rgb(image.convert("RGB"), self.preprocess)
            corrupt = False
        except Exception:
            rgb = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
            prepared = PreparedRGB(
                rgb=rgb,
                canvas=rgb.copy(),
                global_tensor=torch.zeros(3, INPUT_SIZE, INPUT_SIZE),
                trace=trace_from_dimensions(INPUT_SIZE, INPUT_SIZE),
            )
            corrupt = True
        return {
            "prepared": prepared,
            "images": prepared.global_tensor,
            "name": name,
            "corrupt": corrupt,
            "index": index,
        }


def _collate_source(batch: list[dict]) -> dict:
    return {
        "prepared": [row["prepared"] for row in batch],
        "images": torch.stack([row["images"] for row in batch]),
        "name": [row["name"] for row in batch],
        "corrupt": torch.tensor([row["corrupt"] for row in batch], dtype=torch.bool),
        "index": torch.tensor([row["index"] for row in batch], dtype=torch.long),
    }


def _loader(dataset: Dataset, plan: dict) -> DataLoader:
    workers = int(plan["num_workers"])
    return DataLoader(
        dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        timeout=120 if workers else 0,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=1 if workers else None,
        worker_init_fn=seed_worker,
        collate_fn=_collate_source,
    )


def _adapter_logits(model, o3, pta, images: torch.Tensor, part_spec: dict) -> torch.Tensor:
    return adapted_dual_local_view_logits(
        model,
        o3,
        pta,
        images,
        part_top_patches=int(part_spec["top_patches"]),
        part_temperature=float(part_spec["temperature"]),
    )


@torch.no_grad()
def _run_paths(
    plan: dict,
    paths: Sequence[str | Path],
    root: str | Path,
    *,
    names_are_basenames: bool,
    active_candidates: Sequence[str] = CANDIDATES,
    progress_path: Path | None = None,
) -> dict:
    unknown = set(active_candidates) - set(CANDIDATES)
    if unknown:
        raise ValueError(f"Unknown v9 candidates: {sorted(unknown)}")
    started = time.monotonic()
    device = gpu_setup()
    model, preprocess, checkpoint, o3, pta = load_composite(plan["checkpoint"], device)
    model.float().eval().requires_grad_(False)
    o3.float().eval().requires_grad_(False)
    pta.float().eval().requires_grad_(False)
    if any(parameter.requires_grad for module in (model, o3, pta) for parameter in module.parameters()):
        raise RuntimeError("v9 requires every model and adapter parameter to be frozen")
    part_spec = checkpoint["part_token_adapter"]["spec"]["part_pool_spec"]
    tail = native_tail(preprocess)
    dataset = SourceImageDataset(
        paths, root, preprocess, names_are_basenames=names_are_basenames
    )
    stream = _loader(dataset, plan)
    predictions: dict[str, list[torch.Tensor]] = {
        "baseline": [],
        **{name: [] for name in active_candidates},
    }
    prediction_names: list[str] = []
    corrupt_count = 0
    enabled_count = 0
    completed = 0
    for batch in stream:
        images = batch["images"].to(device, non_blocking=True)
        prepared: list[PreparedRGB] = batch["prepared"]
        corrupt_count += int(batch["corrupt"].sum())
        enabled_count += sum(
            item.trace.source_gain >= SOURCE_GAIN_THRESHOLD for item in prepared
        )
        orientations = (images, images.flip(3))
        global_logits: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        native_logits: list[list[torch.Tensor]] = []
        boxes: list[list[list[tuple[int, int, int, int]]]] = []
        # This is the archived L1 call order: original global+four locals,
        # followed by flipped global+four locals.
        for view in orientations:
            current_global, attention = forward_with_last_block_attention(model, view)
            global_logits.append(current_global)
            attentions.append(attention)
            orientation_logits = []
            orientation_boxes = []
            for scale in LOCAL_SCALES:
                local, current_boxes = extract_attention_crops(
                    view, attention, crop_size=scale, top_k=5
                )
                orientation_logits.append(
                    _adapter_logits(model, o3, pta, local, part_spec)
                )
                orientation_boxes.append(current_boxes)
            native_logits.append(orientation_logits)
            boxes.append(orientation_boxes)
        baseline_scores = fuse_global_multilocal_flip_probabilities(
            global_logits[0],
            native_logits[0],
            global_logits[1],
            native_logits[1],
            local_weight=0.4,
            flip_weight=0.5,
            temperature=1.5,
            global_temperature=1.5,
            local_temperature=1.5,
            local_scale_weights=LOCAL_SCALE_WEIGHTS,
        )
        predictions["baseline"].append(baseline_scores.argmax(1).cpu())
        for candidate in active_candidates:
            candidate_logits: list[list[torch.Tensor]] = [[], []]
            for orientation, view in enumerate(orientations):
                for scale_index, scale in enumerate(LOCAL_SCALES):
                    native_local, repeated_boxes = extract_attention_crops(
                        view,
                        attentions[orientation],
                        crop_size=scale,
                        top_k=5,
                    )
                    if repeated_boxes != boxes[orientation][scale_index]:
                        raise RuntimeError("R0/R1 did not share the native L1 boxes")
                    native_cpu = native_local.detach().cpu()
                    recropped = torch.stack(
                        [
                            recrop_local(
                                item,
                                boxes[orientation][scale_index][row],
                                mode=candidate,
                                flipped=bool(orientation),
                                preprocess_tail=tail,
                                native_local=native_cpu[row],
                            )
                            for row, item in enumerate(prepared)
                        ]
                    ).to(device, non_blocking=True)
                    candidate_logits[orientation].append(
                        _adapter_logits(model, o3, pta, recropped, part_spec)
                    )
            scores = fuse_global_multilocal_flip_probabilities(
                global_logits[0],
                candidate_logits[0],
                global_logits[1],
                candidate_logits[1],
                local_weight=0.4,
                flip_weight=0.5,
                temperature=1.5,
                global_temperature=1.5,
                local_temperature=1.5,
                local_scale_weights=LOCAL_SCALE_WEIGHTS,
            )
            predictions[candidate].append(scores.argmax(1).cpu())
        prediction_names.extend(batch["name"])
        completed += len(images)
        if progress_path is not None:
            atomic_json_dump(
                {
                    "status": "running",
                    "completed_samples": completed,
                    "total_samples": len(dataset),
                    "active_candidates": list(active_candidates),
                    "elapsed_seconds": time.monotonic() - started,
                },
                progress_path,
            )
    if corrupt_count:
        raise RuntimeError(
            f"Refusing v9 output: Pillow failed to decode {corrupt_count} images"
        )
    result_predictions = {
        name: torch.cat(parts) for name, parts in predictions.items()
    }
    if any(len(value) != len(paths) for value in result_predictions.values()):
        raise RuntimeError("v9 prediction row count changed")
    return {
        "names": prediction_names,
        "predictions": result_predictions,
        "source_gain_enabled_count": enabled_count,
        "source_gain_enabled_fraction": enabled_count / len(dataset),
        "corrupt_count": corrupt_count,
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_memory_allocated": int(torch.cuda.max_memory_allocated()),
        "host_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "preprocess_contract": validate_native_preprocess(preprocess),
    }


def _module_state_sha256(modules: Sequence[tuple[str, torch.nn.Module]]) -> str:
    digest = hashlib.sha256()
    for prefix, module in modules:
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{prefix}.{name}|{tensor.dtype}|{tuple(tensor.shape)}".encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def smoke_v9(plan: dict) -> dict:
    output = Path(plan["output"])
    destination = output / "smoke"
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    d0 = json.loads((output / "d0/result.json").read_text(encoding="utf-8"))
    if not d0.get("passed"):
        raise ValueError("v9 smoke cannot run before a passing D0 gate")
    selected = list(csv.DictReader(Path(d0["selected_groups_csv"]).open(encoding="utf-8")))
    paths = [row["canonical_path"] for row in selected[:4]]
    device = gpu_setup()
    model, preprocess, checkpoint, o3, pta = load_composite(plan["checkpoint"], device)
    model.float().eval().requires_grad_(False)
    o3.float().eval().requires_grad_(False)
    pta.float().eval().requires_grad_(False)
    modules = (("model", model), ("o3", o3), ("pta", pta))
    before = _module_state_sha256(modules)
    dataset = SourceImageDataset(
        paths, plan["train_root"], preprocess, names_are_basenames=False
    )
    batch = _collate_source([dataset[index] for index in range(len(dataset))])
    if bool(batch["corrupt"].any()):
        raise RuntimeError("v9 smoke selected an undecodable D0 image")
    for item in batch["prepared"]:
        if not torch.equal(item.global_tensor, preprocess(item.rgb)):
            raise RuntimeError("v9 wrapped global tensor changed")
    images = batch["images"].to(device)
    logits, features, attention = forward_features_with_last_block_attention(model, images)
    repeated_logits, repeated_features, repeated_attention = (
        forward_features_with_last_block_attention(model, images)
    )
    torch.testing.assert_close(logits, repeated_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(features, repeated_features, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(attention, repeated_attention, atol=1e-6, rtol=1e-5)
    shared_boxes = {}
    fallback_exact = True
    tail = native_tail(preprocess)
    for orientation, view in enumerate((images, images.flip(3))):
        _, current_attention = forward_with_last_block_attention(model, view)
        for scale in LOCAL_SCALES:
            native, boxes = extract_attention_crops(
                view, current_attention, crop_size=scale, top_k=5
            )
            shared_boxes[f"{orientation}:{scale}"] = boxes
            for candidate in CANDIDATES:
                for row, item in enumerate(batch["prepared"]):
                    result = recrop_local(
                        item,
                        boxes[row],
                        mode=candidate,
                        flipped=bool(orientation),
                        preprocess_tail=tail,
                        native_local=native[row].cpu(),
                    )
                    if item.trace.source_gain < SOURCE_GAIN_THRESHOLD:
                        fallback_exact = fallback_exact and torch.equal(
                            result, native[row].cpu()
                        )
    after = _module_state_sha256(modules)
    if before != after:
        raise RuntimeError("v9 smoke changed a model weight or buffer")
    del modules, model, o3, pta
    torch.cuda.empty_cache()
    run = _run_paths(
        plan,
        paths,
        plan["train_root"],
        names_are_basenames=False,
        active_candidates=CANDIDATES,
        progress_path=destination / "inference_status.json",
    )
    if not fallback_exact:
        raise RuntimeError("v9 low-gain fallback changed the native local tensor")
    result = {
        "status": "passed",
        "plan_id": PLAN_ID,
        "paths": paths,
        "paths_sha256": stable_path_list_sha256(paths),
        "global_tensor_exact": True,
        "global_logits_replay_atol": 1e-5,
        "global_features_replay_atol": 1e-6,
        "attention_replay_atol": 1e-6,
        "shared_native_boxes": True,
        "shared_box_records": shared_boxes,
        "low_gain_native_fallback_exact": fallback_exact,
        "all_weights_and_buffers_unchanged": True,
        "module_state_sha256_before": before,
        "module_state_sha256_after": after,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "test_images_read": False,
        "source_gain_enabled_count": run["source_gain_enabled_count"],
        "max_cuda_memory_allocated": max(
            int(torch.cuda.max_memory_allocated()), run["max_cuda_memory_allocated"]
        ),
        "host_max_rss_kib": run["host_max_rss_kib"],
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json_dump(result, destination / "result.json")
    atomic_json_dump(result, destination / "status.json")
    return result


def diagnostic_v9(plan: dict) -> dict:
    output = Path(plan["output"])
    destination = output / "diagnostic"
    destination.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(plan["val_csv"])
    paths = [canonical_sample_path(value) for value in frame.image_path.astype(str)]
    labels = torch.tensor(frame.label.astype(int).to_numpy(), dtype=torch.long)
    run = _run_paths(
        plan,
        paths,
        plan["train_root"],
        names_are_basenames=False,
        active_candidates=CANDIDATES,
        progress_path=destination / "inference_status.json",
    )
    if run["names"] != paths:
        raise RuntimeError("v9 diagnostic row order changed")
    metrics = {}
    baseline_accuracy = float((run["predictions"]["baseline"] == labels).float().mean())
    metrics["baseline"] = {"raw_micro": baseline_accuracy, "drop_vs_baseline_pp": 0.0}
    stopped = []
    for candidate in CANDIDATES:
        accuracy = float((run["predictions"][candidate] == labels).float().mean())
        delta = 100.0 * (accuracy - baseline_accuracy)
        stop = delta < -float(plan["diagnostic"]["maximum_drop_pp"])
        metrics[candidate] = {
            "raw_micro": accuracy,
            "drop_vs_baseline_pp": delta,
            "engineering_stop": stop,
            "changed_predictions_vs_baseline": int(
                (run["predictions"][candidate] != run["predictions"]["baseline"]).sum()
            ),
        }
        if stop:
            stopped.append(candidate)
    _save_predictions(destination / "predictions.pt", run["names"], run["predictions"])
    result = {
        "status": "complete",
        "plan_id": PLAN_ID,
        "scope": "training_overlap_engineering_only",
        "independent_generalization_claim": False,
        "selection_or_tuning_used": False,
        "rows": len(paths),
        "val_csv_sha256": plan["val_csv_sha256"],
        "metrics": metrics,
        "engineering_stopped_candidates": stopped,
        "active_candidates": [name for name in CANDIDATES if name not in stopped],
        "maximum_allowed_drop_pp": float(plan["diagnostic"]["maximum_drop_pp"]),
        "source_gain_enabled_count": run["source_gain_enabled_count"],
        "source_gain_enabled_fraction": run["source_gain_enabled_fraction"],
        "max_cuda_memory_allocated": run["max_cuda_memory_allocated"],
        "host_max_rss_kib": run["host_max_rss_kib"],
        "elapsed_seconds": run["elapsed_seconds"],
        "optimizer_created": False,
        "optimizer_updates": 0,
        "test_images_read": False,
    }
    atomic_json_dump(result, destination / "result.json")
    atomic_json_dump(result, destination / "status.json")
    return result


def _save_predictions(path: Path, names: Sequence[str], predictions: dict[str, torch.Tensor]) -> None:
    from aegis_clip.checkpoint import _atomic_torch_save

    _atomic_torch_save(
        {"names": list(names), **{key: value.cpu() for key, value in predictions.items()}},
        path,
    )


def _write_submission(
    plan: dict,
    candidate: str,
    names: Sequence[str],
    predicted_indices: torch.Tensor,
    idx_to_class: dict[int, str],
    report_fields: dict,
) -> dict:
    root = Path(plan["output"]) / candidate
    submission = root / "submission"
    if root.exists():
        raise FileExistsError(f"Existing v9 candidate output is never overwritten: {root}")
    submission.mkdir(parents=True, exist_ok=False)
    rows = [(name, idx_to_class[int(index)]) for name, index in zip(names, predicted_indices)]
    if len(rows) != REQUIRED_TEST_ROWS or len({name for name, _ in rows}) != REQUIRED_TEST_ROWS:
        raise ValueError("v9 submission coverage changed")
    if any(len(label) != 4 or not label.isdigit() for _, label in rows):
        raise ValueError("v9 submission contains a malformed class label")
    csv_path = submission / "pred_results.csv"
    csv_path.write_text(
        "".join(f"{name}, {label}\n" for name, label in rows), encoding="utf-8"
    )
    zip_path = submission / "submission.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="pred_results.csv")
    manifest = {
        "format_version": 1,
        "plan_id": PLAN_ID,
        "candidate": candidate,
        "checkpoint": plan["checkpoint"],
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "prediction_csv_sha256": sha256_file(csv_path),
        "submission_zip_sha256": sha256_file(zip_path),
        "csv_delimiter": "comma_space",
        "rows": len(rows),
        "single_checkpoint": True,
        "prior_in_inference": False,
        "optimizer_updates": 0,
        "protocol": INFERENCE_PROTOCOL,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PILLOW_VERSION,
        },
        **report_fields,
    }
    atomic_json_dump(manifest, submission / "manifest.json")
    check_command = [
        sys.executable,
        str(repository_root() / "scripts/check_submission.py"),
        "--test_dir",
        plan["test_root"],
        "--num-classes",
        "500",
        "--csv",
        str(csv_path),
        "--zip",
        str(zip_path),
    ]
    with (root / "submission_validation.log").open("w", encoding="utf-8") as log:
        subprocess.run(check_command, check=True, stdout=log, stderr=subprocess.STDOUT)
    with ZipFile(zip_path) as archive:
        byte_equal = (
            archive.namelist() == ["pred_results.csv"]
            and archive.read("pred_results.csv") == csv_path.read_bytes()
        )
    if not byte_equal:
        raise RuntimeError("v9 ZIP/CSV byte identity failed")
    report = {
        "status": "submission_ready_pending_platform",
        "plan_id": PLAN_ID,
        "candidate": candidate,
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "rows": len(rows),
        "unique_prediction_classes": len({label for _, label in rows}),
        "prediction_coverage_forced_to_500": False,
        "csv_sha256": sha256_file(csv_path),
        "zip_sha256": sha256_file(zip_path),
        "manifest_sha256": sha256_file(submission / "manifest.json"),
        "validation_command": check_command,
        "validation_exit_code": 0,
        "zip_internal_csv_byte_equal": True,
        "online_accuracy": None,
        "online_exact_correct": None,
        "actual_platform_upload_time": None,
        **report_fields,
    }
    atomic_json_dump(report, root / "candidate_report.json")
    atomic_json_dump(report, root / "status.json")
    return report


def infer_v9(plan: dict) -> dict:
    output = Path(plan["output"])
    destination = output / "inference"
    destination.mkdir(parents=True, exist_ok=False)
    diagnostic = json.loads(
        (output / "diagnostic/result.json").read_text(encoding="utf-8")
    )
    active = tuple(diagnostic["active_candidates"])
    test_paths = sorted(
        path
        for path in Path(plan["test_root"]).rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    run = _run_paths(
        plan,
        test_paths,
        plan["test_root"],
        names_are_basenames=True,
        active_candidates=active,
        progress_path=destination / "inference_status.json",
    )
    expected_names = [path.name for path in test_paths]
    if run["names"] != expected_names:
        raise RuntimeError("v9 test row order changed")
    _, idx_to_class = load_class_mapping(plan["class_mapping"])
    archived = dict(_read_submission(plan["fallback_csv"]))
    baseline_labels = [idx_to_class[int(value)] for value in run["predictions"]["baseline"]]
    mismatches = [
        name for name, label in zip(run["names"], baseline_labels)
        if archived.get(name) != label
    ]
    replay = {
        "status": "passed" if not mismatches else "failed",
        "rows": len(run["names"]),
        "mismatch_rows": len(mismatches),
        "first_mismatches": mismatches[:20],
        "archived_csv_sha256": plan["fallback_csv_sha256"],
        "basename_aligned": True,
    }
    atomic_json_dump(replay, destination / "native_replay.json")
    if mismatches:
        result = {
            "status": "failed_native_replay_no_candidate_delivery",
            "native_replay": replay,
            "online_scores": {"R0": None, "R1": None},
        }
        atomic_json_dump(result, destination / "result.json")
        raise RuntimeError(
            f"Archived L1 replay differs on {len(mismatches)} test rows"
        )
    delivered = []
    duplicates = {}
    candidate_reports = {}
    baseline = run["predictions"]["baseline"]
    for candidate in active:
        predictions = run["predictions"][candidate]
        duplicate_of = None
        if torch.equal(predictions, baseline):
            duplicate_of = "L1"
        elif candidate == "R1" and "R0" in active and torch.equal(
            predictions, run["predictions"]["R0"]
        ):
            duplicate_of = "R0"
        if duplicate_of:
            duplicates[candidate] = duplicate_of
            candidate_reports[candidate] = {
                "status": "duplicate_prediction_set_no_new_upload",
                "duplicate_of": duplicate_of,
                "changed_predictions_vs_L1": int((predictions != baseline).sum()),
                "online_accuracy": 69.2794 if duplicate_of == "L1" else None,
                "online_accuracy_source": "existing_identical_prediction_set" if duplicate_of == "L1" else None,
            }
            atomic_json_dump(
                candidate_reports[candidate], output / f"{candidate}_duplicate.json"
            )
            continue
        fields = {
            "pixel_source": "native_224_canvas" if candidate == "R0" else "decoded_source_rgb",
            "changed_predictions_vs_L1": int((predictions != baseline).sum()),
            "changed_predictions_vs_R0": (
                int((predictions != run["predictions"]["R0"]).sum())
                if candidate == "R1" and "R0" in active
                else None
            ),
            "source_gain_threshold": SOURCE_GAIN_THRESHOLD,
            "source_gain_enabled_count": run["source_gain_enabled_count"],
            "source_gain_enabled_fraction": run["source_gain_enabled_fraction"],
            "native_replay_mismatch_rows": 0,
            "global_logits_and_boxes_shared": True,
            "preprocess_contract": run["preprocess_contract"],
            "flip_definition": "unflip_native_box_then_sample_unflipped_source_then_flip_local_output",
            "test_batch_statistics_used_for_selection": False,
        }
        candidate_reports[candidate] = _write_submission(
            plan,
            candidate,
            run["names"],
            predictions,
            idx_to_class,
            fields,
        )
        delivered.append(candidate)
    _save_predictions(destination / "predictions.pt", run["names"], run["predictions"])
    result = {
        "status": "complete_pending_real_platform_feedback" if delivered else "complete_no_unique_new_candidate",
        "plan_id": PLAN_ID,
        "native_replay": replay,
        "diagnostic_stopped_candidates": diagnostic["engineering_stopped_candidates"],
        "active_candidates": list(active),
        "delivered_candidates": delivered,
        "duplicate_prediction_sets": duplicates,
        "candidate_reports": candidate_reports,
        "platform_packages_created": len(delivered),
        "online_scores": {"R0": None, "R1": None},
        "online_exact_correct": {"R0": None, "R1": None},
        "actual_platform_upload_time": {"R0": None, "R1": None},
        "source_gain_enabled_count": run["source_gain_enabled_count"],
        "source_gain_enabled_fraction": run["source_gain_enabled_fraction"],
        "test_dimension_rule_changed_after_observation": False,
        "max_cuda_memory_allocated": run["max_cuda_memory_allocated"],
        "host_max_rss_kib": run["host_max_rss_kib"],
        "elapsed_seconds": run["elapsed_seconds"],
        "optimizer_created": False,
        "optimizer_updates": 0,
        "reference_platform_percent": EXPECTED_REFERENCES,
        "decision_rules": {
            "both_not_above_L1": "keep_L1_close_source_recrop",
            "R0_best_above_L1": "keep_R0_without_source_resolution_claim",
            "R1_best_above_L1": "keep_R1_and_report_R1_minus_R0_and_L1",
            "investment_gate_pp": 0.30,
            "tie_with_L1": "keep_L1",
            "R0_R1_tie_above_L1": "keep_R0",
        },
        "automatic_upload": False,
        "parameter_scan": False,
        "prior_used": False,
    }
    atomic_json_dump(result, destination / "result.json")
    atomic_json_dump(result, destination / "status.json")
    return result
