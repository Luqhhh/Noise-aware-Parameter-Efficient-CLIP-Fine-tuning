"""Stage-internal rebuild pipeline: code travels, fitted thresholds do not.

Every preliminary-stage artifact (fold assignments, OOF logits, trust bundles,
purification manifests, calibration strengths) is rebuilt from the
current-stage official data only.  Thresholds are manifest parameters, never
defaults copied from a previous stage.  The runner records each step's inputs,
outputs, and hashes into ``pipeline_run.json`` so the chain is auditable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from aegis_clip.cli.cache_features import cache_stage_features
from aegis_clip.cli.prepare_final_train import merge_splits
from aegis_clip.cli.prepare_stage import prepare_stage
from aegis_clip.oof_rebuild import load_oof_inputs, rebuild_oof_logits
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.trust import TrustBuildConfig, atomic_torch_save, build_cross_fitted_trust


def _canonical_group_key(path: str, root_name: str) -> str:
    normalized = path.replace("\\", "/")
    prefix = f"{root_name}/"
    if normalized.startswith(prefix):
        return normalized[len(prefix):]
    return normalized


def assign_oof_folds(
    train_csv: str | Path,
    val_csv: str | Path,
    groups_path: str | Path,
    output_csv: str | Path,
    *,
    folds: int,
    seed: int,
    root_name: str,
) -> Path:
    """Assign content-group-aware stratified OOF folds to every official image."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    train_frame = pd.read_csv(train_csv)
    val_frame = pd.read_csv(val_csv)
    frame = pd.concat([train_frame, val_frame], ignore_index=True)
    if set(frame.columns) < {"image_path", "label"}:
        raise ValueError("split CSVs must contain image_path and label")
    groups = json.loads(Path(groups_path).read_text(encoding="utf-8"))
    canonical = [
        _canonical_group_key(path, root_name)
        for path in frame["image_path"].astype(str)
    ]
    missing = [path for path in canonical if path not in groups]
    if missing:
        raise ValueError(f"Content groups miss {len(missing)} samples; first={missing[0]}")
    group_keys = [str(groups[path]) for path in canonical]
    splitter = StratifiedGroupKFold(
        n_splits=int(folds), shuffle=True, random_state=int(seed)
    )
    fold_column = pd.Series(-1, index=frame.index)
    for fold, (_, holdout) in enumerate(
        splitter.split(canonical, frame["label"], groups=group_keys)
    ):
        fold_column.iloc[holdout] = fold
    if (fold_column < 0).any():
        raise RuntimeError("StratifiedGroupKFold left unassigned samples")
    assignments = pd.DataFrame(
        {
            "sample_id": [f"{index:08d}" for index in range(len(frame))],
            "image_path": frame["image_path"].astype(str),
            "label": frame["label"].astype(int),
            "fold": fold_column.astype(int),
        }
    )
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(destination, index=False)
    return destination


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"stage", "seed", "train_root", "output_root", "expected_classes"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Pipeline manifest missing keys: {sorted(missing)}")
    if manifest.get("expected_samples") is None:
        raise ValueError("Pipeline manifest requires expected_samples")
    return manifest


def run_stage_pipeline(manifest_path: str | Path) -> Path:
    manifest = _load_manifest(manifest_path)
    stage = manifest["stage"]
    seed = int(manifest["seed"])
    train_root = Path(manifest["train_root"])
    output_root = Path(manifest["output_root"])
    expected_classes = int(manifest["expected_classes"])
    expected_samples = int(manifest["expected_samples"])
    val_ratio = float(manifest.get("val_ratio", 0.10))
    folds = int(manifest.get("folds", 3))
    device = torch.device(
        manifest.get("device", "cuda")
        if manifest.get("device", "cuda") != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    split_dir = output_root / f"seed{seed}"
    features_dir = split_dir / "features"
    oof_dir = split_dir / "oof"
    trust_dir = split_dir / "trust"
    steps = [str(step) for step in manifest.get("steps", ["split", "features", "folds", "oof", "trust", "final_train"])]
    unknown = set(steps) - {
        "split", "features", "folds", "oof", "trust", "final_train",
    }
    if unknown:
        raise ValueError(f"Unknown pipeline steps: {sorted(unknown)}")

    output_root.mkdir(parents=True, exist_ok=True)
    run_record: dict[str, Any] = {
        "format_version": 1,
        "manifest": str(Path(manifest_path).resolve()),
        "stage": stage,
        "seed": seed,
        "steps": steps,
        "step_results": {},
        "completed": [],
    }

    def record(step: str, **details: Any) -> None:
        run_record["step_results"][step] = details
        run_record["completed"].append(step)

    if "split" in steps:
        result = prepare_stage(
            train_root=train_root,
            output_dir=split_dir,
            stage=stage,
            seed=seed,
            val_ratio=val_ratio,
            expected_classes=expected_classes,
            expected_samples=expected_samples,
            hash_workers=int(manifest.get("hash_workers", 8)),
        )
        record("split", manifest=result)

    features_manifest = None
    if "features" in steps:
        features_manifest = cache_stage_features(
            {
                "data": {
                    "train_root": str(train_root),
                    "expected_official_train_samples": expected_samples,
                },
                "model": {"num_classes": expected_classes},
                "project": {"stage": stage},
            },
            device=device,
            batch_size=int(manifest.get("feature_batch_size", 256)),
            workers=int(manifest.get("feature_workers", 8)),
            output_dir=features_dir,
            overwrite=True,
        )
        record(
            "features",
            output_dir=str(features_dir),
            manifest=features_manifest,
            features_sha256=sha256_file(features_dir / "features.pt"),
        )

    assignments_csv = split_dir / "oof_assignments.csv"
    if "folds" in steps:
        assign_oof_folds(
            split_dir / "train.csv",
            split_dir / "val.csv",
            split_dir / "content_groups.json",
            assignments_csv,
            folds=folds,
            seed=seed,
            root_name=train_root.name,
        )
        record("folds", assignments=str(assignments_csv), folds=folds)

    if "oof" in steps:
        if not features_dir.exists():
            raise RuntimeError("oof step requires the features step")
        inputs = load_oof_inputs(
            assignments_csv,
            features_dir / "features.pt",
            features_dir / "image_paths.json",
            features_dir / "labels.json",
        )
        oof_config = manifest.get("oof", {})
        hashes = {
            "assignments_sha256": sha256_file(assignments_csv),
            "feature_tensor_sha256": sha256_file(features_dir / "features.pt"),
            "feature_paths_sha256": sha256_file(features_dir / "image_paths.json"),
            "feature_labels_sha256": sha256_file(features_dir / "labels.json"),
        }
        result = rebuild_oof_logits(
            inputs,
            oof_dir,
            num_classes=expected_classes,
            epochs=int(oof_config.get("epochs", 50)),
            batch_size=int(oof_config.get("batch_size", 128)),
            infer_batch_size=int(oof_config.get("infer_batch_size", 1024)),
            lr=float(oof_config.get("lr", 0.005)),
            weight_decay=float(oof_config.get("weight_decay", 1.0e-4)),
            warmup_epochs=int(oof_config.get("warmup_epochs", 2)),
            q=float(oof_config.get("q", 0.5)),
            seed=seed,
            device=device,
            input_hashes=hashes,
        )
        record("oof", result=result, output_dir=str(oof_dir))

    if "trust" in steps:
        if not features_dir.exists():
            raise RuntimeError("trust step requires the features step")
        features = torch.load(
            features_dir / "features.pt", map_location="cpu", weights_only=True
        )
        paths = json.loads((features_dir / "image_paths.json").read_text(encoding="utf-8"))
        labels = torch.tensor(
            json.loads((features_dir / "labels.json").read_text(encoding="utf-8")),
            dtype=torch.long,
        )
        groups_path = split_dir / "content_groups.json"
        group_mapping = json.loads(groups_path.read_text(encoding="utf-8"))
        canonical = [_canonical_group_key(path, train_root.name) for path in paths]
        group_keys = [str(group_mapping[path]) for path in canonical]
        trust_config = TrustBuildConfig(
            **{
                key: value
                for key, value in manifest.get("trust", {}).items()
                if key
                in {
                    "folds", "seed", "prototype_temperature", "probe_temperature",
                    "probe_epochs", "probe_batch_size", "probe_lr",
                    "probe_weight_decay", "correction_confidence",
                    "correction_confidence_quantile", "maximum_correction_alpha",
                    "maximum_class_correction_rate", "early_cut_strength",
                    "minimum_clean_probability",
                }
            }
        )
        bundle, summary = build_cross_fitted_trust(
            features,
            labels,
            paths,
            expected_classes,
            groups=group_keys,
            config=trust_config,
            device=str(device),
        )
        trust_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = trust_dir / "trust.pt"
        atomic_torch_save(bundle, bundle_path)
        atomic_json_dump(summary, trust_dir / "trust.summary.json")
        record(
            "trust",
            bundle=str(bundle_path),
            summary=summary,
            bundle_sha256=sha256_file(bundle_path),
        )

    if "final_train" in steps:
        expected_merge = int(manifest["expected_samples"])
        merged = merge_splits(
            split_dir / "train.csv",
            split_dir / "val.csv",
            split_dir / "final_train.csv",
            expected_samples=expected_merge,
        )
        record("final_train", output=str(merged))

    run_path = output_root / "pipeline_run.json"
    atomic_json_dump(run_record, run_path)
    return run_path
