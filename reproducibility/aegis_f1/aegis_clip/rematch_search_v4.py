"""Versioned fail-closed protocol for REMATCH750_SEARCH_V4.

The existing :mod:`aegis_clip.rematch_assets` protocol intentionally froze the
first rematch recipe (224px, trust disabled, no reweighting).  V4 broadens the
explicitly declared search axes without weakening the provenance rules that
make the older baseline comparable.  This module therefore keeps every check
that concerns data identity, official weights, split lineage and test usage,
while only relaxing recipe knobs that are named by an explicit V4 trial.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from aegis_clip.runtime import sha256_file, sha256_lines


PROTOCOL = "rematch750_search_v4"
PROTOCOLS = (
    "rematch750_search_v4",
    "rematch750_search_v5",
    "rematch750_head_l2sp",
    "rematch750_f05_transfer",
    "rematch750_decay_filter",
)
ALLOWED_RESOLUTIONS = (224, 256, 288, 320)
V5_ALLOWED_RESOLUTIONS = (224, 320, 352, 384, 416, 448)
PARENT_KINDS = {
    "shared_lp",
    "same_split_continue",
    "frozen_backbone_head",
    "official_clip_head",
}
V4_TOP_LEVEL = {"project", "data", "features", "model", "trust", "elr", "loss", "longtail",
                "train", "evaluation", "output", "lineage", "promotion", "clean_routing",
                "prototype_contrastive", "dynamic_trust", "diagnostics"}


def rematch_protocol(config: dict[str, Any]) -> str:
    """Return the declared rematch-search protocol, if any."""
    return str(config.get("project", {}).get("protocol", ""))


def is_rematch_search(config: dict[str, Any]) -> bool:
    """Return true for a protocol handled by this module (V4 or V5)."""
    return rematch_protocol(config) in PROTOCOLS


def is_v4(config: dict[str, Any]) -> bool:
    """Backward-compatible helper used by the V4 protocol dispatcher.

    V5 shares this implementation for dataset/cache/checkpoint provenance;
    V5-only semantic declarations are checked in :mod:`rematch_search_v5`.
    """
    return is_rematch_search(config)


def is_v5(config: dict[str, Any]) -> bool:
    return rematch_protocol(config) == "rematch750_search_v5"


def _allowed_resolutions(config: dict[str, Any]) -> tuple[int, ...]:
    return V5_ALLOWED_RESOLUTIONS if is_v5(config) else ALLOWED_RESOLUTIONS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"REMATCH750_SEARCH_V4 provenance: {message}")


def official_weight_hash(config: dict[str, Any]) -> str:
    """Validate and return the OpenAI CLIP ViT-B/32 SHA-256.

    This delegates to the existing implementation so V4 cannot drift from the
    hash used by RM-LP, V3 and all previous checkpoints.
    """
    from aegis_clip.rematch_assets import official_weight_hash as legacy

    return legacy(config)


def _manifest(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(config["data"]["dataset_manifest"])
    require(path.is_file(), f"dataset manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return path, manifest


def validate_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the immutable V4 data/split/official-weight identity."""
    require(is_rematch_search(config), "project.protocol is not a supported rematch search protocol")
    project = config["project"]
    data = config["data"]
    model = config["model"]
    require(project.get("stage") == "repechage", "wrong stage")
    require(
        model.get("backbone") == "ViT-B/32"
        and model.get("pretrained") == "openai",
        "wrong pretrained source",
    )
    resolution = int(model.get("input_resolution", 224))
    allowed_resolutions = _allowed_resolutions(config)
    require(
        resolution in allowed_resolutions,
        f"input resolution must be one of {allowed_resolutions}",
    )
    require(data.get("external_data") is False, "external data forbidden")
    require(
        data.get("test_usage") == "inference_only",
        "test data may only be used for inference",
    )

    path, manifest = _manifest(config)
    require(manifest.get("stage") == "repechage", "dataset belongs to another stage")
    require(
        manifest.get("data_version") == project.get("data_version"),
        "wrong data version",
    )
    require(
        int(manifest.get("num_classes", 0)) == int(model.get("num_classes", 0)),
        "class count mismatch",
    )
    require(
        int(manifest.get("train_samples", -1))
        == int(data.get("expected_official_train_samples", -2)),
        "train size mismatch",
    )
    require(
        int(manifest.get("test_samples", -1))
        == int(data.get("expected_test_samples", -2)),
        "test size mismatch",
    )
    for key in ("train_root", "test_root"):
        require(
            Path(config["data"][key]).resolve() == Path(manifest[key]).resolve(),
            f"{key} changed",
        )
    for name, digest in manifest.get("files", {}).items():
        require(
            sha256_file(path.parent / name) == digest,
            f"dataset asset changed: {name}",
        )
    decode = json.loads((path.parent / "decode_report.json").read_text(encoding="utf-8"))
    require(
        decode.get("status") == "passed" and not decode.get("failures"),
        "decode audit failed",
    )
    require(
        Path(config["data"]["class_mapping"]).resolve()
        == (path.parent / "class_to_idx.json").resolve(),
        "foreign class mapping",
    )
    full = bool(project.get("full_training", False))
    expected_train = "full_train.csv" if full else "train_dev.csv"
    for key, name in (("train_csv", expected_train), ("val_csv", "val_dev.csv")):
        require(
            Path(config["data"][key]).resolve() == (path.parent / name).resolve(),
            f"foreign {key}",
        )
    require(
        bool(data.get("validation_overlap_with_training", False)) == full,
        "overlap declaration mismatch",
    )
    official_weight_hash(config)

    parent_kind = str(project.get("parent_kind", ""))
    if parent_kind:
        require(
            parent_kind in PARENT_KINDS,
            f"unknown parent_kind: {parent_kind}",
        )
    return manifest


def _resolved_reference_resolution(config: dict[str, Any]) -> int:
    """Resolution of the frozen feature tensor used by the feature anchor."""
    search = config.get("project", {}).get("search", {})
    value = search.get("reference_resolution", config.get("features", {}).get("reference_resolution", 224))
    value = int(value)
    require(
        value in _allowed_resolutions(config),
        "reference feature resolution is not declared",
    )
    return value


def _common_binding(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": rematch_protocol(config),
        "stage": "repechage",
        "data_version": manifest["data_version"],
        "dataset_manifest_sha256": sha256_file(config["data"]["dataset_manifest"]),
        "dataset_fingerprint": manifest["train_fingerprint"],
        "class_mapping_sha256": sha256_file(config["data"]["class_mapping"]),
        "official_checkpoint_sha256": official_weight_hash(config),
        "train_csv_sha256": sha256_file(config["data"]["train_csv"]),
        "encoder_precision": "float32",
        "feature_precision": "float32",
        "autocast": False,
    }


def expected_binding(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the checkpoint binding for a V4 trained model."""
    binding = _common_binding(config, manifest)
    resolution = int(config["model"].get("input_resolution", 224))
    binding.update(
        {
            "input_resolution": resolution,
            "reference_resolution": _resolved_reference_resolution(config),
            "train_augmentation": str(
                config["data"].get("train_augmentation", "clip_center_crop")
            ),
            "preprocessing": "OpenAI CLIP resize/center-crop/RGB/CLIP normalization"
            + f" with input_resolution={resolution}",
        }
    )
    return binding


def expected_feature_binding(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the feature-cache binding for a V4 reference anchor."""
    resolution = _resolved_reference_resolution(config)
    binding = _common_binding(config, manifest)
    binding.update(
        {
            "input_resolution": resolution,
            "reference_resolution": resolution,
            "train_augmentation": "feature_cache_fixed_center_crop",
            "preprocessing": "OpenAI CLIP resize/center-crop/RGB/CLIP normalization"
            + f" with input_resolution={resolution}",
        }
    )
    return binding


def _legacy_feature_binding(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Feature-cache binding used by RM-LP and V3 assets already on disk."""
    return {
        "stage": "repechage",
        "data_version": manifest["data_version"],
        "dataset_manifest_sha256": sha256_file(config["data"]["dataset_manifest"]),
        "dataset_fingerprint": manifest["train_fingerprint"],
        "class_mapping_sha256": sha256_file(config["data"]["class_mapping"]),
        "official_checkpoint_sha256": official_weight_hash(config),
        "preprocessing": "OpenAI CLIP 224 bicubic resize/center crop/RGB/CLIP normalization",
        "encoder_precision": "float32",
        "feature_precision": "float32",
        "autocast": False,
    }


def _legacy_checkpoint_binding(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Checkpoint binding used by RM-LP/V3 checkpoints already on disk."""
    return {
        "stage": "repechage",
        "data_version": manifest["data_version"],
        "dataset_manifest_sha256": sha256_file(config["data"]["dataset_manifest"]),
        "class_mapping_sha256": sha256_file(config["data"]["class_mapping"]),
        "train_csv_sha256": sha256_file(config["data"]["train_csv"]),
        "official_checkpoint_sha256": official_weight_hash(config),
        "feature_manifest_sha256": sha256_file(config["features"]["manifest_path"]),
    }


def validate_cache(config: dict[str, Any], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate the frozen feature cache declared as the V4 reference anchor."""
    manifest = manifest or validate_dataset(config)
    features = config["features"]
    base = Path(features["manifest_path"]).resolve().parent
    for key, name in (
        ("tensor_path", "features.pt"),
        ("paths_path", "image_paths.json"),
        ("manifest_path", "manifest.json"),
    ):
        require(
            Path(features[key]).resolve() == (base / name).resolve(),
            f"foreign feature {key}",
        )
    feature_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    expected_legacy = _legacy_feature_binding(config, manifest)
    expected_v4 = expected_feature_binding(config, manifest)
    require(
        feature_manifest.get("rematch_binding") in (expected_legacy, expected_v4),
        "feature provenance mismatch",
    )
    require(
        sha256_file(base / "features.pt") == feature_manifest["tensor_sha256"],
        "feature tensor changed",
    )
    require(
        sha256_file(base / "image_paths.json") == feature_manifest["paths_file_sha256"],
        "feature paths changed",
    )
    dataset_root = Path(config["data"]["dataset_manifest"]).resolve().parent
    with (dataset_root / "full_train.csv").open(encoding="utf-8") as handle:
        paths = [row["image_path"].removeprefix("train/") for row in csv.DictReader(handle)]
    require(
        json.loads((base / "image_paths.json").read_text(encoding="utf-8")) == paths,
        "feature sample order mismatch",
    )
    require(
        feature_manifest.get("path_index_sha256") == sha256_lines(paths),
        "feature order hash mismatch",
    )
    return feature_manifest


def checkpoint_binding(config: dict[str, Any]) -> dict[str, Any]:
    manifest = validate_dataset(config)
    return expected_binding(config, manifest)


def validate_checkpoint(
    path: str | Path,
    config: dict[str, Any],
    *,
    parent: bool = False,
) -> dict[str, Any]:
    """Validate a V4 child checkpoint or a declared V4 parent checkpoint."""
    path = Path(path)
    binding_path = path.with_suffix(".binding.json")
    require(binding_path.is_file(), f"checkpoint binding missing: {binding_path}")
    meta = json.loads(binding_path.read_text(encoding="utf-8"))
    require(meta.get("checkpoint_sha256") == sha256_file(path), "checkpoint hash mismatch")
    manifest = validate_dataset(config)

    if parent:
        expected_legacy = _legacy_checkpoint_binding(config, manifest)
        expected_v4 = expected_binding(config, manifest)
        kind = str(config["project"].get("parent_kind", ""))
        if kind == "frozen_backbone_head":
            # The child head refit uses a separately extracted V3 feature cache.
            # Require the immutable split/data identity but allow the parent
            # checkpoint to retain its historical feature-manifest binding.
            shared_keys = (
                "stage",
                "data_version",
                "dataset_manifest_sha256",
                "class_mapping_sha256",
                "train_csv_sha256",
                "official_checkpoint_sha256",
            )
            current = meta.get("binding", {})
            require(
                all(current.get(key) == expected_legacy.get(key) for key in shared_keys),
                "parent checkpoint data lineage mismatch",
            )
        else:
            require(
                meta.get("binding") in (expected_legacy, expected_v4),
                "parent checkpoint data lineage mismatch",
            )
        experiment_id = str(meta.get("experiment_id", ""))
        if kind == "shared_lp":
            require(experiment_id == "RM_LP", "parent_kind shared_lp requires RM_LP")
        elif kind in {"same_split_continue", "frozen_backbone_head"}:
            declared = str(config["project"].get("parent_experiment_id", ""))
            require(declared, "parent_experiment_id is required for this parent kind")
            require(
                experiment_id == declared,
                "parent experiment_id does not match declaration",
            )
        elif kind == "official_clip_head":
            raise ValueError("official_clip_head does not accept a parent checkpoint")
        return meta

    require(
        meta.get("binding") == expected_binding(config, manifest),
        "checkpoint binding does not match resolved V4 config",
    )
    require(
        meta.get("training_config_sha256") == sha256_file(config["_config_path"]),
        "checkpoint training config mismatch",
    )
    return meta


def validate_training(
    config: dict[str, Any],
    resume: str | None = None,
    init_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Fail closed on lineage before any V4 model or data loading."""
    manifest = validate_dataset(config)
    validate_cache(config, manifest)
    if resume:
        validate_checkpoint(resume, config, parent=False)
    source = init_checkpoint or config.get("train", {}).get("init_checkpoint")
    kind = str(config["project"].get("parent_kind", ""))
    if source:
        validate_checkpoint(source, config, parent=True)
    elif kind not in {"official_clip_head", "frozen_backbone_head"} and config["model"].get("peft_mode") != "frozen":
        raise ValueError(
            "V4 visual fine-tuning must declare a shared LP / same-split parent"
        )
    if kind == "frozen_backbone_head" and config["model"].get("peft_mode") != "frozen":
        raise ValueError("frozen_backbone_head requires model.peft_mode=frozen")
    return manifest
