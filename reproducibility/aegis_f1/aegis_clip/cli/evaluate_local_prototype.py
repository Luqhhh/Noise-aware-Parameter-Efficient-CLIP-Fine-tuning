"""Evaluate the single preregistered N2 trust-weighted local prototype head."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from aegis_clip.balanced_inference import prediction_metrics
from aegis_clip.local_prototype import (
    local_prototype_logits,
    mean_global_prototype_logits,
    trust_weighted_local_prototype_weight,
)
from aegis_clip.local_residual import validate_dual_view_cache
from aegis_clip.runtime import atomic_json_dump, sha256_file


def _metrics(logits: torch.Tensor, cache: dict[str, object]) -> dict[str, float | int]:
    return prediction_metrics(
        logits.argmax(dim=1),
        labels=cache["labels"],
        clean_probability=cache["clean_probability"],
        pseudo_labels=cache["pseudo_labels"],
        correction_alpha=cache["correction_alpha"],
        num_classes=500,
        clean_core_threshold=0.70,
    )


def evaluate_local_prototype(
    train_cache_path: str | Path,
    validation_cache_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    train_cache_path = Path(train_cache_path).resolve()
    validation_cache_path = Path(validation_cache_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    train_cache = torch.load(train_cache_path, map_location="cpu", weights_only=False)
    validation_cache = torch.load(
        validation_cache_path, map_location="cpu", weights_only=False
    )
    validate_dual_view_cache(train_cache, expected_feature_dim=512, expected_num_classes=500)
    validate_dual_view_cache(
        validation_cache, expected_feature_dim=512, expected_num_classes=500
    )
    if set(train_cache["paths"]) & set(validation_cache["paths"]):
        raise ValueError("N2 train and validation caches overlap")
    if train_cache["checkpoint_sha256"] != validation_cache["checkpoint_sha256"]:
        raise ValueError("N2 caches were built from different checkpoints")
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != train_cache["checkpoint_sha256"]:
        raise ValueError("N2 checkpoint does not match the dual-view caches")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    base_weight = state["classifier.weight"].float().cpu()
    base_bias = state["classifier.bias"].float().cpu()
    prototype_weight, class_trust_mass = trust_weighted_local_prototype_weight(
        train_cache["local_features"],
        train_cache["labels"],
        train_cache["clean_probability"],
        base_weight,
    )
    prototype_logits = local_prototype_logits(
        validation_cache["local_features"], prototype_weight, base_bias
    )
    fused_logits = mean_global_prototype_logits(
        validation_cache["global_logits"], prototype_logits
    )
    prototype_norm = F_normalize_norm(prototype_weight, base_weight)
    report = {
        "status": "evaluated",
        "protocol": {
            "experiment": "N2_TRUST_WEIGHTED_LOCAL_CLASS_PROTOTYPE",
            "clean_threshold": 0.70,
            "fusion": "mean_logits_1_to_1",
            "prototype_weighting": "clean_probability",
            "prototype_scale": "per_class_A2_classifier_weight_norm",
            "prototype_bias": "A2_classifier_bias",
            "parameter_scan": False,
            "test_data_used": False,
            "external_data_used": False,
        },
        "audit": {
            "train_samples": len(train_cache["paths"]),
            "validation_samples": len(validation_cache["paths"]),
            "path_overlap": 0,
            "classes": int(class_trust_mass.numel()),
            "minimum_class_trust_mass": float(class_trust_mass.min()),
            "maximum_class_trust_mass": float(class_trust_mass.max()),
            **prototype_norm,
        },
        "global_A2": _metrics(validation_cache["global_logits"], validation_cache),
        "local_prototype_only": _metrics(prototype_logits, validation_cache),
        "N2_mean_logits": _metrics(fused_logits, validation_cache),
        "lineage": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "train_cache": str(train_cache_path),
            "train_cache_sha256": sha256_file(train_cache_path),
            "validation_cache": str(validation_cache_path),
            "validation_cache_sha256": sha256_file(validation_cache_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation.json"
    artifact_path = output_dir / "evaluation.pt"
    atomic_json_dump(report, report_path)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "prototype_weight": prototype_weight,
            "classifier_bias": base_bias,
            "class_trust_mass": class_trust_mass,
            "paths": list(validation_cache["paths"]),
            "labels": validation_cache["labels"],
            "local_prototype_logits": prototype_logits,
            "N2_logits": fused_logits,
            "report": report,
        },
        temporary,
    )
    os.replace(temporary, artifact_path)
    return report_path, artifact_path


def F_normalize_norm(
    prototype_weight: torch.Tensor, base_weight: torch.Tensor
) -> dict[str, float]:
    prototype_direction_norm = torch.nn.functional.normalize(
        prototype_weight.float(), dim=1
    ).norm(dim=1)
    norm_difference = (
        prototype_weight.float().norm(dim=1) - base_weight.float().norm(dim=1)
    ).abs()
    return {
        "prototype_direction_norm_minimum": float(prototype_direction_norm.min()),
        "prototype_direction_norm_maximum": float(prototype_direction_norm.max()),
        "maximum_classifier_norm_difference": float(norm_difference.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for path in evaluate_local_prototype(
        args.train_cache, args.validation_cache, args.checkpoint, args.output_dir
    ):
        print(path)


if __name__ == "__main__":
    main()
