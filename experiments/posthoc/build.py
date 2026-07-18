"""Build one auditable checkpoint containing a soup head and visual prototypes.

Usage:
    python -m experiments.posthoc.build --config configs/e20_posthoc.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from common.posthoc import (
    assert_non_classifier_state_equal,
    fit_weighted_multiprototypes,
    interpolate_linear_heads,
)
from common.utils import load_config


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_key(value: str) -> str:
    parts = [part for part in str(value).replace("\\", "/").split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"image path has fewer than two components: {value}")
    return "/".join(parts[-2:])


def _load_feature_cache(directory: str | Path) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    directory = Path(directory)
    features = torch.load(directory / "features.pt", map_location="cpu", weights_only=True)
    with open(directory / "image_paths.json", "r", encoding="utf-8") as handle:
        paths = json.load(handle)
    with open(directory / "labels.json", "r", encoding="utf-8") as handle:
        labels = torch.as_tensor(json.load(handle), dtype=torch.long)
    if features.ndim != 2 or features.shape[0] != len(paths) or labels.numel() != len(paths):
        raise ValueError(f"invalid or misaligned feature cache: {directory}")
    return features.float(), [_path_key(path) for path in paths], labels


def _select_fit_rows(
    features: torch.Tensor,
    paths: list[str],
    cached_labels: torch.Tensor,
    fit_csv: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    index = {path: position for position, path in enumerate(paths)}
    if len(index) != len(paths):
        raise ValueError("feature cache contains duplicate canonical image paths")
    frame = pd.read_csv(fit_csv)
    if not {"image_path", "label"}.issubset(frame.columns):
        raise ValueError("fit CSV must contain image_path and label columns")
    positions = []
    labels = []
    missing = []
    for path, label in zip(frame["image_path"], frame["label"]):
        key = _path_key(str(path))
        if key not in index:
            missing.append(key)
            continue
        position = index[key]
        label = int(label)
        if int(cached_labels[position]) != label:
            raise ValueError(f"label mismatch for {key}")
        positions.append(position)
        labels.append(label)
    if missing:
        raise ValueError(
            f"fit CSV has {len(missing)} paths absent from cache; first={missing[0]}"
        )
    return features[positions], torch.as_tensor(labels, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    recipe = config["posthoc"]

    first_path = Path(recipe["first_checkpoint"])
    first = torch.load(first_path, map_location="cpu", weights_only=False)
    state = first["model_state_dict"]
    second_path = recipe.get("second_checkpoint")
    if second_path:
        second = torch.load(second_path, map_location="cpu", weights_only=False)
        second_state = second["model_state_dict"]
        assert_non_classifier_state_equal(state, second_state)
        alpha = float(recipe["soup_alpha"])
        weight, bias = interpolate_linear_heads(
            state["classifier.weight"].float(),
            state["classifier.bias"].float(),
            second_state["classifier.weight"].float(),
            second_state["classifier.bias"].float(),
            alpha=alpha,
        )
        state["classifier.weight"] = weight
        state["classifier.bias"] = bias
        first["linear_soup"] = {
            "first_checkpoint": str(first_path.resolve()),
            "first_checkpoint_sha256": _sha256(first_path),
            "second_checkpoint": str(Path(second_path).resolve()),
            "second_checkpoint_sha256": _sha256(second_path),
            "alpha": alpha,
            "single_backbone_verified": True,
        }

    original, original_paths, original_labels = _load_feature_cache(
        recipe["feature_cache_dir"]
    )
    flipped, flipped_paths, flipped_labels = _load_feature_cache(
        recipe["flipped_feature_cache_dir"]
    )
    if original_paths != flipped_paths or not torch.equal(original_labels, flipped_labels):
        raise ValueError("original and horizontal-flip feature caches are not paired")
    fit_features, fit_labels = _select_fit_rows(
        F.normalize((original + flipped) / 2.0, dim=1),
        original_paths,
        original_labels,
        recipe["fit_csv"],
    )
    prototypes = fit_weighted_multiprototypes(
        fit_features,
        fit_labels,
        torch.ones(fit_labels.numel()),
        num_classes=int(config["model"]["num_classes"]),
        prototypes_per_class=int(recipe["prototypes_per_class"]),
        random_state=int(recipe.get("random_state", 42)),
    )
    first["multiprototype_head"] = {
        "prototypes": prototypes,
        "prototypes_per_class": int(recipe["prototypes_per_class"]),
        "aggregation": str(recipe["aggregation"]),
        "softmax_temperature": float(recipe.get("softmax_temperature", 0.05)),
        "alpha": float(recipe["prototype_alpha"]),
        "candidate_scale": float(recipe["candidate_scale"]),
        "trust_power": 0.0,
        "fit_representation": "mean_original_horizontal_flip",
        "fit_csv": str(Path(recipe["fit_csv"]).resolve()),
        "fit_csv_sha256": _sha256(recipe["fit_csv"]),
        "single_checkpoint": True,
    }

    output = Path(recipe["output_checkpoint"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(first, temporary)
    temporary.replace(output)
    report = {
        "experiment_id": config["experiment"]["id"],
        "output_checkpoint": str(output.resolve()),
        "output_checkpoint_sha256": _sha256(output),
        "prototype_shape": list(prototypes.shape),
        "source_checkpoint": str(first_path.resolve()),
        "source_checkpoint_sha256": _sha256(first_path),
        "linear_soup": first.get("linear_soup"),
        "multiprototype": {
            key: value
            for key, value in first["multiprototype_head"].items()
            if key != "prototypes"
        },
    }
    report_path = output.with_suffix(".build.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
