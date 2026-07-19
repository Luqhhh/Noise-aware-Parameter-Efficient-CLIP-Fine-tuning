"""Build split-isolated cross-fitted trust for train and validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import pandas as pd
import torch

from aegis_clip.config import load_config
from aegis_clip.features import FrozenFeatureStore, canonical_sample_path
from aegis_clip.runtime import atomic_json_dump
from aegis_clip.trust import (
    TrustBuildConfig,
    atomic_torch_save,
    build_cross_fitted_trust,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    feature_config = config["features"]
    trust_config = config.get("trust", {})

    train_frame = pd.read_csv(data["train_csv"])
    val_frame = pd.read_csv(data["val_csv"])
    frame = pd.concat([train_frame, val_frame], ignore_index=True)
    paths = [canonical_sample_path(path) for path in frame["image_path"].astype(str)]
    if len(paths) != len(set(paths)):
        raise ValueError("Combined train/validation paths are not unique")
    store = FrozenFeatureStore(
        feature_config["tensor_path"],
        feature_config["paths_path"],
        feature_config.get("manifest_path"),
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    group_mapping = None
    groups_path = trust_config.get("groups_path")
    if groups_path:
        if not Path(groups_path).is_file():
            raise FileNotFoundError(
                f"Configured trust.groups_path does not exist: {groups_path}"
            )
        group_mapping = json.loads(Path(groups_path).read_text(encoding="utf-8"))
        missing_groups = [path for path in paths if path not in group_mapping]
        if missing_groups:
            raise ValueError(
                f"Content groups miss {len(missing_groups)} samples; "
                f"first={missing_groups[0]}"
            )

    valid_names = {field.name for field in fields(TrustBuildConfig)}
    build_values = {
        key: value
        for key, value in trust_config.get("build", {}).items()
        if key in valid_names
    }
    build_config = TrustBuildConfig(**build_values)
    train_bundle, train_summary = _build_split(
        train_frame, store, group_mapping, config, build_config, args.device
    )
    val_bundle, val_summary = _build_split(
        val_frame, store, group_mapping, config, build_config, args.device
    )
    bundle = _combine_bundles(train_bundle, val_bundle)
    summary = {
        "method": "split_isolated_cross_fitted_visual_trust_v1",
        "samples": len(bundle["paths"]),
        "train": train_summary,
        "validation": val_summary,
    }
    output = Path(trust_config["bundle_path"])
    atomic_torch_save(bundle, output)
    atomic_json_dump(summary, output.with_suffix(".summary.json"))

    labels = torch.cat(
        [
            torch.tensor(train_frame["label"].astype(int).to_numpy(copy=True)),
            torch.tensor(val_frame["label"].astype(int).to_numpy(copy=True)),
        ]
    ).long()
    diagnostics = bundle["diagnostics"]
    diagnostic_frame = pd.DataFrame(
        {
            "path": paths,
            "noisy_label": labels.numpy(),
            "fold": diagnostics["fold_id"].numpy(),
            "clean_probability": bundle["clean_probability"].numpy(),
            "prototype_label_probability": diagnostics[
                "prototype_label_probability"
            ].numpy(),
            "probe_label_probability": diagnostics[
                "probe_label_probability"
            ].numpy(),
            "prototype_top1": diagnostics["prototype_top1"].numpy(),
            "probe_top1": diagnostics["probe_top1"].numpy(),
            "prototype_confidence": diagnostics["prototype_confidence"].numpy(),
            "probe_confidence": diagnostics["probe_confidence"].numpy(),
            "pseudo_label": bundle["pseudo_label"].numpy(),
            "correction_alpha": bundle["correction_alpha"].numpy(),
            "mislabeled_easy": diagnostics["mislabeled_easy"].numpy(),
        }
    )
    diagnostic_frame.to_csv(output.with_suffix(".diagnostics.csv"), index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _build_split(
    frame: pd.DataFrame,
    store: FrozenFeatureStore,
    group_mapping: dict[str, str] | None,
    config: dict,
    build_config: TrustBuildConfig,
    device: str,
) -> tuple[dict, dict]:
    paths = [canonical_sample_path(path) for path in frame["image_path"].astype(str)]
    labels = torch.tensor(
        frame["label"].astype(int).to_numpy(copy=True), dtype=torch.long
    )
    groups = [group_mapping[path] for path in paths] if group_mapping else None
    return build_cross_fitted_trust(
        store.get_many(paths),
        labels,
        paths,
        num_classes=int(config["model"]["num_classes"]),
        groups=groups,
        config=build_config,
        device=device,
    )


def _combine_bundles(train: dict, validation: dict) -> dict:
    vector_keys = (
        "clean_probability",
        "pseudo_label",
        "pseudo_confidence",
        "correction_alpha",
    )
    diagnostic_keys = tuple(train["diagnostics"])
    return {
        "paths": list(train["paths"]) + list(validation["paths"]),
        **{
            key: torch.cat([train[key], validation[key]], dim=0)
            for key in vector_keys
        },
        "metadata": {
            "method": "split_isolated_cross_fitted_visual_trust_v1",
            "train": train["metadata"],
            "validation": validation["metadata"],
        },
        "diagnostics": {
            key: torch.cat(
                [train["diagnostics"][key], validation["diagnostics"][key]], dim=0
            )
            for key in diagnostic_keys
        },
    }


if __name__ == "__main__":
    main()
