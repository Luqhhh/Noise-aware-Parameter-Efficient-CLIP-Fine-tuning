"""Audit canonical numeric class prompts against train-only CLIP evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import clip
import torch
import torch.nn.functional as F

from aegis_clip.data import TrustBundle
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.prompt_audit import numeric_prompt_diagnostics
from aegis_clip.runtime import atomic_json_dump, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--feature-tensor", required=True)
    parser.add_argument("--feature-paths", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--trust-bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--template", default="a photo of a {class_id}")
    parser.add_argument("--clean-core-threshold", type=float, default=0.70)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "validation_csv": args.validation_csv,
            "feature_tensor": args.feature_tensor,
            "feature_paths": args.feature_paths,
            "feature_manifest": args.feature_manifest,
            "trust_bundle": args.trust_bundle,
            "checkpoint": args.checkpoint,
        }.items()
    }
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    classifier = checkpoint["model_state_dict"]["classifier.weight"].float()
    num_classes = int(classifier.shape[0])
    prompts = [
        args.template.format(class_id=f"{index:04d}")
        for index in range(num_classes)
    ]
    model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
    model.eval()
    with torch.no_grad():
        text = F.normalize(
            model.encode_text(clip.tokenize(prompts)).float(), dim=1
        )

    feature_store = FrozenFeatureStore(
        tensor_path=paths["feature_tensor"],
        paths_path=paths["feature_paths"],
        manifest_path=paths["feature_manifest"],
        expected_dim=int(classifier.shape[1]),
    )
    trust = TrustBundle(paths["trust_bundle"])
    with paths["validation_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sample_paths = [row["image_path"] for row in rows]
    labels = torch.tensor([int(row["label"]) for row in rows])
    clean = torch.stack(
        [
            trust.values_for(path, int(label))["clean_probability"]
            for path, label in zip(sample_paths, labels)
        ]
    ).float()
    report = numeric_prompt_diagnostics(
        text_features=text,
        image_features=feature_store.get_many(sample_paths),
        labels=labels,
        clean_probability=clean,
        classifier_weights=classifier,
        clean_core_threshold=float(args.clean_core_threshold),
    )
    report.update(
        {
            "status": "analyzed",
            "test_data_used": False,
            "backbone": "ViT-B/32",
            "pretrained": "openai",
            "prompt_template": args.template,
            "first_prompt": prompts[0],
            "last_prompt": prompts[-1],
            "lineage": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        }
    )
    destination = Path(args.output).resolve()
    atomic_json_dump(report, destination)
    print(destination)


if __name__ == "__main__":
    main()
