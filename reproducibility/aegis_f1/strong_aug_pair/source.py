"""Preserve original artifact bytes; verify a separately bound local location map."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from aegis_clip.runtime import sha256_file
from aegis_clip.rematch_assets import validate_dataset, validate_cache, official_weight_hash
from aegis_clip.v1_pipeline import fingerprint, load_artifact, read_rows, validate_rows
from aegis_clip.v1_strategy import load_trainable_state
from v3.plan import normalize_targets
from .core import require

ROOT = Path(__file__).resolve().parents[3]
PARENT_FILES = ("v1_strategy.py", "v1_pipeline.py", "model.py", "submission.py")


def lf_digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_protocol(protocol):
    frozen = json.loads((ROOT / "configs/team_exploration_20261001/machine_b.json").read_text())
    require(protocol == frozen, "Prepared protocol changed")
    required = dict(epochs_per_arm=4, logical_batch_size=32, micro_batch_size=8, num_workers=2,
                    seed=42, image_size=448, lora_lr=5e-5, head_lr=2.5e-4, optimizer="AdamW",
                    lora_weight_decay=0., head_weight_decay=.01, ema_decay=.999, grad_clip=1.,
                    label_smoothing=.1, mixup_alpha=.2, primary_export="ema_swa_2_4",
                    platform_upload=False, remote_execution=False, spawn_subagents=False,
                    wall_clock_time_limits=False)
    require(all(protocol.get(k) == v for k, v in required.items()), "Only the fixed B recipe is supported")
    require(protocol["scheduler"] == dict(name="OneCycleLR", pct_start=.1, anneal_strategy="cos"),
            "Scheduler changed")
    require(protocol["augmentations"] == dict(
        control=dict(crop_min=.8, horizontal_flip=True, randaugment=[2, 7]),
        candidate=dict(crop_min=.35, horizontal_flip=True,
            color_jitter=dict(brightness=.5, contrast=.5, saturation=.5, hue=.125),
            randaugment=[2, 7], random_erasing_probability=.3, cutmix=False, random_erasing_value="random")),
        "Augmentation recipe changed")
    require(protocol["test_decoder"] == dict(scales=[448, 512, 576], flip=True, bias=None,
                                             reduction="sum_logits"), "Decoder changed")
    require(protocol["review_gate"] == dict(net_vs_control_min=75, net_vs_parent_min=75,
        target_net_vs_control_min=25, corrections_to_regressions_min=1.25, automatic_full_training=False)
        and protocol["stop_if_micro_pp_vs_parent_below"] == -2
        and protocol["cost_probe"]["updates"] == 5
        and protocol["target_group"]["minimum_parent_target_errors"] == 75,
        "Quality/review/target gates changed")


class Source:
    def __init__(self, locations):
        self.locations = {k: str(Path(v).resolve()) for k, v in locations.items()}
        self.handoff = Path(self.locations["handoff"])
        assets = self.handoff / "assets"
        whitelist = json.loads((ROOT / "configs/team_exploration_20261001/asset_sources.json").read_text())
        require(json.loads((self.handoff / "manifest.json").read_text()) == whitelist, "Foreign handoff")
        for item in whitelist["files"]:
            path = self.handoff / item["archive_path"]
            require(path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"],
                    f"Changed handoff: {path}")
        self.parent_path, self.targets_path = assets / "common/dev512/selected.pt", assets / "common/targets.pt"
        self.parent_binding = json.loads(self.parent_path.with_suffix(".sha256.json").read_text())["binding"]
        self.parent = load_artifact(self.parent_path, self.parent_binding)
        self.targets = load_artifact(self.targets_path, self.parent_binding)
        recipe = yaml.safe_load((assets / "common/original_v1_config.yaml").read_text())
        require(recipe["source"]["partition"] == "train_dev" and recipe["project"]["stage"] == "repechage"
                and str(recipe["project"]["data_version"]) == "20260921", "Foreign parent population")
        effective = {k: v for k, v in recipe.items() if k not in ("source", "output")}
        effective["partition"] = "train_dev"
        require(fingerprint(effective) == self.parent_binding["recipe_sha256"], "Original recipe mismatch")
        archived = {}
        for name in PARENT_FILES:
            old = assets / "common/archived_v1_code" / name
            current = ROOT / "reproducibility/aegis_f1/aegis_clip" / name
            archived[f"reproducibility/aegis_f1/aegis_clip/{name}"] = sha256_file(old)
            a, b = ast.parse(old.read_text()), ast.parse(current.read_text())
            excluded = {"v1_strategy.py": {"V1Classifier", "build_classifier"},
                        "v1_pipeline.py": {"StageContext", "load_recipe", "calibrate", "infer"}}.get(name, set())
            if excluded:
                for tree in (a, b):
                    tree.body = [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.ClassDef))
                                 or node.name not in excluded]
            require(ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False),
                    f"Effective parent implementation changed: {name}")
        require(fingerprint(archived) == self.parent_binding["implementation_sha256"], "Archived code mismatch")
        # Reconstruct the projection-only native parent using its actual archived
        # classifier; later pre-projection additions in main cannot alter it.
        spec = importlib.util.spec_from_file_location("strong_aug_archived_v1_strategy",
            assets / "common/archived_v1_code/v1_strategy.py")
        self.archived_strategy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.archived_strategy)
        dataset_source = ROOT / "results/rematch750_20260921_dataset_manifest.json"
        feature_source = ROOT / "results/rematch750_20260921_feature_manifest.json"
        require(lf_digest(dataset_source) == self.parent_binding["dataset_manifest_sha256"] and
                lf_digest(feature_source) == self.parent_binding["feature_manifest_sha256"],
                "Original manifest bytes do not match the parent binding")
        stage = Path(self.locations["stage"])
        reference = yaml.safe_load((assets / "common/original_reference_config.yaml").read_text())
        reference["data"].update(train_root=self.locations["train_root"], test_root=self.locations["test_root"],
            train_csv=str(stage / "train_dev.csv"), val_csv=str(stage / "val_dev.csv"),
            dataset_manifest=str(stage / "dataset_manifest.json"), class_mapping=str(stage / "class_to_idx.json"))
        reference["features"].update(tensor_path=str(stage / "features/features.pt"),
            paths_path=str(stage / "features/image_paths.json"), manifest_path=str(stage / "features/manifest.json"))
        reference["model"]["official_checkpoint"] = self.locations["official_checkpoint"]
        self.manifest = validate_dataset(reference)
        original_manifest = json.loads(dataset_source.read_text())
        original_manifest.update(train_root=self.locations["train_root"], test_root=self.locations["test_root"])
        require(self.manifest == original_manifest, "Local dataset differs beyond location migration")
        # This local audited official cache was encoded independently; never pretend
        # its byte hash is identical to the original training cache.
        self.cache = validate_cache(reference, self.manifest)
        require(self.cache["feature_dim"] == 512 and self.cache["normalized"] is True and
                self.cache["augmentation"] == "none" and self.cache["test_data_used"] is False,
                "Only the audited official 224px projected cache is allowed")
        self.official = Path(self.locations["official_checkpoint"])
        require(official_weight_hash(reference) == self.parent_binding["official_checkpoint_sha256"],
                "Official initialization mismatch")
        mapping = json.loads((stage / "class_to_idx.json").read_text())
        self.classes = [k for k, v in sorted(mapping.items(), key=lambda kv: kv[1])]
        require(sorted(mapping.values()) == list(range(750)) and
                sha256_file(stage / "class_to_idx.json") == self.parent_binding["class_mapping_sha256"],
                "Class mapping mismatch")
        self.train, self.val, self.full = [read_rows(stage / name) for name in
                                         ("train_dev.csv", "val_dev.csv", "full_train.csv")]
        validate_rows(self.train, self.val, self.classes)
        require(len(self.train) == 133815 and len(self.val) == 14880 and
                {r["image_path"] for r in self.full} == {r["image_path"] for r in self.train+self.val},
                "Population mismatch")
        self.supervision = normalize_targets(self.targets, self.train, self.classes)
        require(int((self.supervision["weights"] > 0).sum()) == 119074, "Supervision population changed")
        self.active = np.flatnonzero(self.supervision["weights"] > 0)
        require(self.parent["classes"] == self.classes and self.parent["model_config"] == recipe["model"]
                and self.parent["selected_policy"] == "swa_ema"
                and self.parent["swa_epochs"] == list(range(4, 13))
                and self.parent["targets_sha256"] == sha256_file(self.targets_path)
                and self.parent["trajectory"] == fingerprint(dict(binding=self.parent_binding, seed=42)),
                "Incorrect DEV parent or supervision")
        self.model_config = recipe["model"]
        self.train_root, self.test_root = Path(self.locations["train_root"]), Path(self.locations["test_root"])
        self.stage = stage
        archive_predictions = np.load(assets / "common/validation_predictions.npz", allow_pickle=False)
        require(np.array_equal(archive_predictions["labels"], [int(r["label"]) for r in self.val]) and
                np.array_equal(archive_predictions["image_paths"], [r["image_path"] for r in self.val]),
                "Original parent predictions are misaligned")
        self.parent_predictions = archive_predictions["parent512"].copy()
        self.binding = dict(stage="repechage", data_version="20260921", parent_binding=self.parent_binding,
            local_dataset_manifest_sha256=sha256_file(stage / "dataset_manifest.json"),
            local_feature_manifest_sha256=sha256_file(stage / "features/manifest.json"),
            local_feature_tensor_sha256=self.cache["tensor_sha256"],
            class_mapping_sha256=sha256_file(stage / "class_to_idx.json"),
            official_checkpoint_sha256=self.parent_binding["official_checkpoint_sha256"],
            parent_sha256=sha256_file(self.parent_path), targets_sha256=sha256_file(self.targets_path),
            local_cache_is_independent_encoding=True, original_artifacts_modified=False)

    def model(self):
        model = self.archived_strategy.build_classifier(self.official, len(self.classes), self.model_config)
        load_trainable_state(model, self.parent["selected_state"])
        return model

    def features(self):
        tensor = torch.load(self.stage / "features/features.pt", map_location="cpu", weights_only=True)
        require(tensor.shape == (148695, 512) and torch.isfinite(tensor).all(), "Invalid local feature tensor")
        index = {r["image_path"]: i for i, r in enumerate(self.full)}
        return tensor[[index[r["image_path"]] for r in self.train]].float(), \
               tensor[[index[r["image_path"]] for r in self.val]].float()


def implementation_hashes():
    files = [Path(__file__).parent / name for name in ("__init__.py", "source.py", "core.py", "runtime.py")]
    files += [ROOT / "reproducibility/aegis_f1/aegis_clip" / name for name in PARENT_FILES]
    files += [ROOT / "reproducibility/aegis_f1/v1_continuation/runtime.py",
              ROOT / "scripts/check_submission.py", ROOT / "scripts/verify_strong_aug_pair.py"]
    files.append(ROOT / "scripts/run_strong_aug_pair.py")
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in files}
