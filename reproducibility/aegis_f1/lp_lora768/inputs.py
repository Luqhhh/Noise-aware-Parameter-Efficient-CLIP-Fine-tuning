"""Portable locations with immutable source identities and exact parent replay."""
from __future__ import annotations
import importlib.util
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from aegis_clip.config import load_config
from aegis_clip.rematch_assets import validate_dataset
from aegis_clip.runtime import sha256_file, atomic_json_dump
from aegis_clip.v1_pipeline import Images, image_transform, load_artifact, read_rows, validate_rows
from aegis_clip.v1_strategy import build_classifier, load_trainable_state

ROOT = Path(__file__).resolve().parents[3]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class Context:
    def __init__(self, locations):
        self.locations = read_json(locations)
        self.protocol = read_json(ROOT / "configs/team_exploration_20261001/machine_a.json")
        self.assets = Path(self.locations["assets"])
        spec = importlib.util.spec_from_file_location("handoff_verify", ROOT / "scripts/export_team_exploration_assets.py")
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        self.bundle_report = exporter.verify(self.locations["bundle"], ROOT / "configs/team_exploration_20261001/asset_sources.json")
        whitelist = read_json(ROOT / "configs/team_exploration_20261001/asset_sources.json")
        if read_json(self.assets / "manifest.json") != whitelist:
            raise ValueError("Unpacked manifest differs from reviewed whitelist")
        for item in whitelist["files"]:
            path = self.assets / item["archive_path"]
            if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                raise ValueError(f"Unpacked asset changed: {path}")
        self.parent_path = self.assets / self.protocol["parent_asset"]
        meta = read_json(self.parent_path.with_suffix(".sha256.json"))
        self.parent = load_artifact(self.parent_path, meta["binding"])
        targets_path = self.assets / "assets/common/targets.pt"
        self.targets = load_artifact(targets_path, read_json(targets_path.with_suffix(".sha256.json"))["binding"])
        if self.parent["selected_policy"] != "frozen_visual_head_last20" or self.parent["visual_parameter_updates"]:
            raise ValueError("Requires the untouched ordinary DEV768 head parent")
        if self.parent["binding"]["arm"] != "unbalanced" or self.parent["head_epochs"] != 20:
            raise ValueError("Balanced or non-last20 parent forbidden")
        source = Path(self.locations["source_root"])
        reference = load_config(source / "configs/rematch750_lp.yaml")
        self.manifest = validate_dataset(reference)
        self.base = Path(reference["data"]["dataset_manifest"]).parent
        mapping = read_json(self.base / "class_to_idx.json")
        self.classes = [k for k, _ in sorted(mapping.items(), key=lambda p: p[1])]
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("Invalid class mapping")
        self.train, self.val = read_rows(self.base / "train_dev.csv"), read_rows(self.base / "val_dev.csv")
        validate_rows(self.train, self.val, self.classes)
        self.train_root, self.test_root = Path(self.manifest["train_root"]), Path(self.manifest["test_root"])
        self.official = Path(self.locations["official_checkpoint"])
        self.device = self.locations["device"]
        self.frozen = np.load(self.assets / "assets/common/validation_predictions.npz", allow_pickle=False)
        np.testing.assert_array_equal(self.frozen["image_paths"], [r["image_path"] for r in self.val])
        np.testing.assert_array_equal(self.frozen["labels"], [int(r["label"]) for r in self.val])
        if self.targets["image_paths"] != [r["image_path"] for r in self.train]:
            raise ValueError("Train target order differs")
        np.testing.assert_array_equal(self.targets["labels"], [int(r["label"]) for r in self.train])
        if self.classes != self.targets["class_names"] or self.classes != self.parent["classes"]:
            raise ValueError("Artifact classes differ from local mapping")
        if len(self.train) != 133815 or len(self.val) != self.protocol["validation"]["rows"]:
            raise ValueError("Training or validation population changed")
        self.active = torch.where(self.targets["weights"] > 0)[0].numpy()
        if len(self.active) != 119074:
            raise ValueError("Supervision population changed")
        for value in (self.targets["weights"], self.targets["original_alpha"]):
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite supervision")
        if (self.targets["weights"] < 0).any():
            raise ValueError("Negative reliability")
        class_sha = sha256_file(self.base / "class_to_idx.json")
        official_sha = sha256_file(self.official)
        for artifact in (self.parent, self.targets):
            binding = artifact["binding"]
            if binding["stage"] != "repechage" or str(binding["data_version"]) != "20260921":
                raise ValueError("Foreign stage artifact")
            if binding["class_mapping_sha256"] != class_sha or binding["official_checkpoint_sha256"] != official_sha:
                raise ValueError("Official initialization or class identity differs")
        if self.parent["binding"]["targets_sha256"] != sha256_file(targets_path):
            raise ValueError("768 parent supervision differs")
        self.groups_path = ROOT / self.protocol["frozen_target_groups"]
        self.groups = read_json(self.groups_path)
        if self.groups["source_sha256"] != sha256_file(self.assets / "assets/common/validation_predictions.npz"):
            raise ValueError("Frozen confusion group source changed")
        labels, predictions = self.frozen["labels"], self.frozen["unbalanced768"]
        pairs = Counter(tuple(sorted((int(y), int(p)))) for y, p in zip(labels, predictions) if y != p)
        top = [dict(classes=list(k), error_count=n) for k, n in sorted(pairs.items(), key=lambda p: (-p[1], p[0]))[:10]]
        if top != self.groups["top_pairs"]:
            raise ValueError("Frozen confusion ranking differs")
        target = np.isin(labels, self.groups["target_classes"])
        if int(target.sum()) != self.groups["target_rows"] or int(((labels != predictions) & target).sum()) != self.groups["parent_target_errors"]:
            raise ValueError("Frozen target population differs")
        np.testing.assert_array_equal(self.frozen["tail"], self.groups["tail_classes"])
        self.binding = dict(experiment_id=self.protocol["experiment_id"], owner=self.locations["owner"],
            parent_sha256=sha256_file(self.parent_path), source_binding=self.parent["binding"],
            targets_sha256=sha256_file(targets_path), local_dataset_manifest_sha256=sha256_file(self.base / "dataset_manifest.json"),
            local_train_sha256=sha256_file(self.base / "train_dev.csv"), local_val_sha256=sha256_file(self.base / "val_dev.csv"),
            class_mapping_sha256=class_sha, official_checkpoint_sha256=official_sha,
            protocol_sha256=sha256_file(ROOT / "configs/team_exploration_20261001/machine_a.json"), groups_sha256=sha256_file(self.groups_path))
        code_hashes = {name: hashlib.sha256((Path(__file__).parent/name).read_text(encoding="utf-8-sig").encode()).hexdigest()
                       for name in ("inputs.py", "paired.py", "metrics.py", "runtime.py", "__main__.py")}
        self.binding["implementation_sha256"] = hashlib.sha256(json.dumps(code_hashes, sort_keys=True).encode()).hexdigest()
        self.inputs_report = dict(status="verified", all_ordered_labels_and_paths_match=True,
            positive_weight_rows=len(self.active), train_rows=len(self.train), val_rows=len(self.val),
            bundle=self.bundle_report, binding=self.binding, locations=self.locations, original_artifacts_modified=False)


def adapter_head_state(model):
    """Keep frozen LoRA in control; generic trainable_state would silently omit it."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if n.startswith("head.") or "lora_" in n}


def restore_adapter_head(model, state):
    parameters = {n: p for n, p in model.named_parameters() if n.startswith("head.") or "lora_" in n}
    if parameters.keys() != state.keys():
        raise ValueError("Complete adapter/head keys differ")
    with torch.no_grad():
        for name, p in parameters.items():
            if p.shape != state[name].shape:
                raise ValueError(f"Adapter/head shape mismatch: {name}")
            p.copy_(state[name])


def initial_model(context, arm):
    if arm not in context.protocol["arms"]:
        raise ValueError("Unknown arm")
    config = context.parent["model_config"]
    if config["feature_path"] != "pre_projection" or config["blocks"] != 12 or config["rank"] != 32 or config["alpha"] != 64:
        raise ValueError("Frozen 768 parent architecture differs")
    model = build_classifier(context.official, len(context.classes), config, "cpu")
    load_trainable_state(model, context.parent["selected_state"])
    if len(model.adapted_modules) != 48 or model.visual.proj is not None:
        raise ValueError("Incomplete LoRA or unexpected 512 projection")
    for name, p in model.visual.named_parameters():
        p.requires_grad_(arm == "lora_and_head" and "lora_" in name)
    return model.to(context.device).float().eval()


@torch.no_grad()
def center_predictions(context, model, rows, *, batch_size=8, output=None):
    model.eval()
    loader = DataLoader(Images(context.train_root, rows, image_transform(448)), batch_size=batch_size,
                        num_workers=context.protocol["num_workers"], shuffle=False)
    predictions = np.empty(len(rows), dtype=np.int64)
    logits = np.empty((len(rows), len(context.classes)), dtype=np.float32)
    started = time.monotonic()
    for batch, (images, indices) in enumerate(loader, 1):
        values = model(images.to(context.device)).float().cpu()
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite parent/validation output")
        predictions[indices.numpy()] = values.argmax(1).numpy()
        logits[indices.numpy()] = values.numpy()
        if batch == 1 or batch % 100 == 0 or batch == len(loader):
            print(json.dumps(dict(phase="center", completed=min(batch * batch_size, len(rows)), total=len(rows), seconds=time.monotonic()-started)), flush=True)
    if output:
        np.savez_compressed(output, predictions=predictions, logits=logits,
                            labels=[int(r["label"]) for r in rows], image_paths=[r["image_path"] for r in rows])
    return predictions, dict(rows=len(rows), seconds=time.monotonic()-started)

