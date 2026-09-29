"""CPU-only preparation and immutable input/recipe checks. No torch import."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
VENDOR = Path(__file__).resolve().parents[1] / "vendor/palm_v13"
STAGES = ("s1_384", "s2_448", "s3_576", "v13_final")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def json_read(path):
    return json.loads(Path(path).read_text())


def verify_vendor():
    p = json_read(VENDOR / "provenance.json")
    for name, entry in p["files"].items():
        require(sha(VENDOR / name) == entry["sha256"], f"Pinned source changed: {name}")
    return p


def check_split(records, split, require_group_isolation=False):
    train, val = split["train"], split["val"]
    require(train and val, "Empty train/holdout split")
    require(len(set(train)) == len(train) and len(set(val)) == len(val), "Repeated indices")
    require(set(train).isdisjoint(val), "Holdout leaks into training")
    require(set(train) | set(val) == set(range(len(records))), "Split must cover the exact manifest")
    tg = {records[i]["content_group"] for i in train}
    vg = {records[i]["content_group"] for i in val}
    overlap = tg & vg
    require(not require_group_isolation or not overlap, "Frozen content groups overlap")
    return dict(train_rows=len(train), val_rows=len(val), overlapping_content_groups=len(overlap),
                val_rows_in_overlapping_groups=sum(records[i]["content_group"] in overlap for i in val))


def stage_plan(cfg, train_rows, val_rows, parent):
    b = int(cfg["data"]["batch_size"])
    n = train_rows // b
    require(n > 0, "Fewer training images than logical batch")
    epochs = int(cfg["train"]["epochs"])
    return dict(image_size=cfg["data"]["image_size"], epochs=epochs, logical_batch_size=b,
                steps_per_epoch=n, total_updates=n * epochs, train_rows=train_rows,
                val_rows=val_rows, lr_backbone=cfg["train"]["lr_backbone"], lr_head=cfg["train"]["lr_head"],
                warmup_steps=max(int(cfg["train"]["warmup_epochs"] * n), 1),
                parent=parent, optimizer_reset=True, scheduler_reset=True, ema_reset=True,
                holdout_independent=parent is not None and train_rows < cfg["data"]["expected_train_images"])


def cost_estimate(stage, seconds_per_update, validation_seconds, overhead_seconds):
    values = (seconds_per_update, validation_seconds, overhead_seconds)
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) and seconds_per_update > 0 and overhead_seconds > 0 and validation_seconds >= 0,
            "Finite measured update/overhead costs and nonnegative validation cost required")
    # validation_seconds is ONE full raw+EMA holdout evaluation pair per epoch.
    return stage["total_updates"] * seconds_per_update + stage["epochs"] * validation_seconds + overhead_seconds


def authorize(plan_path, authorization, operation, stage=None):
    a = json_read(authorization)
    require(a.get("authorized") is True, "GPU execution is not authorized")
    require(a.get("operation") == operation and a.get("stage") == stage, "Authorization operation/stage mismatch")
    require(a.get("plan_sha256") == sha(plan_path), "Authorization does not bind this plan")
    budget = a.get("max_seconds")
    require(isinstance(budget, (int, float)) and math.isfinite(budget) and budget > 0, "Finite positive budget required")
    plan = verify_prepared(plan_path)
    if operation == "train":
        spec = plan["stages"][stage]
        estimate = cost_estimate(spec, a.get("measured_seconds_per_update"),
                                 a.get("measured_validation_seconds"), a.get("measured_overhead_seconds"))
    else:
        estimate = a.get("estimated_seconds")
        require(isinstance(estimate, (int, float)) and math.isfinite(estimate) and estimate > 0,
                "Measured finite operation estimate required")
    require(estimate <= .8 * budget, "Cost estimate does not leave 20% budget reserve")
    return a


def verify_prepared(plan_path):
    plan_path = Path(plan_path)
    p = json_read(plan_path)
    verify_vendor()
    for name, digest in p["inputs"].items():
        require(sha(name) == digest, f"Frozen input changed: {name}")
    for name, entry in p["stages"].items():
        require(sha(plan_path.parent / entry["config"]) == entry["config_sha256"], f"Stage config changed: {name}")
    return p


def prepare(recipe_path, output):
    recipe_path, output = Path(recipe_path).resolve(), Path(output).resolve()
    require(not output.exists(), "Refusing to overwrite a prepared experiment")
    recipe = json_read(recipe_path)
    source = verify_vendor()
    require(recipe["source_commit"] == source["commit"], "Source revision mismatch")
    require(recipe["execution_authorized"] is False and recipe["num_classes"] == 750 and
            recipe["data_version"] == "20260921", "Unexpected recipe authorization or phase")
    art = Path(recipe["stage_artifacts"])
    metadata = json_read(art / "dataset_manifest.json")
    require(metadata["data_version"] == recipe["data_version"], "Wrong official data version")
    require(metadata["stage"] == "repechage" and metadata["num_classes"] == 750 and
            metadata["train_samples"] == recipe["official_train_rows"] and metadata["test_samples"] == recipe["test_rows"] and
            metadata["external_data"] is False and metadata["test_usage"] == "inference_only", "Wrong official data protocol")
    for name, digest in metadata["files"].items():
        require(sha(art / name) == digest, f"Official audit file changed: {name}")
    decoder = json_read(art / "decode_report.json")
    require(decoder["status"] == "passed" and not decoder["failures"], "Current decode audit is not clean")
    mapping = json_read(art / "class_to_idx.json")
    require(mapping == {f"{i:04d}": i for i in range(750)}, "Nonofficial class mapping")
    with (art / "full_train.csv").open(newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["image_path"])
    require(len(rows) == recipe["official_train_rows"], "Unexpected official train count")
    records = []
    for i, row in enumerate(rows):
        path = Path(row["image_path"])
        label = int(row["label"])
        require(len(path.parts) == 3 and path.parts[0] == "train" and path.parts[1] == f"{label:04d}" and
                0 <= label < 750 and not row["error"] and row["content_group"] and row["file_sha256"],
                "Unsafe/nonofficial manifest row")
        records.append(dict(index=i, relative_path=path.relative_to("train").as_posix(), label=label,
                            source_sha256=row["file_sha256"], content_group=row["content_group"]))
    require(len({r["relative_path"] for r in records}) == len(records), "Duplicate image paths")
    require(len(Counter(r["label"] for r in records)) == 750, "Class coverage incomplete")
    require(recipe["split_mode"] == "frozen_grouped" and recipe["strategy_only"] is True,
            "Strategy transfer uses the frozen grouped local protocol")
    lookup = {"train/" + r["relative_path"]: i for i, r in enumerate(records)}
    split = {}
    for name in ("train", "val"):
        with (art / f"{name}_dev.csv").open(newline="") as f:
            split[name] = sorted(lookup[r["image_path"]] for r in csv.DictReader(f))
        require(len(split[name]) == metadata[f"{name}_dev_samples"], "Frozen split count mismatch")
    mode = "frozen_grouped"
    split_report = check_split(records, split, mode == "frozen_grouped")
    require(Path(recipe["official_checkpoint"]).is_file(), "Local official CLIP is missing")
    require(sha(recipe["official_checkpoint"]) == recipe["official_sha256"], "Official CLIP identity mismatch")
    output.mkdir(parents=True)
    manifest = output / "train_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    dump(output / "split.json", split)
    inputs = {str(recipe_path): sha(recipe_path), str(manifest): sha(manifest),
              str(output / "split.json"): sha(output / "split.json"),
              recipe["official_checkpoint"]: recipe["official_sha256"]}
    for name in ("dataset_manifest.json", "decode_report.json", "class_to_idx.json", "full_train.csv", "train_dev.csv", "val_dev.csv", "test_manifest.csv"):
        inputs[str(art / name)] = sha(art / name)
    plan = dict(experiment_id=recipe["experiment_id"], status="prepared_no_gpu", execution_authorized=False,
                data_version=recipe["data_version"], source_commit=source["commit"], recipe=recipe,
                split_mode=mode, split=split_report, inputs=inputs, stages={},
                runtime_dependencies={k: importlib.util.find_spec(k) is not None for k in ("torch", "torchvision", "clip")},
                strategy_only=True, transfer_notes=recipe["transfer_notes"], local_metrics=None, platform_metrics=None)
    for i, name in enumerate(STAGES):
        rel = "configs/" + ("v13_final.yaml" if i == 3 else "v12/" + name + ".yaml")
        cfg = yaml.safe_load((VENDOR / rel).read_text())
        cfg["paths"]["project_root"] = str(output)
        cfg["data"].update(train_dir=recipe["train_root"], decode_cap=0,
                            manifest="train_manifest.csv", split="split.json", expected_train_images=len(records),
                            num_workers=recipe["num_workers"], prefetch_factor=recipe["prefetch_factor"])
        cfg["model"] = dict(backbone="ViT-B/32", official_checkpoint=recipe["official_checkpoint"], head="linear", dropout=0.0, freeze_first_blocks=0, train_last_blocks=0)
        cfg["train"]["output_dir"] = "runs/" + name
        cfg["train"]["seed"] = recipe["seed"]  # source main reads train.seed, not root seed
        cfg["local_replay"] = dict(micro_batch_size=recipe["micro_batch_size"],
                                   gradient_checkpointing=recipe["gradient_checkpointing"], final_stage=i == 3)
        config = output / "configs" / (name + ".json")
        dump(config, cfg)
        spec = stage_plan(cfg, len(split["train"]) if i < 3 else len(records), len(split["val"]) if i < 3 else 0,
                          "official_openai" if i == 0 else STAGES[i - 1])
        if i == 3:
            spec.update(holdout_independent=False, train_policy="all_official_rows", independent_validation=False)
        spec.update(config=str(config.relative_to(output)), config_sha256=sha(config))
        plan["stages"][name] = spec
    for path in sorted(Path(__file__).parent.glob("*")):
        if path.is_file():
            plan["inputs"][str(path)] = sha(path)
    plan["inputs"][str(VENDOR / "provenance.json")] = sha(VENDOR / "provenance.json")
    plan["inputs"][str(ROOT / "reproducibility/aegis_f1/aegis_clip/model.py")] = sha(ROOT / "reproducibility/aegis_f1/aegis_clip/model.py")
    dump(output / "plan.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--recipe", default=str(ROOT / "configs/palm_v13_20260929/recipe.json"))
    p.add_argument("--output", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--plan", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        p = prepare(args.recipe, args.output)
        print(json.dumps({k: p[k] for k in ("status", "split", "runtime_dependencies", "strategy_only")}, indent=2))
    else:
        p = verify_prepared(args.plan)
        print(json.dumps(dict(status="verified", plan_sha256=sha(args.plan))))


if __name__ == "__main__":
    main()
