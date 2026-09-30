"""CPU preparation and bindings; no implicit training or FULL promotion."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import torch
import yaml

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_pipeline import StageContext, fingerprint, load_artifact, load_recipe
from v3.plan import frozen_groups, normalize_targets

ROOT = Path(__file__).resolve().parents[3]
FIXED = {
    "WFT448": dict(image_size=448, epochs=4, backbone_lr=5e-6, head_lr=5e-5,
                   optimizer="adamw", weight_decay=.05, scheduler="warmup_cosine", warmup_epochs=.25,
                   grad_clip=1., ema_decay=.999, average_epochs=[2, 3, 4], views=[448, 512, 576], flip=True),
    "LR512": dict(image_size=512, epochs=4, backbone_lr=5e-5, head_lr=2.5e-4,
                  optimizer="v1_adamw", weight_decay=0., scheduler="v1_onecycle", warmup_epochs=.4,
                  grad_clip=1., ema_decay=.999, average_epochs=[2, 3, 4], views=[512], flip=True),
}


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != {"experiment_id", "route", "partition", "parent_config", "parent_code_root", "parent_checkpoint", "targets",
                    "recipe", "micro_batch_size", "num_workers", "seed"}:
        raise ValueError("Unexpected continuation configuration keys")
    route, population = cfg["route"], cfg["partition"]
    if route not in FIXED or population not in ("train_dev", "full_train"):
        raise ValueError("Only WFT448/LR512 with dev/full population are supported")
    if cfg["recipe"] != FIXED[route] or cfg["experiment_id"] != route + ("_DEV" if population == "train_dev" else "_FULL"):
        raise ValueError("The four-epoch recipe and export window are frozen; no parameter/window search")
    if cfg["micro_batch_size"] < 1 or cfg["num_workers"] < 0:
        raise ValueError("Invalid micro batch or workers")
    for key in ("parent_config", "parent_code_root", "parent_checkpoint", "targets"):
        cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    return cfg


PARENT_FILES = tuple("reproducibility/aegis_f1/aegis_clip/" + name for name in
                     ("v1_strategy.py", "v1_pipeline.py", "model.py", "submission.py"))


def compatible_parent_code(current_root, archived_root):
    """The archived dev pipeline differs only in unused calibration reporting.

    Verify that difference explicitly, while binding the parent's original code
    hash. All model, targets, transforms, training, and inference code must agree.
    """
    archived = {}
    for name in PARENT_FILES:
        current, old = Path(current_root) / name, Path(archived_root) / name
        archived[name] = sha256_file(old)
        if sha256_file(current) == archived[name]:
            continue
        if not name.endswith("v1_pipeline.py"):
            raise ValueError("Archived parent model/submission code differs")
        trees = [ast.parse(path.read_text()) for path in (current, old)]
        for tree in trees:
            tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef) or node.name != "calibrate"]
        if ast.dump(trees[0], include_attributes=False) != ast.dump(trees[1], include_attributes=False):
            raise ValueError("Archived parent effective pipeline differs outside unused calibration")
    return fingerprint(archived)


def inspect_source(cfg):
    context = StageContext(load_recipe(cfg["parent_config"]))
    implementation = compatible_parent_code(ROOT, cfg["parent_code_root"])
    context.parent_binding = dict(context.binding, implementation_sha256=implementation)
    if context.config["source"]["partition"] != cfg["partition"]:
        raise ValueError("Dev must start from dev SWA, FULL must start from full SWA")
    if context.config["model"] != dict(backbone="ViT-B/32", pretrained="openai", image_size=448,
                                         blocks=12, rank=32, alpha=64.):
        raise ValueError("Parent architecture differs from current v1")
    checkpoint = load_artifact(cfg["parent_checkpoint"], context.parent_binding)
    if (checkpoint.get("classes") != context.classes or checkpoint.get("model_config") != context.config["model"]
            or checkpoint.get("selected_policy") != "swa_ema" or checkpoint.get("swa_epochs") != list(range(4, 13))
            or checkpoint.get("trajectory") != fingerprint(dict(binding=context.parent_binding, seed=context.config["project"]["seed"]))):
        raise ValueError("Parent is not the selected current-stage v1 EMA SWA4-12")
    if checkpoint.get("targets_sha256") != sha256_file(cfg["targets"]):
        raise ValueError("Parent and continuation supervision disagree")
    targets = load_artifact(cfg["targets"], context.parent_binding)
    supervision = normalize_targets(targets, context.train, context.classes)
    if cfg["seed"] != context.config["project"]["seed"]:
        raise ValueError("Keep the parent's fixed seed")
    if cfg["micro_batch_size"] > context.config["train"]["batch_size"]:
        raise ValueError("Micro batch cannot increase the actual v1 logical batch")
    return context, supervision


def source_hashes(cfg):
    inputs = {cfg[k]: sha256_file(cfg[k]) for k in ("parent_config", "parent_checkpoint", "targets")}
    for k in ("parent_checkpoint", "targets"):
        sidecar = str(Path(cfg[k]).with_suffix(".sha256.json"))
        inputs[sidecar] = sha256_file(sidecar)
    for name in PARENT_FILES:
        path = str(Path(cfg["parent_code_root"]) / name)
        inputs[path] = sha256_file(path)
    for name in ("plan.py", "model.py", "runtime.py"):
        path = str(Path(__file__).parent / name)
        inputs[path] = sha256_file(path)
    for name in ("v1_strategy.py", "v1_pipeline.py", "model.py", "submission.py"):
        path = str(ROOT / "reproducibility/aegis_f1/aegis_clip" / name)
        inputs[path] = sha256_file(path)
    for name in ("v3/plan.py", "v3/report.py"):
        path = str(ROOT / "reproducibility/aegis_f1" / name)
        inputs[path] = sha256_file(path)
    return inputs


def validate_support(report_path, route, assessment):
    report = json.loads(Path(report_path).read_text())
    if (report.get("status") != "development_complete" or report.get("route") != route
            or report.get("partition") != "train_dev" or report.get("recipe") != FIXED[route]
            or report.get("validation_is_independent") is not True or report.get("epochs") != 4
            or not report.get("trajectory") or not report.get("paired") or not assessment.strip()):
        raise ValueError("FULL needs a complete matching DEV report and an evidence-based support assessment")
    if route == "LR512" and report.get("baseline_before_training") is not True:
        raise ValueError("LR512 FULL requires an untrained 512 baseline")
    for path, expected in report["artifacts"].items():
        if sha256_file(path) != expected:
            raise ValueError("Development evidence changed")
    return dict(report=str(Path(report_path).resolve()), sha256=sha256_file(report_path),
                assessment=assessment, automatic_promotion=False)


def prepare(config, output, *, development_report=None, support_assessment=""):
    cfg, output = load_config(config), Path(output).resolve()
    support = None
    if cfg["partition"] == "full_train":
        if not development_report:
            raise ValueError("FULL cannot be prepared before matching DEV evidence")
        support = validate_support(development_report, cfg["route"], support_assessment)
    context, supervision = inspect_source(cfg)
    groups = frozen_groups(context.train, supervision, len(context.classes), .1, 5)
    counts = groups["train_rows"]
    groups["head_classes"] = sorted(range(len(counts)), key=lambda c: (-counts[c], c))[:max(1, len(counts)//10)]
    plan = dict(status="prepared_not_started", experiment_id=cfg["experiment_id"], config=cfg,
                source_binding=context.binding, parent_binding=context.parent_binding,
                inputs=source_hashes(cfg), classes=context.classes,
                train_rows=len(context.train), active_train_rows=int((supervision["weights"] > 0).sum()),
                val_rows=len(context.val), logical_batch_size=context.config["train"]["batch_size"],
                inherited_train=context.config["train"], groups=groups, support=support,
                validation_is_independent=cfg["partition"] == "train_dev", bias="disabled",
                test_usage="single_checkpoint_inference_only", platform_score=None)
    if support:
        plan["inputs"][support["report"]] = support["sha256"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json_dump(plan, output / "plan.json")
    return output / "plan.json"


def verify(path):
    plan = json.loads(Path(path).read_text())
    cfg = plan["config"]
    if cfg["route"] not in FIXED or cfg["recipe"] != FIXED[cfg["route"]]:
        raise ValueError("Prepared recipe differs from the fixed protocol")
    for name, expected in plan["inputs"].items():
        if sha256_file(name) != expected:
            raise ValueError(f"Bound source changed: {name}")
    context, supervision = inspect_source(plan["config"])
    if (context.binding != plan["source_binding"] or context.parent_binding != plan["parent_binding"]
            or context.classes != plan["classes"]):
        raise ValueError("Current-stage data or class binding changed")
    if (plan["logical_batch_size"] != context.config["train"]["batch_size"]
            or plan["inherited_train"] != context.config["train"]
            or plan["train_rows"] != len(context.train) or plan["val_rows"] != len(context.val)):
        raise ValueError("Prepared population or inherited training settings changed")
    if cfg["partition"] == "full_train":
        support = plan.get("support")
        if not support:
            raise ValueError("FULL needs matching DEV evidence")
        validate_support(support["report"], cfg["route"], support["assessment"])
    return plan, context, supervision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify"])
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--plan")
    parser.add_argument("--development-report")
    parser.add_argument("--support-assessment", default="")
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.config or not args.output:
            parser.error("prepare requires --config and new --output")
        print(prepare(args.config, args.output, development_report=args.development_report,
                      support_assessment=args.support_assessment))
    else:
        if not args.plan:
            parser.error("verify requires --plan")
        plan, _, _ = verify(args.plan)
        print(json.dumps(dict(status="verified", experiment_id=plan["experiment_id"],
                              train_rows=plan["train_rows"], val_rows=plan["val_rows"])))


if __name__ == "__main__":
    main()
