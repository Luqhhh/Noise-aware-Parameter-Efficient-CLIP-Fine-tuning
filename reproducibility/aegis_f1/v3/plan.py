"""CPU preparation: freeze existing v1 supervision and the common v3 protocol."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import yaml

from v2.plan import ROOT, dump, json_read, require, sha

ARMS = ("original", "v1_supervision")


def normalize_targets(payload, rows, classes):
    paths = payload["image_paths"]
    require(len(paths) == len(set(paths)) and set(paths) == {r["image_path"] for r in rows},
            "v1 targets must cover exactly train_dev, without holdout rows")
    require(payload["class_names"] == classes, "v1 target class mapping mismatch")
    index = {path: i for i, path in enumerate(paths)}
    order = [index[r["image_path"]] for r in rows]
    result = {}
    for key in ("labels", "targets", "weights", "original_alpha", "kept", "pseudo", "agreement"):
        values = payload[key].detach().cpu().numpy()
        require(values.shape == (len(rows),), f"Invalid v1 target shape: {key}")
        result[key] = values[order]
    labels = np.array([int(r["label"]) for r in rows])
    require(np.array_equal(result["labels"], labels), "v1 original labels changed")
    targets = result["targets"]
    require(np.issubdtype(targets.dtype, np.integer) and ((targets >= 0) & (targets < len(classes))).all(),
            "v1 target label out of range")
    for key in ("weights", "original_alpha", "agreement"):
        require(np.isfinite(result[key]).all() and ((result[key] >= 0) & (result[key] <= 1)).all(),
                f"Invalid v1 supervision values: {key}")
    kept, pseudo, active = result["kept"], result["pseudo"], result["weights"] > 0
    require(kept.dtype == np.bool_ and pseudo.dtype == np.bool_ and not (kept & pseudo).any(),
            "Invalid v1 kept/pseudo masks")
    require(np.array_equal(active, kept | pseudo), "v1 active mask and reliability disagree")
    require(np.array_equal(targets[~pseudo], labels[~pseudo]), "Non-pseudo labels were changed")
    coverage = np.bincount(targets[active], minlength=len(classes))
    require((coverage > 0).all(), "v1 supervision empties a class")
    return result


def frozen_groups(rows, supervision, classes, fraction, max_groups):
    labels = np.array([int(r["label"]) for r in rows])
    counts = np.bincount(labels, minlength=classes)
    content_groups = [set() for _ in range(classes)]
    for row in rows:
        content_groups[int(row["label"])].add(row["content_group"])
    groups = [len(value) for value in content_groups]
    tail = sorted(range(classes), key=lambda c: (int(counts[c]), c))[:max(1, math.ceil(classes * fraction))]
    active = supervision["weights"] > 0
    active_original = np.bincount(labels[active], minlength=classes)
    active_targets = np.bincount(supervision["targets"][active], minlength=classes)
    return dict(tail_classes=tail, other_classes=sorted(set(range(classes)) - set(tail)),
                few_support_classes=[c for c, n in enumerate(groups) if n <= max_groups],
                train_rows=counts.tolist(), train_groups=groups,
                active_original_rows=active_original.tolist(), active_target_rows=active_targets.tolist(),
                definition="Original train_dev counts/groups, frozen before either candidate")


def check_recipe(recipe):
    require(recipe.get("strategy_name") == "v3" and recipe.get("execution_authorized") is False,
            "Preparation is v3 engineering only; no GPU execution authorization")
    require(recipe.get("image_size") == 384 and recipe.get("epochs") == 2 and
            recipe.get("logical_batch_size") == 96 and recipe.get("final_weights") == "last_ema",
            "Only the fixed 384px/two-epoch pair is implemented")
    require(recipe.get("sampler_labels") == "original" and recipe.get("resolution_ladder") is False and
            recipe.get("full_train") is False and recipe.get("test_usage") == "inference_only",
            "The pair cannot change sampling labels, use full_train, or start a ladder")
    require(0 < recipe["ema_decay"] < 1 and 0 < recipe["tail_fraction"] < 1 and
            recipe["micro_batch_size"] > 0 and recipe["num_workers"] >= 0 and
            recipe["prefetch_factor"] > 0 and recipe["few_support_max_groups"] >= 1,
            "Invalid pair dimensions or frozen group definition")


def write_rows(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["index"], row["label"] = int(row["index"]), int(row["label"])
    return rows


def prepare(recipe_path, output):
    from aegis_clip.v1_pipeline import StageContext, load_artifact, load_recipe
    from .core import paired_draws

    recipe_path, output = Path(recipe_path).resolve(), Path(output).resolve()
    require(not output.exists(), "Refusing to overwrite a prepared pair")
    recipe = json_read(recipe_path)
    check_recipe(recipe)
    source_cfg = load_recipe(recipe["source_v1_config"])
    require(source_cfg["source"]["partition"] == "train_dev" and
            str(source_cfg["project"]["data_version"]) == recipe["data_version"],
            "v3 requires current-stage, train_dev-only v1 supervision")
    context = StageContext(source_cfg)
    require(context.manifest["stage"] == source_cfg["project"]["stage"],
            "v1 data belongs to a different competition stage")
    payload = load_artifact(recipe["source_v1_targets"], context.binding)
    train = sorted(context.train, key=lambda r: r["image_path"])
    val = sorted(context.val, key=lambda r: r["image_path"])
    supervision = normalize_targets(payload, train, context.classes)
    classes = len(context.classes)
    groups = frozen_groups(train, supervision, classes, recipe["tail_fraction"], recipe["few_support_max_groups"])
    draws = paired_draws([int(r["label"]) for r in train], classes, recipe["seed"],
                         recipe["epochs"], recipe["logical_batch_size"])
    # Reuse the v2 first-stage recipe, changing only the common horizon/seed and
    # fixing both arms' selection to the last EMA instead of validation selection.
    template = ROOT / "configs/v2/stages/s1_384.yaml"
    cfg = yaml.safe_load(template.read_text())
    cfg["data"].update(num_classes=classes, image_size=recipe["image_size"], eval_size=recipe["image_size"],
                       batch_size=recipe["logical_batch_size"], num_workers=recipe["num_workers"],
                       prefetch_factor=recipe["prefetch_factor"])
    cfg["train"].update(epochs=recipe["epochs"], seed=recipe["seed"],
                        ema_enabled=True, ema_decay=recipe["ema_decay"])
    runtime_recipe = dict(official_checkpoint=str(context.official),
                         official_sha256=context.binding["official_checkpoint_sha256"],
                         num_classes=classes, gradient_checkpointing=recipe["gradient_checkpointing"])
    output.mkdir(parents=True)
    for name, rows in (("train", train), ("val", val)):
        records = [dict(index=i, image_path=r["image_path"],
                        relative_path=Path(r["image_path"]).relative_to("train").as_posix(),
                        label=int(r["label"]), source_sha256=r["file_sha256"], content_group=r["content_group"])
                   for i, r in enumerate(rows)]
        write_rows(output / f"{name}_manifest.csv", records)
    np.savez(output / "supervision.npz", **supervision)
    np.save(output / "draws.npy", draws)
    dump(output / "groups.json", groups)
    dump(output / "shared_config.json", cfg)
    inputs = {str(recipe_path): sha(recipe_path), str(template): sha(template)}
    source_files = [Path(recipe["source_v1_config"]), Path(recipe["source_v1_targets"]),
                    Path(recipe["source_v1_targets"]).with_suffix(".sha256.json"), context.official]
    base = Path(context.reference["data"]["dataset_manifest"]).parent
    source_files += [base / name for name in ("dataset_manifest.json", "class_to_idx.json", "decode_report.json",
                                             "full_train.csv", "train_dev.csv", "val_dev.csv", "features/manifest.json")]
    source_files += [Path(source_cfg["source"]["root"]) / source_cfg["source"]["reference_config"]]
    for package in (Path(__file__).parent, ROOT / "reproducibility/aegis_f1/v2"):
        source_files += sorted(package.glob("*.py"))
    source_files += [ROOT / "reproducibility/aegis_f1/aegis_clip" / name for name in
                    ("v1_strategy.py", "v1_pipeline.py", "model.py", "runtime.py", "rematch_assets.py", "config.py")]
    for path in source_files + list(output.iterdir()):
        inputs[str(path.resolve())] = sha(path)
    steps = draws.shape[1] // recipe["logical_batch_size"]
    plan = dict(schema=1, experiment_id=recipe["experiment_id"], strategy_name="v3",
                strategy_revision=recipe["strategy_revision"], status="prepared_no_gpu",
                execution_authorized=False, recipe=recipe, model_recipe=runtime_recipe,
                v1_binding=context.binding, inputs=inputs, arms=list(ARMS),
                train_root=str(context.train_root), classes=context.classes,
                train_rows=len(train), val_rows=len(val), content_group_overlap=0,
                steps_per_epoch=steps, updates_per_arm=steps * recipe["epochs"],
                total_updates=2 * steps * recipe["epochs"], selected_weights="last_ema",
                supervision=dict(original_kept=int(supervision["kept"].sum()),
                                 pseudo=int(supervision["pseudo"].sum()), active=int((supervision["weights"] > 0).sum()),
                                 zero_weight=int((supervision["weights"] == 0).sum()),
                                 clean_labels_known=False),
                tail_classes=groups["tail_classes"], few_support_classes=groups["few_support_classes"],
                gpu_started=False, training_started=False, local_metrics=None, platform_metrics=None)
    dump(output / "plan.json", plan)
    return plan


def verify(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json_read(plan_path)
    check_recipe(plan["recipe"])
    require(plan["v1_binding"]["stage"] == "repechage" and
            plan["v1_binding"]["data_version"] == plan["recipe"]["data_version"],
            "Wrong competition stage/version in frozen v1 supervision")
    require(plan["arms"] == list(ARMS) and plan["execution_authorized"] is False,
            "Unexpected pair or execution authorization")
    for path, digest in plan["inputs"].items():
        require(sha(path) == digest, f"Frozen input changed: {path}")
    for name in ("train_manifest.csv", "val_manifest.csv", "shared_config.json", "supervision.npz", "draws.npy", "groups.json"):
        require(str(plan_path.parent / name) in plan["inputs"], f"Unbound pair input: {name}")
    return plan


def authorize(plan_path, authorization_path, operation):
    auth = json_read(authorization_path)
    require(auth.get("authorized") is True, "v3 GPU execution is not authorized")
    require(auth.get("operation") == operation and auth.get("plan_sha256") == sha(plan_path),
            "Authorization does not bind this operation and plan")
    budget = auth.get("max_seconds")
    require(isinstance(budget, (int, float)) and math.isfinite(budget) and budget > 0, "Finite operation budget required")
    plan = verify(plan_path)
    if operation == "train":
        path = auth.get("cost_report")
        require(path and sha(path) == auth.get("cost_report_sha256"), "Bound measured pair cost report required")
        cost = json_read(path)
        require(cost["status"] == "measured_probe" and cost["plan_sha256"] == sha(plan_path), "Wrong probe evidence")
        require(3 <= cost.get("updates_per_arm", 0) <= 10 and
                all(cost.get(key) is True for key in ("includes_decode_augmentation_wait",
                    "includes_full_raw_ema_validation", "includes_checkpoint_write")) and
                cost.get("usable_as_parent") is False, "Incomplete measured probe protocol")
        values = [cost[k] for k in ("seconds_per_update", "validation_seconds_per_arm_epoch", "overhead_seconds")]
        require(all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values), "Invalid measured costs")
        estimate = (plan["total_updates"] * values[0] + 2 * plan["recipe"]["epochs"] * values[1]
                    + (plan["recipe"]["epochs"] + 1) * values[2])
    else:
        require(operation == "probe", "Only a bounded probe or fixed pair is implemented")
        estimate = auth.get("estimated_seconds")
    require(isinstance(estimate, (int, float)) and math.isfinite(estimate) and 0 < estimate <= .8 * budget,
            "Cost estimate must leave 20% budget reserve")
    return auth, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--recipe", default=str(ROOT / "configs/v3/recipe.json"))
    p.add_argument("--output", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = prepare(args.recipe, args.output) if args.command == "prepare" else verify(args.plan)
    print({key: plan[key] for key in ("status", "train_rows", "val_rows", "supervision", "total_updates")})


if __name__ == "__main__":
    main()
