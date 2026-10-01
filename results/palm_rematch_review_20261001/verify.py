"""Read-only source/configuration review; no model, image or training imports.

Requires the project's existing PyYAML and a clone at the pinned public commit.
Writes one new JSON report. Does not rerun the original submission checker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


SOURCE_COMMIT = "fcc37813c8bea3f2761707a145b2955ce95c097e"
BASE_COMMIT = "7b3bd46"
SOLUTION = Path("solutions/clip-vitb32-full-finetune")
FIELDS = {
    "data": ["image_size", "eval_size", "eval_resize_ratio", "batch_size", "val_batch_size"],
    "model": ["head", "dropout", "freeze_first_blocks", "train_last_blocks"],
    "augment": ["rrc_scale", "color_jitter", "rand_augment", "rand_augment_ops",
                "rand_augment_magnitude", "random_erase", "mixup", "cutmix",
                "mix_prob", "label_smoothing"],
    "sampler": ["scheme", "power"],
    "train": ["epochs", "lr_backbone", "lr_head", "llrd_gamma", "weight_decay",
              "warmup_epochs", "min_lr_ratio", "grad_clip", "ema_decay", "amp_dtype",
              "elr_lambda", "elr_momentum", "early_stop_patience", "log_every"],
}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, repo = args.source.resolve(), args.repo.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    assert git(source, "rev-parse", "HEAD") == SOURCE_COMMIT
    assert not git(source, "status", "--porcelain"), "Public source clone is modified"

    stages = []
    for name, external in [
        ("s1_384", "configs/v12/s1_384.yaml"),
        ("s2_448", "configs/v12/s2_448.yaml"),
        ("s3_576", "configs/v12/s3_576.yaml"),
        ("full_576", "configs/v13_final.yaml"),
    ]:
        upstream = load(source / SOLUTION / external)
        local = load(repo / f"configs/v2/stages/{name}.yaml")
        comparisons = [
            {"field": f"{section}.{field}", "public_value": upstream[section][field],
             "project_value": local[section][field],
             "equal": upstream[section][field] == local[section][field]}
            for section, fields in FIELDS.items() for field in fields
        ]
        assert all(x["equal"] for x in comparisons), name
        stages.append({"stage": name, "public_config": str(SOLUTION / external),
                       "project_config": f"configs/v2/stages/{name}.yaml",
                       "compared_fields": len(comparisons), "comparisons": comparisons})

    train = (source / SOLUTION / "src/aic_clip/train_ft.py").read_text()
    dedup = (source / SOLUTION / "scripts/dedup_train_set.py").read_text()
    split = (source / SOLUTION / "scripts/make_holdout_split.py").read_text()
    infer = (source / SOLUTION / "src/aic_clip/infer_ft.py").read_text()
    local_model = (repo / "reproducibility/aegis_f1/v2/model.py").read_text()
    signatures = {
        "public_projected_feature": ".image_embeds" in train and "self.vision.config.projection_dim" in train,
        "project_projected_512_linear_feature": "return x @ visual.proj" in local_model and "nn.Linear(512," in local_model,
        "public_probability_view_average": "F.softmax(logits, dim=1)" in infer and "np.mean(np.stack(list(view_probs.values())" in infer,
        "public_near_duplicate_uses_top1": "sims.topk(2, dim=1)" in dedup and "top.indices[:, 0]" in dedup,
        "public_conflicting_pair_counter_is_directed": "conflict_pairs += 1" in dedup and "union.union(i, j)" in dedup,
        "public_split_stratifies_rows": "rng.shuffle(items)" in split and 'by_class[int(row["label"])].append(i)' in split,
        "public_full_still_evaluates_validation": "train_idx = list(range(len(records)))" in train and "ema_metrics = evaluate(model, val_loader" in train,
    }
    assert all(signatures.values()), signatures

    readme = (source / SOLUTION / "README.md").read_text()
    latest = git(source, "show", "-s", "--format=%B", "1aae60e")
    v15 = (source / SOLUTION / "configs/v15_strong_reg.yaml").read_text()
    assert "76.5890" in readme and "76.7413" in latest
    assert "+2.08 pp" in v15 and "DOWN 0.58 pp" in v15

    record_path = repo / "results/v1_full_swa_platform_20261001/artifact_verification.json"
    record = json.loads(record_path.read_text())
    csv_path = Path(record["artifacts"]["csv"]["path"])
    zip_path = Path(record["artifacts"]["zip"]["path"])
    csv_bytes, zip_bytes = csv_path.read_bytes(), zip_path.read_bytes()
    csv_sha = hashlib.sha256(csv_bytes).hexdigest()
    zip_sha = hashlib.sha256(zip_bytes).hexdigest()
    assert csv_sha == record["artifacts"]["csv"]["sha256"]
    assert zip_sha == record["artifacts"]["zip"]["sha256"]
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["pred_results.csv"]
        assert archive.read("pred_results.csv") == csv_bytes
    assert len(csv_bytes.splitlines()) == 37444
    assert record["submission_checks_passed"] == 9

    links = []
    for relative in ["docs/palm_rematch_review_20261001.md", "docs/current_execution_plan.md"]:
        path = repo / relative
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            # The first run creates the report linked by the document itself.
            assert resolved.exists() or resolved == args.output.resolve(), (relative, target)
            links.append({"document": relative, "target": target})

    report = {
        "experiment_id": "PALM_REMATCH_REVIEW_20261001",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "project_base_commit": BASE_COMMIT,
        "source_repository": "https://github.com/Palm0palM/aic-noisy-clip",
        "source_commit": SOURCE_COMMIT,
        "source_remote_branches": git(source, "branch", "-r").splitlines(),
        "source_review_scope": str(SOLUTION),
        "configuration_comparison_scope": "selected core fields, not runtime or data equivalence",
        "equal_configuration_fields": sum(x["compared_fields"] for x in stages),
        "stages": stages,
        "static_source_signatures": signatures,
        "public_results_evidence": {
            "readme_v13_self_report_percent": 76.5890,
            "latest_commit_six_view_self_report_percent": 76.7413,
            "v14_holdout_delta_pp_comment": 2.08,
            "v14_test_delta_pp_comment": -0.58,
            "independent_evaluation_or_platform_receipt_available": False,
        },
        "incumbent_reference": {
            "reported_platform_percent": record["platform_score_percent"],
            "csv": str(csv_path), "zip": str(zip_path),
            "csv_sha256": csv_sha, "zip_sha256": zip_sha,
            "rows": 37444, "zip_csv_bytes_identical": True,
            "prior_submission_checks_passed": 9,
            "prior_submission_check": "results/v1_full_swa_platform_20261001/submission_check.log",
            "nine_checks_rerun_in_this_review": False,
        },
        "checked_local_links": len(links),
        "new_training_or_inference_started": False,
        "public_training_code_executed": False,
        "test_images_or_labels_read": False,
        "verification_passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["experiment_id", "equal_configuration_fields",
                                           "checked_local_links", "verification_passed"]}))


if __name__ == "__main__":
    main()
