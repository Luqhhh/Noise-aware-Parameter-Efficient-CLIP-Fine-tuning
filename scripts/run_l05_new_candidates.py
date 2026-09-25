#!/usr/bin/env python3
"""Single-GPU serial driver for the NEW01/NEW02/NEW03 L05 continuation candidates.

Runs at most one GPU job at a time and, per candidate:

1. builds a machine-local runtime config from ``configs/l05_new_candidates/<ID>.yaml``
   and verifies the frozen split, class mapping and the L05 parent lineage before
   any training (the continuation uses ``parent_kind=same_split_continue`` with
   ``parent_experiment_id=RM_V5_L05_CUDA_LOCAL``; L05 is never relabelled as RM-LP);
2. trains the stage-2 budget (3 epochs, GCE + local supervision from epoch 1, no
   CE warmup, backbone LR 3e-6 / head LR 1e-4, 3-epoch cosine, effective batch 1024);
3. caches the candidate's own validation flip-TTA branch logits;
4. evaluates center and the frozen Flip/T=1.4 + validation-fitted prior 0.60
   protocol with corrections/regressions;
5. appends one row to ``results/l05_new_candidates.csv``.

It never uploads to the platform and never re-runs A0/N1/O3/PTA or multi-seed work.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
if str(AEGIS_ROOT) not in sys.path:
    sys.path.insert(0, str(AEGIS_ROOT))

from aegis_clip.config import load_config  # noqa: E402
from aegis_clip.rematch_protocol import validate_checkpoint  # noqa: E402
from aegis_clip.runtime import sha256_file  # noqa: E402

CANDIDATES = ("NEW01", "NEW02", "NEW03")
CONFIG_DIR = ROOT / "configs" / "l05_new_candidates"
STAGE_DIR = Path("/home/lux1/noise/artifacts/stages/repechage/20260921")
L05_CHECKPOINT = (
    ROOT / "outputs/f05_focus_l05/RM_V5_L05_CUDA_LOCAL/seed42/checkpoints/best.pt"
)
OUTPUT_ROOT = ROOT / "outputs" / "l05_new_candidates"
RUNTIME_DIR = OUTPUT_ROOT / "_runtime_configs"
SIDECAR = ROOT / "artifacts" / "l05_new_candidates" / "NEW03_candidate_labels.json"
RESULTS_CSV = ROOT / "results" / "l05_new_candidates.csv"
L05_REFERENCE = ROOT / "results" / "f05_focus_l05_tta_prior_platform_20260925.json"
SPLIT_SEED = 42

RESULTS_FIELDS = [
    "candidate_id",
    "base_checkpoint_sha",
    "train_seed",
    "training_recipe",
    "selected_epoch",
    "center_macro",
    "center_micro",
    "decode_macro",
    "decode_micro",
    "delta_vs_l05",
    "affected_samples",
    "corrections",
    "regressions",
    "status",
    "platform_score",
]


def _abs(value, default=None):
    if value in (None, ""):
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _run(command: list[str]) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(AEGIS_ROOT) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    print("RUN", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _frozen_split_report(stage_dir: Path) -> dict:
    manifest_path = stage_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("seed", -1)) != SPLIT_SEED:
        raise ValueError("dataset manifest split seed is not the frozen 42")
    verified = {}
    for name, digest in manifest.get("files", {}).items():
        path = stage_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"frozen split asset changed: {name}")
        verified[name] = digest
    return {
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "split_seed": int(manifest["seed"]),
        "train_fingerprint": manifest.get("train_fingerprint"),
        "val_dev_sha256": verified.get("val_dev.csv"),
        "train_dev_sha256": verified.get("train_dev.csv"),
        "class_to_idx_sha256": verified.get("class_to_idx.json"),
    }


def build_runtime_config(args, candidate: str) -> Path:
    smoke = int(getattr(args, "smoke_max_steps", 0) or 0) > 0
    if not smoke and int(args.microbatch) * int(args.grad_accum) != 1024:
        raise ValueError("effective batch must remain 1024")
    config = load_config(CONFIG_DIR / f"{candidate}.yaml")
    config.pop("_config_path", None)
    stage_dir = _abs(args.stage_dir, STAGE_DIR)
    train_root = _abs(args.train_root, Path("/home/lux1/noise/train"))
    test_root = _abs(args.test_root, Path("/home/lux1/noise/test"))
    for name in (
        "class_to_idx.json",
        "dataset_manifest.json",
        "train_dev.csv",
        "val_dev.csv",
        "features/features.pt",
        "features/image_paths.json",
        "features/manifest.json",
    ):
        if not (stage_dir / name).is_file():
            raise FileNotFoundError(stage_dir / name)
    parent = _abs(args.l05_checkpoint, L05_CHECKPOINT)
    if not parent.is_file():
        raise FileNotFoundError(parent)

    config["data"].update(
        {
            "class_mapping": str(stage_dir / "class_to_idx.json"),
            "dataset_manifest": str(stage_dir / "dataset_manifest.json"),
            "train_csv": str(stage_dir / "train_dev.csv"),
            "val_csv": str(stage_dir / "val_dev.csv"),
            "train_root": str(train_root),
            "test_root": str(test_root),
        }
    )
    config["features"].update(
        {
            "tensor_path": str(stage_dir / "features/features.pt"),
            "paths_path": str(stage_dir / "features/image_paths.json"),
            "manifest_path": str(stage_dir / "features/manifest.json"),
        }
    )
    config["train"].update(
        {
            "device": str(args.device),
            "batch_size": int(args.microbatch),
            "grad_accum_steps": int(args.grad_accum),
            "effective_batch_size": int(args.microbatch) * int(args.grad_accum),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "loader_timeout": 0 if int(args.num_workers) == 0 else 120,
            "npu_pin_memory": False,
            "optimizer_impl": "foreach",
            "gradient_norm_impl": "foreach",
            "init_checkpoint": str(parent),
        }
    )
    config["evaluation"]["batch_size"] = int(args.eval_batch)
    config["evaluation"]["inference_batch_size"] = int(args.infer_batch)
    config["output"]["root"] = str(_abs(args.output_root, OUTPUT_ROOT))
    config["project"]["seed"] = int(args.train_seed)
    config["project"]["protocol"] = "rematch750_search_v5"
    config["project"]["implementation_status"] = "implemented"
    config["project"]["search"]["micro_batch_candidates"] = [int(args.microbatch)]
    config["project"]["search"]["smoke_required"] = False
    if int(getattr(args, "smoke_max_steps", 0) or 0) > 0:
        config["train"]["max_steps"] = int(args.smoke_max_steps)
        config["train"]["epochs"] = 1
        config["train"]["schedule_epochs"] = 1
        config["project"]["experiment_id"] = f"{config['project']['experiment_id']}_smoke"
    if candidate == "NEW03":
        config["loss"]["set_supervision"]["candidate_labels_path"] = str(
            _abs(args.sidecar, SIDECAR)
        )

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    runtime_path = RUNTIME_DIR / (
        f"{candidate}_{args.device.replace(':', '_')}_mb{args.microbatch}"
        f"_seed{int(args.train_seed)}.yaml"
    )
    runtime_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return runtime_path


def build_sidecar(args) -> tuple[Path, dict]:
    from aegis_clip.duplicate_sets import build_candidate_label_sets, conflict_group_report

    stage_dir = _abs(args.stage_dir, STAGE_DIR)
    sidecar = _abs(args.sidecar, SIDECAR)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    report = conflict_group_report(stage_dir / "train_dev.csv")
    build_candidate_label_sets(
        stage_dir / "train_dev.csv", sidecar, num_classes=750
    )
    return sidecar, report


def preflight(candidate: str, runtime_path: Path, parent: Path) -> dict:
    config = load_config(runtime_path)
    split = _frozen_split_report(Path(config["data"]["dataset_manifest"]).parent)
    meta = validate_checkpoint(parent, config, parent=True)
    report = {
        "candidate": candidate,
        "runtime_config": str(runtime_path),
        "train_seed": int(config["project"]["seed"]),
        "split_seed": split["split_seed"],
        "parent_checkpoint": str(parent),
        "parent_sha256": sha256_file(parent),
        "parent_binding_experiment_id": meta.get("experiment_id"),
        "parent_kind": config["project"].get("parent_kind"),
        "parent_experiment_id": config["project"].get("parent_experiment_id"),
        "frozen_split": split,
        "run_dir": str(
            Path(config["output"]["root"])
            / str(config["project"]["experiment_id"])
            / f"seed{int(config['project']['seed'])}"
        ),
    }
    preflight_path = runtime_path.with_suffix(".preflight.json")
    preflight_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _count_train_rows(train_csv: str | Path) -> int:
    with Path(train_csv).open(encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _status(delta_macro_pp: float, delta_micro_pp: float) -> str:
    if delta_macro_pp >= 1.00:
        return "breakthrough"
    if delta_macro_pp >= 0.50 and delta_micro_pp >= 0.0:
        return "strong"
    if delta_macro_pp >= 0.30 and delta_micro_pp >= -0.10:
        return "promote"
    if delta_macro_pp > 0.0:
        return "weak"
    return "fail"


def _append_row(row: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = RESULTS_CSV.is_file()
    rows = []
    if exists:
        with RESULTS_CSV.open(newline="", encoding="utf-8") as handle:
            rows = [
                existing
                for existing in csv.DictReader(handle)
                if existing.get("candidate_id") != row["candidate_id"]
            ]
    rows.append(row)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_FIELDS)
        writer.writeheader()
        for existing in rows:
            writer.writerow({key: existing.get(key, "") for key in RESULTS_FIELDS})


def run_candidate(args, candidate: str) -> dict:
    parent = _abs(args.l05_checkpoint, L05_CHECKPOINT)
    sidecar_report = None
    if candidate == "NEW03":
        _, sidecar_report = build_sidecar(args)
    runtime_path = build_runtime_config(args, candidate)
    report = preflight(candidate, runtime_path, parent)
    run_dir = Path(report["run_dir"])
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    if int(getattr(args, "smoke_max_steps", 0) or 0) > 0 and run_dir.exists():
        shutil.rmtree(run_dir)

    if not args.skip_train:
        _run(
            [
                sys.executable,
                "-m",
                "aegis_clip.cli.rematch",
                "train",
                "--config",
                str(runtime_path),
                "--device",
                str(args.device),
            ]
        )
    if int(getattr(args, "smoke_max_steps", 0) or 0) > 0:
        checkpoint = run_dir / "checkpoints" / "best.pt"
        report = {
            **report,
            "smoke": True,
            "smoke_checkpoint": str(checkpoint),
            "smoke_checkpoint_exists": checkpoint.is_file(),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    checkpoint = run_dir / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"candidate checkpoint missing: {checkpoint}")
    config = load_config(runtime_path)
    cache_path = run_dir / "val_branch_logits.pt"
    if not args.skip_cache:
        _run(
            [
                sys.executable,
                str(ROOT / "scripts/cache_validation_tta_logits.py"),
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(runtime_path),
                "--output",
                str(cache_path),
                "--tta-fusion",
                "mean_probabilities",
                "--tta-temperature",
                "1.0",
                "--device",
                str(args.device),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
            ]
        )
    evaluation_path = run_dir / "evaluation.json"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_l05_candidate.py"),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(runtime_path),
            "--val-branch-cache",
            str(cache_path),
            "--output",
            str(evaluation_path),
        ]
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    selected = json.loads(
        (run_dir / "checkpoints" / "selected_report.json").read_text(encoding="utf-8")
    )
    reference = json.loads(L05_REFERENCE.read_text(encoding="utf-8"))
    l05_decode_macro = float(reference["local"]["selected_raw_macro"])
    delta_macro_pp = 100.0 * (
        float(evaluation["decode"]["macro"]) - l05_decode_macro
    )
    delta_micro_pp = 100.0 * (
        float(evaluation["decode"]["micro"]) - float(reference["local"]["selected_raw_micro"])
    )
    affected = (
        int(sidecar_report["num_conflict_samples"])
        if sidecar_report
        else _count_train_rows(config["data"]["train_csv"])
    )
    recipe = (
        f"stage2:{int(config['train']['epochs'])}ep,"
        f"backbone_lr={float(config['train']['backbone_lr']):g},"
        f"head_lr={float(config['train']['head_lr']):g},"
        f"cosine,eff_batch={int(config['train']['effective_batch_size'])}"
    )
    row = {
        "candidate_id": candidate,
        "base_checkpoint_sha": sha256_file(parent),
        "train_seed": int(config["project"]["seed"]),
        "training_recipe": recipe,
        "selected_epoch": selected.get("selected_epoch"),
        "center_macro": evaluation["center"]["macro"],
        "center_micro": evaluation["center"]["micro"],
        "decode_macro": evaluation["decode"]["macro"],
        "decode_micro": evaluation["decode"]["micro"],
        "delta_vs_l05": f"{delta_macro_pp:.4f}",
        "affected_samples": affected,
        "corrections": evaluation["decode"]["vs_center"]["corrections"],
        "regressions": evaluation["decode"]["vs_center"]["regressions"],
        "status": _status(delta_macro_pp, delta_micro_pp),
        "platform_score": "",
    }
    _append_row(row)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--infer-batch", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--stage-dir", default=str(STAGE_DIR))
    parser.add_argument("--train-root", default="/home/lux1/noise/train")
    parser.add_argument("--test-root", default="/home/lux1/noise/test")
    parser.add_argument("--l05-checkpoint", default=str(L05_CHECKPOINT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--sidecar", default=str(SIDECAR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--smoke-max-steps",
        type=int,
        default=0,
        help="Implementation smoke only: cap optimizer updates and run one epoch.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()

    selected = [item.strip() for item in str(args.candidates).split(",") if item.strip()]
    unknown = [item for item in selected if item not in CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown candidates: {unknown}")
    if args.check_only:
        args.dry_run = True
    started = time.time()
    summary = [run_candidate(args, candidate) for candidate in selected]
    print(
        json.dumps(
            {"elapsed_seconds": round(time.time() - started, 1), "candidates": len(summary)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
