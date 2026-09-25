#!/usr/bin/env python3
"""Local/GPU1 L05 retraining and desktop submission export.

L05 = R02 (384px global full fine-tune) with confidence-gated attention-local
supervision starting at epoch 5 and ``local_supervision_weight = 0.25``.
The recipe is the existing V5 ``L05`` config; this driver only supplies
machine-local paths, device binding, effective-batch-preserving microbatch
configuration, and post-training submission export.
"""

from __future__ import annotations

import argparse
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
L05_CONFIG = ROOT / "configs" / "rematch750_search_v5" / "L05.yaml"
if str(AEGIS_ROOT) not in sys.path:
    sys.path.insert(0, str(AEGIS_ROOT))

from aegis_clip.config import load_config  # noqa: E402
from aegis_clip.runtime import sha256_file  # noqa: E402

#: The data split is a frozen artifact: ``dataset_manifest.json`` records this
#: seed and the train/val CSVs were written once.  Changing ``--train-seed`` must
#: never re-draw it.
SPLIT_SEED = 42

#: The audited shared RM-LP parent (``parent_kind=shared_lp``).  A different
#: parent changes the experiment and must be opted into explicitly.
DEFAULT_RM_LP_SHA256 = "d5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b"


def _abs(value: str | Path | None, default: Path | None = None) -> Path | None:
    if value in (None, ""):
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _default_desktop() -> Path:
    candidates = sorted(Path("/mnt/c/Users").glob("*/Desktop"))
    if candidates:
        return candidates[0]
    home_desktop = Path.home() / "Desktop"
    return home_desktop


def _frozen_split_report(stage_dir: Path) -> dict:
    """Fail closed unless the frozen train/val content-group split is untouched."""
    manifest_path = stage_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("seed", -1)) != SPLIT_SEED:
        raise ValueError(
            "dataset manifest records split seed "
            f"{manifest.get('seed')!r}, expected the frozen {SPLIT_SEED}; "
            "the training seed must not re-draw the split"
        )
    verified: dict[str, str] = {}
    for name, digest in manifest.get("files", {}).items():
        path = stage_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"frozen split asset changed since the split was built: {name}")
        verified[name] = actual
    return {
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "split_seed": int(manifest["seed"]),
        "train_fingerprint": manifest.get("train_fingerprint"),
        "test_fingerprint": manifest.get("test_fingerprint"),
        "verified_assets": verified,
    }


def _parent_report(rm_lp: Path, expected_sha256: str | None) -> dict:
    actual = sha256_file(rm_lp)
    if expected_sha256 and actual != str(expected_sha256):
        raise ValueError(
            f"RM-LP parent SHA-256 changed: {actual} != expected {expected_sha256}"
        )
    return {
        "parent_checkpoint": str(rm_lp),
        "parent_sha256": actual,
        "expected_parent_sha256": str(expected_sha256) if expected_sha256 else None,
    }


def _normalize_recipe(config: dict) -> dict:
    """Drop the training seed and path metadata so only the recipe remains."""
    normalized = json.loads(json.dumps(config, default=str))
    normalized.pop("_config_path", None)
    normalized.get("project", {}).pop("seed", None)
    return normalized


def _config_diff(left, right, prefix: str = "") -> list[str]:
    diffs: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            diffs.extend(_config_diff(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else str(key)))
    elif left != right:
        diffs.append(f"{prefix}: {left!r} != {right!r}")
    return diffs


def build_runtime_config(args: argparse.Namespace) -> Path:
    if int(args.microbatch) * int(args.grad_accum) != 1024:
        raise ValueError(
            "effective batch must remain 1024: "
            f"{args.microbatch} x {args.grad_accum} != 1024"
        )
    config = load_config(L05_CONFIG)
    config.pop("_config_path", None)

    stage_dir = _abs(args.stage_dir, ROOT / "artifacts/stages/repechage/20260921")
    train_root = _abs(args.train_root, ROOT / "train")
    test_root = _abs(args.test_root, ROOT / "test")
    rm_lp = _abs(args.rm_lp_checkpoint)
    if stage_dir is None or not stage_dir.is_dir():
        raise FileNotFoundError(stage_dir)
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
    if rm_lp is None or not rm_lp.is_file():
        raise FileNotFoundError(rm_lp)

    split_report = _frozen_split_report(stage_dir)
    parent_report = _parent_report(rm_lp, args.expected_parent_sha256)

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
    config["model"]["input_resolution"] = int(args.resolution)
    config["train"].update(
        {
            "device": str(args.device),
            "batch_size": int(args.microbatch),
            "grad_accum_steps": int(args.grad_accum),
            "effective_batch_size": 1024,
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "npu_pin_memory": False,
            "optimizer_impl": "foreach",
            "gradient_norm_impl": "foreach",
            "init_checkpoint": str(rm_lp),
        }
    )
    config["evaluation"]["batch_size"] = int(args.eval_batch)
    config["evaluation"]["inference_batch_size"] = int(args.infer_batch)
    config["output"]["root"] = str(_abs(args.output_root, ROOT / "outputs/f05_focus_l05"))
    config["project"]["experiment_id"] = str(args.experiment_id)
    config["project"]["seed"] = int(args.train_seed)
    config["project"]["protocol"] = "rematch750_search_v5"
    config["project"]["implementation_status"] = "implemented"
    config["project"]["search"]["micro_batch_candidates"] = [int(args.microbatch)]
    config["project"]["search"]["smoke_required"] = False

    runtime_root = _abs(args.runtime_dir, ROOT / "outputs/f05_focus_l05/_runtime_configs")
    runtime_root.mkdir(parents=True, exist_ok=True)
    stem = (
        f"L05_{args.experiment_id}_{args.device.replace(':', '_')}_mb{args.microbatch}"
        f"_seed{int(args.train_seed)}"
    )
    runtime_path = runtime_root / f"{stem}.yaml"

    recipe_reference = _abs(args.recipe_reference_config)
    recipe_diff: list[str] | None = None
    recipe_reference_note: str | None = None
    if recipe_reference is not None:
        if recipe_reference.is_file():
            reference = load_config(recipe_reference)
            recipe_diff = _config_diff(
                _normalize_recipe(config), _normalize_recipe(reference)
            )
            if recipe_diff:
                raise ValueError(
                    "training recipe differs from the reference config beyond the "
                    f"training seed: {recipe_diff[:5]}"
                )
        else:
            recipe_reference_note = f"reference config not found: {recipe_reference}"

    runtime_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    preflight = {
        "experiment_id": str(args.experiment_id),
        "train_seed": int(args.train_seed),
        "split_seed": SPLIT_SEED,
        "device": str(args.device),
        "microbatch": int(args.microbatch),
        "grad_accum": int(args.grad_accum),
        "runtime_config": str(runtime_path),
        "run_dir": str(
            Path(config["output"]["root"]) / str(args.experiment_id) / f"seed{int(args.train_seed)}"
        ),
        "frozen_split": split_report,
        "parent": parent_report,
        "recipe_reference_config": str(recipe_reference) if recipe_reference else None,
        "recipe_diff": recipe_diff,
        "recipe_reference_note": recipe_reference_note,
    }
    preflight_path = runtime_root / f"{stem}.preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preflight, ensure_ascii=False, indent=2, default=str), flush=True)
    return runtime_path


def _run_dir(runtime_config: Path) -> Path:
    config = load_config(runtime_config)
    return (
        Path(config["output"]["root"])
        / str(config["project"]["experiment_id"])
        / f"seed{int(config['project']['seed'])}"
    )


def _run(command: list[str]) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(AEGIS_ROOT) + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    print("RUN", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def export_submission(runtime_config: Path, desktop: Path, experiment_id: str) -> Path:
    config = load_config(runtime_config)
    run_dir = (
        Path(config["output"]["root"])
        / str(config["project"]["experiment_id"])
        / f"seed{int(config['project']['seed'])}"
    )
    submission = run_dir / "submission"
    if not submission.is_dir():
        raise FileNotFoundError(submission)
    required = ["pred_results.csv", "submission.zip", "manifest.json"]
    missing = [name for name in required if not (submission / name).is_file()]
    if missing:
        raise FileNotFoundError(f"submission package missing: {missing}")

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    destination = desktop / f"{experiment_id}_submission_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    for name in required:
        shutil.copy2(submission / name, destination / name)
    receipt = {
        "experiment_id": experiment_id,
        "runtime_config": str(runtime_config),
        "run_dir": str(run_dir),
        "desktop_package": str(destination),
        "files": required,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (destination / "EXPORT_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=256)
    parser.add_argument("--auto-microbatch", action="store_true")
    parser.add_argument("--microbatch-candidates", default="8,4,2")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--infer-batch", type=int, default=32)
    parser.add_argument("--stage-dir")
    parser.add_argument("--train-root")
    parser.add_argument("--test-root")
    parser.add_argument("--rm-lp-checkpoint")
    parser.add_argument("--output-root", default="outputs/f05_focus_l05")
    parser.add_argument("--runtime-dir", default="outputs/f05_focus_l05/_runtime_configs")
    parser.add_argument("--experiment-id", default="RM_V5_L05_CUDA")
    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
        help=(
            "Explicit training seed. The frozen train/val content-group split is "
            f"NOT re-drawn; it stays at split_seed={SPLIT_SEED}."
        ),
    )
    parser.add_argument(
        "--expected-parent-sha256",
        default=DEFAULT_RM_LP_SHA256,
        help="Audited shared RM-LP parent SHA-256; pass '' to skip the check.",
    )
    parser.add_argument(
        "--recipe-reference-config",
        default=(
            "outputs/f05_focus_l05/_runtime_configs/"
            "L05_RM_V5_L05_CUDA_LOCAL_cuda_0_mb4.yaml"
        ),
        help="Config whose recipe (ignoring seed) the new run must match.",
    )
    parser.add_argument("--desktop-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    args = parser.parse_args()

    def execute_candidate(microbatch: int, grad_accum: int) -> Path:
        args.microbatch = int(microbatch)
        args.grad_accum = int(grad_accum)
        runtime = build_runtime_config(args)
        print(
            json.dumps(
                {
                    "runtime_config": str(runtime),
                    "microbatch": int(microbatch),
                    "grad_accum": int(grad_accum),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        train_command = [
            sys.executable,
            "-m",
            "aegis_clip.cli.rematch",
            "train",
            "--config",
            str(runtime),
            "--device",
            str(args.device),
        ]
        infer_command = [
            sys.executable,
            "-m",
            "aegis_clip.cli.rematch",
            "infer",
            "--config",
            str(runtime),
            "--device",
            str(args.device),
        ]
        if args.dry_run:
            print(json.dumps({"train": train_command, "infer": infer_command}, indent=2))
            return runtime
        if not args.skip_train:
            _run(train_command)
        if not args.skip_infer:
            _run(infer_command)
            desktop = _abs(args.desktop_dir, _default_desktop())
            destination = export_submission(runtime, desktop, str(args.experiment_id))
            print(
                json.dumps(
                    {"desktop_package": str(destination)},
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
        return runtime

    if not args.auto_microbatch:
        execute_candidate(args.microbatch, args.grad_accum)
        return 0

    candidates = [
        int(value)
        for value in str(args.microbatch_candidates).split(",")
        if value.strip()
    ]
    if not candidates:
        raise ValueError("--microbatch-candidates is empty")
    last_error: Exception | None = None
    for candidate in candidates:
        if 1024 % candidate != 0:
            continue
        grad_accum = 1024 // candidate
        try:
            execute_candidate(candidate, grad_accum)
            return 0
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(
                json.dumps(
                    {
                        "microbatch_failed": candidate,
                        "return_code": int(exc.returncode),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            # The runtime path is deterministic for the candidate.
            runtime_root = _abs(
                args.runtime_dir, ROOT / "outputs/f05_focus_l05/_runtime_configs"
            )
            runtime_candidate = (
                runtime_root
                / f"L05_{args.experiment_id}_{args.device.replace(':', '_')}"
                f"_mb{candidate}_seed{int(args.train_seed)}.yaml"
            )
            if runtime_candidate.is_file():
                candidate_run_dir = _run_dir(runtime_candidate)
                if (candidate_run_dir / "checkpoints/best.pt").is_file():
                    # Training succeeded; do not discard a valid checkpoint
                    # because a later inference/export step failed.
                    print(
                        json.dumps(
                            {
                                "post_train_failure_with_valid_checkpoint": str(
                                    candidate_run_dir
                                )
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    raise
                shutil.rmtree(candidate_run_dir, ignore_errors=True)
    if last_error is not None:
        raise last_error
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
