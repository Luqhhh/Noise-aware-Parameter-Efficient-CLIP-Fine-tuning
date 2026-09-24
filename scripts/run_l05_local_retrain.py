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
    config["project"]["protocol"] = "rematch750_search_v5"
    config["project"]["implementation_status"] = "implemented"
    config["project"]["search"]["micro_batch_candidates"] = [int(args.microbatch)]
    config["project"]["search"]["smoke_required"] = False

    runtime_root = _abs(args.runtime_dir, ROOT / "outputs/f05_focus_l05/_runtime_configs")
    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_root / f"L05_{args.experiment_id}_{args.device.replace(':', '_')}_mb{args.microbatch}.yaml"
    runtime_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
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
                / f"L05_{args.experiment_id}_{args.device.replace(':', '_')}_mb{candidate}.yaml"
            )
            if runtime_candidate.is_file():
                shutil.rmtree(_run_dir(runtime_candidate), ignore_errors=True)
    if last_error is not None:
        raise last_error
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
