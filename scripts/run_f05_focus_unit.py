#!/usr/bin/env python3
"""Materialize and optionally run one F05-focus unit on a specified worker.

The runner is intentionally path-explicit so the same commit can run on GPU0,
GPU1, or a dry-run host.  C0/N1 are ordinary Aegis training configs; A0/P0 are
Round-0 probes.  D1/D2 composition remains cache-bound and is documented in
``deploy/f05_focus_gpu1`` instead of being guessed here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
if str(AEGIS_ROOT) not in sys.path:
    sys.path.insert(0, str(AEGIS_ROOT))

from aegis_clip.config import load_config  # noqa: E402


UNIT_CONFIG = {
    "C0": ROOT / "configs" / "f05_focus" / "C0_cuda_control.yaml",
    "N1": ROOT / "configs" / "f05_focus" / "N1_hp_noise_drop.yaml",
}
TRAIN_UNITS = {"C0", "N1"}
PROBE_UNITS = {"A0", "P0"}


def _resolve(path: str | Path | None) -> Path | None:
    if path in (None, ""):
        return None
    return Path(str(path)).expanduser().resolve()


def _materialize_train_config(args: argparse.Namespace, unit: str) -> Path:
    config = load_config(UNIT_CONFIG[unit])
    config.pop("_config_path", None)
    config["train"]["device"] = str(args.device)
    config["output"]["root"] = str(Path(args.output_root).expanduser().resolve())
    if args.rm_lp_checkpoint:
        config["train"]["init_checkpoint"] = str(
            Path(args.rm_lp_checkpoint).expanduser().resolve()
        )
    if args.stage_dir:
        stage_dir = Path(args.stage_dir).expanduser().resolve()
        config["data"].update(
            {
                "class_mapping": str(stage_dir / "class_to_idx.json"),
                "dataset_manifest": str(stage_dir / "dataset_manifest.json"),
                "train_csv": str(stage_dir / "train_dev.csv"),
                "val_csv": str(stage_dir / "val_dev.csv"),
            }
        )
        config["features"].update(
            {
                "tensor_path": str(stage_dir / "features" / "features.pt"),
                "paths_path": str(stage_dir / "features" / "image_paths.json"),
                "manifest_path": str(stage_dir / "features" / "manifest.json"),
            }
        )
    if args.train_root:
        config["data"]["train_root"] = str(
            Path(args.train_root).expanduser().resolve()
        )
    if args.test_root:
        config["data"]["test_root"] = str(
            Path(args.test_root).expanduser().resolve()
        )
    if unit == "N1" and args.hp_noise_manifest:
        config["trust"]["sample_weight_path"] = str(
            Path(args.hp_noise_manifest).expanduser().resolve()
        )
    if args.batch_size is not None:
        config["train"]["batch_size"] = int(args.batch_size)
    if args.grad_accum is not None:
        config["train"]["grad_accum_steps"] = int(args.grad_accum)
    effective_batch = int(config["train"]["batch_size"]) * int(
        config["train"]["grad_accum_steps"]
    )
    if effective_batch != 1024:
        raise ValueError(
            "effective batch must remain 1024; got "
            f"{config['train']['batch_size']} x "
            f"{config['train']['grad_accum_steps']} = {effective_batch}"
        )
    if args.num_workers is not None:
        config["train"]["num_workers"] = int(args.num_workers)
    runtime_dir = (
        Path(args.runtime_dir).expanduser()
        if args.runtime_dir
        else Path(args.output_root).expanduser() / "_runtime_configs"
    )
    if not runtime_dir.is_absolute():
        runtime_dir = ROOT / runtime_dir
    runtime_dir = runtime_dir.resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    target = runtime_dir / f"{unit}_{args.device.replace(':', '_')}.yaml"
    target.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def _command(args: argparse.Namespace, unit: str) -> list[str]:
    python = sys.executable
    if unit in TRAIN_UNITS:
        runtime = _materialize_train_config(args, unit)
        return [
            python,
            "-m",
            "aegis_clip.cli.rematch",
            "train",
            "--config",
            str(runtime),
            "--device",
            str(args.device),
        ]
    if unit == "A0":
        if not args.f05_checkpoint:
            raise ValueError("--f05-checkpoint is required for A0")
        command = [
            python,
            str(ROOT / "scripts" / "run_a0_m1_probe.py"),
            "--checkpoint",
            str(Path(args.f05_checkpoint).expanduser().resolve()),
            "--device",
            str(args.device),
            "--output-dir",
            str(Path(args.output_root).expanduser().resolve() / "A0_M1"),
            "--summary",
            str(Path(args.summary).expanduser().resolve()),
        ]
        if args.attention_cache:
            command += [
                "--cache-output",
                str(Path(args.attention_cache).expanduser().resolve()),
            ]
        return command
    if unit == "P0":
        if not args.val_logits:
            raise ValueError("--val-logits is required for P0")
        return [
            python,
            str(ROOT / "scripts" / "run_p0_prior_probe.py"),
            "--validation-logits",
            str(Path(args.val_logits).expanduser().resolve()),
            "--output-dir",
            str(Path(args.output_root).expanduser().resolve() / "P0_PRIOR"),
            "--bias-output",
            str(
                Path(args.prior_bias).expanduser().resolve()
                if args.prior_bias
                else (ROOT / "artifacts" / "f05_focus" / "prior_bias.pt")
            ),
            "--summary",
            str(Path(args.summary).expanduser().resolve()),
        ]
    if unit in {"D1", "D2", "D3"}:
        raise ValueError(
            f"{unit} requires cache precomputation and is not auto-run by this "
            "path runner; use deploy/f05_focus_gpu1/README.md"
        )
    raise ValueError(f"unsupported unit: {unit}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True, choices=sorted(UNIT_CONFIG) + ["A0", "P0", "D1", "D2", "D3"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="outputs/f05_focus")
    parser.add_argument("--summary", default="results/f05_focus_summary.csv")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--f05-checkpoint")
    parser.add_argument("--rm-lp-checkpoint")
    parser.add_argument("--stage-dir")
    parser.add_argument("--train-root")
    parser.add_argument("--test-root")
    parser.add_argument("--hp-noise-manifest")
    parser.add_argument("--val-logits")
    parser.add_argument("--prior-bias")
    parser.add_argument("--attention-cache")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run immediately; default only materializes and prints the command",
    )
    args = parser.parse_args()
    command = _command(args, args.unit)
    environment = dict(os.environ)
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(AEGIS_ROOT) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    rendered = {
        "unit": args.unit,
        "device": args.device,
        "command": command,
        "execute": bool(args.execute),
    }
    print(json.dumps(rendered, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
