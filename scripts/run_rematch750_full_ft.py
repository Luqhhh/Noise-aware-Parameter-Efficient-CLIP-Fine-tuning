#!/usr/bin/env python3
"""Materialize the full-data (148,695) configs and optionally run them on an NPU.

The tracked configs deliberately use ``npu:UNASSIGNED``.  This helper requires the
operator to expose an idle physical NPU, binds the process-local logical device,
retains the exact runtime YAML beside the experiment outputs, and -- unlike a bare
``rematch train`` invocation -- preflights every input path before anything is
written or trained.

That preflight matters because ``aegis_clip.config._resolve_paths`` resolves
relative paths against *the directory holding the config file*, not the repository
root.  ``configs/rematch750_full_ft/FULL01.yaml`` therefore needs ``../../`` where
``configs/rematch750_v3_b1024_e16_lr4.yaml`` uses ``../``.  Copied to a different
depth, those paths silently point somewhere else and the failure only surfaces as a
late ``FileNotFoundError`` deep inside the trainer.

FULL01's parent checkpoint is produced by FULL00 in the same invocation, so the
preflight treats it as satisfied only when FULL00 is part of this run; asking for
FULL01 alone still requires the parent to be on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
if str(AEGIS_ROOT) not in sys.path:
    sys.path.insert(0, str(AEGIS_ROOT))

from aegis_clip.config import load_config  # noqa: E402
from aegis_clip.rematch_assets import validate_dataset  # noqa: E402


TRIALS = ("FULL00", "FULL01")
DEVICE_RE = re.compile(r"^npu:\d+$")
# Every path-shaped input the two configs consume, plus the assets that
# rematch_assets.validate_dataset() reads out of the manifest's own directory.
REQUIRED_INPUTS = (
    ("data", "train_csv"),
    ("data", "val_csv"),
    ("data", "class_mapping"),
    ("data", "dataset_manifest"),
    ("data", "train_root"),
    ("data", "test_root"),
    ("features", "tensor_path"),
    ("features", "paths_path"),
    ("features", "manifest_path"),
    ("model", "official_checkpoint"),
)
EXPECTED_TRAIN_CSV = "full_train.csv"


def run_dir(config: dict) -> Path:
    """Mirror cli/rematch.py: output_root / experiment_id / seed<seed>."""
    return (
        Path(config["output"]["root"])
        / config["project"]["experiment_id"]
        / f"seed{config['project']['seed']}"
    )


def preflight(name: str, config: dict, deferred: tuple[Path, ...] = ()) -> list[str]:
    """Return a list of human-readable problems; empty means the config can run."""
    problems: list[str] = []
    for section, key in REQUIRED_INPUTS:
        value = config.get(section, {}).get(key)
        if not value:
            problems.append(f"{name}: {section}.{key} is unset")
        elif not Path(value).exists():
            problems.append(f"{name}: missing {section}.{key} -> {value}")
    source = config.get("train", {}).get("init_checkpoint")
    if source and not Path(source).exists():
        # A parent produced earlier in this same invocation is legitimately absent.
        if not any(Path(source).is_relative_to(directory) for directory in deferred):
            problems.append(
                f"{name}: missing train.init_checkpoint -> {source}"
                " (run FULL00 in the same invocation, or point at an existing LP)"
            )
    train_csv = config.get("data", {}).get("train_csv")
    if train_csv and Path(train_csv).name != EXPECTED_TRAIN_CSV:
        problems.append(
            f"{name}: data.train_csv must be {EXPECTED_TRAIN_CSV} for a full-data run,"
            f" got {Path(train_csv).name}"
        )
    if problems:
        # Existence first: validate_dataset() would raise FileNotFoundError for the
        # asset directory before reporting the more useful root/scope mismatches.
        return problems
    # Run the real provenance gate now rather than discovering a rejected config
    # after the first epoch has already been paid for.
    try:
        validate_dataset(config)
    except Exception as error:  # noqa: BLE001 - surfaced verbatim to the operator
        problems.append(f"{name}: provenance gate rejected this config: {error}")
    return problems


def materialize_config(
    name: str, config: dict, device: str, runtime_dir: Path, deferred: tuple[Path, ...]
) -> Path:
    if config["train"].get("device") != "npu:UNASSIGNED":
        raise ValueError(f"tracked config lost its fail-closed device: {name}")
    problems = preflight(name, config, deferred)
    if problems:
        raise FileNotFoundError(
            "input preflight failed; nothing was materialized:\n  " + "\n  ".join(problems)
        )
    rendered_config = {key: value for key, value in config.items() if not key.startswith("_")}
    rendered_config["train"]["device"] = device
    runtime_dir.mkdir(parents=True, exist_ok=True)
    target = runtime_dir / f"{name}.yaml"
    rendered = yaml.safe_dump(rendered_config, sort_keys=False, allow_unicode=True)
    if target.exists():
        # A reviewed dry-run must be reusable by the subsequent --execute
        # invocation, while stale or hand-edited runtime YAML remains a hard
        # failure rather than being silently overwritten.
        if target.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(
                f"runtime config already exists with different content: {target}"
            )
    else:
        target.write_text(rendered, encoding="utf-8")
    # A second strict load proves that all now-absolute paths remain valid
    # configuration values when the file moves out of configs/.
    if load_config(target)["train"]["device"] != device:
        raise RuntimeError("runtime device binding did not survive reload")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, help="logical NPU, e.g. npu:0")
    parser.add_argument("--trials", nargs="+", choices=TRIALS, default=list(TRIALS))
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("train", "infer"),
        default=("train", "infer"),
        help="infer only ever runs for FULL01; FULL00 is an initialization",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run after materialization; default is a safe dry run",
    )
    args = parser.parse_args()

    if not DEVICE_RE.fullmatch(args.device):
        parser.error("--device must be explicit, for example npu:0")
    if args.execute and not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("--execute requires ASCEND_RT_VISIBLE_DEVICES to select an idle physical NPU")

    output_root = ROOT / "outputs" / "clairvoyanttt" / "rematch750_full_ft"
    runtime_dir = output_root / "_runtime_configs"
    configs = {
        trial: load_config(ROOT / "configs" / "rematch750_full_ft" / f"{trial}.yaml")
        for trial in args.trials
    }
    runtimes: dict[str, Path] = {}
    for trial in args.trials:
        # Only siblings produced in this same invocation may be missing on disk.
        deferred = tuple(
            run_dir(sibling)
            for other, sibling in configs.items()
            if other != trial
        )
        runtimes[trial] = materialize_config(
            trial, configs[trial], args.device, runtime_dir, deferred
        )

    planned = [
        (trial, stage)
        for trial in args.trials
        for stage in args.stages
        # FULL00 exists only to become FULL01's parent; it has no submission package.
        if not (stage == "infer" and trial != "FULL01")
    ]
    if "infer" in args.stages and not any(stage == "infer" for _, stage in planned):
        print("note: no submission will be produced; only FULL01 has an infer stage")
    commands = [
        [
            sys.executable,
            "-m",
            "aegis_clip.cli.rematch",
            stage,
            "--config",
            str(runtimes[trial]),
        ]
        for trial, stage in planned
    ]

    print(json.dumps({"device": args.device, "commands": commands}, indent=2))
    if not args.execute:
        return 0

    env = dict(os.environ)
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(AEGIS_ROOT) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
