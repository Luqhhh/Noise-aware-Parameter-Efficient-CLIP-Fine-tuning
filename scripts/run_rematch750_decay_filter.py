#!/usr/bin/env python3
"""Materialize and optionally run the REMATCH750 decay-filter factorial."""
from __future__ import annotations

import argparse
import hashlib
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


TRIALS = ("DF01", "DF02", "DF03")
DEVICE_RE = re.compile(r"^npu:\d+$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_trials() -> dict[str, dict]:
    path = ROOT / "configs" / "rematch750_decay_filter" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    trials = {trial["name"]: trial for trial in manifest["trials"]}
    if tuple(trials) != TRIALS:
        raise ValueError("manifest trials differ from runner declaration")
    return trials


def materialize_config(source: Path, device: str, runtime_dir: Path) -> Path:
    if not DEVICE_RE.fullmatch(device):
        raise ValueError("device must be explicit, for example npu:0")
    config = load_config(source)
    if config["train"].get("device") != "npu:UNASSIGNED":
        raise ValueError(f"tracked config lost its fail-closed device: {source}")
    config.pop("_config_path", None)
    config["train"]["device"] = device
    runtime_dir.mkdir(parents=True, exist_ok=True)
    target = runtime_dir / source.name
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if target.exists():
        if target.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(
                f"runtime config already exists with different content: {target}"
            )
    else:
        target.write_text(rendered, encoding="utf-8")
    if load_config(target)["train"]["device"] != device:
        raise RuntimeError("runtime device binding did not survive reload")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, help="logical NPU, e.g. npu:0")
    parser.add_argument("--trials", nargs="+", choices=TRIALS, default=list(TRIALS))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not DEVICE_RE.fullmatch(args.device):
        parser.error("--device must be explicit, for example npu:0")
    if args.execute and not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error(
            "--execute requires ASCEND_RT_VISIBLE_DEVICES to select an idle physical NPU"
        )

    manifest_trials = _manifest_trials()
    runtime_dir = (
        ROOT / "outputs" / "xjn" / "rematch750_decay_filter" / "_runtime_configs"
    )
    commands: list[list[str]] = []
    for trial in args.trials:
        entry = manifest_trials[trial]
        source = ROOT / entry["config"]
        if _sha256(source) != entry["config_sha256"]:
            raise ValueError(f"tracked config hash mismatch: {source}")
        runtime = materialize_config(source, args.device, runtime_dir)
        commands.append(
            [
                sys.executable,
                "-m",
                "aegis_clip.cli.rematch",
                "train",
                "--config",
                str(runtime),
            ]
        )
    print(
        json.dumps(
            {
                "device": args.device,
                "baseline_dependency": "RM_HL00_F05_CONTROL",
                "commands": commands,
            },
            indent=2,
        )
    )
    if not args.execute:
        return 0

    env = dict(os.environ)
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(AEGIS_ROOT) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
