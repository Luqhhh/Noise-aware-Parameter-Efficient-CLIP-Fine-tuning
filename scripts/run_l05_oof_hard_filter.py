#!/usr/bin/env python3
"""Build the pinned L05 runner and, only on explicit request, train filtered LP/FT."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "f050ecb59e0a885da0b43cdc6c8c316953fb5e7d"
SOURCE_V4_SHA256 = "7472f598be2637069bb7471a7c2f556c7e3ebe0bada6233d243862bfc1aa195a"
OUT = ROOT / "outputs/codex/l05_oof_hard_filter"
RUNNER = OUT / "runner"
RUNS = OUT / "runs"
PREP_CONFIG = json.loads((ROOT / "configs/l05_oof_hard_filter/fixed.json").read_text())
ASSETS = Path(PREP_CONFIG["dataset_manifest"]).parent
SELECTION = OUT / "cpu_prepare"
OVERLAY = ROOT / "scripts/l05_oof_hard_filter_rematch_search_v4.py"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git_show(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=ROOT)


def _archive_framework() -> None:
    archive = subprocess.check_output([
        "git", "archive", "--format=tar", SOURCE_COMMIT,
        "reproducibility/aegis_f1/aegis_clip", "scripts/check_submission.py",
    ], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar:
            target = RUNNER / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                assert source is not None
                target.write_bytes(source.read())
            else:
                raise ValueError(f"unexpected archive member type: {member.name}")


def _set_paths(config: dict, *, selected: Path) -> None:
    stage = json.loads((ASSETS / "dataset_manifest.json").read_text())
    config["data"].update(
        train_root=str(Path(stage["train_root"])),
        test_root=str(Path(stage["test_root"])),
        train_csv=str(selected),
        train_selection_manifest=str(SELECTION / "manifest.json"),
        val_csv=str(ASSETS / "val_dev.csv"),
        class_mapping=str(ASSETS / "class_to_idx.json"),
        dataset_manifest=str(ASSETS / "dataset_manifest.json"),
    )
    config["features"].update(
        tensor_path=str(ASSETS / "features/features.pt"),
        paths_path=str(ASSETS / "features/image_paths.json"),
        manifest_path=str(ASSETS / "features/manifest.json"),
    )
    config["model"]["official_checkpoint"] = str(Path.home() / ".cache/clip/ViT-B-32.pt")
    config["output"]["root"] = str(RUNS)


def _configs() -> dict[str, str]:
    selected = SELECTION / "train_filtered.csv"
    lp = yaml.safe_load(git_show("configs/rematch750_lp.yaml"))
    _set_paths(lp, selected=selected)
    lp["project"].update(experiment_id="RM_LP_HF01", protocol="rematch750_search_v4",
                         parent_kind="official_clip_head")
    lp["train"].update(device="cuda:0", num_workers=2, pin_memory=True)

    ft = yaml.safe_load(git_show("configs/rematch750_search_v5/L05.yaml"))
    _set_paths(ft, selected=selected)
    ft["project"].update(experiment_id="RM_V5_L05_HF01", parent_kind="shared_lp")
    ft["train"].update(
        device="cuda:0", batch_size=4, grad_accum_steps=256,
        effective_batch_size=1024, num_workers=2, prefetch_factor=2,
        pin_memory=True, npu_pin_memory=False, optimizer_impl="foreach",
        gradient_norm_impl="foreach", init_checkpoint=str(
            RUNS / "RM_LP_HF01/seed42/checkpoints/best.pt"),
    )
    ft["project"]["search"].update(micro_batch_candidates=[4], smoke_required=False)
    ft["evaluation"].update(batch_size=32, inference_batch_size=32)
    configs = {"LP_HF01.yaml": lp, "L05_HF01.yaml": ft}
    config_dir = RUNNER / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return {name: yaml.safe_dump(value, sort_keys=False) for name, value in configs.items()}


def materialize() -> None:
    existing = RUNNER / "runner_manifest.json"
    if existing.exists():
        manifest = json.loads(existing.read_text())
        require = {
            "source_commit": SOURCE_COMMIT,
            "overlay_sha256": sha256(OVERLAY),
            "builder_sha256": sha256(Path(__file__)),
            "selection_manifest_sha256": sha256(SELECTION / "manifest.json"),
        }
        if any(manifest.get(key) != value for key, value in require.items()):
            raise ValueError("existing pinned runner differs from this source or selection")
        for relative, digest in manifest["files"].items():
            if sha256(RUNNER / relative) != digest:
                raise ValueError(f"pinned runner changed: {relative}")
        return
    if RUNNER.exists() and any(RUNNER.iterdir()):
        raise FileExistsError(f"runner directory is nonempty without a manifest: {RUNNER}")
    RUNNER.mkdir(parents=True, exist_ok=True)
    source = git_show("reproducibility/aegis_f1/aegis_clip/rematch_search_v4.py")
    if hashlib.sha256(source).hexdigest() != SOURCE_V4_SHA256:
        raise ValueError("pinned protocol source hash changed")
    _archive_framework()
    target = RUNNER / "reproducibility/aegis_f1/aegis_clip/rematch_search_v4.py"
    if target.read_bytes() != source:
        raise ValueError("archived protocol source differs from pinned commit")
    target.write_bytes(OVERLAY.read_bytes())
    configs = _configs()
    for name, content in configs.items():
        (RUNNER / "configs" / name).write_text(content, encoding="utf-8")
    files = sorted([
        *(RUNNER / "reproducibility/aegis_f1/aegis_clip").rglob("*.py"),
        RUNNER / "scripts/check_submission.py",
        *(RUNNER / "configs" / name for name in configs),
    ])
    payload = {
        "source_commit": SOURCE_COMMIT,
        "overlay_sha256": sha256(OVERLAY),
        "builder_sha256": sha256(Path(__file__)),
        "selection_manifest_sha256": sha256(SELECTION / "manifest.json"),
        "files": {str(path.relative_to(RUNNER)): sha256(path) for path in files},
    }
    existing.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(RUNNER / "reproducibility/aegis_f1"), str(RUNNER),
        env.get("PYTHONPATH", ""),
    ])
    return env


def verify() -> None:
    script = """
from aegis_clip.config import load_config
from aegis_clip.rematch_protocol import validate_dataset, validate_cache, validate_training, checkpoint_binding
from aegis_clip.rematch_search_v5 import validate_declaration
from pathlib import Path
import json
root = Path(__import__('sys').argv[1])
lp = load_config(root / 'configs/LP_HF01.yaml')
ft = load_config(root / 'configs/L05_HF01.yaml')
validate_training(lp)
validate_dataset(ft)
validate_cache(ft)
validate_declaration(ft, require_implemented=True)
lp_binding = checkpoint_binding(lp)
ft_binding = checkpoint_binding(ft)
assert lp_binding['train_csv_sha256'] == ft_binding['train_csv_sha256']
assert lp_binding['train_selection_manifest_sha256'] == ft_binding['train_selection_manifest_sha256']
assert lp['model']['use_cached_training'] and not ft['model']['use_cached_training']
assert ft['train']['init_checkpoint'].endswith('/RM_LP_HF01/seed42/checkpoints/best.pt')
print(json.dumps({'status':'cpu_protocol_verified',
                  'selected_train_sha256': lp_binding['train_csv_sha256'],
                  'selection_manifest_sha256': lp_binding['train_selection_manifest_sha256'],
                  'lp_experiment': lp['project']['experiment_id'],
                  'ft_experiment': ft['project']['experiment_id']}))
"""
    subprocess.run([sys.executable, "-c", script, str(RUNNER)], cwd=RUNNER,
                   env=environment(), check=True)


def train(phase: str, device: str) -> None:
    config = RUNNER / "configs" / ("LP_HF01.yaml" if phase == "lp" else "L05_HF01.yaml")
    if phase == "ft":
        checkpoint = RUNS / "RM_LP_HF01/seed42/checkpoints/best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError("filtered LP checkpoint missing; run train-lp first")
    subprocess.run([
        sys.executable, "-m", "aegis_clip.cli.rematch", "train",
        "--config", str(config), "--device", device,
    ], cwd=RUNNER, env=environment(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train-lp", "train-ft", "run"))
    parser.add_argument("--device", default="cuda:0",
                        help="Explicit accelerator to use when training; prepare never opens it")
    args = parser.parse_args()
    materialize()
    verify()
    if args.action == "train-lp":
        train("lp", args.device)
    elif args.action == "train-ft":
        train("ft", args.device)
    elif args.action == "run":
        train("lp", args.device)
        train("ft", args.device)


if __name__ == "__main__":
    main()
