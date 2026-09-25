#!/usr/bin/env python3
"""Build one L05 flip-TTA + validation-fitted-prior submission package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
if str(AEGIS_ROOT) not in sys.path:
    sys.path.insert(0, str(AEGIS_ROOT))

from aegis_clip.checkpoint import build_from_checkpoint  # noqa: E402
from aegis_clip.config import load_config  # noqa: E402
from aegis_clip.data import TestImageDataset  # noqa: E402
from aegis_clip.prior_alignment import apply_prior_bias, fit_prior_bias  # noqa: E402
from aegis_clip.runtime import seed_worker, sha256_file  # noqa: E402
from aegis_clip.submission import create_submission  # noqa: E402
from aegis_clip.tta import fuse_paired_logits  # noqa: E402


def _mapping(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): str(value) for key, value in raw.items()}


def _source_hashes(config: dict) -> dict[str, str] | None:
    manifest = config["data"].get("dataset_manifest")
    if not manifest:
        return None
    path = Path(manifest).parent / "test_manifest.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            Path(row["image_path"]).name: row["file_sha256"]
            for row in csv.DictReader(handle)
        }


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val-branch-cache", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--prior-strength", type=float, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-root", default="outputs/f05_focus_l05")
    parser.add_argument("--desktop-dir", default="/mnt/c/Users/lqh22/Desktop")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    val_payload = torch.load(args.val_branch_cache, map_location="cpu", weights_only=False)
    required = {"original_logits", "flip_logits"}
    missing = required - set(val_payload)
    if missing:
        raise ValueError(f"validation branch cache missing: {sorted(missing)}")
    val_fused = fuse_paired_logits(
        val_payload["original_logits"].float(),
        val_payload["flip_logits"].float(),
        mode="mean_probabilities",
        temperature=float(args.temperature),
    )
    bias, fit_report = fit_prior_bias(val_fused, max_iterations=50)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, preprocess, _ = build_from_checkpoint(
        checkpoint_path, device, config_override=config
    )
    model.eval()
    dataset = TestImageDataset(
        config["data"]["test_root"], preprocess, source_hashes=_source_hashes(config)
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
        worker_init_fn=seed_worker,
    )
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    names: list[str] = []
    original_parts: list[torch.Tensor] = []
    flipped_parts: list[torch.Tensor] = []
    for batch in loader:
        if bool(batch["corrupt"].any()):
            raise RuntimeError("corrupt test image detected")
        images = batch["images"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            original = model(images=images)
            flipped = model(images=torch.flip(images, dims=(3,)))
        original_parts.append(original.detach().float().cpu())
        flipped_parts.append(flipped.detach().float().cpu())
        names.extend(str(name) for name in batch["name"])
    original = torch.cat(original_parts, 0)
    flipped = torch.cat(flipped_parts, 0)
    if len(names) != int(config["data"]["expected_test_samples"]):
        raise RuntimeError("test name count mismatch")
    fused = fuse_paired_logits(
        original, flipped, mode="mean_probabilities", temperature=float(args.temperature)
    )
    corrected = apply_prior_bias(fused, bias, strength=float(args.prior_strength))
    mapping = _mapping(Path(config["data"]["class_mapping"]).resolve())
    valid_labels = {str(value).zfill(4) for value in mapping.values()}
    predictions = [
        (name, mapping[int(index)].zfill(4))
        for name, index in zip(names, corrected.argmax(dim=1).tolist())
    ]
    output_root = Path(args.output_root).expanduser().resolve()
    destination = output_root / str(args.tag)
    if destination.exists():
        shutil.rmtree(destination)
    inference_mode = (
        f"horizontal_flip:mean_probabilities:t={float(args.temperature):g}:"
        f"balanced_prior={float(args.prior_strength):g}"
    )
    create_submission(
        predictions,
        names,
        destination,
        checkpoint_path,
        inference_mode=inference_mode,
        tta_risk_acknowledged=True,
        valid_labels=valid_labels,
        extra_manifest={
            "candidate": str(args.tag),
            "temperature": float(args.temperature),
            "prior_strength": float(args.prior_strength),
            "prior_fit_iterations": int(fit_report["iterations"]),
            "config_sha256": sha256_file(config_path),
            "val_branch_cache": str(Path(args.val_branch_cache).resolve()),
        },
        overwrite=True,
        space_after_comma=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_submission.py"),
            "--test_dir",
            str(config["data"]["test_root"]),
            "--class-mapping",
            str(config["data"]["class_mapping"]),
            "--csv",
            str(destination / "pred_results.csv"),
            "--zip",
            str(destination / "submission.zip"),
        ],
        check=True,
    )
    desktop = Path(args.desktop_dir).expanduser().resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    desktop_package = desktop / f"L05_{args.tag}_{stamp}"
    if desktop_package.exists():
        shutil.rmtree(desktop_package)
    desktop_package.mkdir(parents=True)
    for name in ("pred_results.csv", "submission.zip", "manifest.json"):
        shutil.copy2(destination / name, desktop_package / name)
    print(
        json.dumps(
            {
                "candidate": str(args.tag),
                "temperature": float(args.temperature),
                "prior_strength": float(args.prior_strength),
                "package": str(destination),
                "desktop_package": str(desktop_package),
                "csv_sha256": sha256_file(destination / "pred_results.csv"),
                "zip_sha256": sha256_file(destination / "submission.zip"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
