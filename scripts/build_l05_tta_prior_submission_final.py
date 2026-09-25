#!/usr/bin/env python3
"""Build one L05 flip-TTA + validation-fitted-prior submission package.

Fail-closed by default:

* the destination candidate directory is never deleted or overwritten -- a new
  candidate needs a new tag, and a tag listed in the protected-package registry
  is refused outright;
* the validation branch cache is re-bound to the checkpoint, class mapping,
  frozen split, content-group partition, resolution/preprocessing/precision and
  TTA rule before any bias is fitted;
* the fitted bias is recorded with its tensor SHA-256 and the SHA-256 of the
  exact cache bytes it was fitted from, so a new-seed checkpoint cannot silently
  reuse the previous model's prior;
* the frozen L05 recipe (horizontal flip, ``mean_probabilities``, T=1.4,
  strength=0.60) is enforced unless ``--allow-recipe-drift`` is passed.
"""

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
from aegis_clip.prior_alignment import apply_prior_bias  # noqa: E402
from aegis_clip.runtime import seed_worker, sha256_file  # noqa: E402
from aegis_clip.submission import create_submission  # noqa: E402
from aegis_clip.tta import fuse_paired_logits  # noqa: E402
from aegis_clip.tta_prior_binding import (  # noqa: E402
    checkpoint_identity,
    fit_bound_prior,
    resolved_recipe,
    validation_cache_identity,
    verify_prior_record,
)


def _mapping(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): str(value) for key, value in raw.items()}


def _protected_candidates(registry: Path) -> dict[str, dict]:
    if not registry.is_file():
        return {}
    payload = json.loads(registry.read_text(encoding="utf-8"))
    entries = payload.get("protected_packages", [])
    return {str(entry["candidate"]): entry for entry in entries}


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
    parser.add_argument("--fusion", default="mean_probabilities")
    parser.add_argument("--tta", default="horizontal_flip")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-root", default="outputs/f05_focus_l05")
    parser.add_argument("--desktop-dir", default="/mnt/c/Users/lqh22/Desktop")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument(
        "--protected-registry",
        default="results/f05_focus_l05_protected_packages.json",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="DANGEROUS: allow replacing an existing candidate directory.",
    )
    parser.add_argument(
        "--allow-recipe-drift",
        action="store_true",
        help="Record a non-frozen temperature/strength instead of failing closed.",
    )
    parser.add_argument(
        "--skip-desktop-copy",
        action="store_true",
        help="Do not copy the package to the desktop directory.",
    )
    args = parser.parse_args()

    protected = _protected_candidates(Path(args.protected_registry).expanduser())
    if str(args.tag) in protected:
        raise SystemExit(
            f"refusing to rebuild protected candidate {args.tag!r}; "
            "pick a new candidate id"
        )

    output_root = Path(args.output_root).expanduser().resolve()
    destination = output_root / str(args.tag)
    if destination.exists() and not args.allow_overwrite:
        raise SystemExit(
            f"candidate directory already exists and will not be overwritten: "
            f"{destination}. Use a new --tag (or --allow-overwrite to replace it)."
        )

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    cache_path = Path(args.val_branch_cache).expanduser().resolve()

    recipe = resolved_recipe(
        tta=str(args.tta),
        fusion=str(args.fusion),
        temperature=float(args.temperature),
        prior_strength=float(args.prior_strength),
        enforce_fixed=not bool(args.allow_recipe_drift),
    )

    checkpoint = checkpoint_identity(checkpoint_path, config)
    cache_identity = validation_cache_identity(cache_path, config, checkpoint)
    val_payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    bias, fit_report, prior_record = fit_bound_prior(
        val_payload, cache_identity, recipe=recipe, max_iterations=int(args.max_iterations)
    )
    verify_prior_record(prior_record, cache_identity, recipe=recipe, bias=bias)

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
        original, flipped, mode=recipe["fusion"], temperature=recipe["temperature"]
    )
    corrected = apply_prior_bias(fused, bias, strength=recipe["prior_strength"])
    mapping = _mapping(Path(config["data"]["class_mapping"]).resolve())
    valid_labels = {str(value).zfill(4) for value in mapping.values()}
    predictions = [
        (name, mapping[int(index)].zfill(4))
        for name, index in zip(names, corrected.argmax(dim=1).tolist())
    ]
    inference_mode = (
        f"{recipe['tta']}:{recipe['fusion']}:t={recipe['temperature']:g}:"
        f"balanced_prior={recipe['prior_strength']:g}"
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
            "temperature": recipe["temperature"],
            "prior_strength": recipe["prior_strength"],
            "fusion": recipe["fusion"],
            "tta": recipe["tta"],
            "prior_fit_iterations": int(fit_report["iterations"]),
            "config_sha256": sha256_file(config_path),
            "val_branch_cache": str(cache_path),
            "val_branch_cache_sha256": cache_identity["cache_sha256"],
            "class_mapping_sha256": cache_identity["class_mapping_sha256"],
            "dataset_manifest_sha256": cache_identity["dataset_manifest_sha256"],
            "dataset_fingerprint": cache_identity["dataset_fingerprint"],
            "split_seed": cache_identity["split_seed"],
            "sample_order_sha256": cache_identity["sample_order_sha256"],
            "content_group_set_sha256": cache_identity["content_group_set_sha256"],
            "input_resolution": cache_identity["input_resolution"],
            "preprocessing": cache_identity["preprocessing"],
            "encoder_precision": cache_identity["encoder_precision"],
            "feature_precision": cache_identity["feature_precision"],
            "prior_bias_sha256": prior_record["bias_sha256"],
            "prior_record": prior_record,
            "test_data_used": False,
        },
        overwrite=bool(args.allow_overwrite),
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
    desktop_package = None
    if not args.skip_desktop_copy:
        desktop = Path(args.desktop_dir).expanduser().resolve()
        if not desktop.is_dir():
            raise SystemExit(f"desktop directory missing (use --skip-desktop-copy): {desktop}")
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        desktop_package = desktop / f"L05_{args.tag}_{stamp}"
        if desktop_package.exists():
            raise SystemExit(f"desktop package already exists: {desktop_package}")
        desktop_package.mkdir(parents=True)
        for name in ("pred_results.csv", "submission.zip", "manifest.json"):
            shutil.copy2(destination / name, desktop_package / name)
    print(
        json.dumps(
            {
                "candidate": str(args.tag),
                "recipe": recipe,
                "package": str(destination),
                "desktop_package": str(desktop_package) if desktop_package else None,
                "csv_sha256": sha256_file(destination / "pred_results.csv"),
                "zip_sha256": sha256_file(destination / "submission.zip"),
                "checkpoint_sha256": cache_identity["checkpoint_sha256"],
                "cache_sha256": cache_identity["cache_sha256"],
                "prior_bias_sha256": prior_record["bias_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
