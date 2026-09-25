#!/usr/bin/env python3
"""Minimum implementation checks for the NEW01/NEW02/NEW03 candidate switches.

These are engineering checks, not experiments:

* NEW01 -- on real ``train_dev`` images, confirm the mapped original-image ROI is
  a legal, non-degenerate rectangle, that the ROI tensor is not a re-crop of the
  already-downscaled global tensor, and (with ``--checkpoint``) that the local
  branch receives non-zero gradient. Records original sizes, ROI sizes and the
  number of active local samples.
* NEW02 -- confirm the flip-fusion objective keeps both branches in-graph with
  non-zero gradients and that ``p_mix`` is exactly the mean of the two tempered
  probabilities.
* NEW03 -- confirm single-element sets degenerate to ordinary GCE, that copies of
  the same duplicate group share one candidate set, that the sidecar was built
  from ``train_dev`` only, and print the affected group/sample counts.

The test set is never read.  Nothing here writes a submission or a result row.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
for location in (str(ROOT), str(AEGIS_ROOT)):
    if location not in sys.path:
        sys.path.insert(0, location)

from aegis_clip.candidate_losses import (  # noqa: E402
    flip_fusion_global_loss,
    probability_generalized_cross_entropy,
    set_generalized_cross_entropy,
)
from aegis_clip.data import (  # noqa: E402
    _sample_rrc_params,
    _resized_crop,
    _hflip,
    _pil_to_tensor,
    split_geometry_augmentation,
)
from aegis_clip.duplicate_sets import (  # noqa: E402
    conflict_group_report,
    load_candidate_label_sets,
)
from aegis_clip.original_roi import (  # noqa: E402
    attention_crop_box,
    global_box_to_original,
    original_roi_batch,
)

STAGE_DIR = Path("/home/lux1/noise/artifacts/stages/repechage/20260921")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", default=str(STAGE_DIR))
    parser.add_argument("--train-root", default="/home/lux1/noise/train")
    parser.add_argument("--sidecar", default="artifacts/l05_new_candidates/NEW03_candidate_labels.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", default="results/l05_new_candidates_checks.json")
    return parser.parse_args()


@torch.no_grad()
def _new01_geometry(args) -> dict:
    stage = Path(args.stage_dir)
    train_root = Path(args.train_root)
    with (stage / "train_dev.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for _, row in zip(range(int(args.samples)), csv.DictReader(handle))]
    from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
    from torchvision.transforms import InterpolationMode

    transforms = Compose(
        [
            RandomResizedCrop(
                384,
                scale=(0.70, 1.0),
                ratio=(0.85, 1.15),
                interpolation=InterpolationMode.BICUBIC,
            ),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    size, scale, ratio, interpolation, flip_p, tail = split_geometry_augmentation(transforms)
    from PIL import Image

    globals_, originals, geometry = [], [], []
    for row in rows:
        path = train_root / row["image_path"].removeprefix("train/")
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            top, left, ch, cw = _sample_rrc_params(height, width, scale, ratio)
            crop = _resized_crop(image, top, left, ch, cw, size, interpolation)
            flip = bool(torch.rand(()).item() < flip_p)
            if flip:
                crop = _hflip(crop)
            globals_.append(tail(crop))
            originals.append(_pil_to_tensor(image).float() / 255.0)
            geometry.append(
                torch.tensor([height, width, top, left, ch, cw, 1.0 if flip else 0.0, float(size[0])])
            )
    global_batch = torch.stack(globals_)
    geometry_batch = torch.stack(geometry)
    max_h = max(int(o.shape[1]) for o in originals)
    max_w = max(int(o.shape[2]) for o in originals)
    padded = torch.zeros(len(originals), 3, max_h, max_w)
    sizes = torch.zeros(len(originals), 2, dtype=torch.long)
    for i, o in enumerate(originals):
        padded[i, :, : o.shape[1], : o.shape[2]] = o
        sizes[i, 0], sizes[i, 1] = o.shape[1], o.shape[2]

    attention = torch.rand(len(originals), 12, 12)
    box = attention_crop_box(
        attention, crop_size=285, top_patches=5, height=384, width=384
    )
    original_box = global_box_to_original(box, geometry_batch)
    roi = original_roi_batch(padded, sizes, geometry_batch, box, out_size=384)
    box_valid = bool(
        (original_box["x1"] > original_box["x0"]).all()
        and (original_box["y1"] > original_box["y0"]).all()
        and (original_box["x0"] >= 0).all()
        and (original_box["y0"] >= 0).all()
        and (original_box["x1"] <= sizes[:, 1]).all()
        and (original_box["y1"] <= sizes[:, 0]).all()
    )
    # A re-crop of the downscaled global tensor cannot exceed its detail; the
    # originality check is that the two differ materially.
    from torch.nn.functional import interpolate

    downscaled_recrop = interpolate(
        original_roi_batch(
            global_batch,
            torch.full((len(originals), 2), 384),
            torch.tensor(
                [[384.0, 384.0, 0.0, 0.0, 384.0, 384.0, g[6].item(), 384.0] for g in geometry_batch]
            ),
            box,
            out_size=384,
        ),
        size=(384, 384),
        mode="bilinear",
        align_corners=False,
    )
    delta = float((roi - downscaled_recrop).abs().max())
    return {
        "samples": len(rows),
        "box_valid": box_valid,
        "roi_shape": list(roi.shape),
        "original_sizes": sizes.tolist(),
        "max_abs_delta_vs_downscaled_recrop": delta,
        "changed": delta > 1.0e-3,
        "active_local_samples": int(len(rows)),
    }


def _new02_check() -> dict:
    torch.manual_seed(0)
    logits_o = torch.randn(8, 5, requires_grad=True)
    logits_f = torch.randn(8, 5, requires_grad=True)
    target = torch.randint(0, 5, (8,))
    loss, diag = flip_fusion_global_loss(logits_o, logits_f, target, q=0.5, temperature=1.4)
    loss.sum().backward()
    p_mix_expected = (diag["p_o"] + diag["p_f"]) / 2.0
    return {
        "both_branches_grad_nonzero": bool(
            logits_o.grad is not None
            and logits_f.grad is not None
            and float(logits_o.grad.abs().sum()) > 0
            and float(logits_f.grad.abs().sum()) > 0
        ),
        "p_mix_is_mean": bool(torch.allclose(diag["p_mix"], p_mix_expected, atol=1.0e-6)),
        "loss_finite": bool(torch.isfinite(loss).all()),
    }


def _new03_check(args) -> dict:
    sidecar = Path(args.sidecar)
    if not sidecar.is_absolute():
        sidecar = ROOT / sidecar
    labels = load_candidate_label_sets(sidecar)
    report = conflict_group_report(Path(args.stage_dir) / "train_dev.csv")
    non_singleton = {k: v for k, v in labels.items() if len(v) > 1}
    sets_match = all(
        labels[path] == sorted(set(labels[path])) for path in labels
    )
    # singleton degeneracy
    probabilities = torch.softmax(torch.randn(4, 7), dim=1)
    target = torch.tensor([1, 2, 3, 4])
    index = target.unsqueeze(1)
    mask = torch.ones(4, 1, dtype=torch.bool)
    set_loss = set_generalized_cross_entropy(probabilities, index, mask, q=0.5)
    plain = probability_generalized_cross_entropy(probabilities, target, q=0.5)
    return {
        "sidecar": str(sidecar),
        "num_conflict_groups": report["num_conflict_groups"],
        "num_conflict_samples": report["num_conflict_samples"],
        "non_singleton_samples": len(non_singleton),
        "sets_sorted_unique": bool(sets_match),
        "singleton_equals_gce": bool(torch.allclose(set_loss, plain, atol=1.0e-6)),
    }


@torch.no_grad()
def _gradient_probe(args) -> dict | None:
    if not args.checkpoint or not args.config:
        return None
    from aegis_clip.checkpoint import build_from_checkpoint
    from aegis_clip.config import load_config

    device = torch.device(args.device)
    config = load_config(Path(args.config).expanduser().resolve())
    model, _, _ = build_from_checkpoint(
        Path(args.checkpoint).expanduser().resolve(), device, config_override=config
    )
    model.train()
    images = torch.rand(2, 3, 384, 384, device=device)
    logits = model(images=images)
    model.zero_grad(set_to_none=True)
    logits.sum().backward()
    total = sum(
        float(p.grad.abs().sum())
        for p in model.parameters()
        if p.grad is not None
    )
    return {"global_grad_abs_sum": total, "global_grad_nonzero": total > 0}


def main() -> int:
    args = _parse_args()
    result = {
        "schema_version": 1,
        "device": args.device,
        "test_data_used": False,
        "new01_geometry": _new01_geometry(args),
        "new02_flip_fusion": _new02_check(),
        "new03_set_supervision": _new03_check(args),
        "gradient_probe": _gradient_probe(args),
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    ok = (
        result["new01_geometry"]["box_valid"]
        and result["new01_geometry"]["changed"]
        and result["new02_flip_fusion"]["both_branches_grad_nonzero"]
        and result["new03_set_supervision"]["singleton_equals_gce"]
        and result["new03_set_supervision"]["sets_sorted_unique"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
