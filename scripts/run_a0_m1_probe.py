#!/usr/bin/env python3
"""Round-0 A0 probe: F05 native global versus attention-guided local fusion.

This is a training-free probe.  It reconstructs the F05 global path, audits
that the attention pipeline's global logits are bit-identical to the native
global path, then evaluates the two preregistered fusion weights
(A0-50 = global 0.50 / local 0.50, A0-40 = global 0.60 / local 0.40).

No flip, crop-size sweep, or top-k sweep is performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
for location in (str(ROOT), str(AEGIS_ROOT)):
    if location not in sys.path:
        sys.path.insert(0, location)

from aegis_clip.balanced_inference import prediction_metrics  # noqa: E402
from aegis_clip.checkpoint import build_from_checkpoint  # noqa: E402
from aegis_clip.config import load_config  # noqa: E402
from aegis_clip.data import OnlineImageDataset, TrustBundle  # noqa: E402
from aegis_clip.features import FrozenFeatureStore  # noqa: E402
from aegis_clip.local_inference import attention_local_global_logits  # noqa: E402
from aegis_clip.localization import fuse_global_local_probabilities  # noqa: E402
from aegis_clip.runtime import seed_worker, set_seed, sha256_file  # noqa: E402
from f05_focus_summary import append_summary_row  # noqa: E402


FUSIONS = (
    {"tag": "A0-50", "global_weight": 0.50, "local_weight": 0.50},
    {"tag": "A0-40", "global_weight": 0.60, "local_weight": 0.40},
)
F05_MACRO = 0.7443073987960815
F05_MICRO = 0.7544354796409607


def _device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _build_loader(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    device: torch.device,
    *,
    config_override: dict[str, Any] | None,
    input_resize_mode: str,
    batch_size: int,
    num_workers: int | None,
):
    config = config_override or checkpoint["config"]
    model, preprocess, _ = build_from_checkpoint(
        checkpoint_path, device, config_override=config
    )
    from aegis_clip.image_preprocess import select_inference_preprocess

    preprocess = select_inference_preprocess(
        preprocess,
        mode=input_resize_mode,
        input_resolution=int(config["model"].get("input_resolution", 224)),
    )
    feature_config = config["features"]
    feature_store = FrozenFeatureStore(
        feature_config["tensor_path"],
        feature_config["paths_path"],
        feature_config.get("manifest_path"),
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    trust_bundle = (
        TrustBundle(config["trust"]["bundle_path"])
        if config.get("trust", {}).get("enabled", False)
        else None
    )
    dataset = OnlineImageDataset(
        config["data"]["val_csv"],
        config["data"]["train_root"],
        preprocess,
        feature_store,
        trust_bundle,
    )
    workers = (
        int(num_workers)
        if num_workers is not None
        else int(config["train"].get("num_workers", 4))
    )
    loader = DataLoader(
        dataset,
        batch_size=int(
            batch_size
            or config["evaluation"].get("batch_size", config["evaluation"].get("inference_batch_size", 128))
        ),
        shuffle=False,
        num_workers=workers,
        timeout=int(config["train"].get("loader_timeout", 120 if workers else 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )
    return model, config, loader


def _collect_metrics(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    clean_probability: torch.Tensor,
    pseudo_labels: torch.Tensor,
    correction_alpha: torch.Tensor,
    num_classes: int,
) -> dict[str, float | int]:
    return prediction_metrics(
        logits.argmax(dim=1),
        labels=labels,
        clean_probability=clean_probability,
        pseudo_labels=pseudo_labels,
        correction_alpha=correction_alpha,
        num_classes=num_classes,
        clean_core_threshold=0.70,
    )


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


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    set_seed(int(args.seed), deterministic=True)

    config_override = load_config(args.config) if args.config else None
    model, config, loader = _build_loader(
        checkpoint_path,
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
        device,
        config_override=config_override,
        input_resize_mode=str(args.input_resize_mode),
        batch_size=int(args.batch_size or 0),
        num_workers=args.num_workers,
    )
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    num_classes = int(config["model"]["num_classes"])
    model.eval()

    native_parts: list[torch.Tensor] = []
    attention_global_parts: list[torch.Tensor] = []
    local_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    clean_parts: list[torch.Tensor] = []
    pseudo_parts: list[torch.Tensor] = []
    correction_parts: list[torch.Tensor] = []
    paths: list[str] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                native_logits = model(images=images)
                result = attention_local_global_logits(
                    model,
                    images,
                    crop_size=int(args.crop_size),
                    top_patches=int(args.top_patches),
                )
            native_parts.append(native_logits.detach().float().cpu())
            attention_global_parts.append(
                result["global_logits"].detach().float().cpu()
            )
            local_parts.append(result["local_logits"].detach().float().cpu())
            labels_parts.append(batch["label"].long().cpu())
            clean_parts.append(batch["clean_probability"].float().cpu())
            pseudo_parts.append(batch["pseudo_label"].long().cpu())
            correction_parts.append(batch["correction_alpha"].float().cpu())
            paths.extend(str(path) for path in batch["path"])

    native_logits = torch.cat(native_parts, dim=0)
    attention_global_logits = torch.cat(attention_global_parts, dim=0)
    local_logits = torch.cat(local_parts, dim=0)
    labels = torch.cat(labels_parts, dim=0)
    clean_probability = torch.cat(clean_parts, dim=0)
    pseudo_labels = torch.cat(pseudo_parts, dim=0)
    correction_alpha = torch.cat(correction_parts, dim=0)

    difference = (native_logits - attention_global_logits).abs()
    audit = {
        "native_global_vs_attention_pipeline_global": {
            "maximum_absolute_logit_difference": float(difference.max()),
            "prediction_agreement": float(
                (native_logits.argmax(1) == attention_global_logits.argmax(1))
                .float()
                .mean()
            ),
            "tolerance": float(args.audit_tolerance),
            "passed": bool(
                float(difference.max()) <= float(args.audit_tolerance)
                and bool(
                    (
                        native_logits.argmax(1)
                        == attention_global_logits.argmax(1)
                    ).all()
                )
            ),
        }
    }
    if not audit["native_global_vs_attention_pipeline_global"]["passed"]:
        raise RuntimeError(
            "A0 global-logit audit failed: "
            + json.dumps(
                audit["native_global_vs_attention_pipeline_global"],
                ensure_ascii=False,
            )
        )

    metric_args = {
        "labels": labels,
        "clean_probability": clean_probability,
        "pseudo_labels": pseudo_labels,
        "correction_alpha": correction_alpha,
        "num_classes": num_classes,
    }
    native_metrics = _collect_metrics(logits=native_logits, **metric_args)
    local_metrics = _collect_metrics(logits=local_logits, **metric_args)
    candidates: list[dict[str, Any]] = []
    for fusion in FUSIONS:
        fused = fuse_global_local_probabilities(
            attention_global_logits,
            local_logits,
            local_weight=float(fusion["local_weight"]),
            temperature=1.0,
        )
        metrics = _collect_metrics(logits=fused, **metric_args)
        delta_macro_pp = 100.0 * (
            float(metrics["raw_macro"]) - float(native_metrics["raw_macro"])
        )
        delta_micro_pp = 100.0 * (
            float(metrics["raw_micro"]) - float(native_metrics["raw_micro"])
        )
        candidates.append(
            {
                **fusion,
                "local_weight": float(fusion["local_weight"]),
                "global_weight": float(fusion["global_weight"]),
                "metrics": metrics,
                "delta_macro_pp": delta_macro_pp,
                "delta_micro_pp": delta_micro_pp,
                "changed_predictions": int(
                    (native_logits.argmax(1) != fused.argmax(1)).sum()
                ),
                "status": _status(delta_macro_pp, delta_micro_pp),
            }
        )
    best = max(candidates, key=lambda item: (item["metrics"]["raw_macro"], item["metrics"]["raw_micro"]))
    report = {
        "experiment_id": "A0_M1",
        "direction": "attention-local",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "val_csv": str(config["data"]["val_csv"]),
        "val_csv_sha256": sha256_file(config["data"]["val_csv"]),
        "input_resolution": int(config["model"].get("input_resolution", 224)),
        "crop_size": int(args.crop_size),
        "top_patches": int(args.top_patches),
        "flip": False,
        "seed": int(args.seed),
        "num_classes": num_classes,
        "audit": audit,
        "native_global": native_metrics,
        "local_only": local_metrics,
        "candidates": candidates,
        "best": best,
        "passed_promotion": bool(
            best["delta_macro_pp"] >= 0.30 and best["delta_micro_pp"] >= -0.10
        ),
        "selection_rule": "best raw_macro then raw_micro among the two fixed fusions",
    }
    report_path = output_dir / "probe.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.cache_output:
        cache_path = Path(args.cache_output).expanduser().resolve()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "experiment_id": "A0_M1",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "val_csv": str(config["data"]["val_csv"]),
                "val_csv_sha256": sha256_file(config["data"]["val_csv"]),
                "crop_size": int(args.crop_size),
                "top_patches": int(args.top_patches),
                "paths": paths,
                "labels": labels,
                "clean_probability": clean_probability,
                "pseudo_labels": pseudo_labels,
                "correction_alpha": correction_alpha,
                "logits": native_logits,
                "native_global_logits": native_logits,
                "attention_global_logits": attention_global_logits,
                "local_logits": local_logits,
            },
            cache_path,
        )

    append_summary_row(
        args.summary,
        {
            "experiment_id": "A0",
            "parent_sha": report["checkpoint_sha256"],
            "seed": int(args.seed),
            "macro": report["best"]["metrics"]["raw_macro"],
            "micro": report["best"]["metrics"]["raw_micro"],
            "delta_macro": report["best"]["delta_macro_pp"],
            "delta_micro": report["best"]["delta_micro_pp"],
            "changed_predictions": report["best"]["changed_predictions"],
            "selected_epoch": "",
            "reject_rate": "",
            "platform_score": "",
            "status": report["best"]["status"],
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        help="Optional runtime YAML overriding val_csv / train_root / features paths",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--top-patches", type=int, default=5)
    parser.add_argument("--audit-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-resize-mode",
        choices=["clip_center_crop", "clip_letterbox"],
        default="clip_center_crop",
    )
    parser.add_argument(
        "--output-dir", default="outputs/f05_focus/A0_M1"
    )
    parser.add_argument("--cache-output")
    parser.add_argument(
        "--summary", default="results/f05_focus_summary.csv"
    )
    args = parser.parse_args()
    report = run_probe(args)
    print(json.dumps(report["best"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
