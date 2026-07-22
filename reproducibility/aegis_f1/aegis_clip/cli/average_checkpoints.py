"""Same-trajectory checkpoint averaging for LoRA and classifier parameters."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import TrustBundle
from aegis_clip.evaluation import evaluate, format_metrics
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.runtime import atomic_json_dump, seed_worker, set_seed
from aegis_clip.trust import atomic_torch_save
from torch.utils.data import DataLoader

from aegis_clip.data import OnlineImageDataset

# ── helpers ────────────────────────────────────────────────────────────────

LORA_PARAM_SUFFIXES = (
    ".lora_A",
    ".lora_B",
    ".q_A",
    ".q_B",
    ".v_A",
    ".v_B",
)
CLASSIFIER_KEYS = {"classifier.weight", "classifier.bias"}
AVERAGEABLE_KEYS = CLASSIFIER_KEYS  # plus LoRA params matched dynamically


def _is_lora_param(name: str) -> bool:
    return ".parametrizations." in name and name.endswith(LORA_PARAM_SUFFIXES)


def _averageable_keys(state_dict: dict[str, torch.Tensor]) -> set[str]:
    return {k for k in state_dict if k in CLASSIFIER_KEYS or _is_lora_param(k)}


def _validate_identical_non_averageable(
    checkpoints: list[dict[str, torch.Tensor]],
    averageable: set[str],
) -> None:
    """Reject averaging when non-LoRA, non-classifier state differs."""
    reference = checkpoints[0]
    for i, sd in enumerate(checkpoints[1:], start=2):
        for key in sorted(set(reference) | set(sd)):
            if key in averageable:
                continue
            if key not in reference or key not in sd:
                raise ValueError(f"Checkpoint {i} is missing key: {key}")
            if not torch.equal(reference[key].cpu(), sd[key].cpu()):
                raise ValueError(f"Non-averageable state differs: {key}")


def _average_state_dicts(
    checkpoints: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Average averageable parameters across checkpoints."""
    if weights is None:
        weights = [1.0 / len(checkpoints)] * len(checkpoints)
    if len(checkpoints) != len(weights):
        raise ValueError("Mismatched checkpoints and weights")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")

    averageable = _averageable_keys(checkpoints[0])
    _validate_identical_non_averageable(checkpoints, averageable)

    result = copy.deepcopy(checkpoints[0])
    for key in averageable:
        accumulator = torch.zeros_like(result[key], dtype=torch.float32)
        for sd, w in zip(checkpoints, weights):
            accumulator += w * sd[key].float()
        result[key] = (accumulator / total_weight).to(dtype=result[key].dtype)
    return result


def _build_val_loader(config: dict[str, Any]) -> DataLoader:
    data_config = config["data"]
    feature_config = config["features"]
    trust_bundle = (
        TrustBundle(config["trust"]["bundle_path"])
        if config.get("trust", {}).get("enabled", False)
        else None
    )
    feature_store = FrozenFeatureStore(
        tensor_path=feature_config["tensor_path"],
        paths_path=feature_config["paths_path"],
        manifest_path=feature_config.get("manifest_path"),
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    try:
        import clip
    except ImportError:
        raise ImportError("Install the pinned official OpenAI CLIP package")
    _, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
    val_dataset = OnlineImageDataset(
        data_config["val_csv"],
        data_config["train_root"],
        preprocess,
        feature_store,
        trust_bundle,
    )
    eval_config = config["evaluation"]
    workers = min(int(eval_config.get("num_workers", 2)), 4)
    return DataLoader(
        val_dataset,
        batch_size=int(eval_config.get("batch_size", 128)),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        timeout=120 if workers else 0,
        worker_init_fn=seed_worker,
        persistent_workers=workers > 0,
    )


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average LoRA and classifier parameters across same-trajectory checkpoints"
    )
    parser.add_argument("--config", required=True, help="Aegis config YAML")
    parser.add_argument("--checkpoints", required=True, nargs="+", help="Epoch checkpoint paths")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument(
        "--scheme",
        choices=["equal", "greedy_soup"],
        default="equal",
        help="Averaging scheme (default: equal weight)",
    )
    parser.add_argument(
        "--epoch-range",
        help="Comma-separated epoch range filter, e.g. '2,6' for epochs 2-6",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval", action="store_true", help="Evaluate the averaged checkpoint")
    parser.add_argument("--selection-metric", default="clean_core_micro")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    config = load_config(args.config)
    num_classes = int(config["model"]["num_classes"])

    # Load all checkpoints
    checkpoints: list[dict[str, torch.Tensor]] = []
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("model_state"))
        if state is None:
            raise ValueError(f"Checkpoint {path} has no model weights")
        checkpoints.append(dict(state))

    print(f"Loaded {len(checkpoints)} checkpoints")
    averageable = _averageable_keys(checkpoints[0])
    print(f"Averageable parameters: {sorted(averageable)}")
    _validate_identical_non_averageable(checkpoints, averageable)
    print("Non-averageable state: IDENTICAL ✓")

    if args.scheme == "equal":
        averaged = _average_state_dicts(checkpoints)
        print(f"Averaged {len(checkpoints)} checkpoints with equal weights")

    elif args.scheme == "greedy_soup":
        # Build val loader for greedy selection
        val_loader = _build_val_loader(config)
        model, _, _ = build_from_checkpoint(args.checkpoints[0], device)

        # Evaluate each checkpoint individually
        scores: list[tuple[float, dict[str, torch.Tensor], str]] = []
        for i, (sd, path) in enumerate(zip(checkpoints, args.checkpoints)):
            model.load_state_dict(sd, strict=True)
            metrics = evaluate(
                model, val_loader, device=device, num_classes=num_classes,
                use_amp=True,
                drift_budget=float(config["evaluation"].get("drift_budget", 0.01)),
                drift_penalty=float(config["evaluation"].get("drift_penalty", 0.5)),
                selector_metric=args.selection_metric,
                clean_core_threshold=float(
                    config["evaluation"].get("clean_core_threshold", 0.70)
                ),
            )
            score = float(metrics[args.selection_metric])
            scores.append((score, sd, str(Path(path).name)))
            print(f"  [{i}] {Path(path).name}: {args.selection_metric}={score:.6f}")

        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        best_score = scores[0][0]
        soup_sd = copy.deepcopy(scores[0][1])
        soup_ckpts = [scores[0][2]]
        print(f"  Greedy start: {soup_ckpts[0]} ({args.selection_metric}={best_score:.6f})")

        for score, sd, name in scores[1:]:
            candidate = _average_state_dicts([soup_sd, sd])
            model.load_state_dict(candidate, strict=True)
            metrics = evaluate(
                model, val_loader, device=device, num_classes=num_classes,
                use_amp=True,
                drift_budget=float(config["evaluation"].get("drift_budget", 0.01)),
                drift_penalty=float(config["evaluation"].get("drift_penalty", 0.5)),
                selector_metric=args.selection_metric,
                clean_core_threshold=float(
                    config["evaluation"].get("clean_core_threshold", 0.70)
                ),
            )
            new_score = float(metrics[args.selection_metric])
            if new_score > best_score:
                soup_sd = candidate
                best_score = new_score
                soup_ckpts.append(name)
                print(f"  + {name}: {args.selection_metric}={new_score:.6f} (ACCEPT)")
            else:
                print(f"  - {name}: {args.selection_metric}={new_score:.6f} (SKIP)")

        averaged = soup_sd
        print(f"Greedy soup: {len(soup_ckpts)}/{len(checkpoints)} checkpoints retained")

    # Build output checkpoint preserving original metadata
    reference = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    reference["model_state_dict"] = averaged
    reference["checkpoint_averaging"] = {
        "scheme": args.scheme,
        "source_checkpoints": [str(Path(p).resolve()) for p in args.checkpoints],
        "num_checkpoints": len(args.checkpoints),
    }
    atomic_torch_save(reference, args.output)
    print(f"Saved averaged checkpoint → {args.output}")

    # Optional evaluation
    if args.eval:
        model, _, _ = build_from_checkpoint(args.output, device)
        val_loader = _build_val_loader(config)
        metrics = evaluate(
            model, val_loader, device=device, num_classes=num_classes,
            use_amp=True,
            drift_budget=float(config["evaluation"].get("drift_budget", 0.01)),
            drift_penalty=float(config["evaluation"].get("drift_penalty", 0.5)),
            selector_metric=args.selection_metric,
            clean_core_threshold=float(
                config["evaluation"].get("clean_core_threshold", 0.70)
            ),
            measure_flip_consistency=bool(
                config["evaluation"].get("measure_flip_consistency", False)
            ),
        )
        print(format_metrics(metrics))


if __name__ == "__main__":
    main()
