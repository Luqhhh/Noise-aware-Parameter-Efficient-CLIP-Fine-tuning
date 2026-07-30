"""Train a tiny local-only residual adapter behind a frozen Aegis model."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from aegis_clip.evaluation import weighted_accuracy, weighted_macro_accuracy
from aegis_clip.local_adapter import (
    build_local_adapter,
    classifier_parameters_from_checkpoint,
    classify_adapted_local_features,
    validate_local_view_cache,
)
from aegis_clip.localization import fuse_global_local_probabilities
from aegis_clip.losses import soft_generalized_cross_entropy
from aegis_clip.runtime import atomic_json_dump, set_seed, sha256_file
from aegis_clip.trust import atomic_torch_save


def _classification_metrics(
    prediction: torch.Tensor,
    *,
    labels: torch.Tensor,
    clean_probability: torch.Tensor,
    pseudo_label: torch.Tensor,
    correction_alpha: torch.Tensor,
    num_classes: int,
    clean_core_threshold: float,
) -> dict[str, float | int]:
    proxy = torch.where(correction_alpha > 0.0, pseudo_label, labels)
    proxy_weight = torch.maximum(clean_probability, correction_alpha)
    unit_weight = torch.ones_like(clean_probability)
    clean_core_weight = (clean_probability >= clean_core_threshold).float()
    return {
        "predicted_class_count": int(prediction.unique().numel()),
        "raw_micro": float((prediction == labels).float().mean()),
        "raw_macro": weighted_macro_accuracy(
            prediction, labels, unit_weight, num_classes
        ),
        "trusted_micro": weighted_accuracy(
            prediction, labels, clean_probability
        ),
        "trusted_macro": weighted_macro_accuracy(
            prediction, labels, clean_probability, num_classes
        ),
        "proxy_micro": weighted_accuracy(prediction, proxy, proxy_weight),
        "proxy_macro": weighted_macro_accuracy(
            prediction, proxy, proxy_weight, num_classes
        ),
        "clean_core_micro": weighted_accuracy(
            prediction, labels, clean_core_weight
        ),
        "clean_core_macro": weighted_macro_accuracy(
            prediction, labels, clean_core_weight, num_classes
        ),
        "clean_core_samples": int(clean_core_weight.sum()),
    }


@torch.no_grad()
def _evaluate(
    adapter: torch.nn.Module,
    cache: dict[str, Any],
    *,
    classifier_weight: torch.Tensor,
    classifier_bias: torch.Tensor,
    device: torch.device,
    local_weight: float,
    clean_core_threshold: float,
    batch_size: int,
    num_classes: int,
) -> dict[str, float | int]:
    adapter.eval()
    global_features = torch.as_tensor(cache["global_features"]).float()
    local_features = torch.as_tensor(cache["local_features"]).float()
    loader = DataLoader(
        TensorDataset(torch.arange(global_features.shape[0])),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions: list[torch.Tensor] = []
    drifts: list[torch.Tensor] = []
    for (indices,) in loader:
        global_batch = global_features[indices].to(device)
        local_batch = local_features[indices].to(device)
        global_logits = F.linear(
            F.normalize(global_batch, dim=1),
            classifier_weight,
            classifier_bias,
        )
        local_logits, adapted = classify_adapted_local_features(
            adapter,
            local_batch,
            classifier_weight,
            classifier_bias,
        )
        fused = fuse_global_local_probabilities(
            global_logits,
            local_logits,
            local_weight=local_weight,
        )
        predictions.append(fused.argmax(dim=1).cpu())
        drifts.append(
            (
                1.0
                - F.cosine_similarity(
                    adapted.float(),
                    F.normalize(local_batch.float(), dim=1),
                    dim=1,
                )
            ).cpu()
        )
    metrics = _classification_metrics(
        torch.cat(predictions),
        labels=torch.as_tensor(cache["labels"]).long(),
        clean_probability=torch.as_tensor(cache["clean_probability"]).float(),
        pseudo_label=torch.as_tensor(cache["pseudo_label"]).long(),
        correction_alpha=torch.as_tensor(cache["correction_alpha"]).float(),
        num_classes=num_classes,
        clean_core_threshold=clean_core_threshold,
    )
    metrics["mean_local_feature_drift"] = float(torch.cat(drifts).mean())
    return metrics


def _promotion(
    baseline: dict[str, float | int],
    candidate: dict[str, float | int],
    *,
    minimum_clean_core_gain: float,
    minimum_raw_gain: float,
    maximum_drift: float,
    required_class_count: int,
) -> dict[str, Any]:
    checks = {
        "clean_core_gain": (
            float(candidate["clean_core_micro"])
            - float(baseline["clean_core_micro"])
            >= minimum_clean_core_gain
        ),
        "trusted_macro_non_decrease": (
            float(candidate["trusted_macro"])
            >= float(baseline["trusted_macro"])
        ),
        "raw_gain_floor": (
            float(candidate["raw_micro"]) - float(baseline["raw_micro"])
            >= minimum_raw_gain
        ),
        "drift_budget": (
            float(candidate["mean_local_feature_drift"]) <= maximum_drift
        ),
        "class_coverage": (
            int(candidate["predicted_class_count"]) == required_class_count
        ),
    }
    return {
        "promoted": all(checks.values()),
        "checks": checks,
        "clean_core_gain": (
            float(candidate["clean_core_micro"])
            - float(baseline["clean_core_micro"])
        ),
        "trusted_macro_gain": (
            float(candidate["trusted_macro"])
            - float(baseline["trusted_macro"])
        ),
        "raw_gain": (
            float(candidate["raw_micro"]) - float(baseline["raw_micro"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--experiment-id",
        default="A2_STRICT_LOCAL_FEATURE_ADAPTER_R1",
    )
    parser.add_argument("--bottleneck-dim", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--local-weight", type=float, default=0.35)
    parser.add_argument("--trusted-threshold", type=float, default=0.70)
    parser.add_argument("--clean-core-threshold", type=float, default=0.70)
    parser.add_argument("--gce-q", type=float, default=0.5)
    parser.add_argument("--local-loss-weight", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= args.local_weight <= 1.0:
        raise ValueError("local-weight must be in [0, 1]")
    if not 0.0 <= args.trusted_threshold <= 1.0:
        raise ValueError("trusted-threshold must be in [0, 1]")
    if not 0.0 <= args.clean_core_threshold <= 1.0:
        raise ValueError("clean-core-threshold must be in [0, 1]")
    if not 0.0 < args.gce_q <= 1.0:
        raise ValueError("gce-q must be in (0, 1]")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")

    set_seed(args.seed, deterministic=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    base_checkpoint_sha256 = sha256_file(args.base_checkpoint)
    base_checkpoint = torch.load(
        args.base_checkpoint, map_location="cpu", weights_only=False
    )
    classifier_weight, classifier_bias = classifier_parameters_from_checkpoint(
        base_checkpoint
    )
    classifier_weight = classifier_weight.to(device)
    classifier_bias = classifier_bias.to(device)
    num_classes, feature_dim = classifier_weight.shape

    train_cache = torch.load(
        args.train_cache, map_location="cpu", weights_only=False
    )
    val_cache = torch.load(
        args.val_cache, map_location="cpu", weights_only=False
    )
    train_size = validate_local_view_cache(
        train_cache,
        expected_checkpoint_sha256=base_checkpoint_sha256,
        expected_split="train",
    )
    val_size = validate_local_view_cache(
        val_cache,
        expected_checkpoint_sha256=base_checkpoint_sha256,
        expected_split="val",
    )
    if (
        int(train_cache["crop_size"]) != int(val_cache["crop_size"])
        or int(train_cache["top_k"]) != int(val_cache["top_k"])
    ):
        raise ValueError("Train and validation local-view protocols differ")
    if torch.as_tensor(train_cache["local_features"]).shape[1] != feature_dim:
        raise ValueError("Cache feature dimension does not match classifier")

    adapter = build_local_adapter(
        feature_dim=feature_dim,
        bottleneck_dim=args.bottleneck_dim,
        residual_scale=args.residual_scale,
    ).to(device)
    baseline = _evaluate(
        adapter,
        val_cache,
        classifier_weight=classifier_weight,
        classifier_bias=classifier_bias,
        device=device,
        local_weight=args.local_weight,
        clean_core_threshold=args.clean_core_threshold,
        batch_size=args.eval_batch_size,
        num_classes=num_classes,
    )

    clean_probability = torch.as_tensor(
        train_cache["clean_probability"]
    ).float()
    selected = torch.nonzero(
        clean_probability >= args.trusted_threshold, as_tuple=False
    ).flatten()
    if selected.numel() == 0:
        raise ValueError("Trusted threshold selected no training samples")
    selected_labels = torch.as_tensor(train_cache["labels"]).long()[selected]
    if selected_labels.unique().numel() != num_classes:
        raise ValueError("Trusted local-adapter training does not cover every class")
    train_dataset = TensorDataset(
        torch.as_tensor(train_cache["global_features"]).float()[selected],
        torch.as_tensor(train_cache["local_features"]).float()[selected],
        selected_labels,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, Any]] = [
        {"epoch": 0, "train_loss": None, **baseline}
    ]
    best_metrics: dict[str, float | int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_clean_core = float("-inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        loss_sum = 0.0
        sample_count = 0
        for global_features, local_features, labels in train_loader:
            global_features = global_features.to(device)
            local_features = local_features.to(device)
            labels = labels.to(device)
            targets = F.one_hot(labels, num_classes=num_classes).float()
            with torch.no_grad():
                global_logits = F.linear(
                    F.normalize(global_features, dim=1),
                    classifier_weight,
                    classifier_bias,
                )
            local_logits, adapted = classify_adapted_local_features(
                adapter,
                local_features,
                classifier_weight,
                classifier_bias,
            )
            fused = fuse_global_local_probabilities(
                global_logits,
                local_logits,
                local_weight=args.local_weight,
            )
            fused_loss = soft_generalized_cross_entropy(
                fused, targets, q=args.gce_q
            ).mean()
            local_loss = soft_generalized_cross_entropy(
                local_logits, targets, q=args.gce_q
            ).mean()
            anchor_loss = (
                1.0
                - F.cosine_similarity(
                    adapted,
                    F.normalize(local_features.float(), dim=1),
                    dim=1,
                )
            ).mean()
            loss = (
                fused_loss
                + args.local_loss_weight * local_loss
                + args.anchor_weight * anchor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            batch_size = labels.numel()
            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size

        metrics = _evaluate(
            adapter,
            val_cache,
            classifier_weight=classifier_weight,
            classifier_bias=classifier_bias,
            device=device,
            local_weight=args.local_weight,
            clean_core_threshold=args.clean_core_threshold,
            batch_size=args.eval_batch_size,
            num_classes=num_classes,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(sample_count, 1),
                **metrics,
            }
        )
        clean_core = float(metrics["clean_core_micro"])
        safety_eligible = (
            float(metrics["trusted_macro"]) >= float(baseline["trusted_macro"])
            and float(metrics["raw_micro"]) >= float(baseline["raw_micro"]) - 0.001
            and float(metrics["mean_local_feature_drift"]) <= 0.01
            and int(metrics["predicted_class_count"]) == num_classes
        )
        if safety_eligible and clean_core > best_clean_core:
            best_clean_core = clean_core
            best_metrics = copy.deepcopy(metrics)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if args.patience > 0 and stale_epochs >= args.patience:
            break

    if best_metrics is None or best_state is None:
        best_metrics = history[-1].copy()
        best_metrics.pop("epoch", None)
        best_metrics.pop("train_loss", None)
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in adapter.state_dict().items()
        }
        best_epoch = int(history[-1]["epoch"])
    promotion = _promotion(
        baseline,
        best_metrics,
        minimum_clean_core_gain=0.002,
        minimum_raw_gain=-0.001,
        maximum_drift=0.01,
        required_class_count=num_classes,
    )
    report = {
        "format_version": 1,
        "experiment_id": args.experiment_id,
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "train_cache": str(Path(args.train_cache).resolve()),
        "train_cache_sha256": sha256_file(args.train_cache),
        "val_cache": str(Path(args.val_cache).resolve()),
        "val_cache_sha256": sha256_file(args.val_cache),
        "train_samples": train_size,
        "trusted_train_samples": int(selected.numel()),
        "val_samples": val_size,
        "crop_size": int(train_cache["crop_size"]),
        "top_k": int(train_cache["top_k"]),
        "adapter": {
            "feature_dim": feature_dim,
            "bottleneck_dim": args.bottleneck_dim,
            "residual_scale": args.residual_scale,
            "parameters": sum(
                parameter.numel() for parameter in adapter.parameters()
            ),
        },
        "training": {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "epochs_completed": int(history[-1]["epoch"]),
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "trusted_threshold": args.trusted_threshold,
            "clean_core_threshold": args.clean_core_threshold,
            "gce_q": args.gce_q,
            "local_weight": args.local_weight,
            "local_loss_weight": args.local_loss_weight,
            "anchor_weight": args.anchor_weight,
        },
        "baseline": baseline,
        "best": best_metrics,
        "promotion": promotion,
        "history": history,
    }
    atomic_json_dump(report, output_dir / "report.json")
    checkpoint_payload = {
        "format_version": 1,
        "experiment_id": report["experiment_id"],
        "base_checkpoint": report["base_checkpoint"],
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "adapter_state_dict": best_state,
        "adapter_spec": report["adapter"],
        "inference": {
            "crop_size": report["crop_size"],
            "top_k": report["top_k"],
            "local_weight": args.local_weight,
        },
        "epoch": best_epoch,
        "metrics": best_metrics,
        "promotion": promotion,
    }
    atomic_torch_save(checkpoint_payload, output_dir / "candidate.pt")
    if promotion["promoted"]:
        atomic_torch_save(checkpoint_payload, output_dir / "best.pt")
    if args.summary_only:
        print(
            json.dumps(
                {
                    "experiment_id": report["experiment_id"],
                    "best_epoch": report["training"]["best_epoch"],
                    "baseline": report["baseline"],
                    "best": report["best"],
                    "promotion": report["promotion"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
