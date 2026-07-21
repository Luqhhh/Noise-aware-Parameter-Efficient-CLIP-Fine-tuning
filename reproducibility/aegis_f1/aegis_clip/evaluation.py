"""Automatic clean-proxy evaluation for noisy-label model selection."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from aegis_clip.multiprototype import blend_multiprototype_logits
from aegis_clip.tta import TTA_FUSION_MODES, fuse_paired_logits


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
    drift_budget: float = 0.01,
    drift_penalty: float = 0.5,
    selector_metric: str = "proxy_macro",
    tta_mode: str = "none",
    tta_fusion: str = "mean_logits",
    tta_temperature: float = 1.0,
    multiprototype_head: dict | None = None,
    clean_core_threshold: float = 0.70,
    measure_flip_consistency: bool = False,
) -> dict[str, float | int | str]:
    if tta_mode not in {"none", "horizontal_flip"}:
        raise ValueError(f"Unsupported TTA mode: {tta_mode}")
    if tta_fusion not in TTA_FUSION_MODES:
        raise ValueError(f"Unsupported TTA fusion mode: {tta_fusion}")
    model.eval()
    predictions: list[torch.Tensor] = []
    noisy_targets: list[torch.Tensor] = []
    proxy_targets: list[torch.Tensor] = []
    clean_probabilities: list[torch.Tensor] = []
    proxy_weights: list[torch.Tensor] = []
    feature_cosines: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    flip_agreements: list[torch.Tensor] = []

    for batch in loader:
        labels = batch["label"].to(device)
        clean = batch["clean_probability"].to(device).float()
        pseudo = batch["pseudo_label"].to(device).long()
        correction = batch["correction_alpha"].to(device).float()
        model_arguments: dict[str, torch.Tensor]
        if "images" in batch:
            images = batch["images"].to(device, non_blocking=True)
            model_arguments = {"images": images}
        else:
            if tta_mode != "none":
                raise ValueError("TTA requires online image batches")
            model_arguments = {
                "features": batch["features"].to(device, non_blocking=True)
            }
        with torch.autocast(
            device_type=device.type, enabled=use_amp and device.type == "cuda"
        ):
            logits, encoded = model(**model_arguments, return_features=True)
            if multiprototype_head is not None:
                logits = blend_multiprototype_logits(
                    logits, encoded, multiprototype_head
                )
            if tta_mode == "horizontal_flip" or (
                measure_flip_consistency and "images" in batch
            ):
                flipped_logits, flipped_encoded = model(
                    images=torch.flip(images, dims=(3,)),
                    return_features=True,
                )
                if multiprototype_head is not None:
                    flipped_logits = blend_multiprototype_logits(
                        flipped_logits, flipped_encoded, multiprototype_head
                    )
                flip_agreements.append(
                    (logits.argmax(dim=1) == flipped_logits.argmax(dim=1)).cpu()
                )
            if tta_mode == "horizontal_flip":
                logits = fuse_paired_logits(
                    logits,
                    flipped_logits,
                    mode=tta_fusion,
                    temperature=tta_temperature,
                )
                encoded = F.normalize(
                    encoded.float() + flipped_encoded.float(), dim=1
                )
        predictions.append(logits.argmax(dim=1).cpu())
        noisy_targets.append(labels.cpu())
        proxy = torch.where(correction > 0.0, pseudo, labels)
        proxy_targets.append(proxy.cpu())
        clean_probabilities.append(clean.cpu())
        proxy_weights.append(torch.maximum(clean, correction).cpu())
        losses.append(F.cross_entropy(logits.float(), labels, reduction="none").cpu())
        reference = batch.get("reference_features")
        if reference is not None:
            reference = F.normalize(reference.to(device).float(), dim=1)
            feature_cosines.append(
                F.cosine_similarity(encoded.float(), reference, dim=1).cpu()
            )

    prediction = torch.cat(predictions)
    noisy = torch.cat(noisy_targets)
    proxy = torch.cat(proxy_targets)
    clean_weight = torch.cat(clean_probabilities)
    proxy_weight = torch.cat(proxy_weights)
    per_sample_loss = torch.cat(losses)

    raw_micro = float((prediction == noisy).float().mean())
    raw_macro = weighted_macro_accuracy(
        prediction, noisy, torch.ones_like(clean_weight), num_classes
    )
    trusted_micro = weighted_accuracy(prediction, noisy, clean_weight)
    trusted_macro = weighted_macro_accuracy(
        prediction, noisy, clean_weight, num_classes
    )
    proxy_micro = weighted_accuracy(prediction, proxy, proxy_weight)
    proxy_macro = weighted_macro_accuracy(
        prediction, proxy, proxy_weight, num_classes
    )
    clean_core_weight = (clean_weight >= float(clean_core_threshold)).float()
    clean_core_micro = weighted_accuracy(prediction, noisy, clean_core_weight)
    clean_core_macro = weighted_macro_accuracy(
        prediction, noisy, clean_core_weight, num_classes
    )
    mean_cosine = (
        float(torch.cat(feature_cosines).mean()) if feature_cosines else 1.0
    )
    mean_drift = 1.0 - mean_cosine
    metrics = {
        "samples": int(prediction.numel()),
        "predicted_class_count": int(prediction.unique().numel()),
        "raw_micro": raw_micro,
        "raw_macro": raw_macro,
        "trusted_micro": trusted_micro,
        "trusted_macro": trusted_macro,
        "proxy_micro": proxy_micro,
        "proxy_macro": proxy_macro,
        "clean_core_micro": clean_core_micro,
        "clean_core_macro": clean_core_macro,
        "clean_core_threshold": float(clean_core_threshold),
        "clean_core_samples": int(clean_core_weight.sum()),
        "flip_prediction_agreement": (
            float(torch.cat(flip_agreements).float().mean())
            if flip_agreements
            else 1.0
        ),
        "mean_cross_entropy": float(per_sample_loss.mean()),
        "mean_feature_cosine": mean_cosine,
        "mean_feature_drift": mean_drift,
        "inference_mode": tta_mode,
        "tta_fusion": tta_fusion if tta_mode != "none" else "none",
        "tta_temperature": (
            float(tta_temperature) if tta_mode != "none" else 1.0
        ),
        "prediction_head": (
            "linear_plus_multiprototype"
            if multiprototype_head is not None
            else "linear"
        ),
    }
    if selector_metric not in {
        "raw_micro",
        "raw_macro",
        "trusted_micro",
        "trusted_macro",
        "proxy_micro",
        "proxy_macro",
        "clean_core_micro",
        "clean_core_macro",
    }:
        raise ValueError(f"Unsupported selector metric: {selector_metric}")
    selector = float(metrics[selector_metric]) - drift_penalty * max(
        0.0, mean_drift - drift_budget
    )
    metrics["selector_metric"] = selector_metric
    metrics["selector"] = selector
    return metrics


def weighted_accuracy(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> float:
    weight = weight.float().clamp_min(0.0)
    denominator = weight.sum().clamp_min(1.0e-12)
    correct = (prediction == target).float()
    return float((correct * weight).sum() / denominator)


def weighted_macro_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    num_classes: int,
) -> float:
    values = []
    for class_index in range(num_classes):
        mask = target == class_index
        class_weight = weight[mask].float().clamp_min(0.0)
        if class_weight.sum() <= 0.0:
            continue
        correct = (prediction[mask] == target[mask]).float()
        values.append((correct * class_weight).sum() / class_weight.sum())
    if not values:
        return 0.0
    return float(torch.stack(values).mean())


def format_metrics(metrics: dict[str, Any]) -> str:
    return " | ".join(
        f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in metrics.items()
    )
