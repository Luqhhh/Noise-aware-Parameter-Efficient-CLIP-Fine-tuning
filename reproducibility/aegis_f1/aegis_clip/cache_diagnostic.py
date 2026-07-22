"""Train-only visual-memory diagnostics for a frozen CLIP representation."""

from __future__ import annotations

import torch

from aegis_clip.evaluation import weighted_accuracy, weighted_macro_accuracy


@torch.no_grad()
def topk_cache_predictions(
    query_features: torch.Tensor,
    bank_features: torch.Tensor,
    bank_labels: torch.Tensor,
    *,
    num_classes: int,
    k: int = 10,
    beta: float = 20.0,
    query_batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return similarity-weighted top-k cache predictions and their margins."""
    if query_features.ndim != 2 or bank_features.ndim != 2:
        raise ValueError("Query and bank features must both be rank-2")
    if query_features.shape[1] != bank_features.shape[1]:
        raise ValueError("Query and bank feature dimensions differ")
    if len(bank_features) != len(bank_labels):
        raise ValueError("Bank features and labels have different lengths")
    if not 1 <= int(k) <= len(bank_features):
        raise ValueError("k must be between 1 and the bank size")
    if int(query_batch_size) < 1:
        raise ValueError("query_batch_size must be positive")
    labels = bank_labels.long()
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError("Bank labels fall outside the class range")

    compute_device = torch.device(device)
    bank = torch.nn.functional.normalize(bank_features.float(), dim=1).to(
        compute_device
    )
    labels = labels.to(compute_device)
    predictions: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    for start in range(0, len(query_features), int(query_batch_size)):
        query = torch.nn.functional.normalize(
            query_features[start : start + int(query_batch_size)].float(), dim=1
        ).to(compute_device)
        similarity, neighbor_indices = (query @ bank.T).topk(int(k), dim=1)
        neighbor_labels = labels[neighbor_indices]
        # Subtracting each row maximum is algebraically prediction-preserving and
        # prevents underflow while retaining Tip-Adapter-style similarity weights.
        weights = torch.exp(float(beta) * (similarity - similarity[:, :1]))
        votes = torch.zeros(
            len(query), int(num_classes), device=compute_device, dtype=torch.float32
        )
        votes.scatter_add_(1, neighbor_labels, weights)
        values, classes = votes.topk(2, dim=1)
        predictions.append(classes[:, 0].cpu())
        margins.append(((values[:, 0] - values[:, 1]) / values[:, 0].clamp_min(1e-12)).cpu())
    return torch.cat(predictions), torch.cat(margins)


def prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    clean_probability: torch.Tensor,
    *,
    num_classes: int,
    clean_threshold: float = 0.70,
) -> dict[str, float | int]:
    prediction = prediction.long().cpu()
    target = target.long().cpu()
    clean = clean_probability.float().cpu()
    if not (len(prediction) == len(target) == len(clean)):
        raise ValueError("Prediction, target, and trust arrays must align")
    clean_core = (clean >= float(clean_threshold)).float()
    return {
        "samples": len(prediction),
        "raw_micro": float((prediction == target).float().mean()),
        "raw_macro": weighted_macro_accuracy(
            prediction, target, torch.ones_like(clean), int(num_classes)
        ),
        "trusted_micro": weighted_accuracy(prediction, target, clean),
        "trusted_macro": weighted_macro_accuracy(
            prediction, target, clean, int(num_classes)
        ),
        "clean_core_micro": weighted_accuracy(prediction, target, clean_core),
        "clean_core_macro": weighted_macro_accuracy(
            prediction, target, clean_core, int(num_classes)
        ),
        "clean_core_samples": int(clean_core.sum()),
    }


def complementarity_metrics(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
    clean_probability: torch.Tensor,
    *,
    clean_threshold: float = 0.70,
) -> dict[str, float | int]:
    baseline = baseline.long().cpu()
    candidate = candidate.long().cpu()
    target = target.long().cpu()
    clean = clean_probability.float().cpu()
    if not (len(baseline) == len(candidate) == len(target) == len(clean)):
        raise ValueError("Complementarity arrays must align")
    mask = clean >= float(clean_threshold)
    baseline_correct = baseline.eq(target)
    candidate_correct = candidate.eq(target)
    denominator = int(mask.sum())
    if denominator == 0:
        raise ValueError("No clean-core samples are available")
    oracle = (baseline_correct | candidate_correct) & mask
    rescue = (~baseline_correct & candidate_correct) & mask
    damage = (baseline_correct & ~candidate_correct) & mask
    return {
        "clean_core_samples": denominator,
        "prediction_disagreement": float((baseline[mask] != candidate[mask]).float().mean()),
        "oracle_clean_core_micro": float(oracle.sum() / denominator),
        "candidate_rescues": int(rescue.sum()),
        "candidate_damages": int(damage.sum()),
        "net_if_oracle": int(rescue.sum()),
    }
