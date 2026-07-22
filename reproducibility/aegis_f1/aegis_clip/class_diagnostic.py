"""Class-level diagnostics for aligned validation-logit caches.

These routines are deliberately test-set agnostic. They compare predictions
only on the fixed noisy-label validation split and expose both the evidence and
its limitations instead of turning the proxy into a new tuning objective.
"""

from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Any, Sequence

import torch


def _require_vector(name: str, value: object, length: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or len(value) != length:
        raise ValueError(f"{name} must be a tensor of shape ({length},)")
    return value.cpu()


def validate_aligned_logit_caches(
    caches: Sequence[dict[str, Any]], *, num_classes: int
) -> None:
    """Require identical sample order and labels across prediction caches."""
    if len(caches) < 2:
        raise ValueError("at least two caches are required")
    reference = caches[0]
    if "paths" not in reference or "logits" not in reference:
        raise ValueError("reference cache is missing paths or logits")
    paths = list(reference["paths"])
    labels = _require_vector("labels", reference.get("labels"), len(paths))
    _require_vector(
        "clean_probability", reference.get("clean_probability"), len(paths)
    )
    for index, cache in enumerate(caches):
        logits = cache.get("logits")
        if not isinstance(logits, torch.Tensor) or logits.shape != (
            len(paths),
            num_classes,
        ):
            raise ValueError(
                f"cache {index} logits must have shape ({len(paths)}, {num_classes})"
            )
        if list(cache.get("paths", [])) != paths:
            raise ValueError(f"cache {index} path order differs from the reference")
        other_labels = _require_vector(
            f"cache {index} labels", cache.get("labels"), len(paths)
        )
        if not torch.equal(other_labels, labels):
            raise ValueError(f"cache {index} labels differ from the reference")


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Return tie-aware Spearman correlation without a SciPy dependency."""
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx == 0.0 or dy == 0.0:
        return None
    return numerator / sqrt(dx * dy)


def paired_bootstrap_delta_pp(
    baseline_correct: torch.Tensor,
    candidate_correct: torch.Tensor,
    *,
    seed: int = 42,
    draws: int = 5000,
) -> dict[str, float | int]:
    delta = candidate_correct.float().cpu() - baseline_correct.float().cpu()
    generator = torch.Generator().manual_seed(seed)
    sample_count = int(delta.numel())
    chunks: list[torch.Tensor] = []
    remaining = draws
    while remaining:
        current = min(remaining, 250)
        indices = torch.randint(
            sample_count, (current, sample_count), generator=generator
        )
        chunks.append(delta[indices].mean(dim=1))
        remaining -= current
    samples = torch.cat(chunks)
    return {
        "estimate_pp": float(delta.mean() * 100.0),
        "ci95_low_pp": float(torch.quantile(samples, 0.025) * 100.0),
        "ci95_high_pp": float(torch.quantile(samples, 0.975) * 100.0),
        "bootstrap_draws": draws,
        "seed": seed,
    }


def _model_summary(
    correct: torch.Tensor, labels: torch.Tensor, *, num_classes: int
) -> tuple[dict[str, float | int], torch.Tensor, torch.Tensor]:
    support = torch.bincount(labels, minlength=num_classes)
    correct_count = torch.bincount(labels[correct], minlength=num_classes)
    accuracy = correct_count.float() / support.clamp_min(1)
    error_count = support - correct_count
    nonempty = support > 0
    summary: dict[str, float | int] = {
        "samples": int(support.sum()),
        "correct": int(correct_count.sum()),
        "errors": int(error_count.sum()),
        "micro_accuracy": float(correct.float().mean()),
        "macro_accuracy": float(accuracy[nonempty].mean()),
        "classes_below_50pct": int((accuracy[nonempty] < 0.50).sum()),
        "classes_below_70pct": int((accuracy[nonempty] < 0.70).sum()),
        "classes_below_80pct": int((accuracy[nonempty] < 0.80).sum()),
        "classes_at_least_90pct": int((accuracy[nonempty] >= 0.90).sum()),
    }
    return summary, accuracy, error_count


def _top_share(counts: torch.Tensor, top_n: int) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    return float(torch.topk(counts, min(top_n, len(counts))).values.sum() / total)


def diagnose_class_errors(
    center_cache: dict[str, Any],
    m1_cache: dict[str, Any],
    m3_cache: dict[str, Any],
    train_cache: dict[str, Any],
    *,
    num_classes: int = 500,
    clean_core_threshold: float = 0.70,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Diagnose error distribution and intervention complementarity."""
    caches = [center_cache, m1_cache, m3_cache]
    validate_aligned_logit_caches(caches, num_classes=num_classes)
    labels = center_cache["labels"].long().cpu()
    clean_probability = center_cache["clean_probability"].float().cpu()
    clean_mask = clean_probability >= clean_core_threshold
    if int(clean_mask.sum()) == 0:
        raise ValueError("clean-core cohort is empty")
    clean_labels = labels[clean_mask]
    predictions = {
        "a2_center": center_cache["logits"].argmax(dim=1).cpu(),
        "m1_attention": m1_cache["logits"].argmax(dim=1).cpu(),
        "m3_complementary": m3_cache["logits"].argmax(dim=1).cpu(),
    }
    clean_predictions = {name: pred[clean_mask] for name, pred in predictions.items()}
    clean_correct = {
        name: pred.eq(clean_labels) for name, pred in clean_predictions.items()
    }

    summaries: dict[str, dict[str, float | int]] = {}
    accuracies: dict[str, torch.Tensor] = {}
    errors: dict[str, torch.Tensor] = {}
    for name in predictions:
        summaries[name], accuracies[name], errors[name] = _model_summary(
            clean_correct[name], clean_labels, num_classes=num_classes
        )

    train_paths = list(train_cache.get("paths", []))
    train_labels = _require_vector(
        "train labels", train_cache.get("labels"), len(train_paths)
    ).long()
    train_clean = _require_vector(
        "train clean_probability", train_cache.get("clean_probability"), len(train_paths)
    ).float()
    train_counts = torch.bincount(train_labels, minlength=num_classes)
    train_trust_mass = torch.bincount(
        train_labels, weights=train_clean, minlength=num_classes
    )
    validation_support = torch.bincount(clean_labels, minlength=num_classes)

    class_rows: list[dict[str, Any]] = []
    for class_id in range(num_classes):
        support = int(validation_support[class_id])
        row: dict[str, Any] = {
            "class_id": class_id,
            "clean_core_samples": support,
            "high_clean_train_samples": int(train_counts[class_id]),
            "high_clean_train_trust_mass": float(train_trust_mass[class_id]),
        }
        for name in predictions:
            correct_count = support - int(errors[name][class_id])
            row[f"{name}_correct"] = correct_count
            row[f"{name}_errors"] = int(errors[name][class_id])
            row[f"{name}_accuracy"] = (
                float(accuracies[name][class_id]) if support else None
            )
        row["m1_vs_center_delta_pp"] = (
            100.0
            * float(
                accuracies["m1_attention"][class_id]
                - accuracies["a2_center"][class_id]
            )
            if support
            else None
        )
        row["m3_vs_center_delta_pp"] = (
            100.0
            * float(
                accuracies["m3_complementary"][class_id]
                - accuracies["a2_center"][class_id]
            )
            if support
            else None
        )
        class_rows.append(row)

    base_correct = clean_correct["a2_center"]
    comparisons: dict[str, Any] = {}
    for name, delta_key in (
        ("m1_attention", "m1_vs_center_delta_pp"),
        ("m3_complementary", "m3_vs_center_delta_pp"),
    ):
        candidate_correct = clean_correct[name]
        predictions_changed = clean_predictions[name].ne(clean_predictions["a2_center"])
        comparisons[f"{name}_vs_a2_center"] = {
            "changed_predictions": int(predictions_changed.sum()),
            "changed_fraction": float(predictions_changed.float().mean()),
            "corrected": int((~base_correct & candidate_correct).sum()),
            "harmed": int((base_correct & ~candidate_correct).sum()),
            "net_correct": int(candidate_correct.sum() - base_correct.sum()),
            "classes_improved": sum(
                row[delta_key] > 0 for row in class_rows if row[delta_key] is not None
            ),
            "classes_harmed": sum(
                row[delta_key] < 0 for row in class_rows if row[delta_key] is not None
            ),
            "paired_bootstrap_micro_delta": paired_bootstrap_delta_pp(
                base_correct, candidate_correct
            ),
        }

    any_correct = torch.stack(list(clean_correct.values())).any(dim=0)
    base_wrong = ~base_correct
    oracle_recoverable = base_wrong & any_correct

    base_error_counts = errors["a2_center"]
    pair_counts = Counter(
        (int(label), int(prediction))
        for label, prediction, correct in zip(
            clean_labels,
            clean_predictions["a2_center"],
            base_correct,
            strict=True,
        )
        if not bool(correct)
    )
    confusion_rows = [
        {"true_class": true, "predicted_class": predicted, "errors": count}
        for (true, predicted), count in pair_counts.most_common(50)
    ]
    confusion_total = sum(pair_counts.values())

    train_count_values = [float(value) for value in train_counts]
    base_accuracy_values = [float(value) for value in accuracies["a2_center"]]
    m3_delta_values = [
        float(a - b)
        for a, b in zip(
            accuracies["m3_complementary"],
            accuracies["a2_center"],
            strict=True,
        )
    ]
    report: dict[str, Any] = {
        "cohort": {
            "definition": f"validation clean_probability >= {clean_core_threshold:.2f}",
            "samples": int(clean_mask.sum()),
            "excluded_samples": int((~clean_mask).sum()),
            "classes": int((validation_support > 0).sum()),
            "minimum_class_support": int(validation_support[validation_support > 0].min()),
            "maximum_class_support": int(validation_support.max()),
            "important_limitation": "clean-core is a trust proxy derived from noisy validation labels, not ground-truth clean validation",
        },
        "models": summaries,
        "comparisons": comparisons,
        "error_concentration": {
            "a2_center_top_10_classes_share": _top_share(base_error_counts, 10),
            "a2_center_top_50_classes_share": _top_share(base_error_counts, 50),
            "a2_center_top_100_classes_share": _top_share(base_error_counts, 100),
            "a2_center_top_10_directed_confusions_share": (
                sum(row["errors"] for row in confusion_rows[:10]) / confusion_total
                if confusion_total
                else 0.0
            ),
            "a2_center_top_50_directed_confusions_share": (
                sum(row["errors"] for row in confusion_rows) / confusion_total
                if confusion_total
                else 0.0
            ),
        },
        "complementarity_ceiling": {
            "a2_center_errors": int(base_wrong.sum()),
            "a2_errors_corrected_by_at_least_one_of_m1_m3": int(
                oracle_recoverable.sum()
            ),
            "a2_error_recovery_fraction": float(
                oracle_recoverable.sum() / base_wrong.sum().clamp_min(1)
            ),
            "all_three_wrong": int((~any_correct).sum()),
            "oracle_micro_accuracy": float(any_correct.float().mean()),
            "oracle_gain_vs_a2_center_pp": float(
                (any_correct.float().mean() - base_correct.float().mean()) * 100.0
            ),
        },
        "training_support_relationship": {
            "high_clean_train_samples": int(train_counts.sum()),
            "minimum_per_class": int(train_counts.min()),
            "maximum_per_class": int(train_counts.max()),
            "spearman_train_count_vs_a2_clean_core_accuracy": spearman_correlation(
                train_count_values, base_accuracy_values
            ),
            "spearman_train_count_vs_m3_delta": spearman_correlation(
                train_count_values, m3_delta_values
            ),
        },
    }
    return report, class_rows, confusion_rows
