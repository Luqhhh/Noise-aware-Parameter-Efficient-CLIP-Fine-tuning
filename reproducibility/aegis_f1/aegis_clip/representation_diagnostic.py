"""Measure whether a visual adaptation recovers previously persistent errors."""

from __future__ import annotations

from typing import Any

import torch

from aegis_clip.class_diagnostic import (
    paired_bootstrap_delta_pp,
    spearman_correlation,
    validate_aligned_logit_caches,
)


def _transition(
    source_correct: torch.Tensor, candidate_correct: torch.Tensor
) -> dict[str, int]:
    corrected = ~source_correct & candidate_correct
    harmed = source_correct & ~candidate_correct
    return {
        "corrected": int(corrected.sum()),
        "harmed": int(harmed.sum()),
        "net_correct": int(corrected.sum() - harmed.sum()),
    }


def diagnose_representation_shift(
    a2_center: dict[str, Any],
    a2_m1: dict[str, Any],
    a2_m3: dict[str, Any],
    n3_center: dict[str, Any],
    n3_m3: dict[str, Any],
    train_cache: dict[str, Any],
    *,
    num_classes: int = 500,
    clean_core_threshold: float = 0.70,
) -> dict[str, Any]:
    caches = [a2_center, a2_m1, a2_m3, n3_center, n3_m3]
    validate_aligned_logit_caches(caches, num_classes=num_classes)
    labels = torch.as_tensor(a2_center["labels"]).long().cpu()
    clean = (
        torch.as_tensor(a2_center["clean_probability"]).float().cpu()
        >= clean_core_threshold
    )
    y = labels[clean]
    predictions = {
        "a2_center": torch.as_tensor(a2_center["logits"]).argmax(dim=1)[clean],
        "a2_m1": torch.as_tensor(a2_m1["logits"]).argmax(dim=1)[clean],
        "a2_m3": torch.as_tensor(a2_m3["logits"]).argmax(dim=1)[clean],
        "n3_center": torch.as_tensor(n3_center["logits"]).argmax(dim=1)[clean],
        "n3_m3": torch.as_tensor(n3_m3["logits"]).argmax(dim=1)[clean],
    }
    correct = {name: prediction.eq(y) for name, prediction in predictions.items()}
    a2_persistent_wrong = ~torch.stack(
        [correct["a2_center"], correct["a2_m1"], correct["a2_m3"]]
    ).any(dim=0)
    recovered_center = a2_persistent_wrong & correct["n3_center"]
    recovered_m3 = a2_persistent_wrong & correct["n3_m3"]

    support = torch.bincount(y, minlength=num_classes)
    class_accuracy: dict[str, torch.Tensor] = {}
    for name in ("a2_m3", "n3_m3"):
        class_correct = torch.bincount(y[correct[name]], minlength=num_classes)
        class_accuracy[name] = class_correct.float() / support.clamp_min(1)
    class_delta = class_accuracy["n3_m3"] - class_accuracy["a2_m3"]
    train_labels = torch.as_tensor(train_cache["labels"]).long().cpu()
    train_counts = torch.bincount(train_labels, minlength=num_classes)

    ordered = torch.argsort(train_counts)
    quartiles = []
    for index, class_indices in enumerate(torch.tensor_split(ordered, 4), start=1):
        quartiles.append(
            {
                "quartile": index,
                "classes": int(class_indices.numel()),
                "minimum_train_samples": int(train_counts[class_indices].min()),
                "maximum_train_samples": int(train_counts[class_indices].max()),
                "a2_m3_macro_accuracy": float(
                    class_accuracy["a2_m3"][class_indices].mean()
                ),
                "n3_m3_macro_accuracy": float(
                    class_accuracy["n3_m3"][class_indices].mean()
                ),
                "n3_vs_a2_m3_delta_pp": float(
                    class_delta[class_indices].mean() * 100.0
                ),
            }
        )

    combined_oracle = torch.stack(
        [correct["a2_m3"], correct["n3_m3"]]
    ).any(dim=0)
    return {
        "cohort": {
            "samples": int(clean.sum()),
            "classes": int((support > 0).sum()),
            "definition": f"clean_probability >= {clean_core_threshold:.2f}",
        },
        "accuracy": {
            name: float(value.float().mean()) for name, value in correct.items()
        },
        "transitions": {
            "a2_center_to_n3_center": _transition(
                correct["a2_center"], correct["n3_center"]
            ),
            "a2_m3_to_n3_m3": _transition(correct["a2_m3"], correct["n3_m3"]),
            "a2_m3_to_n3_m3_paired_bootstrap_delta_pp": paired_bootstrap_delta_pp(
                correct["a2_m3"], correct["n3_m3"]
            ),
        },
        "previously_persistent_errors": {
            "a2_center_m1_m3_all_wrong": int(a2_persistent_wrong.sum()),
            "recovered_by_n3_center": int(recovered_center.sum()),
            "recovered_by_n3_m3": int(recovered_m3.sum()),
            "recovery_fraction_by_n3_m3": float(
                recovered_m3.sum() / a2_persistent_wrong.sum().clamp_min(1)
            ),
            "remaining_wrong_after_n3_m3": int(
                (a2_persistent_wrong & ~correct["n3_m3"]).sum()
            ),
        },
        "complementarity": {
            "a2_m3_plus_n3_m3_oracle_accuracy": float(
                combined_oracle.float().mean()
            ),
            "oracle_gain_over_n3_m3_pp": float(
                (combined_oracle.float().mean() - correct["n3_m3"].float().mean())
                * 100.0
            ),
        },
        "class_effects": {
            "classes_improved": int((class_delta > 0).sum()),
            "classes_harmed": int((class_delta < 0).sum()),
            "classes_unchanged": int((class_delta == 0).sum()),
            "spearman_train_count_vs_n3_gain": spearman_correlation(
                [float(value) for value in train_counts],
                [float(value) for value in class_delta],
            ),
            "train_support_quartiles": quartiles,
        },
        "interpretation_boundary": "descriptive fixed-validation diagnostic; not a platform-score estimate",
        "test_data_used": False,
    }
