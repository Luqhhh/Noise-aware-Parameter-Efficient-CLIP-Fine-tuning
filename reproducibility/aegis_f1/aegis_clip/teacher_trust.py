"""Conservative teacher augmentation for an existing cross-fitted trust bundle."""

from __future__ import annotations

import copy
from typing import Any

import torch


def augment_teacher_trust(
    base: dict[str, Any],
    noisy_labels: torch.Tensor,
    center_logits: torch.Tensor,
    flip_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    minimum_confidence: float = 0.90,
    minimum_margin: float = 0.75,
    maximum_clean_probability: float = 0.60,
    admission_clean_probability: float = 0.65,
    correction_alpha: float = 0.50,
    maximum_class_fraction: float = 0.08,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add bounded two-view teacher corrections without replacing OOF corrections."""

    labels = torch.as_tensor(noisy_labels, dtype=torch.long).flatten().cpu()
    center = torch.as_tensor(center_logits, dtype=torch.float32).cpu()
    flip = torch.as_tensor(flip_logits, dtype=torch.float32).cpu()
    sample_count = labels.numel()
    if center.ndim != 2 or center.shape != flip.shape:
        raise ValueError("teacher logits must have equal [N,C] shapes")
    if center.shape[0] != sample_count:
        raise ValueError("teacher logits and noisy labels have different lengths")
    if not float(temperature) > 0.0:
        raise ValueError("temperature must be positive")
    for name, value in {
        "minimum_confidence": minimum_confidence,
        "minimum_margin": minimum_margin,
        "maximum_clean_probability": maximum_clean_probability,
        "admission_clean_probability": admission_clean_probability,
        "correction_alpha": correction_alpha,
        "maximum_class_fraction": maximum_class_fraction,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    if admission_clean_probability < maximum_clean_probability:
        raise ValueError(
            "admission_clean_probability must not be below maximum_clean_probability"
        )

    required = {
        "paths",
        "clean_probability",
        "pseudo_label",
        "pseudo_confidence",
        "correction_alpha",
    }
    missing = required - set(base)
    if missing:
        raise ValueError(f"base trust bundle is missing fields: {sorted(missing)}")
    if len(base["paths"]) != sample_count:
        raise ValueError("base trust paths and teacher samples have different lengths")

    clean = torch.as_tensor(base["clean_probability"], dtype=torch.float32).flatten()
    pseudo = torch.as_tensor(base["pseudo_label"], dtype=torch.long).flatten()
    pseudo_confidence = torch.as_tensor(
        base["pseudo_confidence"], dtype=torch.float32
    ).flatten()
    alpha = torch.as_tensor(base["correction_alpha"], dtype=torch.float32).flatten()
    if {clean.numel(), pseudo.numel(), pseudo_confidence.numel(), alpha.numel()} != {
        sample_count
    }:
        raise ValueError("base trust vectors have inconsistent lengths")

    center_probability = torch.softmax(center / float(temperature), dim=1)
    flip_probability = torch.softmax(flip / float(temperature), dim=1)
    center_prediction = center_probability.argmax(dim=1)
    flip_prediction = flip_probability.argmax(dim=1)
    fused_probability = (center_probability + flip_probability) / 2.0
    top = fused_probability.topk(2, dim=1)
    teacher_prediction = top.indices[:, 0]
    confidence = top.values[:, 0]
    margin = top.values[:, 0] - top.values[:, 1]

    eligible = (
        (center_prediction == flip_prediction)
        & (teacher_prediction == center_prediction)
        & (teacher_prediction != labels)
        & (confidence >= float(minimum_confidence))
        & (margin >= float(minimum_margin))
        & (clean < float(maximum_clean_probability))
        & (alpha <= 0.0)
        & ((pseudo < 0) | (pseudo == teacher_prediction))
    )

    num_classes = center.shape[1]
    noisy_counts = torch.bincount(labels, minlength=num_classes)
    per_class_limit = torch.clamp(
        torch.ceil(noisy_counts.float() * float(maximum_class_fraction)).long(),
        min=1,
    )
    source_used = torch.zeros(num_classes, dtype=torch.long)
    target_used = torch.zeros(num_classes, dtype=torch.long)
    accepted = torch.zeros(sample_count, dtype=torch.bool)
    ranking = torch.where(eligible)[0]
    if ranking.numel():
        score = confidence[ranking] + margin[ranking] * 1.0e-3
        ranking = ranking[torch.argsort(score, descending=True, stable=True)]
    for index in ranking.tolist():
        source = int(labels[index])
        target = int(teacher_prediction[index])
        if source_used[source] >= per_class_limit[source]:
            continue
        if target_used[target] >= per_class_limit[target]:
            continue
        accepted[index] = True
        source_used[source] += 1
        target_used[target] += 1

    output = copy.deepcopy(base)
    output_clean = clean.clone()
    output_pseudo = pseudo.clone()
    output_confidence = pseudo_confidence.clone()
    output_alpha = alpha.clone()
    output_clean[accepted] = torch.maximum(
        output_clean[accepted],
        torch.full_like(output_clean[accepted], float(admission_clean_probability)),
    )
    output_pseudo[accepted] = teacher_prediction[accepted]
    output_confidence[accepted] = confidence[accepted]
    output_alpha[accepted] = float(correction_alpha)
    output["clean_probability"] = output_clean
    output["pseudo_label"] = output_pseudo
    output["pseudo_confidence"] = output_confidence
    output["correction_alpha"] = output_alpha

    existing_corrections = int((alpha > 0.0).sum())
    accepted_count = int(accepted.sum())
    audit = {
        "sample_count": sample_count,
        "num_classes": num_classes,
        "two_view_agreement": int((center_prediction == flip_prediction).sum()),
        "teacher_noisy_disagreements": int((teacher_prediction != labels).sum()),
        "eligible_before_class_caps": int(eligible.sum()),
        "accepted_teacher_corrections": accepted_count,
        "existing_oof_corrections": existing_corrections,
        "total_corrections": int((output_alpha > 0.0).sum()),
        "maximum_source_class_corrections": int(source_used.max()),
        "maximum_target_class_corrections": int(target_used.max()),
        "maximum_per_class_limit": int(per_class_limit.max()),
        "mean_accepted_confidence": (
            float(confidence[accepted].mean()) if accepted_count else None
        ),
        "mean_accepted_margin": float(margin[accepted].mean()) if accepted_count else None,
        "parameters": {
            "temperature": float(temperature),
            "minimum_confidence": float(minimum_confidence),
            "minimum_margin": float(minimum_margin),
            "maximum_clean_probability": float(maximum_clean_probability),
            "admission_clean_probability": float(admission_clean_probability),
            "correction_alpha": float(correction_alpha),
            "maximum_class_fraction": float(maximum_class_fraction),
        },
    }
    return output, audit
