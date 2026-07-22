"""Strict, preregistered gate for the paired T0/T1 subspace experiment."""

from __future__ import annotations

import math
from typing import Any

import torch

from aegis_clip.balanced_inference import prediction_metrics


VALIDATION_FIELDS = (
    "paths",
    "labels",
    "clean_probability",
    "pseudo_labels",
    "correction_alpha",
)
METRIC_NAMES = (
    "raw_micro",
    "raw_macro",
    "trusted_micro",
    "trusted_macro",
    "proxy_micro",
    "proxy_macro",
    "clean_core_micro",
    "clean_core_macro",
)


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return torch.equal(torch.as_tensor(left), torch.as_tensor(right))
    return left == right


def _require_aligned(caches: dict[str, dict[str, Any]]) -> None:
    names = list(caches)
    reference = caches[names[0]]
    for name in names[1:]:
        candidate = caches[name]
        for field in VALIDATION_FIELDS:
            if field not in reference or field not in candidate:
                raise ValueError(f"Cache is missing validation field {field}")
            if not _same_value(reference[field], candidate[field]):
                raise ValueError(
                    f"Validation field {field} differs for cache {name}"
                )


def _cache_metrics(cache: dict[str, Any], threshold: float) -> dict[str, Any]:
    logits = torch.as_tensor(cache["logits"]).float()
    if logits.ndim != 2 or logits.shape[1] <= 1:
        raise ValueError("Validation logits must have shape [N,C]")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Validation logits contain NaN or Inf")
    labels = torch.as_tensor(cache["labels"]).long().flatten()
    pseudo_labels = torch.as_tensor(cache["pseudo_labels"]).long().flatten()
    clean = torch.as_tensor(cache["clean_probability"]).float().flatten()
    correction = torch.as_tensor(cache["correction_alpha"]).float().flatten()
    if not (
        logits.shape[0]
        == labels.numel()
        == pseudo_labels.numel()
        == clean.numel()
        == correction.numel()
        == len(cache["paths"])
    ):
        raise ValueError("Validation cache fields have inconsistent lengths")
    if not torch.isfinite(clean).all() or not torch.isfinite(correction).all():
        raise FloatingPointError("Validation trust fields contain NaN or Inf")
    if (
        labels.min() < 0
        or labels.max() >= logits.shape[1]
        or pseudo_labels.min() < 0
        or pseudo_labels.max() >= logits.shape[1]
    ):
        raise ValueError("Validation labels are outside the classifier range")
    return prediction_metrics(
        logits.argmax(dim=1),
        labels=labels,
        clean_probability=clean,
        pseudo_labels=pseudo_labels,
        correction_alpha=correction,
        num_classes=logits.shape[1],
        clean_core_threshold=float(threshold),
    )


def _delta_pp(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        name: 100.0 * (float(candidate[name]) - float(baseline[name]))
        for name in METRIC_NAMES
    }


def _number(row: dict[str, Any], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Training metrics are missing numeric field {name}") from exc
    if not math.isfinite(value):
        raise FloatingPointError(f"Training metric {name} is non-finite")
    return value


def evaluate_trust_subspace_gate(
    *,
    original_m1: dict[str, Any],
    t0_center: dict[str, Any],
    t0_m1: dict[str, Any],
    t1_center: dict[str, Any],
    t1_m1: dict[str, Any],
    t0_initial: dict[str, Any],
    t1_initial: dict[str, Any],
    t0_evaluation: dict[str, Any],
    t1_evaluation: dict[str, Any],
    t0_last_metrics: dict[str, Any],
    t1_last_metrics: dict[str, Any],
    clean_core_threshold: float = 0.70,
) -> dict[str, Any]:
    """Evaluate every mechanical and performance condition without test data."""
    caches = {
        "original_m1": original_m1,
        "t0_center": t0_center,
        "t0_m1": t0_m1,
        "t1_center": t1_center,
        "t1_m1": t1_m1,
    }
    _require_aligned(caches)
    expected_views = {
        "t0_center": "center",
        "t0_m1": "attention_local_global",
        "t1_center": "center",
        "t1_m1": "attention_local_global",
    }
    for name, expected in expected_views.items():
        if caches[name].get("view_mode") != expected:
            raise ValueError(f"{name} must use view_mode={expected}")
    if t0_center.get("checkpoint_sha256") != t0_m1.get("checkpoint_sha256"):
        raise ValueError("T0 center and M1 caches use different checkpoints")
    if t1_center.get("checkpoint_sha256") != t1_m1.get("checkpoint_sha256"):
        raise ValueError("T1 center and M1 caches use different checkpoints")

    metrics = {
        name: _cache_metrics(cache, clean_core_threshold)
        for name, cache in caches.items()
    }
    deltas = {
        "t1_m1_minus_t0_m1_pp": _delta_pp(metrics["t1_m1"], metrics["t0_m1"]),
        "t1_center_minus_t0_center_pp": _delta_pp(
            metrics["t1_center"], metrics["t0_center"]
        ),
        "t1_m1_minus_original_m1_pp": _delta_pp(
            metrics["t1_m1"], metrics["original_m1"]
        ),
    }

    t0_steps = _number(t0_last_metrics, "train_trust_subspace_steps")
    t1_steps = _number(t1_last_metrics, "train_trust_subspace_steps")
    t1_projection_steps = _number(
        t1_last_metrics, "train_trust_subspace_projection_steps"
    )
    retained_ratio = _number(
        t1_last_metrics, "train_trust_subspace_retained_norm_ratio"
    )
    checks = {
        "initial_evaluation_exact_match": t0_initial == t1_initial,
        "epoch2_control": _number(t0_last_metrics, "epoch") == 2.0,
        "epoch2_treatment": _number(t1_last_metrics, "epoch") == 2.0,
        "paired_step_count": t0_steps > 0.0 and t0_steps == t1_steps,
        "paired_trusted_examples": _number(
            t0_last_metrics, "train_trust_subspace_trusted_examples"
        )
        == _number(t1_last_metrics, "train_trust_subspace_trusted_examples"),
        "paired_uncertain_examples": _number(
            t0_last_metrics, "train_trust_subspace_uncertain_examples"
        )
        == _number(t1_last_metrics, "train_trust_subspace_uncertain_examples"),
        "control_projection_zero": _number(
            t0_last_metrics, "train_trust_subspace_projection_steps"
        )
        == 0.0,
        "control_basis_rank8": _number(
            t0_last_metrics, "train_trust_subspace_basis_rank"
        )
        == 8.0,
        "treatment_basis_rank8": _number(
            t1_last_metrics, "train_trust_subspace_basis_rank"
        )
        == 8.0,
        "no_skipped_steps": _number(
            t0_last_metrics, "train_trust_subspace_skipped_steps"
        )
        == 0.0
        and _number(t1_last_metrics, "train_trust_subspace_skipped_steps") == 0.0,
        "treatment_projection_coverage_95pct": (
            t1_steps > 0.0 and t1_projection_steps / t1_steps >= 0.95
        ),
        "retained_ratio_strictly_between_zero_and_one": 0.0 < retained_ratio < 1.0,
        "m1_clean_core_gain_vs_t0": deltas["t1_m1_minus_t0_m1_pp"][
            "clean_core_micro"
        ]
        >= 0.20,
        "m1_trusted_macro_safety_vs_t0": deltas["t1_m1_minus_t0_m1_pp"][
            "trusted_macro"
        ]
        >= -0.05,
        "m1_raw_micro_safety_vs_t0": deltas["t1_m1_minus_t0_m1_pp"][
            "raw_micro"
        ]
        >= -0.10,
        "center_clean_core_safety_vs_t0": deltas[
            "t1_center_minus_t0_center_pp"
        ]["clean_core_micro"]
        >= -0.20,
        "m1_clean_core_gain_vs_original_f1": deltas[
            "t1_m1_minus_original_m1_pp"
        ]["clean_core_micro"]
        >= 0.10,
        "feature_drift_within_one_percent": float(
            t1_evaluation["mean_feature_drift"]
        )
        <= 0.01,
        "t1_m1_no_empty_prediction_class": metrics["t1_m1"][
            "prediction_empty_classes"
        ]
        == 0,
    }
    if not math.isfinite(float(t0_evaluation["mean_feature_drift"])) or not math.isfinite(
        float(t1_evaluation["mean_feature_drift"])
    ):
        raise FloatingPointError("Evaluation feature drift is non-finite")
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "delta_pp": deltas,
        "mechanics": {
            "t0_last_metrics": t0_last_metrics,
            "t1_last_metrics": t1_last_metrics,
            "t0_mean_feature_drift": float(t0_evaluation["mean_feature_drift"]),
            "t1_mean_feature_drift": float(t1_evaluation["mean_feature_drift"]),
            "projection_coverage": (
                t1_projection_steps / t1_steps if t1_steps > 0.0 else 0.0
            ),
            "retained_norm_ratio": retained_ratio,
        },
        "thresholds": {
            "t1_vs_t0_m1_clean_core_micro_pp": 0.20,
            "t1_vs_t0_m1_trusted_macro_pp": -0.05,
            "t1_vs_t0_m1_raw_micro_pp": -0.10,
            "t1_vs_t0_center_clean_core_micro_pp": -0.20,
            "t1_vs_original_m1_clean_core_micro_pp": 0.10,
            "maximum_feature_drift": 0.01,
            "minimum_projection_coverage": 0.95,
        },
    }
