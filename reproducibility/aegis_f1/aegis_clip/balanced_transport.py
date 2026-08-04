"""Preregistered balanced-prior transport diagnostics for fixed logits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from aegis_clip.structured_allocation import (
    log_sinkhorn_allocation,
    log_sinkhorn_allocation_until_converged,
)


V1_TEMPERATURE = 1.0
V1_ITERATIONS = 100
V1_NUM_CLASSES = 500
V1_CLEAN_CORE_THRESHOLD = 0.70
V2_MINIMUM_ITERATIONS = 100
V2_MAXIMUM_ITERATIONS = 2000
V2_CHECK_INTERVAL = 10
V2_CONVERGENCE_TOLERANCE = 1.0e-5


def confidence_selected_near_uniform_quotas(
    scores: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Choose integer floor/ceil quotas using each class's marginal support.

    When ``N`` is not divisible by ``C``, exactly ``N % C`` classes receive
    ``ceil(N/C)`` slots.  The extra slots go to classes with the strongest
    score at their prospective final slot, so the unknown missing test items
    are inferred from the model rather than tied to class index order.
    """
    values = torch.as_tensor(scores).detach().float().cpu()
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("scores must have non-empty [N,C] shape with C>=2")
    if not torch.isfinite(values).all():
        raise ValueError("scores must be finite")
    sample_count, num_classes = values.shape
    base, remainder = divmod(int(sample_count), int(num_classes))
    quotas = torch.full((num_classes,), base, dtype=torch.long)
    support_rank = base + 1
    support = values.topk(support_rank, dim=0).values[-1]
    extra_classes = torch.argsort(
        support, descending=True, stable=True
    )[:remainder]
    quotas[extra_classes] += 1
    lower_quota_classes = torch.nonzero(quotas == base).flatten().tolist()
    return quotas, {
        "base_quota": base,
        "extra_slot_classes": int(remainder),
        "lower_quota_classes": lower_quota_classes,
        "quota_min": int(quotas.min()),
        "quota_max": int(quotas.max()),
        "quota_sum": int(quotas.sum()),
    }


def exact_quota_prediction(
    scores: torch.Tensor,
    quotas: torch.Tensor,
    *,
    initial_prediction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Repair an argmax assignment to exact integer quotas deterministically.

    Only rows from overfull source classes are moved.  Scarce target classes
    are filled first, and each move takes the available row with the smallest
    score loss while respecting the remaining source capacity.
    """
    values = torch.as_tensor(scores).detach().float().cpu()
    quota = torch.as_tensor(quotas).detach().long().flatten().cpu()
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("scores must have non-empty [N,C] shape with C>=2")
    if not torch.isfinite(values).all():
        raise ValueError("scores must be finite")
    if quota.numel() != values.shape[1] or bool((quota < 0).any()):
        raise ValueError("quotas must be non-negative with one value per class")
    if int(quota.sum()) != values.shape[0]:
        raise ValueError("quotas must sum to the number of rows")

    if initial_prediction is None:
        current = values.argmax(dim=1)
    else:
        current = torch.as_tensor(initial_prediction).detach().long().flatten().cpu()
        if current.numel() != values.shape[0]:
            raise ValueError("initial_prediction must contain one label per row")
        if bool((current < 0).any()) or bool((current >= values.shape[1]).any()):
            raise ValueError("initial_prediction contains an invalid class")
        current = current.clone()

    initial_counts = torch.bincount(current, minlength=values.shape[1])
    excess = (initial_counts - quota).clamp_min(0)
    deficit = (quota - initial_counts).clamp_min(0)
    if int(excess.sum()) != int(deficit.sum()):
        raise RuntimeError("quota excess and deficit do not balance")

    used = torch.zeros(values.shape[0], dtype=torch.bool)
    target_difficulty: list[tuple[float, int]] = []
    for target in torch.nonzero(deficit, as_tuple=False).flatten().tolist():
        eligible = excess[current] > 0
        rows = torch.nonzero(eligible, as_tuple=False).flatten()
        needed = int(deficit[target])
        if rows.numel() < needed:
            raise RuntimeError("not enough overfull rows to fill a quota")
        losses = values[rows, current[rows]] - values[rows, target]
        boundary_loss = losses.topk(needed, largest=False).values[-1]
        target_difficulty.append((float(boundary_loss), int(target)))

    total_score_loss = 0.0
    maximum_score_loss = 0.0
    for _, target in sorted(
        target_difficulty,
        key=lambda item: (-item[0], item[1]),
    ):
        for _ in range(int(deficit[target])):
            eligible = (~used) & (excess[current] > 0)
            rows = torch.nonzero(eligible, as_tuple=False).flatten()
            if rows.numel() == 0:
                raise RuntimeError("quota repair exhausted eligible source rows")
            losses = values[rows, current[rows]] - values[rows, target]
            local_index = int(torch.argmin(losses))
            row = int(rows[local_index])
            source = int(current[row])
            loss = float(losses[local_index])
            current[row] = int(target)
            used[row] = True
            excess[source] -= 1
            total_score_loss += loss
            maximum_score_loss = max(maximum_score_loss, loss)

    final_counts = torch.bincount(current, minlength=values.shape[1])
    exact = bool(torch.equal(final_counts, quota))
    if not exact:
        raise RuntimeError("quota repair did not produce the requested counts")
    return current, {
        "exact_quota_match": exact,
        "moved_rows": int(used.sum()),
        "initial_count_min": int(initial_counts.min()),
        "initial_count_max": int(initial_counts.max()),
        "final_count_min": int(final_counts.min()),
        "final_count_max": int(final_counts.max()),
        "total_score_loss": total_score_loss,
        "maximum_score_loss": maximum_score_loss,
    }


def exact_near_uniform_transport_prediction(
    logits: torch.Tensor,
    *,
    temperature: float = 0.5,
    iterations: int = V1_ITERATIONS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sinkhorn-calibrate logits, then enforce confidence-selected 49/50 quotas."""
    values = torch.as_tensor(logits).detach().float().cpu()
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("logits must have non-empty [N,C] shape with C>=2")
    if not torch.isfinite(values).all():
        raise ValueError("logits must be finite")
    target_counts = uniform_target_counts(values.shape[0], values.shape[1])
    allocation, sinkhorn = log_sinkhorn_allocation(
        values,
        target_counts,
        temperature=float(temperature),
        iterations=int(iterations),
    )
    scores = allocation.clamp_min(torch.finfo(allocation.dtype).tiny).log()
    quotas, quota_selection = confidence_selected_near_uniform_quotas(scores)
    prediction, repair = exact_quota_prediction(scores, quotas)
    return prediction, {
        "sinkhorn": sinkhorn,
        "quota_selection": quota_selection,
        "repair": repair,
        "hard_balance": hard_balance_diagnostics(
            prediction, num_classes=values.shape[1]
        ),
    }


def uniform_target_counts(
    sample_count: int,
    num_classes: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return an exactly uniform *soft* target marginal summing to N rows."""
    if int(sample_count) < 1:
        raise ValueError("sample_count must be positive")
    if int(num_classes) < 2:
        raise ValueError("num_classes must be at least two")
    return torch.full(
        (int(num_classes),),
        float(sample_count) / float(num_classes),
        dtype=torch.float32,
        device=device,
    )


def hard_balance_diagnostics(
    prediction: torch.Tensor,
    *,
    num_classes: int,
) -> dict[str, float | int]:
    values = torch.as_tensor(prediction).long().flatten().cpu()
    if values.numel() == 0:
        raise ValueError("prediction cannot be empty")
    if (values < 0).any() or (values >= int(num_classes)).any():
        raise ValueError("prediction contains a class outside the declared range")
    counts = torch.bincount(values, minlength=int(num_classes)).float()
    target = float(values.numel()) / float(num_classes)
    return {
        "prediction_count_min": int(counts.min()),
        "prediction_count_max": int(counts.max()),
        "prediction_empty_classes": int((counts == 0).sum()),
        "prediction_count_cv": float(
            counts.std(unbiased=False) / counts.mean().clamp_min(1.0e-12)
        ),
        "prediction_count_l1_from_uniform": float((counts - target).abs().sum()),
        "prediction_count_max_abs_from_uniform": float(
            (counts - target).abs().max()
        ),
    }


def balanced_transport_prediction(
    logits: torch.Tensor,
    *,
    temperature: float = V1_TEMPERATURE,
    iterations: int = V1_ITERATIONS,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Apply fixed soft-uniform Sinkhorn transport and return hard argmax labels."""
    values = torch.as_tensor(logits).float()
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("logits must have non-empty [N,C] shape with C>=2")
    if not torch.isfinite(values).all():
        raise ValueError("logits must be finite")
    target_counts = uniform_target_counts(
        values.shape[0], values.shape[1], device=values.device
    )
    allocation, diagnostics = log_sinkhorn_allocation(
        values,
        target_counts,
        temperature=float(temperature),
        iterations=int(iterations),
    )
    prediction = allocation.argmax(dim=1).cpu()
    diagnostics = {
        **diagnostics,
        "target_count_per_class": float(target_counts[0]),
        "soft_target_count_sum": float(target_counts.sum()),
    }
    return prediction, diagnostics


def converged_balanced_transport_prediction(
    logits: torch.Tensor,
    *,
    temperature: float = V1_TEMPERATURE,
    minimum_iterations: int = V2_MINIMUM_ITERATIONS,
    maximum_iterations: int = V2_MAXIMUM_ITERATIONS,
    check_interval: int = V2_CHECK_INTERVAL,
    tolerance: float = V2_CONVERGENCE_TOLERANCE,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Apply the preregistered convergence-driven balanced transport."""
    values = torch.as_tensor(logits).float()
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("logits must have non-empty [N,C] shape with C>=2")
    if not torch.isfinite(values).all():
        raise ValueError("logits must be finite")
    target_counts = uniform_target_counts(
        values.shape[0], values.shape[1], device=values.device
    )
    allocation, diagnostics = log_sinkhorn_allocation_until_converged(
        values,
        target_counts,
        temperature=float(temperature),
        minimum_iterations=int(minimum_iterations),
        maximum_iterations=int(maximum_iterations),
        check_interval=int(check_interval),
        tolerance=float(tolerance),
    )
    prediction = allocation.argmax(dim=1).cpu()
    diagnostics = {
        **diagnostics,
        "target_count_per_class": float(target_counts[0]),
        "soft_target_count_sum": float(target_counts.sum()),
    }
    return prediction, diagnostics


def relative_cv_reduction(baseline_cv: float, candidate_cv: float) -> float:
    baseline = float(baseline_cv)
    candidate = float(candidate_cv)
    if baseline <= 0.0:
        return 0.0 if candidate == baseline else float("-inf")
    return (baseline - candidate) / baseline


def paired_change_summary(
    baseline_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    labels: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, int | float]:
    baseline = torch.as_tensor(baseline_prediction).long().flatten().cpu()
    candidate = torch.as_tensor(candidate_prediction).long().flatten().cpu()
    targets = torch.as_tensor(labels).long().flatten().cpu()
    if not (baseline.numel() == candidate.numel() == targets.numel()):
        raise ValueError("paired predictions and labels must have equal length")
    selected = torch.ones_like(targets, dtype=torch.bool)
    if mask is not None:
        selected = torch.as_tensor(mask).bool().flatten().cpu()
        if selected.numel() != targets.numel():
            raise ValueError("paired mask must align with labels")
    baseline = baseline[selected]
    candidate = candidate[selected]
    targets = targets[selected]
    baseline_correct = baseline.eq(targets)
    candidate_correct = candidate.eq(targets)
    fixed = (~baseline_correct) & candidate_correct
    broken = baseline_correct & (~candidate_correct)
    changed = baseline.ne(candidate)
    return {
        "samples": int(targets.numel()),
        "changed": int(changed.sum()),
        "changed_fraction": float(changed.float().mean()) if changed.numel() else 0.0,
        "wrong_to_correct": int(fixed.sum()),
        "correct_to_wrong": int(broken.sum()),
        "net_correct": int(fixed.sum() - broken.sum()),
    }


def v1_gate_decision(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the immutable cross-checkpoint V1 promotion gate."""
    required_reports = {"f1_m1", "a2_m1"}
    missing = required_reports - set(reports)
    if missing:
        raise ValueError(f"V1 gate reports missing: {sorted(missing)}")
    checks: dict[str, dict[str, Any]] = {}

    def add(
        name: str,
        actual: float | int,
        operator: str,
        required: float | int,
    ) -> None:
        if operator == ">=":
            passed = float(actual) >= float(required)
        elif operator == "<=":
            passed = float(actual) <= float(required)
        elif operator == "==":
            passed = float(actual) == float(required)
        else:
            raise ValueError(f"Unsupported gate operator: {operator}")
        checks[name] = {
            "actual": actual,
            "operator": operator,
            "required": required,
            "passed": bool(passed),
        }

    accuracy_thresholds = {
        "f1_m1": {
            "clean_core_micro": 0.20,
            "trusted_macro": -0.05,
            "raw_micro": -0.10,
        },
        "a2_m1": {
            "clean_core_micro": 0.10,
            "trusted_macro": -0.05,
            "raw_micro": -0.10,
        },
    }
    for name in ("f1_m1", "a2_m1"):
        report = reports[name]
        delta = report["delta_pp"]
        for metric, threshold in accuracy_thresholds[name].items():
            add(
                f"{name}_{metric}_delta_pp",
                float(delta[metric]),
                ">=",
                threshold,
            )
        add(
            f"{name}_prediction_count_cv_relative_reduction",
            float(report["prediction_count_cv_relative_reduction"]),
            ">=",
            0.20,
        )
        add(
            f"{name}_prediction_empty_classes",
            int(report["transport"]["prediction_empty_classes"]),
            "==",
            0,
        )
        add(
            f"{name}_sinkhorn_maximum_row_absolute_error",
            float(report["sinkhorn"]["maximum_row_absolute_error"]),
            "<=",
            1.0e-4,
        )
        add(
            f"{name}_sinkhorn_maximum_column_absolute_error",
            float(report["sinkhorn"]["maximum_column_absolute_error"]),
            "<=",
            0.05,
        )
    passed = all(bool(item["passed"]) for item in checks.values())
    return {
        "protocol": "V1_F1_M1_KNOWN_BALANCED_PRIOR_TRANSPORT",
        "passed": passed,
        "decision": (
            "eligible_for_compliance_review"
            if passed
            else "closed_no_test_inference"
        ),
        "checks": checks,
        "failed_checks": [
            name for name, item in checks.items() if not bool(item["passed"])
        ],
    }


def v2_gate_decision(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate V1 accuracy checks plus the stricter V2 convergence checks."""
    result = v1_gate_decision(reports)
    checks = dict(result["checks"])
    for name in ("f1_m1", "a2_m1"):
        report = reports[name]
        for metric in (
            "maximum_row_absolute_error",
            "maximum_column_absolute_error",
        ):
            key = f"{name}_sinkhorn_{metric}"
            actual = float(report["sinkhorn"][metric])
            checks[key] = {
                "actual": actual,
                "operator": "<=",
                "required": V2_CONVERGENCE_TOLERANCE,
                "passed": actual <= V2_CONVERGENCE_TOLERANCE,
            }
        converged = bool(report["sinkhorn"].get("converged", False))
        checks[f"{name}_sinkhorn_converged"] = {
            "actual": converged,
            "operator": "==",
            "required": True,
            "passed": converged,
        }
    passed = all(bool(item["passed"]) for item in checks.values())
    return {
        "protocol": "V2_F1_M1_CONVERGED_KNOWN_BALANCED_PRIOR_TRANSPORT",
        "passed": passed,
        "decision": (
            "eligible_for_compliance_review"
            if passed
            else "closed_no_test_inference"
        ),
        "checks": checks,
        "failed_checks": [
            name for name, item in checks.items() if not bool(item["passed"])
        ],
    }
