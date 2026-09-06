"""Deterministic conditional grouped nested OOF for SCOPE-K2."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from aegis_clip.runtime import sha256_file
from aegis_clip.scope_cache import (
    atomic_save_scope_cache,
    load_scope_cache,
    semantic_sha256,
    tensor_sha256,
)
from aegis_clip.scope_protocol import ScopeProtocol


@dataclass(frozen=True)
class BetaFit:
    beta: float
    intercept: float
    objective: float
    gradient: float
    iterations: int
    upper_bound: float


@dataclass(frozen=True)
class OOFThreshold:
    mode: str
    gamma: float | None
    k_oof: int
    n_oof: int
    rho: float | None
    no_switch_reason: str | None
    eligible_score_hash: str
    candidate_count: int
    switch_mask: torch.Tensor


@dataclass(frozen=True)
class DeployedThreshold:
    mode: str
    gamma: float | None
    k_refit: int
    n_refit: int
    refit_fraction: float | None
    no_switch_reason: str | None
    eligible_score_hash: str


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
    draws: int
    seed: int


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    gates: dict[str, bool]


@dataclass(frozen=True)
class MethodOOFResult:
    method: str
    predictions: torch.Tensor
    switches: torch.Tensor
    outer_fold_id: torch.Tensor
    beta_by_outer_fold: tuple[float | None, ...]
    oof_thresholds: tuple[OOFThreshold, ...]
    mapped_thresholds: tuple[DeployedThreshold, ...]
    failure_reasons: tuple[str | None, ...]


METHODS = ("parent", "margin_only", "pace", "gate_only", "no_topology", "scope")


def stratified_group_folds(
    labels: torch.Tensor, groups: Sequence[str], *, folds: int, seed: int,
) -> torch.Tensor:
    """Return deterministic SGKF IDs and prove that no group is split."""
    labels = torch.as_tensor(labels, dtype=torch.int64).flatten().cpu()
    groups = [str(value) for value in groups]
    if labels.numel() != len(groups) or labels.numel() < int(folds) or int(folds) < 2:
        raise ValueError("fold inputs are misaligned or too small")
    splitter = StratifiedGroupKFold(
        n_splits=int(folds), shuffle=True, random_state=int(seed),
    )
    indices = np.arange(labels.numel(), dtype=np.int64)
    result = torch.full((labels.numel(),), -1, dtype=torch.int64)
    for fold, (_, holdout) in enumerate(splitter.split(indices, labels.numpy(), groups)):
        rows = torch.as_tensor(holdout, dtype=torch.int64)
        if (result[rows] >= 0).any():
            raise ValueError("a formal row received multiple folds")
        result[rows] = int(fold)
    if (result < 0).any():
        raise ValueError("fold assignment is incomplete")
    memberships: dict[str, set[int]] = {}
    for group, fold in zip(groups, result.tolist()):
        memberships.setdefault(group, set()).add(int(fold))
    if any(len(value) != 1 for value in memberships.values()):
        raise ValueError("duplicate group crosses fold boundaries")
    return result


def candidate_pair_training_mask(
    candidates: torch.Tensor, labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return y-in-{a,b} and the b-target; third classes remain false targets."""
    candidates = torch.as_tensor(candidates, dtype=torch.int64).cpu()
    labels = torch.as_tensor(labels, dtype=torch.int64).flatten().cpu()
    if candidates.shape != (labels.numel(), 2):
        raise ValueError("candidate/label shapes disagree")
    top = candidates[:, 0].eq(labels)
    runner = candidates[:, 1].eq(labels)
    return top | runner, runner


def _validate_fold_payload(
    payload: Mapping[str, Any], parent: Mapping[str, Any], groups: Sequence[str],
    protocol: ScopeProtocol,
) -> int:
    if payload.get("schema") != "scope_nested_folds_v1":
        raise ValueError("fold artifact schema mismatch")
    paths = [str(value) for value in parent["paths"]]
    labels = torch.as_tensor(parent["label"], dtype=torch.int64).cpu()
    groups = [str(value) for value in groups]
    n = len(paths)
    if len(groups) != n or labels.shape != (n,):
        raise ValueError("fold bindings are misaligned")
    if payload.get("paths") != paths or payload.get("groups") != groups:
        raise ValueError("fold path/group binding mismatch")
    for key, expected in (
        ("formal_row_id", torch.arange(n, dtype=torch.int64)),
        ("labels", labels),
    ):
        current = payload.get(key)
        if not isinstance(current, torch.Tensor) or current.dtype != expected.dtype or not torch.equal(current, expected):
            raise ValueError(f"fold {key} binding mismatch")
    if payload.get("formal_row_binding_sha256") != parent.get("formal_row_binding_sha256"):
        raise ValueError("fold formal row binding mismatch")
    if payload.get("parent_semantic_sha256") != semantic_sha256(parent):
        raise ValueError("fold parent semantic mismatch")
    if payload.get("parent_cache_sha256") != parent.get("_cache_sha256"):
        raise ValueError("fold parent file binding mismatch")
    if payload.get("group_artifact_sha256") != protocol.assets.group_artifact_sha256:
        raise ValueError("fold group artifact mismatch")
    if payload.get("protocol_sha256") != sha256_file(protocol.config_path):
        raise ValueError("fold protocol hash mismatch")
    crossfit = protocol.fixed["crossfit"]
    if payload.get("crossfit") != crossfit or payload.get("conditional_parent") is not True:
        raise ValueError("fold crossfit protocol mismatch")
    outer = torch.as_tensor(payload.get("outer_fold_id"), dtype=torch.int64).cpu()
    if outer.shape != (n,) or sorted(outer.unique().tolist()) != list(range(int(crossfit["outer_folds"]))):
        raise ValueError("outer fold IDs are malformed")
    for group in sorted(set(groups)):
        rows = torch.tensor([i for i, value in enumerate(groups) if value == group])
        if outer[rows].unique().numel() != 1:
            raise ValueError("group crosses outer folds")
    inner = payload.get("inner_fold_id")
    if not isinstance(inner, Mapping) or set(inner) != set(range(int(crossfit["outer_folds"]))):
        raise ValueError("inner fold mapping is incomplete")
    for outer_fold in range(int(crossfit["outer_folds"])):
        ids = torch.as_tensor(inner[outer_fold], dtype=torch.int64).cpu()
        if ids.shape != (n,) or not torch.all(ids[outer == outer_fold] == -1):
            raise ValueError("inner fold leaks into outer holdout")
        train = outer != outer_fold
        if sorted(ids[train].unique().tolist()) != list(range(int(crossfit["inner_folds"]))):
            raise ValueError("inner fold IDs are malformed")
        for group in sorted(set(groups[i] for i in torch.nonzero(train).flatten().tolist())):
            rows = torch.tensor([i for i, value in enumerate(groups) if value == group and bool(train[i])])
            if ids[rows].unique().numel() != 1:
                raise ValueError("group crosses inner folds")
    return n


def freeze_fold_artifact(
    parent: Mapping[str, Any], groups: Sequence[str], protocol: ScopeProtocol,
    destination: str | Path,
) -> dict[str, Any]:
    """Create folds once; an existing artifact is only loaded and verified."""
    destination = Path(destination)
    groups = [str(value) for value in groups]
    if destination.exists():
        payload = load_scope_cache(destination)
        _validate_fold_payload(payload, parent, groups, protocol)
        return payload
    labels = torch.as_tensor(parent["label"], dtype=torch.int64).cpu()
    n = int(labels.numel())
    crossfit = protocol.fixed["crossfit"]
    outer = stratified_group_folds(
        labels, groups, folds=int(crossfit["outer_folds"]), seed=int(crossfit["seed"]),
    )
    inner: dict[int, torch.Tensor] = {}
    for outer_fold in range(int(crossfit["outer_folds"])):
        train_rows = torch.nonzero(outer != outer_fold, as_tuple=False).flatten()
        train_groups = [groups[i] for i in train_rows.tolist()]
        train_inner = stratified_group_folds(
            labels[train_rows], train_groups, folds=int(crossfit["inner_folds"]),
            seed=int(crossfit["seed"]) + int(crossfit["inner_seed_offset"]) + outer_fold,
        )
        ids = torch.full((n,), -1, dtype=torch.int64)
        ids[train_rows] = train_inner
        inner[outer_fold] = ids
    try:
        import sklearn
    except Exception as error:  # pragma: no cover - required dependency
        raise RuntimeError("scikit-learn is required") from error
    payload = {
        "schema": "scope_nested_folds_v1",
        "conditional_parent": True,
        "formal_row_id": torch.arange(n, dtype=torch.int64),
        "paths": list(parent["paths"]),
        "labels": labels,
        "groups": groups,
        "formal_row_binding_sha256": parent["formal_row_binding_sha256"],
        "path_label_group_binding_sha256": semantic_sha256(
            list(zip(parent["paths"], labels.tolist(), groups))
        ),
        "outer_fold_id": outer,
        "inner_fold_id": inner,
        "crossfit": dict(crossfit),
        "parent_cache_sha256": parent["_cache_sha256"],
        "parent_semantic_sha256": semantic_sha256(parent),
        "group_artifact_sha256": protocol.assets.group_artifact_sha256,
        "protocol_sha256": sha256_file(protocol.config_path),
        "lineage": dict(parent["lineage"]),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    _validate_fold_payload(payload, parent, groups, protocol)
    atomic_save_scope_cache(payload, destination)
    stored = load_scope_cache(destination)
    _validate_fold_payload(stored, parent, groups, protocol)
    return stored


def _stable_sigmoid(value: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(value)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + torch.exp(-value[positive]))
    exponent = torch.exp(value[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _beta_gradient(
    beta: float, margin: torch.Tensor, evidence: torch.Tensor, target: torch.Tensor,
) -> float:
    score = margin + float(beta) * evidence
    data_gradient = (evidence * (_stable_sigmoid(score) - target)).sum().item()
    # J adds beta^2/(2n), hence the regularizer contributes beta/n.
    return float((data_gradient + float(beta)) / margin.numel())


def _beta_objective(
    beta: float, margin: torch.Tensor, evidence: torch.Tensor, target: torch.Tensor,
) -> float:
    score = margin + float(beta) * evidence
    loss = torch.nn.functional.softplus(score) - target * score
    n = margin.numel()
    return float(loss.sum().item() / n + float(beta) ** 2 / (2.0 * n))


def fit_shared_beta(
    margin: torch.Tensor,
    evidence: torch.Tensor,
    target: torch.Tensor,
    row_ids: torch.Tensor,
    solver: Mapping[str, Any],
) -> BetaFit:
    """Fit the preregistered non-negative, zero-intercept scalar in CPU FP64."""
    margin = torch.as_tensor(margin, dtype=torch.float64).flatten().cpu()
    evidence = torch.as_tensor(evidence, dtype=torch.float64).flatten().cpu()
    target = torch.as_tensor(target, dtype=torch.int64).flatten().cpu()
    row_ids = torch.as_tensor(row_ids, dtype=torch.int64).flatten().cpu()
    n = int(margin.numel())
    if n == 0 or evidence.numel() != n or target.numel() != n or row_ids.numel() != n:
        raise ValueError("beta inputs are misaligned or empty")
    if n > 1 and not torch.all(row_ids[1:] > row_ids[:-1]):
        raise ValueError("beta rows must be formal_row_id ascending")
    if not torch.isfinite(margin).all() or not torch.isfinite(evidence).all():
        raise ValueError("beta values must be finite")
    if not torch.all((target == 0) | (target == 1)) or torch.unique(target).numel() != 2:
        raise ValueError("beta target must contain both 0 and 1")
    initial = float(solver.get("initial_upper", 1.0))
    maximum = float(solver.get("maximum_upper", 2**20))
    maximum_iterations = int(solver.get("maximum_iterations", 100))
    tolerance = float(solver.get("interval_tolerance", 1.0e-12))
    if not 0.0 < initial <= maximum or maximum_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("beta solver constants are invalid")
    gradient_zero = _beta_gradient(0.0, margin, evidence, target)
    if gradient_zero >= 0.0:
        return BetaFit(
            0.0, 0.0, _beta_objective(0.0, margin, evidence, target),
            gradient_zero, 0, 0.0,
        )
    upper = initial
    upper_gradient = _beta_gradient(upper, margin, evidence, target)
    while upper_gradient < 0.0 and upper < maximum:
        upper = min(maximum, upper * 2.0)
        upper_gradient = _beta_gradient(upper, margin, evidence, target)
    if upper_gradient < 0.0:
        raise ValueError("beta derivative could not be bracketed")
    lower = 0.0
    iterations = 0
    while iterations < maximum_iterations and upper - lower > tolerance:
        midpoint = (lower + upper) / 2.0
        if _beta_gradient(midpoint, margin, evidence, target) < 0.0:
            lower = midpoint
        else:
            upper = midpoint
        iterations += 1
    beta = (lower + upper) / 2.0
    return BetaFit(
        beta, 0.0, _beta_objective(beta, margin, evidence, target),
        _beta_gradient(beta, margin, evidence, target), iterations, upper,
    )


def _wilson_lower(successes: int, total: int, z: float) -> float:
    if total <= 0:
        return float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    spread = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return (center - spread) / denominator


def _switch_statistics(
    switch: torch.Tensor, labels: torch.Tensor, candidates: torch.Tensor,
) -> tuple[int, int, int, int]:
    parent_correct = candidates[:, 0].eq(labels)
    challenger_correct = candidates[:, 1].eq(labels)
    corrections = int((switch & ~parent_correct & challenger_correct).sum())
    regressions = int((switch & parent_correct & ~challenger_correct).sum())
    neutral = int((switch & (parent_correct == challenger_correct)).sum())
    return corrections, regressions, neutral, int(switch.sum())


def select_oof_threshold(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    labels: torch.Tensor,
    candidates: torch.Tensor,
    policy: Mapping[str, Any],
) -> OOFThreshold:
    """Select all/finite/no-switch by net gain, then fewer switches/higher cut."""
    scores = torch.as_tensor(scores, dtype=torch.float64).flatten().cpu()
    eligible = torch.as_tensor(eligible, dtype=torch.bool).flatten().cpu()
    labels = torch.as_tensor(labels, dtype=torch.int64).flatten().cpu()
    candidates = torch.as_tensor(candidates, dtype=torch.int64).cpu()
    n = int(scores.numel())
    if eligible.numel() != n or labels.numel() != n or candidates.shape != (n, 2):
        raise ValueError("threshold inputs are misaligned")
    if not torch.isfinite(scores).all():
        raise ValueError("threshold scores must be finite")
    eligible_scores = scores[eligible]
    score_hash = tensor_sha256(eligible_scores)
    n_eligible = int(eligible.sum())
    empty = torch.zeros(n, dtype=torch.bool)
    if n_eligible == 0:
        return OOFThreshold(
            "no_switch", None, 0, 0, None, "no_eligible", score_hash, 0, empty,
        )
    minimum_precision = float(policy["minimum_accuracy_changing_precision"])
    minimum_wilson = float(policy["minimum_wilson_lower"])
    z = float(policy["wilson_z"])
    choices: list[tuple[str, float | None, torch.Tensor]] = [
        ("all_switch", None, eligible.clone()),
    ]
    unique = torch.unique(eligible_scores, sorted=True)
    for lower, upper in zip(unique[:-1], unique[1:]):
        gamma = float(lower / 2.0 + upper / 2.0)
        choices.append(("finite", gamma, eligible & (scores > gamma)))
    qualified: list[tuple[int, int, float, str, float | None, torch.Tensor]] = []
    for mode, gamma, switch in choices:
        corrections, regressions, _, switched = _switch_statistics(switch, labels, candidates)
        changed = corrections + regressions
        precision = corrections / changed if changed else 0.0
        wilson = _wilson_lower(corrections, changed, z)
        if changed and precision >= minimum_precision and wilson >= minimum_wilson:
            cut = float("-inf") if gamma is None else gamma
            qualified.append((corrections - regressions, -switched, cut, mode, gamma, switch))
    if not qualified:
        return OOFThreshold(
            "no_switch", None, 0, n_eligible, 0.0, "no_qualified_candidate",
            score_hash, len(choices), empty,
        )
    _, _, _, mode, gamma, switch = max(qualified, key=lambda item: item[:3])
    switched = int(switch.sum())
    return OOFThreshold(
        mode, gamma, switched, n_eligible, switched / n_eligible, None,
        score_hash, len(choices), switch,
    )


def map_deployed_threshold(
    oof: OOFThreshold, scores: torch.Tensor, eligible: torch.Tensor,
) -> DeployedThreshold:
    """Map frozen inner-OOF k/n to refit scores without holdout information."""
    scores = torch.as_tensor(scores, dtype=torch.float64).flatten().cpu()
    eligible = torch.as_tensor(eligible, dtype=torch.bool).flatten().cpu()
    if scores.numel() != eligible.numel() or not torch.isfinite(scores).all():
        raise ValueError("refit threshold inputs are invalid")
    eligible_scores = scores[eligible]
    score_hash = tensor_sha256(eligible_scores)
    n_refit = int(eligible.sum())
    if oof.mode == "all_switch":
        return DeployedThreshold(
            "all_switch", None, n_refit, n_refit,
            1.0 if n_refit else None, None, score_hash,
        )
    if oof.mode == "no_switch":
        return DeployedThreshold(
            "no_switch", None, 0, n_refit,
            0.0 if n_refit else None, oof.no_switch_reason, score_hash,
        )
    if oof.mode != "finite" or oof.k_oof <= 0 or oof.n_oof <= 0 or n_refit == 0:
        return DeployedThreshold(
            "no_switch", None, 0, n_refit, 0.0 if n_refit else None,
            "finite_mapping_failed", score_hash,
        )
    target_numerator = int(oof.k_oof) * n_refit
    target_denominator = int(oof.n_oof)
    unique = torch.unique(eligible_scores, sorted=True)
    options: list[tuple[int, int, float]] = []
    for lower, upper in zip(unique[:-1], unique[1:]):
        gamma = float(lower / 2.0 + upper / 2.0)
        count = int((eligible & (scores > gamma)).sum())
        distance = abs(count * target_denominator - target_numerator)
        # Ties: fewer switches, then higher cut.
        options.append((distance, count, -gamma))
    if not options:
        return DeployedThreshold(
            "no_switch", None, 0, n_refit, 0.0, "finite_mapping_failed", score_hash,
        )
    _, count, negative_gamma = min(options)
    gamma = -negative_gamma
    if count <= 0:
        return DeployedThreshold(
            "no_switch", None, 0, n_refit, 0.0, "finite_mapping_failed", score_hash,
        )
    return DeployedThreshold(
        "finite", gamma, count, n_refit, count / n_refit, None, score_hash,
    )


def _no_switch_oof(length: int, eligible_count: int, reason: str) -> OOFThreshold:
    empty_scores = torch.empty(0, dtype=torch.float64)
    return OOFThreshold(
        "no_switch", None, 0, int(eligible_count), 0.0, reason,
        tensor_sha256(empty_scores), 0, torch.zeros(int(length), dtype=torch.bool),
    )


def _no_switch_deployed(eligible_count: int, reason: str) -> DeployedThreshold:
    return DeployedThreshold(
        "no_switch", None, 0, int(eligible_count),
        0.0 if eligible_count else None, reason, tensor_sha256(torch.empty(0, dtype=torch.float64)),
    )


def _method_inputs(
    parent: Mapping[str, Any], evidence: Mapping[str, Any], method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if method not in METHODS:
        raise ValueError(f"unsupported SCOPE method: {method}")
    n = len(parent["paths"])
    corrupt = torch.as_tensor(parent["corrupt"], dtype=torch.bool).cpu()
    branches = torch.as_tensor(parent["constituent_top1"], dtype=torch.int64).cpu()
    if corrupt.shape != (n,) or branches.shape != (n, 4):
        raise ValueError("parent method inputs are malformed")
    conflict = torch.any(branches != branches[:, :1], dim=1) & ~corrupt
    zero = torch.zeros(n, dtype=torch.float64)
    if method == "parent":
        return zero, torch.zeros(n, dtype=torch.bool)
    if method == "margin_only":
        return zero, conflict
    if method == "gate_only":
        family = evidence["scope"]
        return zero, torch.as_tensor(family["eligibility"], dtype=torch.bool).cpu()
    family_name = {"pace": "pace", "no_topology": "no_topology", "scope": "scope"}[method]
    family = evidence[family_name]
    values = torch.as_tensor(family["aggregate"], dtype=torch.float64).cpu()
    eligible = torch.as_tensor(family["eligibility"], dtype=torch.bool).cpu()
    if values.shape != (n,) or eligible.shape != (n,):
        raise ValueError(f"{family_name} evidence shapes are malformed")
    return values, eligible


def run_conditional_nested_oof(
    parent: Mapping[str, Any],
    evidence: Mapping[str, Any],
    folds: Mapping[str, Any],
    method: str,
    solver: Mapping[str, Any],
    threshold_policy: Mapping[str, Any],
) -> MethodOOFResult:
    """Run one preregistered method with inner calibration and outer evaluation."""
    candidates = torch.as_tensor(parent["candidate_indices"], dtype=torch.int64).cpu()
    labels = torch.as_tensor(parent["label"], dtype=torch.int64).cpu()
    margin = torch.as_tensor(parent["parent_margin"], dtype=torch.float64).cpu()
    row_ids = torch.as_tensor(parent["formal_row_id"], dtype=torch.int64).cpu()
    outer = torch.as_tensor(folds["outer_fold_id"], dtype=torch.int64).cpu()
    inner_by_outer = {
        int(key): torch.as_tensor(value, dtype=torch.int64).cpu()
        for key, value in folds["inner_fold_id"].items()
    }
    n = int(labels.numel())
    if candidates.shape != (n, 2) or margin.shape != (n,) or row_ids.shape != (n,) or outer.shape != (n,):
        raise ValueError("OOF row bindings are malformed")
    if not torch.equal(row_ids, torch.arange(n, dtype=torch.int64)):
        raise ValueError("OOF formal rows are not canonical")
    evidence_values, eligibility = _method_inputs(parent, evidence, method)
    predictions = candidates[:, 0].clone()
    switches = torch.zeros(n, dtype=torch.bool)
    if method == "parent":
        return MethodOOFResult(
            method, predictions, switches, outer, (), (), (), (),
        )
    learned = method in {"pace", "no_topology", "scope"}
    pair_mask, runner_target = candidate_pair_training_mask(candidates, labels)
    betas: list[float | None] = []
    thresholds: list[OOFThreshold] = []
    mapped_thresholds: list[DeployedThreshold] = []
    failures: list[str | None] = []
    fold_values = sorted(int(value) for value in outer.unique().tolist())
    for outer_fold in fold_values:
        outer_train = outer != outer_fold
        outer_holdout = outer == outer_fold
        inner_ids = inner_by_outer.get(outer_fold)
        if inner_ids is None or inner_ids.shape != (n,) or not torch.all(inner_ids[outer_holdout] == -1):
            raise ValueError("inner fold artifact is missing or leaks")
        train_rows = torch.nonzero(outer_train, as_tuple=False).flatten()
        inner_scores = torch.full((n,), float("nan"), dtype=torch.float64)
        inner_failed: str | None = None
        for inner_fold in sorted(int(value) for value in inner_ids[outer_train].unique().tolist()):
            inner_train = outer_train & (inner_ids != inner_fold)
            inner_holdout = outer_train & (inner_ids == inner_fold)
            beta_inner = 0.0
            if learned:
                fit_rows = inner_train & pair_mask
                try:
                    beta_inner = fit_shared_beta(
                        margin[fit_rows], evidence_values[fit_rows],
                        runner_target[fit_rows].to(torch.int64), row_ids[fit_rows], solver,
                    ).beta
                except ValueError as error:
                    inner_failed = f"inner_fit_failed:{error}"
                    break
            inner_scores[inner_holdout] = (
                margin[inner_holdout] + beta_inner * evidence_values[inner_holdout]
            )
        if inner_failed is not None or not torch.isfinite(inner_scores[outer_train]).all():
            reason = inner_failed or "inner_oof_incomplete"
            thresholds.append(_no_switch_oof(len(train_rows), int(eligibility[outer_train].sum()), reason))
            mapped_thresholds.append(_no_switch_deployed(int(eligibility[outer_train].sum()), reason))
            betas.append(None)
            failures.append(reason)
            continue
        threshold = select_oof_threshold(
            inner_scores[outer_train], eligibility[outer_train], labels[outer_train],
            candidates[outer_train], threshold_policy,
        )
        beta_outer = 0.0
        if learned:
            fit_rows = outer_train & pair_mask
            try:
                beta_outer = fit_shared_beta(
                    margin[fit_rows], evidence_values[fit_rows],
                    runner_target[fit_rows].to(torch.int64), row_ids[fit_rows], solver,
                ).beta
            except ValueError as error:
                reason = f"outer_refit_failed:{error}"
                thresholds.append(threshold)
                mapped_thresholds.append(_no_switch_deployed(int(eligibility[outer_train].sum()), reason))
                betas.append(None)
                failures.append(reason)
                continue
        refit_scores = margin + beta_outer * evidence_values
        mapped = map_deployed_threshold(threshold, refit_scores[outer_train], eligibility[outer_train])
        if mapped.mode == "all_switch":
            selected = eligibility[outer_holdout]
        elif mapped.mode == "finite" and mapped.gamma is not None:
            selected = eligibility[outer_holdout] & (refit_scores[outer_holdout] > mapped.gamma)
        else:
            selected = torch.zeros(int(outer_holdout.sum()), dtype=torch.bool)
        holdout_rows = torch.nonzero(outer_holdout, as_tuple=False).flatten()
        selected_rows = holdout_rows[selected]
        switches[selected_rows] = True
        predictions[selected_rows] = candidates[selected_rows, 1]
        thresholds.append(threshold)
        mapped_thresholds.append(mapped)
        betas.append(beta_outer)
        failures.append(None)
    return MethodOOFResult(
        method, predictions, switches, outer, tuple(betas), tuple(thresholds),
        tuple(mapped_thresholds), tuple(failures),
    )


def method_metrics(
    result: MethodOOFResult, parent: Mapping[str, Any], clean_threshold: float = 0.70,
) -> dict[str, Any]:
    """Compute paired all-row and clean-core metrics for one OOF method."""
    labels = torch.as_tensor(parent["label"], dtype=torch.int64).cpu()
    candidates = torch.as_tensor(parent["candidate_indices"], dtype=torch.int64).cpu()
    clean = torch.as_tensor(parent["clean_probability"], dtype=torch.float64).cpu() >= float(clean_threshold)
    parent_prediction = candidates[:, 0]
    parent_correct = parent_prediction.eq(labels)
    method_correct = result.predictions.eq(labels)
    corrections = result.switches & ~parent_correct & method_correct
    regressions = result.switches & parent_correct & ~method_correct
    neutral = result.switches & (parent_correct == method_correct)
    fold_rows: list[dict[str, int]] = []
    for fold in sorted(int(value) for value in result.outer_fold_id.unique().tolist()):
        rows = result.outer_fold_id == fold
        fold_rows.append({
            "outer_fold": fold,
            "rows": int(rows.sum()),
            "parent_correct": int(parent_correct[rows].sum()),
            "method_correct": int(method_correct[rows].sum()),
            "delta_correct": int(method_correct[rows].sum() - parent_correct[rows].sum()),
            "switches": int(result.switches[rows].sum()),
        })
    class_accuracies = []
    for label in torch.unique(labels, sorted=True):
        rows = labels == label
        class_accuracies.append(float(method_correct[rows].double().mean()))
    changed = int(corrections.sum() + regressions.sum())
    return {
        "method": result.method,
        "rows": int(labels.numel()),
        "parent_correct": int(parent_correct.sum()),
        "correct": int(method_correct.sum()),
        "accuracy": float(method_correct.double().mean()),
        "delta_accuracy": float((method_correct.double() - parent_correct.double()).mean()),
        "clean_rows": int(clean.sum()),
        "clean_parent_correct": int(parent_correct[clean].sum()),
        "clean_correct": int(method_correct[clean].sum()),
        "clean_accuracy": float(method_correct[clean].double().mean()),
        "clean_delta_accuracy": float((method_correct[clean].double() - parent_correct[clean].double()).mean()),
        "corrections": int(corrections.sum()),
        "regressions": int(regressions.sum()),
        "neutral_switches": int(neutral.sum()),
        "switches": int(result.switches.sum()),
        "switch_precision": float(corrections.sum()) / changed if changed else None,
        "oracle_availability": int(candidates[:, 1].eq(labels).sum()),
        "macro_accuracy": float(np.mean(class_accuracies)),
        "folds": fold_rows,
    }


def fit_full_scope_deployment(
    parent: Mapping[str, Any], evidence: Mapping[str, Any], groups: Sequence[str],
    protocol: ScopeProtocol,
) -> dict[str, Any]:
    """Fit the frozen full-validation 3-fold calibration after promotion only."""
    labels = torch.as_tensor(parent["label"], dtype=torch.int64).cpu()
    candidates = torch.as_tensor(parent["candidate_indices"], dtype=torch.int64).cpu()
    margin = torch.as_tensor(parent["parent_margin"], dtype=torch.float64).cpu()
    rows = torch.as_tensor(parent["formal_row_id"], dtype=torch.int64).cpu()
    evidence_values, eligible = _method_inputs(parent, evidence, "scope")
    pair_mask, target = candidate_pair_training_mask(candidates, labels)
    crossfit = protocol.fixed["crossfit"]
    full_fold = stratified_group_folds(
        labels, groups, folds=int(crossfit["inner_folds"]), seed=int(crossfit["seed"]),
    )
    oof_scores = torch.full_like(margin, float("nan"))
    beta_by_fold: list[float] = []
    for fold in range(int(crossfit["inner_folds"])):
        train = full_fold != fold
        holdout = full_fold == fold
        fit_rows = train & pair_mask
        beta = fit_shared_beta(
            margin[fit_rows], evidence_values[fit_rows], target[fit_rows].to(torch.int64),
            rows[fit_rows], protocol.fixed["beta_solver"],
        ).beta
        beta_by_fold.append(beta)
        oof_scores[holdout] = margin[holdout] + beta * evidence_values[holdout]
    if not torch.isfinite(oof_scores).all():
        raise ValueError("full validation OOF scores are incomplete")
    threshold = select_oof_threshold(
        oof_scores, eligible, labels, candidates, protocol.fixed["threshold"],
    )
    final_beta = fit_shared_beta(
        margin[pair_mask], evidence_values[pair_mask], target[pair_mask].to(torch.int64),
        rows[pair_mask], protocol.fixed["beta_solver"],
    ).beta
    refit_scores = margin + final_beta * evidence_values
    mapped = map_deployed_threshold(threshold, refit_scores, eligible)
    return {
        "beta": final_beta,
        "intercept": 0.0,
        "beta_by_full_oof_fold": beta_by_fold,
        "full_oof_fold_id": full_fold,
        "threshold": threshold,
        "mapped_threshold": mapped,
        "eligible_count": int(eligible.sum()),
        "pair_training_count": int(pair_mask.sum()),
        "oof_score_sha256": tensor_sha256(oof_scores),
        "refit_score_sha256": tensor_sha256(refit_scores),
    }


def validate_fold_artifact(
    payload: Mapping[str, Any], parent: Mapping[str, Any], groups: Sequence[str],
    protocol: ScopeProtocol,
) -> int:
    """Public fail-closed validation used by the evaluator."""
    return _validate_fold_payload(payload, parent, groups, protocol)


def cluster_bootstrap_delta(
    parent_correct: torch.Tensor,
    method_correct: torch.Tensor,
    groups: Sequence[str],
    labels: torch.Tensor,
    outer_fold_id: torch.Tensor,
    *,
    draws: int,
    seed: int,
    quantile_method: str,
) -> Interval:
    """Bootstrap exact groups within (outer fold, group-majority-label) strata."""
    parent = torch.as_tensor(parent_correct, dtype=torch.bool).flatten().cpu()
    method = torch.as_tensor(method_correct, dtype=torch.bool).flatten().cpu()
    labels = torch.as_tensor(labels, dtype=torch.int64).flatten().cpu()
    outer = torch.as_tensor(outer_fold_id, dtype=torch.int64).flatten().cpu()
    groups = [str(value) for value in groups]
    n = int(parent.numel())
    if n == 0 or method.numel() != n or labels.numel() != n or outer.numel() != n or len(groups) != n:
        raise ValueError("bootstrap inputs are misaligned")
    rows_by_group: dict[str, np.ndarray] = {}
    for group in sorted(set(groups)):
        rows = np.asarray([i for i, value in enumerate(groups) if value == group], dtype=np.int64)
        if torch.unique(outer[torch.as_tensor(rows)]).numel() != 1:
            raise ValueError("bootstrap group crosses outer folds")
        rows_by_group[group] = rows
    strata: dict[tuple[int, int], list[str]] = {}
    for group, rows in rows_by_group.items():
        row_tensor = torch.as_tensor(rows, dtype=torch.int64)
        majority = int(labels[row_tensor].bincount(minlength=500).argmax())
        key = (int(outer[row_tensor[0]]), majority)
        strata.setdefault(key, []).append(group)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    delta_rows = (method.to(torch.float64) - parent.to(torch.float64)).numpy()
    ordered_groups = sorted(rows_by_group)
    group_index = {name: index for index, name in enumerate(ordered_groups)}
    group_delta = np.asarray(
        [delta_rows[rows_by_group[name]].sum() for name in ordered_groups], dtype=np.float64,
    )
    group_size = np.asarray(
        [rows_by_group[name].size for name in ordered_groups], dtype=np.int64,
    )
    stratum_indices = [
        np.asarray([group_index[name] for name in sorted(strata[key])], dtype=np.int64)
        for key in sorted(strata)
    ]
    sampled = np.empty(int(draws), dtype=np.float64)
    # Chunked vectorization preserves whole-group resampling without allocating
    # the full draws x groups index matrix.
    chunk_size = 256
    for start in range(0, int(draws), chunk_size):
        width = min(chunk_size, int(draws) - start)
        numerator = np.zeros(width, dtype=np.float64)
        denominator = np.zeros(width, dtype=np.int64)
        for indices in stratum_indices:
            selected_local = rng.integers(0, len(indices), size=(width, len(indices)))
            selected = indices[selected_local]
            numerator += group_delta[selected].sum(axis=1)
            denominator += group_size[selected].sum(axis=1)
        sampled[start:start + width] = numerator / denominator
    lower, upper = np.quantile(sampled, [0.025, 0.975], method=str(quantile_method))
    return Interval(
        float(delta_rows.mean()), float(lower), float(upper), int(draws), int(seed),
    )


def promotion_gate(metrics: Mapping[str, Any]) -> PromotionDecision:
    """Evaluate the eight preregistered promotion requirements as strict AND."""
    raw_total = int(metrics["raw_total"])
    clean_total = int(metrics["clean_total"])
    if raw_total <= 0 or clean_total <= 0:
        raise ValueError("promotion denominators must be positive")
    raw_delta_pp = (
        int(metrics["raw_scope_correct"]) - int(metrics["raw_parent_correct"])
    ) * 100.0 / raw_total
    clean_delta_pp = (
        int(metrics["clean_scope_correct"]) - int(metrics["clean_parent_correct"])
    ) * 100.0 / clean_total
    fold_deltas = [int(value) for value in metrics["fold_deltas"]]
    gates = {
        "raw_delta_at_least_0_20pp": raw_delta_pp >= 0.2,
        "clean_delta_at_least_0_20pp": clean_delta_pp >= 0.2,
        "net_correct_at_least_21": int(metrics["corrections"]) - int(metrics["regressions"]) >= 21,
        "four_of_five_outer_nonnegative": len(fold_deltas) == 5 and sum(value >= 0 for value in fold_deltas) >= 4,
        "bootstrap_lower_strictly_positive": float(metrics["bootstrap_lower"]) > 0.0,
        "strictly_better_raw_than_pace_and_no_topology": int(metrics["raw_scope_correct"]) > int(metrics["pace_raw_correct"]) and int(metrics["raw_scope_correct"]) > int(metrics["no_topology_raw_correct"]),
        "strictly_better_clean_than_pace_and_no_topology": int(metrics["clean_scope_correct"]) > int(metrics["pace_clean_correct"]) and int(metrics["clean_scope_correct"]) > int(metrics["no_topology_clean_correct"]),
        "all_audits_passed": bool(metrics["audits_passed"]),
    }
    return PromotionDecision(all(gates.values()), gates)
