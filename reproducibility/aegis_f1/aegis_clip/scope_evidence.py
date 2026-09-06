"""SCOPE-K2 canonical pair residuals, fixed topology evidence, and gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from aegis_clip.scope_protocol import EVIDENCE_VIEW_WEIGHTS, four_neighbor_edges


@dataclass(frozen=True)
class ScopeParentAudit:
    peft_mode: str
    classifier_mode: str
    feature_adapter: str
    num_classes: int
    feature_dim: int
    has_local_feature_adapter: bool
    has_part_token_adapter: bool


@dataclass(frozen=True)
class ClassifierSpaceBatchAudit:
    base_max_abs_error: float
    dual_max_abs_error: float | None


@dataclass(frozen=True)
class PairResidualGrid:
    residual: torch.Tensor
    weight_norm: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class FamilyEvidenceSummary:
    per_view: torch.Tensor
    weights: torch.Tensor
    total: torch.Tensor
    original: torch.Tensor
    flipped: torch.Tensor
    leave_one_scale: torch.Tensor
    positive_view_count: torch.Tensor


@dataclass(frozen=True)
class FamilyEligibilityAudit:
    eligible: torch.Tensor
    branch_conflict: torch.Tensor
    non_corrupt: torch.Tensor
    weight_norm_valid: torch.Tensor
    support_count_positive: torch.Tensor
    total_positive: torch.Tensor
    orientation_positive: torch.Tensor
    leave_one_scale_positive: torch.Tensor


def validate_scope_parent_model(
    model: nn.Module, checkpoint_payload: Mapping[str, Any]
) -> ScopeParentAudit:
    """Fail closed unless the model is the frozen FULLFT dual-adapter parent."""
    peft_mode = str(getattr(model, "peft_mode", ""))
    if peft_mode != "full_finetune":
        raise ValueError("SCOPE requires peft_mode=full_finetune")
    classifier_mode = str(getattr(model, "classifier_mode", "linear"))
    if classifier_mode != "linear":
        raise ValueError("SCOPE requires classifier_mode=linear")
    feature_adapter = getattr(model, "feature_adapter", None)
    if not isinstance(feature_adapter, nn.Identity):
        raise ValueError("SCOPE requires feature_adapter=Identity")
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Linear):
        raise ValueError("SCOPE requires one torch.nn.Linear classifier")
    checkpoint_mode = str(
        checkpoint_payload.get("config", {}).get("model", {}).get("peft_mode", "")
    )
    if checkpoint_mode != peft_mode:
        raise ValueError("SCOPE checkpoint peft_mode disagrees with model")
    local_payload = checkpoint_payload.get("local_feature_adapter")
    part_payload = checkpoint_payload.get("part_token_adapter")
    if not _complete_adapter_payload(local_payload):
        raise ValueError("SCOPE checkpoint local feature adapter is missing or incomplete")
    if not _complete_adapter_payload(part_payload):
        raise ValueError("SCOPE checkpoint part token adapter is missing or incomplete")
    if not torch.isfinite(classifier.weight.detach().float()).all():
        raise ValueError("SCOPE classifier weight contains non-finite values")
    if classifier.bias is not None and not torch.isfinite(classifier.bias.detach().float()).all():
        raise ValueError("SCOPE classifier bias contains non-finite values")
    return ScopeParentAudit(
        peft_mode=peft_mode,
        classifier_mode=classifier_mode,
        feature_adapter=type(feature_adapter).__name__,
        num_classes=int(classifier.weight.shape[0]),
        feature_dim=int(classifier.weight.shape[1]),
        has_local_feature_adapter=True,
        has_part_token_adapter=True,
    )


def validate_classifier_space_batch(
    model: nn.Module,
    base_logits: torch.Tensor,
    base_cls: torch.Tensor,
    *,
    dual_logits: torch.Tensor | None = None,
    dual_cls: torch.Tensor | None = None,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> ClassifierSpaceBatchAudit:
    """Audit the base linear identity and the anchored dual residual identity."""
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Linear):
        raise ValueError("classifier-space audit requires nn.Linear")
    if (dual_logits is None) != (dual_cls is None):
        raise ValueError("dual logits and features must be provided together")
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in (atol, rtol)):
        raise ValueError("classifier-space tolerances must be finite and non-negative")
    base_values = torch.as_tensor(base_logits).detach().float()
    cls_values = torch.as_tensor(base_cls).detach().float()
    if base_values.ndim != 2 or cls_values.ndim != 2 or base_values.shape[0] == 0:
        raise ValueError("base classifier-space values must be non-empty matrices")
    if (
        base_values.shape[0] != cls_values.shape[0]
        or base_values.shape[1] != classifier.weight.shape[0]
        or cls_values.shape[1] != classifier.weight.shape[1]
    ):
        raise ValueError("base classifier-space shapes disagree")
    if not torch.isfinite(base_values).all() or not torch.isfinite(cls_values).all():
        raise ValueError("base classifier-space values contain non-finite data")
    weight = classifier.weight.detach().float()
    bias = None if classifier.bias is None else classifier.bias.detach().float()
    rebuilt = F.linear(cls_values, weight, bias)
    base_error = float((base_values - rebuilt).abs().max().item())
    if not torch.allclose(base_values, rebuilt, atol=float(atol), rtol=float(rtol)):
        raise ValueError(f"base classifier-space logits mismatch: max_abs_error={base_error:.9g}")
    dual_error: float | None = None
    if dual_logits is not None and dual_cls is not None:
        dual_values = torch.as_tensor(dual_logits).detach().float()
        dual_features = torch.as_tensor(dual_cls).detach().float()
        if dual_values.shape != base_values.shape or dual_features.shape != cls_values.shape:
            raise ValueError("dual classifier-space shapes disagree")
        if not torch.isfinite(dual_values).all() or not torch.isfinite(dual_features).all():
            raise ValueError("dual classifier-space values contain non-finite data")
        expected = base_values + F.linear(dual_features - cls_values, weight, None)
        dual_error = float((dual_values - expected).abs().max().item())
        if not torch.allclose(dual_values, expected, atol=float(atol), rtol=float(rtol)):
            raise ValueError(f"dual classifier-space logits mismatch: max_abs_error={dual_error:.9g}")
    return ClassifierSpaceBatchAudit(
        base_max_abs_error=base_error, dual_max_abs_error=dual_error
    )


def pairwise_residual_grid(
    cls_features: torch.Tensor,
    patch_features: torch.Tensor,
    classifier_weight: torch.Tensor,
    candidates: torch.Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> PairResidualGrid:
    """Compute a canonical unordered direction, then orient it as (a,b)."""
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("weight norm epsilon must be finite and positive")
    cls = torch.as_tensor(cls_features).detach().to(device="cpu", dtype=torch.float64)
    patches = torch.as_tensor(patch_features).detach().to(device="cpu", dtype=torch.float64)
    weight = torch.as_tensor(classifier_weight).detach().to(device="cpu", dtype=torch.float64)
    pair = torch.as_tensor(candidates).detach().to(device="cpu")
    if cls.ndim != 2 or patches.ndim != 3 or patches.shape[1] != 49:
        raise ValueError("SCOPE requires CLS [N,D] and exactly 49 patch tokens")
    if patches.shape[0] != cls.shape[0] or patches.shape[2] != cls.shape[1]:
        raise ValueError("CLS and patch feature shapes disagree")
    if weight.ndim != 2 or weight.shape[1] != cls.shape[1]:
        raise ValueError("classifier weight does not share the feature space")
    if pair.ndim != 2 or pair.shape != (cls.shape[0], 2):
        raise ValueError("candidates must have shape [N,2]")
    if pair.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError("candidates must use an integer dtype")
    pair = pair.to(torch.int64)
    if pair.numel() and (int(pair.min()) < 0 or int(pair.max()) >= weight.shape[0]):
        raise ValueError("candidate class index is out of range")
    if torch.any(pair[:, 0] == pair[:, 1]):
        raise ValueError("candidate classes must be distinct")
    for name, tensor in (("CLS", cls), ("patch", patches), ("classifier", weight)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} values contain non-finite data")

    first, second = pair[:, 0], pair[:, 1]
    lower, upper = torch.minimum(first, second), torch.maximum(first, second)
    canonical_difference = weight[upper] - weight[lower]
    norm = torch.linalg.vector_norm(canonical_difference, dim=1)
    valid = torch.isfinite(norm) & (norm >= float(epsilon))
    safe_norm = torch.where(valid, norm, torch.ones_like(norm))
    canonical_direction = canonical_difference / safe_norm[:, None]
    canonical_residual = torch.einsum(
        "npd,nd->np", patches - cls[:, None, :], canonical_direction
    )
    sign = torch.where(second == upper, 1.0, -1.0).to(torch.float64)
    oriented = canonical_residual * sign[:, None]
    oriented = torch.where(valid[:, None], oriented, torch.zeros_like(oriented))
    if not torch.isfinite(oriented).all():
        raise ValueError("SCOPE residual contains non-finite values")
    return PairResidualGrid(residual=oriented, weight_norm=norm, valid=valid)


def scope_energy(values: torch.Tensor) -> torch.Tensor:
    """Evaluate fixed node plus four-neighbor positive-part energy H."""
    grid = _validated_grid(values)
    positive = grid.clamp_min(0.0)
    edges = four_neighbor_edges()
    first = torch.tensor([edge[0] for edge in edges], dtype=torch.int64)
    second = torch.tensor([edge[1] for edge in edges], dtype=torch.int64)
    node_term = positive.sum(dim=1) / 49.0
    edge_term = torch.minimum(positive[:, first], positive[:, second]).sum(dim=1) / 84.0
    return node_term + edge_term


def scope_view_evidence(residual: torch.Tensor) -> torch.Tensor:
    grid = _validated_grid(residual)
    return scope_energy(grid) - scope_energy(-grid)


def no_topology_view_evidence(residual: torch.Tensor) -> torch.Tensor:
    grid = _validated_grid(residual)
    return grid.mean(dim=1)


def matched_pace_evidence(residual: torch.Tensor, *, tail_size: int = 7) -> torch.Tensor:
    grid = _validated_grid(residual)
    if not 1 <= int(tail_size) <= 49:
        raise ValueError("PACE tail size is out of range")
    ordered = torch.sort(grid, dim=1, descending=True, stable=True).values
    count = int(tail_size)
    # Pair each high tail value with the mirrored low tail value before the
    # reduction.  For -r the paired vector is bitwise -paired(r), avoiding the
    # reduction-order ULP drift of two independently computed means.
    paired_tails = ordered[:, :count] + ordered[:, -count:].flip(dims=(1,))
    return 0.5 * paired_tails.mean(dim=1)


def aggregate_family_evidence(per_view: torch.Tensor) -> FamilyEvidenceSummary:
    values = torch.as_tensor(per_view).detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("SCOPE evidence must contain exactly six views")
    if not torch.isfinite(values).all():
        raise ValueError("SCOPE evidence contains non-finite values")
    weights = torch.tensor(EVIDENCE_VIEW_WEIGHTS, dtype=torch.float64)
    total = (values * weights).sum(dim=1)
    orientation_weights = weights[:3] / weights[:3].sum()
    original = (values[:, :3] * orientation_weights).sum(dim=1)
    flipped = (values[:, 3:] * orientation_weights).sum(dim=1)
    leave: list[torch.Tensor] = []
    for scale in range(3):
        keep = torch.ones(6, dtype=torch.bool)
        keep[scale] = False
        keep[scale + 3] = False
        retained = weights[keep]
        leave.append((values[:, keep] * (retained / retained.sum())).sum(dim=1))
    return FamilyEvidenceSummary(
        per_view=values, weights=weights, total=total,
        original=original, flipped=flipped,
        leave_one_scale=torch.stack(leave, dim=1),
        positive_view_count=(values > 0.0).sum(dim=1).to(torch.int64),
    )


def family_eligibility(
    *,
    constituent_top1: torch.Tensor,
    parent_corrupt: torch.Tensor,
    evidence_corrupt: torch.Tensor,
    weight_norm_valid: torch.Tensor,
    summary: FamilyEvidenceSummary,
) -> FamilyEligibilityAudit:
    count = int(summary.per_view.shape[0])
    branches = torch.as_tensor(constituent_top1, dtype=torch.int64).cpu()
    parent_bad = torch.as_tensor(parent_corrupt, dtype=torch.bool).flatten().cpu()
    evidence_bad = torch.as_tensor(evidence_corrupt, dtype=torch.bool).flatten().cpu()
    norm_valid = torch.as_tensor(weight_norm_valid, dtype=torch.bool).flatten().cpu()
    if branches.shape != (count, 4):
        raise ValueError("constituent top1 must have shape [N,4]")
    if any(value.numel() != count for value in (parent_bad, evidence_bad, norm_valid)):
        raise ValueError("eligibility inputs are row-misaligned")
    if not torch.equal(parent_bad, evidence_bad):
        raise ValueError("parent/evidence corrupt states disagree")
    conflict = torch.any(branches != branches[:, :1], dim=1)
    non_corrupt = ~parent_bad
    support = summary.positive_view_count >= 4
    total = summary.total > 0.0
    orientation = (summary.original > 0.0) & (summary.flipped > 0.0)
    leave = torch.all(summary.leave_one_scale > 0.0, dim=1)
    eligible = conflict & non_corrupt & norm_valid & support & total & orientation & leave
    return FamilyEligibilityAudit(
        eligible=eligible, branch_conflict=conflict, non_corrupt=non_corrupt,
        weight_norm_valid=norm_valid, support_count_positive=support,
        total_positive=total, orientation_positive=orientation,
        leave_one_scale_positive=leave,
    )


def apply_scope_decision(
    candidates: torch.Tensor,
    parent_margin: torch.Tensor,
    evidence: torch.Tensor,
    eligible: torch.Tensor,
    *,
    beta: float,
    threshold: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair = torch.as_tensor(candidates, dtype=torch.int64).cpu()
    margin = torch.as_tensor(parent_margin, dtype=torch.float64).flatten().cpu()
    values = torch.as_tensor(evidence, dtype=torch.float64).flatten().cpu()
    gate = torch.as_tensor(eligible, dtype=torch.bool).flatten().cpu()
    if pair.ndim != 2 or pair.shape[1] != 2 or any(
        value.numel() != pair.shape[0] for value in (margin, values, gate)
    ):
        raise ValueError("decision inputs are row-misaligned")
    if not math.isfinite(float(beta)) or float(beta) < 0.0:
        raise ValueError("beta must be finite and non-negative")
    if not torch.isfinite(margin).all() or not torch.isfinite(values).all():
        raise ValueError("decision scores contain non-finite values")
    eta = margin + float(beta) * values
    switch = gate if threshold is None else gate & (eta > float(threshold))
    predictions = pair[:, 0].clone()
    predictions[switch] = pair[switch, 1]
    return predictions, eta


def _validated_grid(value: torch.Tensor) -> torch.Tensor:
    grid = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
    if grid.ndim != 2 or grid.shape[1] != 49:
        raise ValueError("SCOPE residual grid must have shape [N,49]")
    if not torch.isfinite(grid).all():
        raise ValueError("SCOPE residual grid contains non-finite values")
    return grid


def _complete_adapter_payload(value: Any) -> bool:
    return isinstance(value, Mapping) and isinstance(value.get("spec"), Mapping) and isinstance(value.get("state_dict"), Mapping)
