"""Candidate objectives for the L05 new-candidate spec (NEW02 / NEW03).

Self-contained and differentiable: nothing here detaches, calls ``.item()`` on a
graph value, or touches a prior bias.  The trainer wires these functions; the
module only defines the objectives and their fail-closed input validation.
"""

from __future__ import annotations

import numbers
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


def _validate_q(q: float) -> float:
    q = float(q)
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0,1]")
    return q


def _validate_epsilon(epsilon: float) -> float:
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return epsilon


def _as_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Fail-closed rank-2 probability matrix (finite, non-negative)."""
    values = probabilities
    if values.ndim != 2:
        raise ValueError("probabilities must be rank-2 [N,C]")
    if values.shape[1] <= 1:
        raise ValueError("probabilities must have at least two classes")
    if not torch.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    if bool((values < 0.0).any()):
        raise ValueError("probabilities must be non-negative")
    return values


def _hard_targets(target: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    labels = target.long().flatten()
    if labels.numel() != values.shape[0]:
        raise ValueError("hard targets must align with the probability rows")
    if bool(((labels < 0) | (labels >= values.shape[1])).any()):
        raise ValueError("hard target outside the class range")
    return labels


def probability_generalized_cross_entropy(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    *,
    q: float = 0.5,
    epsilon: float = 1.0e-7,
) -> torch.Tensor:
    """Per-sample GCE in the probability domain: ``l_q(p,y) = (1 - p_y**q)/q``.

    ``probabilities`` is ``[N,C]`` already softmax-normalised.  ``target`` is
    either ``[N]`` hard labels or ``[N,C]`` soft targets.  Hard targets match
    :func:`aegis_clip.losses.soft_generalized_cross_entropy` on
    ``log(probabilities)`` under the same epsilon convention.
    """
    q = _validate_q(q)
    epsilon = _validate_epsilon(epsilon)
    values = _as_probabilities(probabilities)
    clamped = values.clamp_min(epsilon)
    if target.ndim == 1:
        labels = _hard_targets(target, values)
        selected = clamped.gather(1, labels.unsqueeze(1)).squeeze(1)
        return (1.0 - selected.pow(q)) / q
    if target.ndim == 2:
        if target.shape != values.shape:
            raise ValueError("soft targets must match the probability shape")
        if not torch.isfinite(target).all():
            raise ValueError("soft targets must be finite")
        return (target * (1.0 - clamped.pow(q)) / q).sum(dim=1)
    raise ValueError("target must be [N] hard labels or [N,C] soft targets")


def flip_fusion_global_loss(
    logits_original: torch.Tensor,
    logits_flipped: torch.Tensor,
    target: torch.Tensor,
    *,
    q: float = 0.5,
    temperature: float = 1.4,
    epsilon: float = 1.0e-7,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """NEW02 flip-fusion objective over the two view branches and their mean.

    ``p_o = softmax(z(x)/T)``, ``p_f = softmax(z(F(x))/T)``,
    ``p_mix = (p_o + p_f)/2`` and
    ``L = 0.5 * (l_q(p_o,y) + l_q(p_f,y)) / 2 + 0.5 * l_q(p_mix,y)``.
    Both branches stay in the graph and no prior bias is applied.
    """
    q = _validate_q(q)
    epsilon = _validate_epsilon(epsilon)
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if logits_original.ndim != 2 or logits_flipped.ndim != 2:
        raise ValueError("logits must be rank-2 [N,C]")
    if logits_original.shape != logits_flipped.shape:
        raise ValueError("original and flipped logits must share a shape")
    if not torch.isfinite(logits_original).all() or not torch.isfinite(
        logits_flipped
    ).all():
        raise ValueError("logits must be finite")

    p_o = F.softmax(logits_original / float(temperature), dim=1)
    p_f = F.softmax(logits_flipped / float(temperature), dim=1)
    p_mix = (p_o + p_f) / 2.0
    term_o = probability_generalized_cross_entropy(p_o, target, q=q, epsilon=epsilon)
    term_f = probability_generalized_cross_entropy(p_f, target, q=q, epsilon=epsilon)
    term_mix = probability_generalized_cross_entropy(
        p_mix, target, q=q, epsilon=epsilon
    )
    loss = 0.5 * (term_o + term_f) / 2.0 + 0.5 * term_mix
    diagnostics = {
        "p_o": p_o,
        "p_f": p_f,
        "p_mix": p_mix,
        "term_o": term_o,
        "term_f": term_f,
        "term_mix": term_mix,
    }
    return loss, diagnostics


def set_generalized_cross_entropy(
    probabilities: torch.Tensor,
    candidate_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    q: float = 0.5,
    epsilon: float = 1.0e-7,
) -> torch.Tensor:
    """NEW03 set supervision: ``r = sum_{c in S} p_c``, ``loss = (1 - r**q)/q``.

    ``candidate_index`` is ``[N,K]`` long class ids with padded values ignored
    where ``candidate_mask`` is falsy.  Every row needs at least one valid
    candidate and the resulting set mass must lie in ``[epsilon, 1]``.
    """
    q = _validate_q(q)
    epsilon = _validate_epsilon(epsilon)
    values = _as_probabilities(probabilities)
    index = torch.as_tensor(candidate_index).long()
    mask = torch.as_tensor(candidate_mask).bool()
    if index.ndim != 2 or mask.ndim != 2:
        raise ValueError("candidate_index and candidate_mask must be rank-2 [N,K]")
    if index.shape != mask.shape:
        raise ValueError("candidate_index and candidate_mask must share a shape")
    if index.shape[0] != values.shape[0]:
        raise ValueError("candidate rows must align with the probability rows")
    if index.shape[1] == 0:
        raise ValueError("candidate sets must not be empty")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every sample needs at least one valid candidate")
    if bool((mask & ((index < 0) | (index >= values.shape[1]))).any()):
        raise ValueError("valid candidate class id outside the class range")

    safe_index = index.clamp(0, values.shape[1] - 1)
    gathered = values.gather(1, safe_index)
    mass = (gathered * mask.to(values.dtype)).sum(dim=1)
    if bool((mass < epsilon).any()):
        raise ValueError("candidate set mass falls below epsilon")
    if bool((mass > 1.0 + 1.0e-5).any()):
        raise ValueError("candidate set mass exceeds one (duplicate classes?)")
    return (1.0 - mass.clamp_min(epsilon).pow(q)) / q


def build_padded_candidate_tensors(
    paths: Iterable[str],
    candidate_labels: Mapping[str, Sequence[int]],
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad per-path candidate sets into dense tensors for set supervision.

    Returns ``(candidate_index [N,K], candidate_mask [N,K], group_weight [N])``
    where ``K`` is the largest set size among ``paths`` and
    ``group_weight[i] = 1/len(set_i)``.  A path without a candidate set, or
    with an empty/duplicated/out-of-range one, raises ``ValueError``.
    """
    if not isinstance(num_classes, numbers.Integral) or int(num_classes) <= 1:
        raise ValueError("num_classes must be an integer greater than one")
    num_classes = int(num_classes)
    path_list = list(paths)
    if not path_list:
        empty_index = torch.zeros((0, 0), dtype=torch.long)
        empty_mask = torch.zeros((0, 0), dtype=torch.bool)
        empty_weight = torch.zeros((0,), dtype=torch.float32)
        return empty_index, empty_mask, empty_weight

    sets: list[list[int]] = []
    for path in path_list:
        labels = candidate_labels[path] if path in candidate_labels else None
        if labels is None:
            key = str(path)
            if key in candidate_labels:
                labels = candidate_labels[key]
        if labels is None:
            raise ValueError(f"missing candidate set for path: {path}")
        cleaned: list[int] = []
        for label in labels:
            if isinstance(label, bool) or not isinstance(label, numbers.Integral):
                raise ValueError(f"candidate label must be an integer: {label!r}")
            value = int(label)
            if not 0 <= value < num_classes:
                raise ValueError(
                    f"candidate label {value} outside [0, {num_classes})"
                )
            cleaned.append(value)
        if not cleaned:
            raise ValueError(f"empty candidate set for path: {path}")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"duplicate candidate label for path: {path}")
        sets.append(cleaned)

    width = max(len(item) for item in sets)
    rows = len(sets)
    candidate_index = torch.zeros((rows, width), dtype=torch.long)
    candidate_mask = torch.zeros((rows, width), dtype=torch.bool)
    group_weight = torch.empty((rows,), dtype=torch.float32)
    for row, labels in enumerate(sets):
        candidate_index[row, : len(labels)] = torch.tensor(labels, dtype=torch.long)
        candidate_mask[row, : len(labels)] = True
        group_weight[row] = 1.0 / float(len(labels))
    return candidate_index, candidate_mask, group_weight
