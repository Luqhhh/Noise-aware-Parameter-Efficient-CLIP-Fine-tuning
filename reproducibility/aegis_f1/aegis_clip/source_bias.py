"""Bounded class-bias fitting from a frozen, explicitly bound training source.

The fitting objective is class balanced through the existing sample weights and
soft targets.  Model parameters and image scores stay frozen.  This module also
owns the inference-protocol descriptor shared by the producer and consumer so a
calibration record cannot be rebound to a merely similar inference command.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aegis_clip.calibration_binding import protocol_sha256
from aegis_clip.runtime import sha256_file


SCORE_SEMANTICS = "log_final_fused_probabilities"
DEFAULT_BOUND = math.log(4.0)
DEFAULT_REGULARIZATION = 0.01
DEFAULT_CHUNK_SIZE = 4096


def build_inference_protocol_descriptor(
    args: Any,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    preprocess: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Build the exact descriptor used by both source production and inference."""
    excluded = {
        "checkpoint",
        "config",
        "output_dir",
        "prior_config",
        "prior_alignment_strength",
        "prior_alignment_iterations",
        "overwrite",
        "dump_logits",
        "dump_branch_logits",
    }
    code_root = Path(__file__).resolve().parent
    implementation = {
        str(path.relative_to(code_root)): sha256_file(path)
        for path in sorted(code_root.rglob("*.py"))
    }
    return {
        "cli": {
            key: value
            for key, value in vars(args).items()
            if key not in excluded
        },
        "implementation_sha256": protocol_sha256(implementation),
        "model": checkpoint.get("effective_model_spec", config["model"]),
        "preprocess": repr(preprocess),
        "amp": bool(config["train"].get("amp", True)) and device.type == "cuda",
        "device": device.type,
        "default_inference_batch_size": config["evaluation"].get(
            "inference_batch_size",
            min(int(config["evaluation"].get("batch_size", 256)), 256),
        ),
    }


def balanced_sample_weights(
    weights: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return effective class support ``n_c`` and the derived row weights ``u_i``."""
    weights = torch.as_tensor(weights).detach().cpu()
    targets = torch.as_tensor(targets).detach().cpu()
    if targets.ndim != 2 or weights.shape != (targets.shape[0],):
        raise ValueError("weights/targets must have shapes [N] and [N,C]")
    if targets.shape[0] == 0 or targets.shape[1] <= 1:
        raise ValueError("balanced source must have non-empty N and C > 1")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    if (
        not torch.isfinite(weights).all()
        or not torch.isfinite(targets).all()
        or bool((weights < 0).any())
        or bool((targets < 0).any())
    ):
        raise ValueError("weights and targets must be finite and non-negative")
    row_sums = targets.double().sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6):
        raise ValueError("soft-target rows must sum to one")

    rows, classes = targets.shape
    counts = torch.zeros(classes, dtype=torch.float64)
    for start in range(0, rows, int(chunk_size)):
        stop = min(start + int(chunk_size), rows)
        counts += (
            weights[start:stop].double().unsqueeze(1)
            * targets[start:stop].double()
        ).sum(dim=0)
    if not torch.isfinite(counts).all() or bool((counts <= 0).any()):
        raise ValueError("every class must have strictly positive effective supervision")

    row_weights = torch.empty(rows, dtype=torch.float64)
    for start in range(0, rows, int(chunk_size)):
        stop = min(start + int(chunk_size), rows)
        row_weights[start:stop] = (
            weights[start:stop].double()
            * (targets[start:stop].double() / counts.unsqueeze(0)).sum(dim=1)
            / float(classes)
        )
    if not torch.isfinite(row_weights).all() or bool((row_weights < 0).any()):
        raise ValueError("derived calibration row weights are invalid")
    if not math.isclose(float(row_weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("derived calibration row weights must sum to one")
    return counts.numpy(), row_weights.numpy()


def _baseline_logsumexp(scores: np.ndarray, chunk_size: int) -> np.ndarray:
    rows = scores.shape[0]
    result = np.empty(rows, dtype=np.float64)
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        block = np.asarray(scores[start:stop], dtype=np.float64)
        maximum = block.max(axis=1)
        result[start:stop] = maximum + np.log(
            np.exp(block - maximum[:, None]).sum(axis=1)
        )
    return result


def objective_delta_and_gradient(
    bias: np.ndarray,
    scores: np.ndarray | torch.Tensor,
    row_weights: np.ndarray,
    *,
    regularization: float = DEFAULT_REGULARIZATION,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    baseline_logsumexp: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Evaluate the registered objective difference and its analytic gradient."""
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores)
    bias = np.asarray(bias, dtype=np.float64)
    row_weights = np.asarray(row_weights, dtype=np.float64)
    if scores.ndim != 2 or bias.shape != (scores.shape[1],):
        raise ValueError("scores/bias must have shapes [N,C] and [C]")
    if row_weights.shape != (scores.shape[0],):
        raise ValueError("row_weights must have shape [N]")
    if scores.shape[0] == 0 or scores.shape[1] <= 1:
        raise ValueError("scores must have non-empty N and C > 1")
    if int(chunk_size) <= 0 or float(regularization) < 0.0:
        raise ValueError("invalid chunk_size or regularization")
    if (
        not np.isfinite(scores).all()
        or not np.isfinite(bias).all()
        or not np.isfinite(row_weights).all()
        or np.any(row_weights < 0.0)
    ):
        raise ValueError("objective inputs must be finite with non-negative row weights")
    if not math.isclose(float(row_weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("row_weights must sum to one")
    if baseline_logsumexp is None:
        baseline_logsumexp = _baseline_logsumexp(scores, int(chunk_size))
    baseline_logsumexp = np.asarray(baseline_logsumexp, dtype=np.float64)
    if baseline_logsumexp.shape != (scores.shape[0],) or not np.isfinite(baseline_logsumexp).all():
        raise ValueError("invalid baseline logsumexp")

    value = 0.0
    gradient = np.zeros(scores.shape[1], dtype=np.float64)
    for start in range(0, scores.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), scores.shape[0])
        shifted = np.asarray(scores[start:stop], dtype=np.float64) + bias[None, :]
        maximum = shifted.max(axis=1)
        exponentials = np.exp(shifted - maximum[:, None])
        denominators = exponentials.sum(axis=1)
        logsumexp = maximum + np.log(denominators)
        current_weights = row_weights[start:stop]
        value += float(
            np.dot(current_weights, logsumexp - baseline_logsumexp[start:stop])
        )
        gradient += (exponentials / denominators[:, None]).T @ current_weights
    classes = scores.shape[1]
    value -= float(bias.mean())
    value += float(regularization) * float(np.dot(bias, bias)) / (2.0 * classes)
    gradient -= 1.0 / classes
    gradient += float(regularization) * bias / classes
    if not math.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("non-finite source-bias objective")
    return value, gradient


def projected_gradient_inf_norm(
    bias: np.ndarray,
    gradient: np.ndarray,
    *,
    bound: float = DEFAULT_BOUND,
    tolerance: float = 1e-10,
) -> float:
    bias = np.asarray(bias, dtype=np.float64)
    projected = np.asarray(gradient, dtype=np.float64).copy()
    projected[(bias <= -float(bound) + tolerance) & (projected > 0.0)] = 0.0
    projected[(bias >= float(bound) - tolerance) & (projected < 0.0)] = 0.0
    return float(np.max(np.abs(projected)))


def fit_bounded_source_bias(
    scores: torch.Tensor,
    weights: torch.Tensor,
    targets: torch.Tensor,
    *,
    regularization: float = DEFAULT_REGULARIZATION,
    bound: float = DEFAULT_BOUND,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    maxiter: int = 100,
    maxfun: int = 300,
    maxls: int = 30,
    maxcor: int = 10,
    ftol: float = 1e-12,
    gtol: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit and validate the one registered bounded L-BFGS-B problem."""
    from scipy import __version__ as scipy_version
    from scipy.optimize import minimize

    scores = torch.as_tensor(scores).detach().cpu()
    if scores.ndim != 2 or not torch.isfinite(scores).all():
        raise ValueError("scores must be a finite [N,C] tensor")
    if scores.shape != targets.shape:
        raise ValueError("scores and targets must have the same [N,C] shape")
    if float(bound) <= 0.0 or float(regularization) < 0.0:
        raise ValueError("bound must be positive and regularization non-negative")
    counts, row_weights = balanced_sample_weights(
        weights, targets, chunk_size=int(chunk_size)
    )
    score_array = scores.numpy()
    baseline = _baseline_logsumexp(score_array, int(chunk_size))

    def evaluate(value: np.ndarray) -> tuple[float, np.ndarray]:
        return objective_delta_and_gradient(
            value,
            score_array,
            row_weights,
            regularization=float(regularization),
            chunk_size=int(chunk_size),
            baseline_logsumexp=baseline,
        )

    initial = np.zeros(scores.shape[1], dtype=np.float64)
    initial_value, initial_gradient = evaluate(initial)
    if abs(initial_value) > 1e-12:
        raise RuntimeError("registered objective difference is not zero at initialization")
    result = minimize(
        evaluate,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(-float(bound), float(bound))] * scores.shape[1],
        options={
            "maxiter": int(maxiter),
            "maxfun": int(maxfun),
            "maxls": int(maxls),
            "maxcor": int(maxcor),
            "ftol": float(ftol),
            "gtol": float(gtol),
        },
    )
    bias = np.asarray(result.x, dtype=np.float64)
    final_value, final_gradient = evaluate(bias)
    projected_norm = projected_gradient_inf_norm(
        bias, final_gradient, bound=float(bound)
    )
    if not bool(result.success):
        raise RuntimeError(f"L-BFGS-B source-bias fit failed: {result.message}")
    if final_value > initial_value + 1e-12:
        raise RuntimeError("source-bias objective increased")
    if projected_norm > 1e-6:
        raise RuntimeError("source-bias projected gradient exceeds 1e-6")
    if not np.isfinite(bias).all() or np.any(np.abs(bias) > float(bound) + 1e-10):
        raise RuntimeError("source-bias result is non-finite or outside registered bounds")

    corrected_marginal = np.zeros(scores.shape[1], dtype=np.float64)
    for start in range(0, scores.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), scores.shape[0])
        shifted = np.asarray(score_array[start:stop], dtype=np.float64) + bias[None, :]
        maximum = shifted.max(axis=1)
        exponential = np.exp(shifted - maximum[:, None])
        corrected_marginal += (
            exponential / exponential.sum(axis=1, keepdims=True)
        ).T @ row_weights[start:stop]

    report = {
        "status": "converged_checks_passed",
        "solver": "scipy.optimize.minimize:L-BFGS-B",
        "scipy_version": scipy_version,
        "numpy_version": np.__version__,
        "dtype": "float64",
        "rows": int(scores.shape[0]),
        "classes": int(scores.shape[1]),
        "chunk_size": int(chunk_size),
        "regularization": float(regularization),
        "bound": float(bound),
        "maxiter": int(maxiter),
        "maxfun": int(maxfun),
        "maxls": int(maxls),
        "maxcor": int(maxcor),
        "ftol": float(ftol),
        "gtol": float(gtol),
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "gradient_evaluations": int(result.njev),
        "initial_objective_delta": float(initial_value),
        "final_objective_delta": float(final_value),
        "initial_gradient_inf_norm": float(np.max(np.abs(initial_gradient))),
        "projected_gradient_inf_norm": projected_norm,
        "bias_min": float(bias.min()),
        "bias_max": float(bias.max()),
        "bias_l2": float(np.linalg.norm(bias)),
        "classes_at_lower_bound": int(np.count_nonzero(bias <= -float(bound) + 1e-10)),
        "classes_at_upper_bound": int(np.count_nonzero(bias >= float(bound) - 1e-10)),
        "effective_class_support_min": float(counts.min()),
        "effective_class_support_max": float(counts.max()),
        "row_weight_sum": float(row_weights.sum()),
        "weighted_corrected_marginal_l1_to_uniform": float(
            np.abs(corrected_marginal - 1.0 / scores.shape[1]).sum()
        ),
        "objective_is_difference_not_absolute_ce": True,
    }
    return torch.from_numpy(bias.astype(np.float32)), report
