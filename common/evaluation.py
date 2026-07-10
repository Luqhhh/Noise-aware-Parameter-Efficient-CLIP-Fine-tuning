"""
Multi-split evaluation and candidate selection.

Provides functions for:
    - Loading evaluation result JSON files
    - Computing paired deltas between candidates and a baseline (E0)
    - Applying candidate selection rules with fallback

Design:
    - Paired deltas are computed per-split: Delta_i = Acc_candidate,i - Acc_E0,i
    - Mean delta, sample std (ddof=1), min, max, confirmation wins are reported
    - Candidate elimination uses explicit pre-defined rules
    - Fallback to E0 if no candidate survives
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default threshold: candidate degrades more than 0.2pp vs E0 on any split -> eliminated
DEFAULT_MIN_DELTA_THRESHOLD = -0.002


def load_eval_json(path: str) -> Dict[str, Any]:
    """Load an evaluation results JSON file.

    Expected format:
    {
        "accuracy": 0.85,
        "loss": 0.45,
        "total_samples": 1000,
        "correct_samples": 850
    }

    Args:
        path: Path to the evaluation JSON file.

    Returns:
        Dictionary with evaluation results.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If required fields are missing.
    """
    eval_path = Path(path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation result not found: {path}")

    with open(eval_path, "r") as f:
        results = json.load(f)

    required = {"accuracy", "loss"}
    missing = required - set(results.keys())
    if missing:
        raise ValueError(
            f"Evaluation result missing required fields: {missing}. "
            f"Found: {list(results.keys())}"
        )

    return results


def compute_paired_deltas(
    e0_results: Dict[int, Dict[str, Any]],
    candidate_results: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute paired deltas between a candidate and E0 across splits.

    Args:
        e0_results: {split_seed: {"accuracy": ..., "loss": ...}} for E0.
        candidate_results: {split_seed: {"accuracy": ..., "loss": ...}} for candidate.

    Returns:
        Dictionary with:
            - "per_split": {split_seed: delta_accuracy}
            - "mean_delta": float
            - "std_delta": float (sample std, ddof=1)
            - "min_delta": float
            - "max_delta": float
            - "confirmation_wins": int (count of splits where delta > -0.002)
            - "num_splits": int

    Raises:
        ValueError: If no shared split seeds are found.
    """
    shared_seeds = sorted(
        set(e0_results.keys()) & set(candidate_results.keys())
    )

    if not shared_seeds:
        raise ValueError(
            "No shared split seeds between E0 and candidate. "
            f"E0 seeds: {sorted(e0_results.keys())}, "
            f"Candidate seeds: {sorted(candidate_results.keys())}"
        )

    per_split = {}
    for seed in shared_seeds:
        e0_acc = e0_results[seed]["accuracy"]
        cand_acc = candidate_results[seed]["accuracy"]
        per_split[seed] = cand_acc - e0_acc

    deltas = list(per_split.values())
    n = len(deltas)
    mean_delta = sum(deltas) / n

    if n > 1:
        variance = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
        std_delta = variance ** 0.5
    else:
        std_delta = 0.0

    min_delta = min(deltas)
    max_delta = max(deltas)
    confirmation_wins = sum(
        1 for d in deltas if d > DEFAULT_MIN_DELTA_THRESHOLD
    )

    result = {
        "per_split": per_split,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "min_delta": min_delta,
        "max_delta": max_delta,
        "confirmation_wins": confirmation_wins,
        "num_splits": n,
    }

    logger.info(
        f"Paired deltas over {n} splits: "
        f"mean={mean_delta:.6f}, std={std_delta:.6f}, "
        f"min={min_delta:.6f}, max={max_delta:.6f}, "
        f"wins={confirmation_wins}/{n}"
    )

    return result


def apply_candidate_rules(
    candidates: Dict[str, Dict[str, Any]],
    fallback: str = "E0",
) -> str:
    """Apply candidate selection rules to choose the final method.

    Rules:
        1. Eliminate candidates with any split delta < min_delta_threshold.
        2. Eliminate candidates with mean_delta <= 0.
        3. If no survivors, return fallback.
        4. If exactly one survivor, return it.
        5. If multiple survivors, apply tiebreakers.

    Args:
        candidates: {candidate_name: compute_paired_deltas_output}.
        fallback: Experiment ID to fallback to if no candidate passes.

    Returns:
        Selected experiment ID string.

    Raises:
        ValueError: If candidates is empty.
    """
    if not candidates:
        raise ValueError("No candidates provided for selection")

    logger.info(f"Applying candidate rules to {len(candidates)} candidates")

    # Stage 1: Eliminate by min_delta threshold
    survivors = {}
    for name, deltas in candidates.items():
        if deltas["min_delta"] < DEFAULT_MIN_DELTA_THRESHOLD:
            logger.info(
                f"  Eliminated {name}: min_delta={deltas['min_delta']:.6f} "
                f"< {DEFAULT_MIN_DELTA_THRESHOLD}"
            )
        else:
            survivors[name] = deltas

    # Stage 2: Eliminate by mean_delta <= 0
    for name in list(survivors.keys()):
        if survivors[name]["mean_delta"] <= 0:
            logger.info(
                f"  Eliminated {name}: mean_delta={survivors[name]['mean_delta']:.6f} <= 0"
            )
            del survivors[name]

    # Stage 3: Fallback check
    if not survivors:
        logger.info(f"No survivors. Falling back to {fallback}.")
        return fallback

    # Stage 4: Single survivor
    if len(survivors) == 1:
        selected = next(iter(survivors))
        logger.info(f"Single survivor: {selected}")
        return selected

    # Stage 5: Tiebreakers
    # Tiebreaker 1: higher min_delta
    sorted_candidates = sorted(
        survivors.items(),
        key=lambda x: (-x[1]["min_delta"], -x[1]["mean_delta"]),
    )

    # Check if clear winner by min_delta
    best_name, best_deltas = sorted_candidates[0]
    second_name, second_deltas = sorted_candidates[1]

    delta_diff = abs(best_deltas["mean_delta"] - second_deltas["mean_delta"])

    if delta_diff >= 0.001:  # 0.1pp threshold
        logger.info(
            f"Selected {best_name} by higher min_delta "
            f"({best_deltas['min_delta']:.6f} vs {second_deltas['min_delta']:.6f})"
        )
        return best_name

    # Tiebreaker 2: lower std_delta
    sorted_by_std = sorted(
        survivors.items(),
        key=lambda x: (x[1]["std_delta"], -x[1]["min_delta"]),
    )
    selected = sorted_by_std[0][0]
    logger.info(
        f"Selected {selected} by lower std_delta "
        f"({sorted_by_std[0][1]['std_delta']:.6f})"
    )

    return selected
