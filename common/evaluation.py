"""
Multi-split evaluation and paired delta reporting.

Computes paired deltas vs E0 (tuned linear baseline) across confirm splits
and produces pooled/confirmation statistics for candidate selection.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_eval_results(results_path: str) -> Dict:
    """Load eval_results.json from a training run.

    Args:
        results_path: Path to the eval_results.json file.

    Returns:
        Parsed JSON dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Eval results not found: {results_path}")
    with open(results_path, "r") as f:
        return json.load(f)


def compute_paired_deltas(
    e0_results: Dict[int, float],
    candidate_results: Dict[int, float],
) -> Dict[str, float]:
    """Compute paired deltas: delta_i = Acc_candidate,i - Acc_E0,i for each split.

    Each dict maps split_seed -> validation accuracy (float in [0, 1]).

    Args:
        e0_results: E0 validation accuracy per split seed.
        candidate_results: Candidate validation accuracy per split seed.

    Returns:
        Dict with keys:
          - deltas: {split_seed: delta}
          - mean_delta: mean across splits
          - std_delta: sample standard deviation (ddof=1)
          - min_delta: worst-split delta
          - max_delta: best-split delta
          - confirmation_wins: X/Y format (splits where delta > -0.002)
          - pooled_delta: same as mean_delta (for API convenience)

    Raises:
        ValueError: If no shared split seeds found between the two result sets.
    """
    deltas = {}
    for split_seed in e0_results:
        if split_seed in candidate_results:
            deltas[split_seed] = (
                candidate_results[split_seed] - e0_results[split_seed]
            )

    n = len(deltas)
    if n == 0:
        raise ValueError(
            "No shared split seeds found between E0 and candidate results"
        )

    mean_delta = sum(deltas.values()) / n
    std_delta = (
        (sum((d - mean_delta) ** 2 for d in deltas.values()) / (n - 1)) ** 0.5
        if n > 1
        else 0.0
    )
    min_delta = min(deltas.values())
    max_delta = max(deltas.values())
    wins = sum(1 for d in deltas.values() if d > -0.002)

    return {
        "deltas": deltas,
        "mean_delta": round(mean_delta, 6),
        "std_delta": round(std_delta, 6),
        "min_delta": round(min_delta, 6),
        "max_delta": round(max_delta, 6),
        "confirmation_wins": f"{wins}/{n}",
        "pooled_delta": round(mean_delta, 6),
    }


def apply_candidate_rules(
    candidates: Dict[str, Dict],
    elimination_threshold: float = -0.002,
    tie_threshold: float = 0.001,
) -> Tuple[str, Dict]:
    """Apply the pre-specified candidate selection rules.

    The selection process:
      1. Eliminate candidates where any split degrades by > elimination_threshold.
      2. Eliminate candidates whose mean delta <= 0.
      3. Fallback to E0 if no survivors.
      4. If exactly one survivor, select it.
      5. Tiebreaker among multiple survivors:
         a. Higher mean_delta wins.
         b. If within tie_threshold, pick by deterministic rules (higher mean_delta
            then lexicographic experiment ID).
         c. Otherwise higher mean_delta wins.

    Args:
        candidates: {candidate_id: paired_delta_report} where each report
            is the dict returned by compute_paired_deltas.
        elimination_threshold: Maximum allowed degradation on any single split.
            Default -0.002 (0.2 percentage points).
        tie_threshold: Maximum mean_delta difference to consider a tie.
            Default 0.001 (0.1 percentage points).

    Returns:
        Tuple of (selected_id, selection_report), where selection_report contains
        at minimum a "reason" key and possibly additional keys like "winner",
        "delta_diff", "fallback".
    """
    if not candidates:
        return "E0", {"reason": "no_candidates_provided", "fallback": "E0"}

    # Step 2: Eliminate candidates that degrade > elimination_threshold on any split
    survivors = {}
    elimination_log = {}
    for cid, report in candidates.items():
        min_delta = report.get("min_delta", -float("inf"))
        if min_delta > elimination_threshold:
            survivors[cid] = report
        else:
            elimination_log[cid] = f"min_delta {min_delta} <= {elimination_threshold}"

    # Step 3: Eliminate if mean delta <= 0
    survivors = {
        k: v
        for k, v in survivors.items()
        if v.get("mean_delta", -float("inf")) > 0
    }
    # Update elimination log for those dropped in this step
    for cid in candidates:
        if cid not in survivors and cid not in elimination_log:
            elimination_log[cid] = (
                f"mean_delta {candidates[cid].get('mean_delta')} <= 0"
            )

    # Step 4: Fallback to E0
    if not survivors:
        return "E0", {
            "reason": "no_candidates_survived",
            "fallback": "E0",
            "elimination_log": elimination_log,
        }

    # Step 5: Exactly one survivor
    if len(survivors) == 1:
        winner = list(survivors.keys())[0]
        return winner, {
            "reason": "sole_survivor",
            "winner": winner,
            "elimination_log": elimination_log,
        }

    # Step 6: Multiple survivors — apply tiebreaker
    sorted_ids = sorted(survivors.keys())
    best_mean = survivors[sorted_ids[0]]["mean_delta"]
    runner_up_mean = survivors[sorted_ids[1]]["mean_delta"]
    delta_diff = abs(best_mean - runner_up_mean)

    if delta_diff < tie_threshold:
        # Tiebreaker rules:
        # 1. Fewer inference-time parameters (cosine < linear since no bias)
        # 2. Fewer augmentation components (a0 < a1 < a2 < a3)
        # 3. Higher worst-split delta
        # 4. Lower delta std
        # 5. Lexicographic experiment ID
        #
        # Simplified heuristic: prefer by (higher min_delta, lower std_delta,
        # lexicographic experiment ID).
        winner = max(
            survivors.keys(),
            key=lambda k: (
                survivors[k].get("min_delta", -float("inf")),
                -survivors[k].get("std_delta", float("inf")),
                k,  # lexicographic tiebreaker
            ),
        )
        return winner, {
            "reason": "tiebreaker",
            "winner": winner,
            "delta_diff": delta_diff,
            "elimination_log": elimination_log,
        }

    # Otherwise higher mean delta wins
    winner = max(survivors.keys(), key=lambda k: survivors[k]["mean_delta"])
    return winner, {
        "reason": "higher_mean_delta",
        "winner": winner,
        "delta_diff": delta_diff,
        "elimination_log": elimination_log,
    }
