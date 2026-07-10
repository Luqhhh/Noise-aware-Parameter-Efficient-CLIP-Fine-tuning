# Task 11 Report: Multi-Split Evaluation Module

## Summary

Created `common/evaluation.py` with three functions for multi-split evaluation
and paired delta reporting. Modified `scripts/split_data.py` to accept
`--split-seeds` for generating multiple seed-separated splits.

## Files Created/Modified

| File | Lines | Description |
|------|-------|-------------|
| `common/evaluation.py` | ~220 | Paired delta computation, candidate selection rules, JSON loader |
| `tests/test_evaluation.py` | ~240 | Unit tests for all three functions |
| `scripts/split_data.py` | Modified | Added `--split-seeds` argument + refactored into `_generate_single_split()` |

## Key Design Decisions

1. **Refactored split generation**: Moved single-split logic into
   `_generate_single_split()` so both single-seed and multi-seed modes share
   the same code path. The `main()` function resolves parameters then dispatches.

2. **Multi-seed output layout**: Seeds produce splits in
   `{split_dir}/seed_{N}/` subdirectories, keeping each seed's output isolated.

3. **compute_paired_deltas**: Computes per-split deltas, mean, sample std
   (ddof=1), min/max, and `confirmation_wins` (count of splits where delta >
   -0.002). Raises `ValueError` on no shared seeds.

4. **apply_candidate_rules**: Four-stage elimination:
   - Eliminate by min_delta threshold (default -0.002)
   - Eliminate by mean_delta <= 0
   - Fallback to E0 if no survivors
   - Tiebreaker: higher min_delta, then lower std_delta, then lexicographic

5. **Explicit exceptions**: Uses `FileNotFoundError`, `ValueError`, and
   `json.JSONDecodeError` (raised naturally by json.load) in accordance
   with the project's exception policy.

## Usage

```bash
# Single seed (unchanged)
python scripts/split_data.py --config configs/baseline.yaml

# Multi-seed
python scripts/split_data.py --config configs/baseline.yaml \
    --split-seeds 42,43,44
```

## Dependencies

- `json`, `pathlib.Path` (stdlib)
