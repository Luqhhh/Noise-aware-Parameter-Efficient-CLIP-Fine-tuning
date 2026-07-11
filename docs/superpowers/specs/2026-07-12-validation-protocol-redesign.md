# Validation Protocol Redesign: Rebuilding Trusted Baselines

## Goal

Fix a critical stage-to-stage validation leak (88% of F1's validation images were seen in D3's training set) and rebuild all baselines on a single, reusable master split for fair comparison.

## Current State

| Experiment | Best Val Acc | Status | Verdict |
|---|---|---|---|
| E0 | 69.86% | `historical_valid` | Needs re-run on unified split |
| D3 | 70.53% | `pending_fair_comparison` | Different val set from E0 |
| D4A | 70.09% | `historical_valid` | dropout 0.3, below D3 |
| D4B | 69.39% | `historical_valid` | dropout 0.5 |
| D4C | — | `incomplete` | Paused |
| F0 | 69.84% (1 epoch) | `incomplete` | Control experiment missing |
| F1 | **80.13%** | `invalid_stage_leakage` | 88% val seen by D3 train |
| F1b | **80.13%** | `invalid_stage_leakage` | Same leak |
| F2-* | — | `blocked_by_invalid_parent` | Paused |

## Root Cause

F1 validation contained 10,116 images, of which 8,902 (88.0%) were in D3's training split. Each experiment generated its own random split, and subsequent experiments did not check for train/val overlap with their parent checkpoint's data.

## Scope: In-Scope vs Out-of-Scope

**In scope (this phase):**
1. Fix D3 → F1 stage-to-stage validation leak
2. Establish a single, reusable master split
3. Fair comparison of E0, D3, F0, F1 on the same validation set
4. Confirm D3 cleaning and F1 partial unfreeze are real
5. Build traceable training → inference → submission pipeline
6. Multi-seed confirmation (seed=42, 3407, 2026)

**Out of scope (deferred):**
- Label Smoothing, GCE, SCE
- EMA loss sample weighting
- MixUp
- EMA Teacher
- LoRA
- F2 large-scale parameter search
- New robustness strategy combinations

## Execution Order

```text
1. Freeze + mark existing results
2. Build unique master split
3. Add parent-child split audit
4. Refactor D3 to train-only cleaning
5. Re-run E0 and D3
6. Complete F0 control experiment
7. Re-run F1 on same split
8. seed=42 decision gate
9. Multi-seed confirmation
10. Submission pipeline audit
```

## Key Design Constraints

1. **One split per seed** — All experiments with the same seed share identical train/val splits
2. **Train-only cleaning** — D3 cleaning operates only on training data; validation is untouched
3. **Parent-child audit** — Every experiment initialized from a parent checkpoint MUST verify `child_val ∩ parent_train = ∅` and `child_val = parent_val`
4. **Hard exit on violation** — Training aborts on any split integrity violation
5. **Epoch-0 validation** — Every experiment must run validation before first optimizer step and match parent accuracy within 0.05pp

## Submission Discipline

- `results/submission_registry.csv` tracks every submission with checkpoint SHA256, prediction SHA256, and online accuracy
- Dual-implementation consistency check: top-1 match rate = 100%, logit max abs error ≤ 1e-5
- T0 = strict verified baseline; T1 = best multi-seed-confirmed model

## Directory Structure

```
outputs/
├── master_splits/
│   ├── seed42/
│   ├── seed3407/
│   └── seed2026/
├── e0_strict/
│   ├── seed42/
│   ├── seed3407/
│   └── seed2026/
├── d3_strict/
│   ├── seed42/
├── f0_strict/
│   ├── seed42/
└── f1_strict/
    ├── seed42/
```
