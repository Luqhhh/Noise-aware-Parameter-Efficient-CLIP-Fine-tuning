# S_MIXUP Provenance Audit

**Date:** 2026-07-17
**Scope:** A04, P04 — four-source config verification + AMP/FP32 precision separation

## 1. Config Provenance (Four-Source Audit)

| Source | A04 α | A04 p | A04 id | P04 α | P04 p | P04 id |
|--------|-------|-------|--------|-------|-------|--------|
| [1] Git config | 0.4 | 0.2 | S_MIXUP_A04 | 0.2 | 0.4 | S_MIXUP_P04 |
| [2] resolved_config.yaml | 0.4 | 0.2 | S_MIXUP_A04 | 0.2 | 0.4 | S_MIXUP_P04 |
| [3] config_snapshot | 0.4 | 0.2 | S_MIXUP_A04 | 0.2 | 0.4 | S_MIXUP_P04 |
| [4] best.pt internal | 0.4 | 0.2 | S_MIXUP_A04 | 0.2 | 0.4 | S_MIXUP_P04 |

**Verdict: ALL FOUR SOURCES CONSISTENT for both experiments.**

- **A04**: α=0.4, p=0.2 — varies alpha (0.2→0.4) with p fixed at 0.2
- **P04**: α=0.2, p=0.4 — varies probability (0.2→0.4) with α fixed at 0.2

A04 and P04 are NOT directly comparable — each tests a different parameter
against the parent baseline A02 (α=0.2, p=0.2, not yet available).

## 2. AMP Precision (mixed-precision validation, eval_results.json)

| Metric | A04 (α=0.4,p=0.2) | P04 (α=0.2,p=0.4) |
|--------|---------------------|---------------------|
| n_correct | 7265 | 7263 |
| Micro | 0.70424583 | 0.70405196 |
| Macro | 0.70369768 | 0.70354456 |
| Bottom-10% | 0.31570005 | 0.31317559 |
| Micro-Macro gap | 0.00054815 | 0.00050740 |
| Best epoch | 37 | 39 |

**Δ n_correct (P04−A04): −2 images (−0.0194 pp)**
McNemar: NOT AVAILABLE for AMP (no per-sample AMP prediction records).

## 3. FP32 Precision (post-training evaluation, prediction_records.csv)

| Metric | A04 (α=0.4,p=0.2) | P04 (α=0.2,p=0.4) |
|--------|---------------------|---------------------|
| n_correct | 7266 | 7261 |
| Micro | 0.70434277 | 0.70385808 |

Contingency (n=10316):
- Both correct: 7215 (69.94%)
- Both wrong: 3004 (29.12%)
- A04 correct, P04 wrong: 51
- A04 wrong, P04 correct: 46
- Mismatch: 97 (0.94%)

**Δ n_correct (P04−A04): −5 images (−0.0485 pp)**

Cross-validation: 7266 − 7261 = 5 = 51 − 46 ✅

McNemar exact p = 0.6849 → NOT significant at α=0.05.

## 4. Conclusions

1. P04 shows no measurable improvement over A04 in either AMP (−2 images)
   or FP32 (−5 images) precision; the difference is not significant
   (McNemar p = 0.6849).

2. A04 (α=0.4, p=0.2) and P04 (α=0.2, p=0.4) vary different parameters.
   Neither can be declared "champion" without comparison against the
   shared parent A02 (α=0.2, p=0.2).

3. P04 was resumed from epoch 36 after a crash. Its RNG trajectory is
   discontinuous. The result is valid but the resume must be disclosed
   in any reporting.

4. Awaiting A01 (α=0.1, p=0.2) and A02 (α=0.2, p=0.2) before declaring
   final ranking, parameter effects, or alpha-vs-probability attribution.

## 5. P04 Recovery Audit (epoch 36 checkpoint)

- last.pt SHA-256: `bbb967e6e5fc3c3240f424c6767daeaf214f8683f86ca95619befded2cb132df`
- epoch=36, global_step=25704, best_val_acc=0.70279178
- Pre-resume gate: CPU strict-load verified (model 154 keys, optimizer 2 states, scheduler last_epoch=24276, scaler scale=4194304)
- Config lineage: all 21 key fields match between checkpoint and current config
- Git HEAD at resume: 5037f94 (unchanged throughout)
- Resumed epoch 37: global_step→26418, LR 0.00101832→0.00089312
- RNG state NOT saved; shuffle/augmentation trajectory differs from uninterrupted run
- No code changes, no checkpoint patching
