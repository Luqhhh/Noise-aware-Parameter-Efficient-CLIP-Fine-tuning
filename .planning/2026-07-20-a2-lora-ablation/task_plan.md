# A2 LoRA Ablation Spec Plan

**Status:** training and platform evaluation complete; documentation updated  
**Date:** 2026-07-20  
**Goal:** Define two controlled LoRA ablations on the A2 parent checkpoint, isolating adapter capacity while retaining the same cleaned data, parent model, loss, distillation, split, seed, and evaluation protocol.

## Phases

- [x] Phase 0 — Recover context and inspect existing A2/AEGIS assets.
- [x] Phase 0b — Pull latest `origin/main` (blocked by WSL GitHub TLS/443 failures; retry before implementation).
- [x] Phase 1 — Translate the image requirements into a pre-registered experiment matrix.
- [x] Phase 2 — Audit implementation readiness and record required code/config gaps.
- [x] Phase 3 — Draft the reproducibility, acceptance, and stop rules.
- [x] Phase 4 — Implement the generic visual-LoRA/distillation runner only after the user approves this spec.
- [x] Phase 5 — Run smoke tests, then sequential full runs: `A2_LORA_MIN` → `A2_LORA_FULL`.
- [x] Phase 6 — Re-evaluate checkpoints, generate bare/TTA submissions, validate, and register results.

## Locked experiment matrix

| ID | Parent | Rank | Target blocks | Target modules | Distillation |
|---|---|---:|---|---|---|
| `A2_LORA_MIN` | A2 `best.pt` | 4 | block 11 only | attention Q/V/out | yes |
| `A2_LORA_FULL` | A2 `best.pt` | 8 | blocks 8–11 | attention Q/V/out | yes |

Only rank and target-block scope differ. Alpha, dropout, optimizer, augmentation, clean-data manifest, split, seed, epoch budget, and selection rule are shared. Use alpha/r scaling equivalent to the AEGIS F1 reference (`r=4, alpha=8`; `r=8, alpha=16`) unless the parent runner documents a different fixed convention.

## Required gates before training

1. Pull/rebase latest `main` and record the actual source commit.
2. Locate the exact A2 `best.pt`, A2 train/val CSVs, class mapping, cleaning manifest, and parent config; record SHA-256 values.
3. Verify the child split is byte-identical to A2's split and that no validation image is in the parent's training set.
4. Verify epoch-0 child logits match the parent checkpoint before any LoRA update; fail closed on mismatch.
5. Verify the implementation can attach Q/V/out LoRA to arbitrary visual blocks and can compute cosine feature distillation against a frozen parent model.
6. Run a synthetic unit test for block selection, parameter freezing, state-dict save/load, and finite distillation loss.

## Future execution order

1. Run a short smoke test for each config (one epoch or a bounded batch count) and inspect trainable-parameter audit, loss finiteness, and checkpoint reload.
2. Run `A2_LORA_MIN` to the fixed AEGIS-style budget (6 epochs unless the recovered A2 protocol specifies another pre-registered budget).
3. Run `A2_LORA_FULL` with every non-LoRA variable unchanged.
4. Re-load each `best.pt` in a fresh process; produce raw validation metrics, per-class metrics, feature drift, flip agreement, and parent prediction-change rate.
5. Generate and validate both bare and horizontal-flip TTA submissions. Bare is the compliance-primary result; TTA is secondary and carries the same two-forward-pass compliance caveat documented for AEGIS F1.

## Decision rules

- Primary gate: `A2_LORA_MIN` must improve bare accuracy by at least `+0.30 pp` over the A2 parent to justify the minimal adapter.
- If MIN passes and FULL adds no material bare gain (pre-register `≤0.10 pp`), retain MIN for complexity and overfitting reasons.
- If MIN fails but FULL passes, retain FULL as the capacity-needed configuration and report the failed minimal ablation.
- If both fail, close this LoRA route; do not search rank/alpha/dropout/block combinations.
- Never select a winner using TTA alone. Report bare and TTA separately, with platform score and local metrics kept distinct.

## Required artifacts

For each run, retain resolved config, parent/config/split/manifest hashes, runtime protocol audit, train log, `best.pt`/`last.pt`, re-evaluation JSON, per-class CSV, prediction records, bare submission CSV/ZIP, TTA submission CSV/ZIP, and `check_submission.py` output. Checkpoints remain local/ignored unless a later explicit request changes repository policy.

## Errors / blockers

| Error | Attempt | Resolution |
|---|---|---|
| GitHub HTTPS pull: `GnuTLS recv error (-110)` | `git pull --rebase --autostash origin main` | Retry with HTTP/1.1; second attempt timed out connecting to port 443. Retry after network/VPN recovery. |

## Actual platform outcome (2026-07-22)

| Experiment | Inference | Platform score |
|---|---|---:|
| A2_LORA_MIN | bare | **61.1167%** |
| A2_LORA_MIN | horizontal_flip TTA | **61.6574%** |
| A2_LORA_FULL | bare | **61.5733%** |
| A2_LORA_FULL | horizontal_flip TTA | **62.1781%** |

The full adapter with TTA is the best result in this ablation (+0.5207 pp over MIN TTA). The bare FULL package remains unsubmitted, so no local metric is treated as a platform score.
