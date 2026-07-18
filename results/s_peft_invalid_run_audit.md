# S-PEFT Invalid Run Audit

Date: 2026-07-16
Archived: 2026-07-16

## Invalid Experiments

| Directory | Config | Problem |
|---|---|---|
| outputs/peft/s_peft/e1_ln_1e6_invalid_freeze_clip_true/ | freeze_clip: true + peft: visual_layernorm_only | LN gradients blocked |
| outputs/peft/s_peft/e2_ln_5e7_invalid_freeze_clip_true/ | freeze_clip: true + peft: visual_layernorm_only | LN gradients blocked |

## Root Cause

`model.py:188` — `torch.set_grad_enabled(not self.freeze_clip)` globally disables
visual gradients when `freeze_clip=true`, regardless of per-parameter `requires_grad`.

The PEFT interface (`common/peft.py`) correctly sets `requires_grad=True` for LN
parameters, but the model's `encode_image` method overrides this at the context level.

Result: E1/E2 trained classifier only — structurally identical to E0. All three
experiments produced identical metrics (micro=0.73207).

## Evidence

- Optimizer state: backbone group parameters 0/52 had `exp_avg` entries
- Fixed-batch gradient test: 0/52 LN parameters had non-zero gradients
- Parameter diff vs parent: all LN parameters bitwise identical to parent
- Prediction change E1-E0, E2-E0, E1-E2: 0 mismatches

## Remediation

Configs corrected to `freeze_clip: false`. Training re-run successfully.
Valid results at `outputs/peft/s_peft/e1_ln_1e6/` and `outputs/peft/s_peft/e2_ln_5e7/`.

## Public Interface Gap

The PEFT system has no conflict gate to reject `freeze_clip=true` + `peft.type`
targeting visual parameters. This gate should be added to `common/peft.py` or
`common/config_schema.py` to prevent re-occurrence (deferred to team).
