# S-PEFT E0/E1/E2 — Final Verification Report

Generated: 2026-07-16

## Parent

- experiment_id: W1_CE5_GCE05
- checkpoint: outputs/gce/ce5_gce_q05/seed42/checkpoints/best.pt
- checkpoint SHA-256: cea35b25b1135c8ea32f1cfdf32e30eda077b6bd95b43bc95ceb5f61f1bf96a2
- parent_recipe: CE epochs 1-5, GCE q=0.5 epochs 6-50, no MixUp
- metrics: micro=0.7314, macro=0.7309, bottom-10%=0.3367

## Continuation Recipe (E0/E1/E2 unified)

- loss: GCE q=0.5 (no CE warmup repeat)
- mixup: alpha=0.2, probability=0.2
- classifier_lr: 1e-4
- scheduler: cosine
- warmup_epochs: 2
- epochs: 15
- early_stop_patience: 5
- split: outputs/baselines/ref/seed42 (seed=42)

## Experiment Configurations

| | E0 | E1 | E2 |
|---|---|---|---|
| experiment_id | S_PEFT_E0_FROZEN | S_PEFT_E1_LN_1E6 | S_PEFT_E2_LN_5E7 |
| freeze_clip | true | false | false |
| peft.type | (none) | visual_layernorm_only | visual_layernorm_only |
| backbone_lr | — | 1e-6 | 5e-7 |
| trainable params | 256,500 | 296,436 | 296,436 |

## Epoch-0 Gates (all PASSED)

All three experiments reproduced parent validation accuracy within 0.0005 threshold.

## First-Run Invalid Result (archived)

E1/E2 were initially run with `freeze_clip: true` (following the C_EXP5 config pattern).
This configuration is incompatible with `peft.type: visual_layernorm_only`:

- `model.py:188` uses `torch.set_grad_enabled(not self.freeze_clip)` in `encode_image`
- When `freeze_clip=true`, all visual gradients are blocked regardless of per-parameter `requires_grad`
- The PEFT interface has no conflict gate to reject this combination
- LN parameters had `requires_grad=True` but received `grad=None` at every step
- Optimizer state showed 0/52 backbone params ever stepped

Invalid directories archived as:
- outputs/peft/s_peft/e1_ln_1e6_invalid_freeze_clip_true/
- outputs/peft/s_peft/e2_ln_5e7_invalid_freeze_clip_true/

Root cause recorded as: "E1/E2 configs with freeze_clip=true + peft visual_layernorm_only — PEFT
interface lacks a conflict gate for this combination."

## Valid Results (freeze_clip=false, re-run)

### Metrics (from deterministic re-inference of best.pt, 2026-07-16)

| Metric | E0 (Frozen) | E1 (LN 1e-6) | E2 (LN 5e-7) |
|---|---|---|---|
| n_correct / N | 7549/10316 | 7548/10316 | 7544/10316 |
| Micro Accuracy | 0.7317758821 | 0.7316789453 | 0.7312911981 |
| Macro Accuracy | 0.731316 | 0.731254 | 0.730867 |
| Bottom-10% Accuracy | 0.336710 | 0.336932 | 0.336932 |
| Best Epoch | 1 | 3 | 3 |
| Epochs Run | 6 (ES) | 8 (ES) | 8 (ES) |

> **修正说明 (2026-07-16):** 旧 prediction_records 与对应 best.pt 的确定性复评
> 不一致，具体生成链路原因尚未确认。现已按照 checkpoint SHA、同一 val.csv 和
> class mapping 重新推理，修正后的记录作为权威结果。
> （训练期 `eval_results.json` 中的 best_val_acc 为 E0=0.732067、E1=0.731776、
> E2=0.731679，与权威复评值相差 3/1/4 个样本。）
> 所有成对恒等式在修正后数值上成立，结论不变。

### E1 vs E0

- Micro: −0.0097pp — within ±0.10pp, no measurable local gain
- Prediction mismatch: 56/10316 (0.54%), McNemar exact p=1.00 (not significant)
- Paired identity: acc_E0 − acc_E1 = (13−12)/10316 = +0.0000969 ✓
- 12 wrong→correct, 13 correct→wrong

### E2 vs E0

- Micro: −0.0485pp — within ±0.10pp, no measurable local gain
- Prediction mismatch: 44/10316 (0.43%), McNemar exact p=0.36 (not significant)
- Paired identity: acc_E0 − acc_E2 = (12−7)/10316 = +0.0004847 ✓
- 7 wrong→correct, 12 correct→wrong
- LR halving (1e-6→5e-7) did not improve; no further ordinary LR grid search recommended

### E1 vs E2

- Micro: E1 − E2 = +0.0388pp — within ±0.10pp
- Prediction mismatch: 15/10316 (0.15%), McNemar exact p=0.29 (not significant)
- Paired identity: acc_E1 − acc_E2 = (6−2)/10316 = +0.0003877 ✓

**结论：三组配对差异均在 ±0.10pp 内，McNemar 三组均不显著，无可测本地收益。**

### Gradient and Parameter Verification (all PASSED)

- G1-G10 fixed-batch gates: all passed for both E1 and E2
- 52/52 LN gradients finite and non-zero
- 52/52 LN optimizer steps confirmed
- visual.proj and other non-LN visual parameters frozen and bitwise unchanged
- Feature drift confirmed non-zero but small (cos_dist E1: 2.9e-5, E2: 7.4e-6)

## Prediction Records Audit

All three prediction_records.csv contain exactly 10,316 unique samples matching
val.csv (outputs/baselines/ref/seed42/val.csv). Zero duplicates, zero missing, zero extras.

Regenerated 2026-07-16 from best.pt (SHA-256 verified) via deterministic inference.
Per-class metrics recomputed and verified. All McNemar pairwise identities hold.
Audit script: `scripts/audit_consistency.py`.

## Public Wiring Issue (documented, not patched)

When `freeze_clip=false`, `encode_image()` (model.py:188) calls
`torch.set_grad_enabled(True)` during the visual forward pass. During validation
(wrapped in `torch.no_grad()`), this inner context re-enables autograd for the
visual encoder, causing intermediate activations to be stored unnecessarily.

Impact: validation/inference VRAM and time are higher than needed for PEFT experiments.
Numerical results are correct. Fix (deferred to team): encode_image should respect
the caller's grad context when individual visual parameters are trainable.

## PEFT Gate Assessment

Per plan §6.12 (Platform Bare Gate): NOT evaluated — no platform submission was made.

Local assessment: E1/E2 show results within ±0.10pp of paired frozen control E0,
with no measurable local gain. McNemar tests show no statistically significant
prediction differences.

Decision:
- E0 (frozen control): valid paired baseline, results stand
- E1/E2 (LN-only PEFT): valid results, no positive local signal
- E3 (backbone-only LN): diagnostic experiment — team decision pending
- E4 (LN + feature distillation): deferred pending E3 outcome
- E5 (second seed): not applicable (no candidate meets gate)
- LoRA (§9): gated — minimum LN-only PEFT not proven effective
- No TTA, no platform submission for any S-PEFT experiment

## Staging Whitelist (do NOT auto-commit)

Configs:
  configs/s_peft_e0_frozen.yaml
  configs/s_peft_e1_ln_1e6.yaml
  configs/s_peft_e2_ln_5e7.yaml

Text results (per experiment):
  outputs/peft/s_peft/{e0_frozen,e1_ln_1e6,e2_ln_5e7}/seed42/checkpoints/
    eval_results.json
    artifact_manifest.json
    per_class_metrics.csv
    prediction_records.csv
    resolved_config.yaml
    config_snapshot_*.yaml
    reeval_best.json
    eval_last.json
  outputs/peft/s_peft/{e0_frozen,e1_ln_1e6,e2_ln_5e7}/seed42/
    split_lineage_audit.json

Excluded from commit:
  *.pt (best.pt, best_raw.pt, last.pt)
  *.log, train_log.csv
  submission.zip
  outputs/peft/s_peft/e*_invalid_freeze_clip_true/ (archived, not committed)
  scripts/s_peft_queue.sh, scripts/s_peft_rerun.sh (local temporary scripts)
