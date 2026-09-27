# K05 submission campaign — 2026-09-27

## Decision summary

Use a two-part portfolio for the remaining platform opportunities:

1. Keep `L05_T14_P060` as the protected score anchor. Its package is already platform-validated and must not be rebuilt or overwritten.
2. Spend the exploratory slot only on a format-audited K05 checkpoint that has progressed far enough to provide useful local-to-platform transfer information. Continue training after packaging; do not treat an intermediate submission as the final K05 verdict.

This is intentionally more aggressive than the normal promotion gate, but the risk is bounded: the protected L05 package remains available, the experimental package changes only the model checkpoint, and every generated file is hash-bound and independently checked.

## Repository and source binding

- Team remote: `https://github.com/Luqhhh/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning.git`
- Team `origin/main` after fetch at 2026-09-27 21:26 CST: `b48a0af`
- K05 implementation commit: `2940a14f3f4b9e607ea38668a163608a1a9015fe`
- K05 implementation source branch: `origin/codex/rematch750_search_v5`
- Active local execution branch: `codex/k05_cuda_local_20260927`
- Experiment: `RM_V5_K05_CUDA_LOCAL_R1`, seed 42
- Runtime config SHA-256: `e2b27bd8104f0e002ff57e49d1f73b3937de993a22c0dfb151da30612b4067c5`
- Protocol/data: `rematch750_search_v5`, data version `20260921`
- Model: OpenAI CLIP ViT-B/32, input 384, full fine-tuning, feature-anchor weight 0.5
- Training: 16 epochs, batch 4, gradient accumulation 256, effective batch 1024, evaluation every 2 epochs
- Selection: raw validation macro accuracy, tie-break raw micro accuracy

## Evidence available before the K05 run

| Candidate | Inference | Local macro | Local micro | Platform | Interpretation |
|---|---:|---:|---:|---:|---|
| L05 center | center crop | 75.2497% | 76.2970% | 66.3898% | Fair same-inference local comparator for K05 |
| L05_T14_P060 | flip TTA + temperature 1.4 + prior 0.6 | 75.8265% | 76.6532% | 66.9480% | Protected score anchor; not a fair raw-center comparator |

Protected L05 package hashes:

- `manifest.json`: `3712f796bf8736207c5dc28c38b0b4364248f4ba32d09ffaf4a475a8762ef79b`
- `pred_results.csv`: `51e0efe7178528d23993a44351069d2829b0cd3669c195a77d8d879e66776a75`
- `submission.zip`: `e788f07636b80abb61685335cd8df108803863ea36ffbde8b353f8e8317fcd7f`

## K05 observed learning curve

| Epoch | Raw macro | Raw micro | Predicted classes | Head macro | Medium macro | Tail macro | Decision |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 69.9229% | 70.8535% | 746/750 | 71.0909% | 73.1806% | 65.4973% | Reject for platform; save as curve anchor |

At epoch 2, K05 is 5.327 percentage points below L05 center on macro and 5.443 points below it on micro. The shortfall is too large for a useful platform probe. Its main weakness is the tail segment, not class collapse: 746/750 classes are still predicted.

Epoch-2 snapshot:

- Checkpoint SHA-256: `efe392e5ba033f0e6c04a618a84d224f31fef7deeb8ab93fb378a64b6cbbc55e`
- Binding SHA-256: `663b8b0dcdc7a9c06e682ecab52c1297e97238228e11ee2c859b57811d2736f9`
- Local path: `/home/x28639/projects/Noise-aware-K05-CUDA/outputs/k05_submission_snapshots/epoch02`

## Preregistered decision rule for tonight

- Evaluate epoch 4 first.
- If epoch-4 raw macro is at least 73.5% and raw micro is at least 74.5%, generate the center-only E04 platform package immediately.
- If either threshold is missed, continue to epoch 6 while the projected package-completion time remains before 23:25 CST.
- In all cases require: 37,444 rows, zero corrupt images, labels within the official 750-class mapping, a ZIP containing only the root-level `pred_results.csv`, byte-identical CSV content inside/outside the ZIP, and recorded SHA-256 hashes.
- K05 rematch inference remains global center-crop only: no TTA, local crop, test-time prior fitting, or checkpoint blend. This keeps the platform observation attributable to K05 training rather than a confounded inference recipe.

The thresholds are looser than the final promotion threshold because this submission has a different purpose: estimate the platform transfer of the K05 family while training continues. They are still strict enough to avoid spending a slot on the clearly undertrained epoch-2 model.

## What is still missing

- No K05 platform score yet, so the local-to-platform transfer factor is uncalibrated.
- K05 has only one seed and an incomplete learning curve.
- The full 16-epoch selector is not available tonight.
- The tail-class deficit remains material at epoch 2.
- The current-stage protocol deliberately forbids applying the proven L05 TTA/prior recipe to K05, so an apples-to-apples comparison must use L05 center metrics.
- The platform's score retention behavior has not been assumed; the protected L05 package remains the rollback artifact.

## Strategy after the platform receipt

1. Record submission ID, timestamp, platform period, score, package hash, and whether it became the highest score.
2. Finish K05 to epoch 16 without restarting and retain the best raw-macro checkpoint.
3. Compare platform delta versus L05 center with local raw-center delta. This gives a direct transfer residual and tells us whether 384px + feature anchoring generalizes beyond local validation.
4. If transfer is favorable, prioritize tail recovery while preserving the same inference protocol. If transfer is unfavorable despite local gains, stop model-side expansion and spend later opportunities on independently audited L05 inference variants.
5. Never overwrite a submitted package; every later checkpoint receives a new candidate ID and directory.

## Verification status

- Epoch-2 training/validation evidence: verified from the live run logs and saved checkpoint binding.
- L05 package: locally hash-verified and previously platform-validated.
- K05 platform package: pending a later checkpoint and submission audit.
- K05 platform effectiveness: unverified until the platform receipt is recorded.
