# K05 submission campaign — 2026-09-27/28

## Decision summary

Use both remaining platform opportunities on new information:

1. `K05_E04_CENTER_20260928` has been frozen, inferred, and audited as the first genuinely new package.
2. Resume through the trainer's native full-state path from the last complete epoch-4 checkpoint, then freeze, infer, and audit epoch 6 as `K05_E06_CENTER_20260928`.

`L05_T14_P060` remains the protected score anchor and rollback artifact, but must not be resubmitted because its exact predictions already have a platform receipt. A duplicate would spend a slot without adding evidence.

This is intentionally more aggressive than the normal promotion gate, but the risk is bounded: both candidates change only the checkpoint along one K05 lineage, every generated file is hash-bound and independently checked, and the platform comparison directly measures whether additional K05 training transfers. The two packages may be used together only after confirming that their test prediction vectors differ; a distinct checkpoint alone is not enough.

## Repository and source binding

- Team remote: `https://github.com/Luqhhh/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning.git`
- Team `origin/main` after fetch on 2026-09-28: `37dad13`
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

## Comparison with the team's other current candidates

| Direction | Audited result versus L05 | New package available? | Decision |
|---|---:|---:|---|
| Three-crop TTA | +0.138 pp macro, +0.081 pp micro | No | Below the preregistered promotion gate; useful evidence but not today's package |
| Class-wise vector calibration | -0.212 pp macro, -0.363 pp micro in cross-fit | No | Close; calibration overfit did not transfer to held-out folds |
| RSC continuation | -0.040 pp macro, -0.020 pp micro | No | Close; below the parent after audit |
| Fourier amplitude mixing | 0.000 pp after selector retained epoch 0 | No | Close; augmentation did not beat the parent |
| L05 prior 0.70 probe | local macro +0.031 pp, platform -0.067 pp / -25 correct | Yes, already submitted | Closed; keep prior 0.60 |
| L05 TTA temperature 1.1 probe | local macro -0.081 pp, platform -0.206 pp / -77 correct | Yes, already submitted | Closed; keep temperature 1.4 |
| Stochastic depth | -1.519 pp macro, -1.526 pp micro | Yes, unpromoted | Closed below gate |
| OOF hard filtering | LP complete at 62.6255% macro; full fine-tune still running at last team update | No | Wait for audited full-run result |
| P75 patch readout | GPU training running; first-step gradient audit passed | No | Stronger mechanism candidate, but no checkpoint/validation/package yet |

The comparison leaves K05 as the only independently different, locally running model family with a complete new package now. P75 may be the more promising medium-term mechanism, but at the latest remote update it had no validated checkpoint or package; OOF hard filtering was also still training. Re-submitting unchanged L05 has zero information value. The useful near-term strategy is therefore two audited K05 checkpoints from one controlled trajectory, while keeping P75/OOF behind their own validation gates.

Two existing within-lineage transitions show that local macro deltas have transferred approximately one-for-one to the platform: F05 to L05 center was +0.819 pp locally and +0.919 pp on platform; L05 center to the selected TTA/prior decode was +0.577 pp locally and +0.558 pp on platform. This supports using the local raw-macro gap as a real risk signal. It does not prove the same transfer for the K family, which is precisely the uncertainty the exploratory submission is meant to measure.

Related K-family full-run results also bound expectations: K02 (320px, anchor 0.5) reached 74.7375% macro / 75.7594% micro, and K09 (448px, stronger same-view anchor) reached 75.0306% / 76.0215%. K05 (384px, anchor 0.5) is therefore scientifically plausible but not guaranteed to beat L05; the platform probe is valuable mainly as a cross-family transfer measurement.

## K05 observed learning curve

| Epoch | Raw macro | Raw micro | Predicted classes | Head macro | Medium macro | Tail macro | Decision |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 69.9229% | 70.8535% | 746/750 | 71.0909% | 73.1806% | 65.4973% | Reject for platform; save as curve anchor |
| 4 | 72.6436% | 73.6089% | 746/750 | 73.9456% | 75.3923% | 68.5930% | Package as aggressive information probe; below ordinary promotion gate |

At epoch 2, K05 is 5.327 percentage points below L05 center on macro and 5.443 points below it on micro. The shortfall is too large for a useful platform probe. Its main weakness is the tail segment, not class collapse: 746/750 classes are still predicted.

Epoch 4 gains **+2.7207 pp macro / +2.7554 pp micro** over epoch 2, confirming that K05 is still learning rapidly, but remains **-2.6061 pp macro / -2.6881 pp micro** below L05 center. Its first test package changes **6,919 / 37,444 predictions (18.478261%)** relative to the previously submitted L05 package, so it is new information rather than a duplicate.

Epoch-2 snapshot:

- Checkpoint SHA-256: `efe392e5ba033f0e6c04a618a84d224f31fef7deeb8ab93fb378a64b6cbbc55e`
- Binding SHA-256: `663b8b0dcdc7a9c06e682ecab52c1297e97238228e11ee2c859b57811d2736f9`
- Local path: `/home/x28639/projects/Noise-aware-K05-CUDA/outputs/k05_submission_snapshots/epoch02`

Epoch-4 snapshot and audited package:

- Checkpoint SHA-256: `3a1393e0fb464b90e7722cec90e998061b969ab55231c21bdfe5ee4600b32dd6`
- Binding SHA-256: `3b2bff0f95c6b412c914eaafe64b5864ec9b2b326db3e6fba123f828bb84f96b`
- Prediction CSV SHA-256: `b0692d7efb6b87f2be895d2d8c4c23236004298efa887e20062712094e63a3f1`
- Submission ZIP SHA-256: `008be8da3d0787d4b5418ed5baa75aaf565e39b2b85c1939b53e9afce6bd4e83`
- Submission audit: 37,444/37,444 images, no duplicates, no missing images, labels in `0000`-`0749`, ZIP contains only root-level `pred_results.csv`
- Windows delivery: `C:\Users\28639\Documents\New project 3\submission_packages\2026-09-28_K05_E04_CENTER`

The original Windows-host process stopped during epoch 5 without a logged Python exception, before producing another complete epoch checkpoint. On 2026-09-28 the run was resumed from epoch 4 using the trainer's native restore of model, optimizer, scheduler, AMP scaler, RNG, and data-generator state. Resume helper SHA-256: `3ca59a59f14c3eafdf9c4d8e99ea77e2e50b2eac361a0c4519aebb172b675c4f`. The trainer logged `Resumed from epoch 4`; the partial epoch-5 state was discarded and epoch 5 was rerun. Because the OS process and data-loader workers restarted, this is a full saved-state continuation of the same checkpoint lineage, not a claim of byte-for-byte equivalence to the counterfactual uninterrupted stochastic augmentation stream.

## Preregistered decision rule for today's two new submissions

- Evaluate epoch 4, freeze it, and generate the center-only E04 package regardless of whether it clears the ordinary promotion gate; label it explicitly as an intermediate information probe.
- Continue the native full-state epoch-4 resume to epoch 6, freeze it, and generate a second center-only E06 package. Do not retune or alter the data/inference recipe between the two checkpoints.
- Use 73.5% raw macro and 74.5% raw micro as the interpretation threshold for E04, not as a packaging threshold. Below it, E04 is a learning-curve calibration point rather than a score candidate.
- Prefer E06 for the later submission if its selector improves; retain E04 as the earlier independent platform point. If E06's test labels are identical to E04, do not spend the second slot: continue to the next completed evaluation checkpoint instead.
- In all cases require: 37,444 rows, zero corrupt images, labels within the official 750-class mapping, a ZIP containing only the root-level `pred_results.csv`, byte-identical CSV content inside/outside the ZIP, and recorded SHA-256 hashes.
- K05 rematch inference remains global center-crop only: no TTA, local crop, test-time prior fitting, or checkpoint blend. This keeps the platform observation attributable to K05 training rather than a confounded inference recipe.

Epoch 2 remains rejected because it is clearly undertrained. Epochs 4 and 6 are late enough to measure the post-warmup GCE trajectory, while their paired platform scores estimate the marginal value of two additional epochs without changing any other factor.

## What is still missing

- No K05 platform score yet, so the local-to-platform transfer factor is uncalibrated.
- K05 has only one seed and an incomplete learning curve.
- The full 16-epoch selector is not available for today's submission window.
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
- K05 E04 package: locally verified and structurally audited; platform effectiveness remains unverified.
- K05 E06 package: pending epoch-6 validation, inference, prediction-difference check, and submission audit.
- K05 platform effectiveness: unverified until the platform receipt is recorded.
