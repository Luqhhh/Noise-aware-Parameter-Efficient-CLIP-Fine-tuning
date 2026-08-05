# F1 R3 multiscale-teacher preparation (2026-08-05)

## Objective

Move the platform-positive 128/144/160 attention-localization signal into
training supervision. The earlier R3-wide run merely relaxed thresholds on the
same center/flip teacher and was platform-neutral. This candidate instead uses
two deterministic attention-multiscale views of every official training image
and keeps the R2 checkpoint, optimizer, learning rates, schedule, and fixed
three-epoch policy unchanged.

## Implementation

`aegis_clip.cli.build_teacher_trust` now supports
`--teacher-view attention_multiscale`. For each original and flipped image it:

1. obtains last-block attention and global logits;
2. extracts deterministic 128/144/160 attention crops with top-k 5;
3. fuses global and mean local probabilities with local weight 0.4 and
   temperature 1.5;
4. stores the two view-specific log-probability tensors.

The cache records the full view specification. A legacy center/flip cache is
accepted only for the default mode; attempting to reuse it as a multiscale
cache fails closed. Backward compatibility was verified by reproducing the
historical R3-wide audit exactly: 851 eligible, 840 accepted, and 6,403 total
corrections.

## Exact cache command

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.build_teacher_trust \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --train-csv artifacts/stages/preliminary/final_full_train.csv \
  --base-trust artifacts/trust/selftrain_r1_teacher_v2_relaxed.pt \
  --output artifacts/trust/selftrain_r2_teacher_v3_multiscale_probe.pt \
  --audit-output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/teacher_trust_multiscale_probe_audit.json \
  --teacher-logits-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/teacher_train_multiscale_128_144_160_logits.pt \
  --teacher-view attention_multiscale --local-crop-sizes 128,144,160 \
  --local-top-k 5 --local-weight 0.4 --local-temperature 1.5 \
  --batch-size 128 --num-workers 4 --temperature 1.0 \
  --minimum-confidence 0.75 --minimum-margin 0.45 \
  --maximum-clean-probability 0.60 --admission-clean-probability 0.65 \
  --correction-alpha 0.20 --maximum-class-fraction 0.10
```

- Runtime: approximately 19 minutes
- Samples: 103,218; classes: 500
- Center and flip tensor shapes: `[103218, 500]`
- Canonical paths: 103,218 unique and complete
- All logits finite
- Cache SHA-256:
  `99abea908f15ae38dcb0884ab32fd858a2bd42d2080e282d040a95b2e052c021`

The initial center-teacher thresholds accepted only two samples because the
multiscale cache contains already temperature-flattened fused probabilities.
The disposable 12 MiB probe trust bundle was removed after this diagnosis.

## Threshold analysis and formal trust bundle

Among 8,954 samples satisfying the non-confidence eligibility conditions, the
multiscale confidence median is 0.2359 and its 90th percentile is 0.4399.
The selected thresholds `confidence=0.45, margin=0.15` therefore select a
strict upper-confidence subset on the correct probability scale.

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.build_teacher_trust \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --train-csv artifacts/stages/preliminary/final_full_train.csv \
  --base-trust artifacts/trust/selftrain_r1_teacher_v2_relaxed.pt \
  --output artifacts/trust/selftrain_r2_teacher_v3_multiscale.pt \
  --audit-output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R3_MULTISCALE_FP32/teacher_trust_audit.json \
  --teacher-logits-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/teacher_train_multiscale_128_144_160_logits.pt \
  --teacher-view attention_multiscale --local-crop-sizes 128,144,160 \
  --local-top-k 5 --local-weight 0.4 --local-temperature 1.5 \
  --batch-size 128 --num-workers 4 --temperature 1.0 \
  --minimum-confidence 0.45 --minimum-margin 0.15 \
  --maximum-clean-probability 0.60 --admission-clean-probability 0.65 \
  --correction-alpha 0.25 --maximum-class-fraction 0.08
```

- Two-view agreement: 96,702
- Teacher/noisy-label disagreements: 21,382
- Eligible and accepted additions: 769
- Existing corrections preserved element-for-element: 5,563
- Total corrections: 6,332
- Mean accepted confidence: 0.529909
- Mean accepted margin: 0.364028
- Maximum source/target additions per class: 14 / 10
- Agreement with center teacher prediction: 754 / 769 (98.05%)
- Overlap with R3-wide additions: 360
- New versus R3-wide additions: 409
- Trust SHA-256:
  `6868041cc7b995a3e8e557ae925d1d25160acf23af09202f46911ce92125b30f`

## Training candidate

- Config: `configs/f1_flat_mlp_lora_selftrain_r3_multiscale.yaml`
- Experiment: `F1_FLAT_MLP_LORA_SELFTRAIN_R3_MULTISCALE_FP32`
- Parent: R2 epoch 3 checkpoint SHA-256
  `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- Fixed three epochs, FP32, batch 32
- Head LR `1.5e-7`; MLP-LoRA LR `3e-6`
- Full tests: 268 passed, 8 warnings

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_mlp_lora_selftrain_r3_multiscale.yaml --overwrite
```

## Status

Implementation, cache, trust bundle, and resolved config paths are audited.
Training has not yet started. The current desktop submission remains unchanged.
